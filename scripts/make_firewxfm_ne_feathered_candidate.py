#!/usr/bin/env python3
"""Create a Northeast/coastal-East FireWx-FM calibration candidate.

This script is intentionally light-weight: it does not run model inference and
does not touch production serving artifacts. It applies a smooth, regional
logit adjustment to an existing probability map, with optional USGS same-grid
rasters used only as a soft spatial prior inside the Northeast/coastal-East
repair region.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform as transform_coords
from scipy.ndimage import distance_transform_edt


CHANNEL_NAMES = [
    "t2m",
    "d2m",
    "u10",
    "v10",
    "cape",
    "sp",
    "blh",
    "vis",
    "prate",
    "tp",
    "dynamic_valid",
    "static_valid",
    "fuel_fbfm40",
    "canopy_cover",
    "housing_density",
    "population",
]

REGIONS = {
    "all_lower48": None,
    "west_lon_lt_minus105": (-125.0, -105.0, 24.0, 50.0),
    "central_lon_minus105_to_minus90": (-105.0, -90.0, 24.0, 50.0),
    "east_lon_ge_minus90": (-90.0, -66.0, 24.0, 50.0),
    "seam_west_minus92_to_minus90": (-92.0, -90.0, 24.0, 50.0),
    "seam_east_minus90_to_minus88": (-90.0, -88.0, 24.0, 50.0),
    "southeast": (-88.0, -75.0, 24.0, 37.0),
    "florida": (-88.0, -79.0, 24.0, 31.5),
    "mid_atlantic": (-82.0, -74.0, 32.0, 41.0),
    "northeast": (-82.0, -66.0, 37.0, 50.0),
    "coastal_atlantic": (-82.0, -66.0, 24.0, 45.0),
    "repair_region_bbox": (-84.0, -66.0, 35.5, 50.0),
}

PROB_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


def summarize(values: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"cells": 0}
    out: dict[str, Any] = {
        "cells": int(vals.size),
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "q10": float(np.quantile(vals, 0.10)),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
        "q999": float(np.quantile(vals, 0.999)),
        "max": float(vals.max()),
    }
    for threshold in PROB_THRESHOLDS:
        out[f"frac_ge_{threshold:.2f}"] = float((vals >= threshold).mean())
    return out


def lon_lat_from_tif(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with rasterio.open(path) as src:
        rows = np.arange(src.height)
        cols = np.arange(src.width)
        xs = src.transform.c + (cols + 0.5) * src.transform.a
        ys = src.transform.f + (rows + 0.5) * src.transform.e
        xx, yy = np.meshgrid(xs, ys)
        lon_values, lat_values = transform_coords(
            src.crs,
            "EPSG:4326",
            xx.ravel().tolist(),
            yy.ravel().tolist(),
        )
        lon = np.asarray(lon_values, dtype=np.float32).reshape(src.height, src.width)
        lat = np.asarray(lat_values, dtype=np.float32).reshape(src.height, src.width)
        return lon, lat, str(src.crs)


def region_masks(lon: np.ndarray, lat: np.ndarray, lower48: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, bbox in REGIONS.items():
        if bbox is None:
            masks[name] = lower48.copy()
            continue
        lon_min, lon_max, lat_min, lat_max = bbox
        masks[name] = (
            lower48
            & (lon >= lon_min)
            & (lon < lon_max)
            & (lat >= lat_min)
            & (lat <= lat_max)
        )
    return masks


def rank01(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.float32)
    vals = values[valid]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return out
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(vals.size, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, vals.size, dtype=np.float32)
    out_values = np.zeros(int(valid.sum()), dtype=np.float32)
    finite = np.isfinite(values[valid])
    out_values[finite] = ranks
    out[valid] = out_values
    return out


def load_usgs_prior(paths: list[Path], lower48: np.ndarray) -> np.ndarray | None:
    if not paths:
        return None
    ranks: list[np.ndarray] = []
    for path in paths:
        arr = np.load(path).astype(np.float32)
        if arr.shape != lower48.shape:
            raise ValueError(f"USGS prior {path} shape {arr.shape} != mask {lower48.shape}")
        valid = lower48 & np.isfinite(arr)
        if int(valid.sum()) == 0:
            continue
        ranks.append(rank01(arr, valid))
    if not ranks:
        return None
    return np.mean(np.stack(ranks, axis=0), axis=0).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def feathered_repair_weight(
    lon: np.ndarray,
    lat: np.ndarray,
    lower48: np.ndarray,
    repair_bbox_mask: np.ndarray,
    edge_feather_cells: float,
) -> np.ndarray:
    # Smooth entry from the Mid-Atlantic into the Northeast/coastal East. The
    # hard bbox only bounds the repair area; the effective adjustment is feathered.
    lon_gate = sigmoid((lon + 82.5) / 1.5)
    lat_gate = sigmoid((lat - 37.0) / 1.3)
    raw = (lon_gate * lat_gate).astype(np.float32)
    raw *= repair_bbox_mask.astype(np.float32)

    if edge_feather_cells > 0:
        dist_inside = distance_transform_edt(repair_bbox_mask)
        edge_weight = np.clip(dist_inside / float(edge_feather_cells), 0.0, 1.0).astype(np.float32)
        raw *= edge_weight

    raw[~lower48] = 0.0
    raw[raw < 0.01] = 0.0
    return np.clip(raw, 0.0, 1.0).astype(np.float32)


def logit_adjust(
    probability: np.ndarray,
    weight: np.ndarray,
    max_shift: float,
    usgs_prior: np.ndarray | None,
    prior_floor: float,
    prior_power: float,
    cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(probability.astype(np.float32), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).astype(np.float32)
    if usgs_prior is None:
        prior = np.ones_like(probability, dtype=np.float32)
    else:
        prior = np.clip(usgs_prior.astype(np.float32), 0.0, 1.0)
        prior = np.clip(prior_floor + (1.0 - prior_floor) * np.power(prior, prior_power), 0.0, 1.0)
    shift = (float(max_shift) * weight * prior).astype(np.float32)
    adjusted = (1.0 / (1.0 + np.exp(-(logits + shift)))).astype(np.float32)
    repaired = shift > 0.0
    adjusted[repaired] = np.minimum(adjusted[repaired], np.float32(cap))
    adjusted[~repaired] = probability[~repaired].astype(np.float32)
    return adjusted, shift


def write_probability_tif(path: Path, data: np.ndarray, reference_tif: Path) -> None:
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="float32", nodata=0.0, compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def colorize(prob: np.ndarray, lower48: np.ndarray) -> np.ndarray:
    rgb = np.full(prob.shape + (3,), [244, 244, 239], dtype=np.uint8)
    palette = np.array(
        [
            [45, 103, 70],
            [79, 148, 78],
            [148, 180, 67],
            [219, 183, 70],
            [225, 127, 57],
            [207, 72, 54],
            [123, 28, 36],
        ],
        dtype=np.float32,
    )
    bins = np.array([0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.32], dtype=np.float32)
    valid = lower48 & np.isfinite(prob) & (prob > 0)
    idx = np.clip(np.searchsorted(bins, prob, side="right"), 0, len(palette) - 1)
    rgb[valid] = palette[idx[valid]].astype(np.uint8)
    return rgb


def write_quicklook(path: Path, baseline: np.ndarray, candidate: np.ndarray, lower48: np.ndarray) -> None:
    left = colorize(baseline, lower48)
    right = colorize(candidate, lower48)
    diff = np.clip((candidate - baseline) / 0.08, 0.0, 1.0)
    diff_rgb = np.full(left.shape, [244, 244, 239], dtype=np.uint8)
    diff_rgb[lower48] = np.stack(
        [
            np.rint(255 * diff[lower48]),
            np.rint(255 * (1.0 - diff[lower48])),
            np.full(int(lower48.sum()), 80),
        ],
        axis=1,
    ).astype(np.uint8)
    separator = np.full((left.shape[0], 4, 3), 30, dtype=np.uint8)
    canvas = np.concatenate([left, separator, right, separator, diff_rgb], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def paired_seam_summary(prob: np.ndarray, lon: np.ndarray, lower48: np.ndarray, seam_lon: float) -> dict[str, Any]:
    col_lon = np.nanmedian(np.where(lower48, lon, np.nan), axis=0)
    col = int(np.nanargmin(np.abs(col_lon - seam_lon)))
    if col <= 0 or col >= prob.shape[1] - 1:
        return {"column": col, "error": "seam column at edge"}
    mask = lower48[:, col - 1] & lower48[:, col] & np.isfinite(prob[:, col - 1]) & np.isfinite(prob[:, col])
    diff = prob[:, col] - prob[:, col - 1]
    vals = diff[mask]
    abs_vals = np.abs(vals)
    if vals.size == 0:
        return {"column": col, "cells": 0}
    return {
        "column": col,
        "median_lon": float(col_lon[col]),
        "cells": int(vals.size),
        "east_minus_west_mean": float(vals.mean()),
        "abs_diff_q50": float(np.quantile(abs_vals, 0.50)),
        "abs_diff_q90": float(np.quantile(abs_vals, 0.90)),
        "abs_diff_q95": float(np.quantile(abs_vals, 0.95)),
        "abs_diff_max": float(abs_vals.max()),
    }


def channel_summaries(stack_path: Path | None, masks: dict[str, np.ndarray]) -> dict[str, Any] | None:
    if stack_path is None:
        return None
    stack = np.load(stack_path, mmap_mode="r")
    selected = [4, 7, 10, 11, 12, 13]
    out: dict[str, Any] = {}
    for idx in selected:
        name = CHANNEL_NAMES[idx]
        out[name] = {
            region: summarize(np.asarray(stack[idx][mask], dtype=np.float32))
            for region, mask in masks.items()
            if region in {"west_lon_lt_minus105", "central_lon_minus105_to_minus90", "southeast", "mid_atlantic", "northeast"}
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-npy", type=Path, required=True)
    parser.add_argument("--reference-tif", type=Path, required=True)
    parser.add_argument("--lower48-mask-npy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stack-npy", type=Path)
    parser.add_argument("--old-npy", type=Path)
    parser.add_argument("--usgs-prior-npy", type=Path, action="append", default=[])
    parser.add_argument("--max-shift", type=float, default=0.75)
    parser.add_argument("--prior-floor", type=float, default=0.35)
    parser.add_argument("--prior-power", type=float, default=0.65)
    parser.add_argument("--edge-feather-cells", type=float, default=8.0)
    parser.add_argument("--cap", type=float, default=0.36)
    parser.add_argument("--prefix", default="firewxfm_20260707_t12_staticfix_ne_feathered")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = np.load(args.baseline_npy).astype(np.float32)
    lower48 = np.load(args.lower48_mask_npy).astype(bool)
    if baseline.shape != lower48.shape:
        raise ValueError(f"baseline {baseline.shape} != lower48 mask {lower48.shape}")
    lon, lat, crs = lon_lat_from_tif(args.reference_tif)
    if lon.shape != baseline.shape:
        raise ValueError(f"reference grid {lon.shape} != baseline {baseline.shape}")

    masks = region_masks(lon, lat, lower48)
    usgs_prior = load_usgs_prior(args.usgs_prior_npy, lower48)
    repair_weight = feathered_repair_weight(
        lon=lon,
        lat=lat,
        lower48=lower48,
        repair_bbox_mask=masks["repair_region_bbox"],
        edge_feather_cells=float(args.edge_feather_cells),
    )
    candidate, shift = logit_adjust(
        probability=baseline,
        weight=repair_weight,
        max_shift=float(args.max_shift),
        usgs_prior=usgs_prior,
        prior_floor=float(args.prior_floor),
        prior_power=float(args.prior_power),
        cap=float(args.cap),
    )
    candidate[~lower48] = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.output_dir / f"{args.prefix}_probability_lower48.npy"
    tif_path = args.output_dir / f"{args.prefix}_probability_lower48.tif"
    shift_path = args.output_dir / f"{args.prefix}_logit_shift.npy"
    summary_path = args.output_dir / f"{args.prefix}_audit.json"
    quicklook_path = args.output_dir / f"{args.prefix}_quicklook.png"
    np.save(npy_path, candidate.astype(np.float32))
    np.save(shift_path, shift.astype(np.float32))
    write_probability_tif(tif_path, candidate, args.reference_tif)
    write_quicklook(quicklook_path, baseline, candidate, lower48)

    old_summary = None
    if args.old_npy is not None and args.old_npy.exists():
        old = np.load(args.old_npy).astype(np.float32)
        old_summary = {name: summarize(old[mask]) for name, mask in masks.items()}

    delta = candidate - baseline
    summary = {
        "status": "ok",
        "diagnosis": {
            "current_ground_truth_note": "No local 2026-07 FIRMS target was found during this pass; USGS rasters are used only as a same-day spatial prior/sanity reference, not as active-fire ground truth.",
            "model_side_interpretation": "The staticfix-balanced model has repaired the previous broad eastern suppression, especially Southeast/Florida. The remaining weak visual response is concentrated in the Northeast/coastal-East, consistent with regional calibration/generalization rather than a frontend, mask, projection, or lon=-90 tile artifact.",
            "production_safety": "Outputs are standalone files and are not deployed or copied into the website pipeline.",
        },
        "inputs": {
            "baseline_npy": str(args.baseline_npy),
            "old_npy": str(args.old_npy) if args.old_npy else None,
            "reference_tif": str(args.reference_tif),
            "lower48_mask_npy": str(args.lower48_mask_npy),
            "stack_npy": str(args.stack_npy) if args.stack_npy else None,
            "usgs_prior_npy": [str(p) for p in args.usgs_prior_npy],
            "crs": crs,
            "shape": [int(v) for v in baseline.shape],
        },
        "method": {
            "kind": "feathered Northeast/coastal-East logit calibration candidate",
            "max_shift": float(args.max_shift),
            "prior_floor": float(args.prior_floor),
            "prior_power": float(args.prior_power),
            "edge_feather_cells": float(args.edge_feather_cells),
            "cap": float(args.cap),
            "repair_bbox": REGIONS["repair_region_bbox"],
            "hard_unchanged_regions": "Central and West are outside the repair mask except for numerical zero shift.",
        },
        "outputs": {
            "candidate_npy": str(npy_path),
            "candidate_tif": str(tif_path),
            "shift_npy": str(shift_path),
            "quicklook_png": str(quicklook_path),
            "summary_json": str(summary_path),
        },
        "baseline_regions": {name: summarize(baseline[mask]) for name, mask in masks.items()},
        "candidate_regions": {name: summarize(candidate[mask]) for name, mask in masks.items()},
        "delta_regions": {name: summarize(delta[mask]) for name, mask in masks.items()},
        "shift_regions": {name: summarize(shift[mask]) for name, mask in masks.items()},
        "old_regions": old_summary,
        "seam_checks": {
            "baseline_lon_minus90": paired_seam_summary(baseline, lon, lower48, -90.0),
            "candidate_lon_minus90": paired_seam_summary(candidate, lon, lower48, -90.0),
            "baseline_lon_minus84": paired_seam_summary(baseline, lon, lower48, -84.0),
            "candidate_lon_minus84": paired_seam_summary(candidate, lon, lower48, -84.0),
        },
        "input_channel_regions": channel_summaries(args.stack_npy, masks),
    }
    if usgs_prior is not None:
        summary["usgs_prior_regions"] = {name: summarize(usgs_prior[mask]) for name, mask in masks.items()}

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_tif": str(tif_path),
                "summary_json": str(summary_path),
                "quicklook_png": str(quicklook_path),
                "northeast_baseline_frac_ge_0p10": summary["baseline_regions"]["northeast"]["frac_ge_0.10"],
                "northeast_candidate_frac_ge_0p10": summary["candidate_regions"]["northeast"]["frac_ge_0.10"],
                "central_mean_abs_delta": summary["delta_regions"]["central_lon_minus105_to_minus90"]["mean"],
                "west_mean_abs_delta": summary["delta_regions"]["west_lon_lt_minus105"]["mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
