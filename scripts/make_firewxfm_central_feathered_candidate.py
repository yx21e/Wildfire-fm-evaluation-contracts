#!/usr/bin/env python3
"""Create a source-level central-US seam calibration candidate.

This script does not smooth tiles or alter the frontend. It applies a smooth
longitude-gated logit adjustment directly to the FireWx-FM probability raster
for the Rockies/Great-Plains transition where the raw candidate has a visible
low-probability trough. Cell-level heterogeneity is preserved because every
cell keeps its relative logit structure inside the calibrated region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import transform as transform_coords


PROB_THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20)


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
        "max": float(vals.max()),
    }
    for threshold in PROB_THRESHOLDS:
        out[f"frac_ge_{threshold:.2f}"] = float((vals >= threshold).mean())
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def load_probability(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def load_lon_lat(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
    return lon, lat


def feather_weight(
    lon: np.ndarray,
    lat: np.ndarray,
    lower48: np.ndarray,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    peak_lon: float,
    sigma_lon: float,
    edge_width_deg: float,
) -> np.ndarray:
    bbox = lower48 & (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    center = np.exp(-0.5 * np.square((lon - peak_lon) / sigma_lon)).astype(np.float32)
    west_edge = sigmoid((lon - lon_min) / edge_width_deg)
    east_edge = sigmoid((lon_max - lon) / edge_width_deg)
    south_edge = sigmoid((lat - lat_min) / edge_width_deg)
    north_edge = sigmoid((lat_max - lat) / edge_width_deg)
    weight = center * west_edge * east_edge * south_edge * north_edge
    weight[~bbox] = 0.0
    weight[weight < 0.01] = 0.0
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def apply_logit_shift(prob: np.ndarray, weight: np.ndarray, shift: float, cap: float) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(prob.astype(np.float32), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).astype(np.float32)
    applied_shift = (float(shift) * weight).astype(np.float32)
    out = (1.0 / (1.0 + np.exp(-(logits + applied_shift)))).astype(np.float32)
    changed = applied_shift > 0.0
    out[changed] = np.minimum(out[changed], np.float32(cap))
    out[~changed] = prob[~changed]
    return out, applied_shift


def write_tif(path: Path, arr: np.ndarray, reference_tif: Path) -> None:
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="float32", nodata=0.0, compress="deflate", predictor=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def region_mask(lon: np.ndarray, lat: np.ndarray, lower48: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    lo, hi, south, north = bbox
    return lower48 & (lon >= lo) & (lon < hi) & (lat >= south) & (lat <= north)


def diff_summary(candidate: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    diff = np.abs(candidate[mask] - baseline[mask]).astype(np.float32)
    if diff.size == 0:
        return {"cells": 0}
    return {
        "cells": int(diff.size),
        "mean_abs_diff": float(diff.mean()),
        "q90_abs_diff": float(np.quantile(diff, 0.90)),
        "q99_abs_diff": float(np.quantile(diff, 0.99)),
        "max_abs_diff": float(diff.max()),
        "changed_fraction_gt_1e-6": float((diff > 1e-6).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tif", type=Path, required=True)
    parser.add_argument("--lower48-mask-npy", type=Path, required=True)
    parser.add_argument("--output-tif", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--shift", type=float, default=0.95)
    parser.add_argument("--cap", type=float, default=0.28)
    parser.add_argument("--lon-min", type=float, default=-111.0)
    parser.add_argument("--lon-max", type=float, default=-97.0)
    parser.add_argument("--lat-min", type=float, default=25.0)
    parser.add_argument("--lat-max", type=float, default=50.0)
    parser.add_argument("--peak-lon", type=float, default=-102.8)
    parser.add_argument("--sigma-lon", type=float, default=3.0)
    parser.add_argument("--edge-width-deg", type=float, default=1.2)
    args = parser.parse_args()

    if not 0.0 < args.cap <= 1.0:
        raise ValueError("--cap must be in (0, 1].")
    baseline, _ = load_probability(args.baseline_tif)
    lower48 = np.load(args.lower48_mask_npy).astype(bool)
    if baseline.shape != lower48.shape:
        raise ValueError(f"baseline {baseline.shape} != mask {lower48.shape}")
    lon, lat = load_lon_lat(args.baseline_tif)
    weight = feather_weight(
        lon,
        lat,
        lower48,
        lon_min=float(args.lon_min),
        lon_max=float(args.lon_max),
        lat_min=float(args.lat_min),
        lat_max=float(args.lat_max),
        peak_lon=float(args.peak_lon),
        sigma_lon=float(args.sigma_lon),
        edge_width_deg=float(args.edge_width_deg),
    )
    candidate, applied_shift = apply_logit_shift(
        baseline,
        weight,
        shift=float(args.shift),
        cap=float(args.cap),
    )
    candidate[~lower48] = 0.0
    write_tif(args.output_tif, candidate, args.baseline_tif)

    bboxes = {
        "all_lower48": (-125.0, -66.0, 24.0, 50.0),
        "west_lon_lt_minus111": (-125.0, -111.0, 24.0, 50.0),
        "central_calibration_region": (float(args.lon_min), float(args.lon_max), float(args.lat_min), float(args.lat_max)),
        "rockies_west_band_minus110_to_minus105": (-110.0, -105.0, 24.0, 50.0),
        "central_low_band_minus105_to_minus100": (-105.0, -100.0, 24.0, 50.0),
        "plains_east_band_minus100_to_minus95": (-100.0, -95.0, 24.0, 50.0),
        "outside_calibration_region": (-125.0, -66.0, 24.0, 50.0),
    }
    summaries: dict[str, Any] = {}
    calibration_mask = weight > 0.0
    for name, bbox in bboxes.items():
        mask = region_mask(lon, lat, lower48, bbox)
        if name == "outside_calibration_region":
            mask = lower48 & ~calibration_mask
        summaries[name] = {
            "baseline": summarize(baseline[mask]),
            "candidate": summarize(candidate[mask]),
            "diff": diff_summary(candidate, baseline, mask),
        }

    summary = {
        "baseline_tif": str(args.baseline_tif),
        "output_tif": str(args.output_tif),
        "lower48_mask_npy": str(args.lower48_mask_npy),
        "shape": [int(v) for v in baseline.shape],
        "method": "source_probability_logit_shift_with_smooth_lon_lat_gate",
        "parameters": {
            "shift": float(args.shift),
            "cap": float(args.cap),
            "lon_min": float(args.lon_min),
            "lon_max": float(args.lon_max),
            "lat_min": float(args.lat_min),
            "lat_max": float(args.lat_max),
            "peak_lon": float(args.peak_lon),
            "sigma_lon": float(args.sigma_lon),
            "edge_width_deg": float(args.edge_width_deg),
        },
        "calibration_cells": int(calibration_mask.sum()),
        "calibration_weight": summarize(weight[calibration_mask]),
        "applied_logit_shift": summarize(applied_shift[calibration_mask]),
        "regions": summaries,
        "note": "This is a source-probability calibration candidate, not frontend/tile smoothing.",
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.output_tif), "summary": str(args.summary_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
