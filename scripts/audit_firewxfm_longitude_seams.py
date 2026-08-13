#!/usr/bin/env python3
"""Audit longitude-band discontinuities in a FireWx-FM probability GeoTIFF.

The audit is intentionally source-product focused. It reads the probability
GeoTIFF before tiling and reports broad longitude-band summaries, adjacent-band
jumps, and optional input-stack channel summaries. This catches central-US seams
that would otherwise be hidden by frontend smoothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import transform as transform_coords


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

DEFAULT_BANDS = [
    (-125.0, -115.0),
    (-115.0, -110.0),
    (-110.0, -105.0),
    (-105.0, -100.0),
    (-100.0, -95.0),
    (-95.0, -90.0),
    (-90.0, -85.0),
    (-85.0, -75.0),
    (-75.0, -66.0),
]

DEFAULT_CHANNELS = [4, 7, 10, 11, 12, 13]
PROBABILITY_THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20)


def parse_bands(text: str | None) -> list[tuple[float, float]]:
    if not text:
        return DEFAULT_BANDS
    bands: list[tuple[float, float]] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        lo, hi = item.split(":")
        bands.append((float(lo), float(hi)))
    return bands


def parse_channels(text: str | None) -> list[int]:
    if not text:
        return DEFAULT_CHANNELS
    return [int(v) for v in text.split(",") if v.strip()]


def band_label(band: tuple[float, float]) -> str:
    lo, hi = band
    return f"lon_{lo:g}_to_{hi:g}"


def summarize(values: np.ndarray, *, probability: bool = False) -> dict[str, Any]:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"cells": 0}
    out: dict[str, Any] = {
        "cells": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "q10": float(np.quantile(vals, 0.10)),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
        "max": float(vals.max()),
    }
    if probability:
        for threshold in PROBABILITY_THRESHOLDS:
            out[f"frac_ge_{threshold:.2f}"] = float((vals >= threshold).mean())
    else:
        out["frac_zero"] = float((vals == 0.0).mean())
        out["frac_le_0p25"] = float((vals <= 0.25).mean())
    return out


def load_lon_lat(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
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


def read_probability(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def read_stack(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        stack = np.load(path, mmap_mode="r")
        if stack.ndim != 3:
            raise ValueError(f"Expected [C,H,W] stack in {path}, got {stack.shape}")
        return stack
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def adjacent_band_jumps(band_summaries: dict[str, Any]) -> list[dict[str, Any]]:
    labels = list(band_summaries)
    jumps: list[dict[str, Any]] = []
    for left, right in zip(labels[:-1], labels[1:]):
        a = band_summaries[left].get("probability", {})
        b = band_summaries[right].get("probability", {})
        if not a.get("cells") or not b.get("cells"):
            continue
        jumps.append(
            {
                "left_band": left,
                "right_band": right,
                "mean_delta_right_minus_left": float(b["mean"] - a["mean"]),
                "q90_delta_right_minus_left": float(b["q90"] - a["q90"]),
                "q95_delta_right_minus_left": float(b["q95"] - a["q95"]),
                "frac_ge_0p10_delta_right_minus_left": float(b["frac_ge_0.10"] - a["frac_ge_0.10"]),
                "mean_ratio_right_over_left": float(b["mean"] / max(a["mean"], 1e-8)),
                "q90_ratio_right_over_left": float(b["q90"] / max(a["q90"], 1e-8)),
            }
        )
    return jumps


def evaluate_central_seam(jumps: list[dict[str, Any]], max_abs_mean_delta: float, min_mean_ratio: float) -> dict[str, Any]:
    central_pairs = {
        ("lon_-110_to_-105", "lon_-105_to_-100"),
        ("lon_-105_to_-100", "lon_-100_to_-95"),
    }
    flagged = []
    for jump in jumps:
        pair = (jump["left_band"], jump["right_band"])
        if pair not in central_pairs:
            continue
        mean_delta = abs(float(jump["mean_delta_right_minus_left"]))
        ratio = float(jump["mean_ratio_right_over_left"])
        inverse_ratio = 1.0 / max(ratio, 1e-8)
        if mean_delta > max_abs_mean_delta or min(ratio, inverse_ratio) < min_mean_ratio:
            flagged.append(jump)
    return {
        "central_pairs_checked": [list(v) for v in sorted(central_pairs)],
        "max_abs_mean_delta": float(max_abs_mean_delta),
        "min_mean_ratio": float(min_mean_ratio),
        "flagged_pairs": flagged,
        "pass": len(flagged) == 0,
        "note": "Pass means no large adjacent-band probability jump in the two Rockies/Great-Plains seam pairs.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability-tif", type=Path, required=True)
    parser.add_argument("--lower48-mask-npy", type=Path, required=True)
    parser.add_argument("--input-stack", type=Path, help="Optional [16,H,W] NPY or 16-band GeoTIFF input stack.")
    parser.add_argument("--bands", help="Comma list of lon_min:lon_max bands. Defaults to CONUS bands.")
    parser.add_argument("--channels", help="Comma list of stack channel indices to summarize.")
    parser.add_argument("--max-abs-central-mean-delta", type=float, default=0.018)
    parser.add_argument("--min-central-mean-ratio", type=float, default=0.55)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    probability = read_probability(args.probability_tif)
    lower48 = np.load(args.lower48_mask_npy).astype(bool)
    if probability.shape != lower48.shape:
        raise ValueError(f"probability {probability.shape} != mask {lower48.shape}")
    lon, lat, crs = load_lon_lat(args.probability_tif)
    if lon.shape != probability.shape:
        raise ValueError(f"lon/lat grid {lon.shape} != probability {probability.shape}")

    stack = read_stack(args.input_stack) if args.input_stack else None
    if stack is not None and tuple(stack.shape[1:]) != tuple(probability.shape):
        raise ValueError(f"input stack grid {stack.shape[1:]} != probability {probability.shape}")
    channels = parse_channels(args.channels)
    bands = parse_bands(args.bands)

    band_summaries: dict[str, Any] = {}
    for band in bands:
        lo, hi = band
        mask = lower48 & (lon >= lo) & (lon < hi)
        label = band_label(band)
        entry: dict[str, Any] = {
            "longitude_range": [float(lo), float(hi)],
            "cells": int(mask.sum()),
            "probability": summarize(probability[mask], probability=True),
        }
        if stack is not None:
            entry["input_channels"] = {}
            for channel in channels:
                if not 0 <= channel < stack.shape[0]:
                    raise ValueError(f"channel {channel} outside stack channel count {stack.shape[0]}")
                name = CHANNEL_NAMES[channel] if channel < len(CHANNEL_NAMES) else f"channel_{channel}"
                entry["input_channels"][name] = summarize(np.asarray(stack[channel][mask], dtype=np.float32))
        band_summaries[label] = entry

    jumps = adjacent_band_jumps(band_summaries)
    result = {
        "probability_tif": str(args.probability_tif),
        "lower48_mask_npy": str(args.lower48_mask_npy),
        "input_stack": str(args.input_stack) if args.input_stack else None,
        "crs": crs,
        "shape": [int(v) for v in probability.shape],
        "lower48_cells": int(lower48.sum()),
        "bands": band_summaries,
        "adjacent_band_jumps": jumps,
        "central_seam_check": evaluate_central_seam(
            jumps,
            max_abs_mean_delta=float(args.max_abs_central_mean_delta),
            min_mean_ratio=float(args.min_central_mean_ratio),
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.output_json), "central_seam_pass": result["central_seam_check"]["pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
