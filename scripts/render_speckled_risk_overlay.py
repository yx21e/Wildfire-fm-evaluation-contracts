#!/usr/bin/env python3
"""Render a discrete speckled wildfire-risk overlay preview.

This script intentionally changes the *spatial rendering* of the visual preview:
it does not blur or interpolate the probability raster into a smooth heatmap.
Each input grid cell is rendered as a small categorical mark with deterministic
cell-level texture, fragmented high-risk pockets, and a green-dominant palette.
The quantitative probability raster is not modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin


PALETTE = np.array(
    [
        [42, 104, 62],    # very low: muted green
        [65, 141, 70],    # low: green
        [104, 170, 77],   # low-moderate: light green
        [194, 201, 78],   # moderate: yellow-green
        [235, 177, 67],   # elevated: orange
        [224, 92, 61],    # high: red-orange
        [125, 31, 31],    # extreme: dark red
    ],
    dtype=np.float32,
)

BACKGROUND = np.array([244, 244, 239], dtype=np.float32)


def empirical_rank(values: np.ndarray) -> np.ndarray:
    """Return normalized empirical ranks for a one-dimensional array."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float32)
    if values.shape[0] == 1:
        ranks[0] = 0.5
        return ranks
    ranks[order] = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    return ranks


def longitude_band_rank(arr: np.ndarray, valid: np.ndarray, bands: int = 18) -> np.ndarray:
    """Compute local ranks in vertical geographic bands without smoothing."""
    out = np.zeros(arr.shape, dtype=np.float32)
    h, w = arr.shape
    edges = np.linspace(0, w, bands + 1, dtype=int)
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        band_valid = valid[:, left:right]
        if not np.any(band_valid):
            continue
        vals = arr[:, left:right][band_valid]
        ranks = empirical_rank(vals)
        tmp = out[:, left:right]
        tmp[band_valid] = ranks
        out[:, left:right] = tmp
    return out


def shifted(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift an array with zero fill instead of wraparound."""
    out = np.zeros_like(mask, dtype=np.float32)
    src_y0 = max(0, -dy)
    src_y1 = mask.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = mask.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = mask.shape[0] - max(0, -dy)
    dst_x0 = max(0, dx)
    dst_x1 = mask.shape[1] - max(0, -dx)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


def build_display_score(probability: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    """Build a textured display score while preserving the source raster grid."""
    values = probability[valid]
    global_rank = np.zeros(probability.shape, dtype=np.float32)
    global_rank[valid] = empirical_rank(values)
    local_rank = longitude_band_rank(probability, valid)

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.075, 0.075, size=probability.shape).astype(np.float32)

    # Deterministic micro-clusters and short streaks: these fragment large areas
    # without Gaussian smoothing or radial kernels.
    seed_probability = 0.0035 + 0.018 * np.square(global_rank)
    cluster_seed = (rng.random(probability.shape) < seed_probability) & valid
    cluster = np.zeros(probability.shape, dtype=np.float32)
    for dy, dx, weight in [
        (0, 0, 0.10),
        (1, 0, 0.055),
        (-1, 0, 0.045),
        (0, 1, 0.055),
        (0, -1, 0.045),
        (1, 1, 0.030),
        (-1, 1, 0.025),
        (2, 0, 0.020),
        (0, 2, 0.020),
    ]:
        cluster += shifted(cluster_seed.astype(np.float32), dy, dx) * weight

    streak_seed = (rng.random(probability.shape) < (0.0015 + 0.0075 * global_rank)) & valid
    streak = np.zeros(probability.shape, dtype=np.float32)
    for dy, dx, weight in [(0, 0, 0.050), (0, 1, 0.035), (0, 2, 0.020), (1, 0, 0.025)]:
        streak += shifted(streak_seed.astype(np.float32), dy, dx) * weight

    score = 0.64 * global_rank + 0.28 * local_rank + jitter + cluster + streak
    score = np.clip(score, 0.0, 1.0)
    score[~valid] = 0.0
    return score.astype(np.float32)


def categorize(score: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    """Convert display scores to fragmented categorical risk levels."""
    bins = np.array([0.42, 0.64, 0.78, 0.885, 0.955, 0.988], dtype=np.float32)
    category = np.digitize(score, bins).astype(np.uint8)
    category[~valid] = 255

    rng = np.random.default_rng(seed + 17)
    texture = rng.random(score.shape)

    # Fragment warm colors so the West does not become one giant continuous blob.
    high = (category >= 5) & valid
    category[high & (texture > 0.62 + 0.28 * score)] = 4
    elevated = (category == 4) & valid
    category[elevated & (texture < 0.22)] = 3
    moderate = (category == 3) & valid
    category[moderate & (texture < 0.16)] = 2

    # Natural sparse gaps in the lowest-risk cells, not in high-risk pockets.
    very_low = (category == 0) & valid
    low = (category == 1) & valid
    category[very_low & (texture < 0.020)] = 255
    category[low & (texture < 0.006)] = 255
    return category


def render_rgb(category: np.ndarray) -> np.ndarray:
    """Render categorical cells with opacity-like blending on a light background."""
    rgb = np.broadcast_to(BACKGROUND, category.shape + (3,)).copy()
    valid = category != 255
    colors = PALETTE[np.clip(category[valid], 0, len(PALETTE) - 1)]
    alpha = np.array([0.56, 0.58, 0.61, 0.66, 0.72, 0.78, 0.84], dtype=np.float32)
    a = alpha[np.clip(category[valid], 0, len(alpha) - 1)][:, None]
    rgb[valid] = colors * a + BACKGROUND * (1.0 - a)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def save_geotiff_like(rgb: np.ndarray, template_tif: Path, output_tif: Path) -> None:
    """Write an RGB TIFF while preserving core GeoTIFF tags from a template."""
    template = Image.open(template_tif)
    image = Image.fromarray(rgb)
    if image.size != template.size:
        raise ValueError(f"output size {image.size} does not match template {template.size}")

    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    for code in [33550, 33922, 34735, 34736, 34737, 42112, 42113]:
        if code in template.tag_v2:
            ifd[code] = template.tag_v2[code]
    image.save(output_tif, format="TIFF", compression="tiff_adobe_deflate", tiffinfo=ifd)


def write_summary(path: Path, score: np.ndarray, category: np.ndarray, valid: np.ndarray, seed: int) -> None:
    counts = {}
    labels = [
        "very_low_green",
        "low_green",
        "low_moderate_green",
        "moderate_yellow_green",
        "elevated_orange",
        "high_red_orange",
        "extreme_dark_red",
    ]
    total = int(np.count_nonzero(valid))
    for idx, label in enumerate(labels):
        count = int(np.count_nonzero((category == idx) & valid))
        counts[label] = {"cells": count, "fraction_of_valid": count / total if total else 0.0}
    gaps = int(np.count_nonzero((category == 255) & valid))
    summary = {
        "renderer": "discrete_speckled_risk_overlay",
        "seed": seed,
        "spatial_method": "cell-level categorical rendering with deterministic jitter, micro-clusters, short streaks, fragmented warm classes, and no Gaussian blur/interpolation",
        "palette": labels,
        "valid_cells": total,
        "within_mask_visual_gaps": {"cells": gaps, "fraction_of_valid": gaps / total if total else 0.0},
        "category_counts": counts,
        "display_score_quantiles": {
            "q05": float(np.quantile(score[valid], 0.05)),
            "q50": float(np.quantile(score[valid], 0.50)),
            "q95": float(np.quantile(score[valid], 0.95)),
            "q99": float(np.quantile(score[valid], 0.99)),
        },
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability-npy", type=Path, required=True)
    parser.add_argument("--mask-npy", type=Path, required=True)
    parser.add_argument("--template-tif", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-tif", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--seed", type=int, default=20260709)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probability = np.load(args.probability_npy).astype(np.float32)
    mask = np.load(args.mask_npy).astype(bool)
    valid = np.isfinite(probability) & mask & (probability > 0)
    if probability.shape != mask.shape:
        raise ValueError(f"probability shape {probability.shape} != mask shape {mask.shape}")
    if not np.any(valid):
        raise ValueError("no valid cells to render")

    score = build_display_score(probability, valid, seed=args.seed)
    category = categorize(score, valid, seed=args.seed)
    rgb = render_rgb(category)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_tif.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output_png)
    save_geotiff_like(rgb, args.template_tif, args.output_tif)

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        write_summary(args.summary_json, score, category, valid, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
