from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform, transform_bounds
from sklearn.neighbors import NearestNeighbors

from cache_helpers import ensure_dirs, resampling_from_name, split_for_timestamp


HRRR_RE = re.compile(r"hrrr\.(\d{8})\.t(\d{2})z\.wrfsfcf00\.grib2$")

VAR_GROUPS = {
    "level2": {
        "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 2},
        "vars": ["t2m", "d2m", "sh2", "r2", "pt"],
    },
    "level10": {
        "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10},
        "vars": ["u10", "v10", "max_10si"],
    },
    "height_0_3000": {
        "filter_by_keys": {"typeOfLevel": "heightAboveGroundLayer", "bottomLevel": 0, "topLevel": 3000},
        "vars": ["cape"],
    },
    "surface_instant": {
        "filter_by_keys": {"typeOfLevel": "surface", "stepType": "instant"},
        "vars": ["sp", "blh", "vis", "prate", "gust", "t"],
    },
    "surface_accum": {
        "filter_by_keys": {"typeOfLevel": "surface", "stepType": "accum"},
        "vars": ["tp", "ssrun", "bgrun"],
    },
}


@dataclass
class SampleRecord:
    issue_ts: pd.Timestamp
    target_ts: pd.Timestamp
    hrrr_path: Path
    label_path: Path


@dataclass
class RegionalContext:
    config: Dict[str, object]
    cache_root: Path
    firms_dir: Path
    hrrr_roots: List[Path]
    static_layers: Dict[str, Dict[str, str]]
    sample_dir: Path
    static_dir: Path
    manifest_dir: Path
    split_dir: Path


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_context(config: Dict[str, object]) -> RegionalContext:
    cache_root = Path(config["cache_root"])
    sample_dir = cache_root / "inputs"
    static_dir = cache_root / "static"
    manifest_dir = cache_root / "manifests"
    split_dir = cache_root / "splits"
    ensure_dirs([sample_dir, static_dir, manifest_dir, split_dir])
    return RegionalContext(
        config=config,
        cache_root=cache_root,
        firms_dir=Path(config["firms_dir"]),
        hrrr_roots=[Path(p) for p in config["hrrr_roots"]],
        static_layers=config["static_layers"],
        sample_dir=sample_dir,
        static_dir=static_dir,
        manifest_dir=manifest_dir,
        split_dir=split_dir,
    )


def build_projected_grid(bounds: Dict[str, float], target_crs: str, resolution_m: float) -> Tuple[np.ndarray, np.ndarray]:
    west, south, east, north = transform_bounds(
        "EPSG:4326",
        target_crs,
        float(bounds["lon_min"]),
        float(bounds["lat_min"]),
        float(bounds["lon_max"]),
        float(bounds["lat_max"]),
        densify_pts=21,
    )
    res = float(resolution_m)
    x_asc = np.arange(west + 0.5 * res, east, res, dtype=np.float32)
    y_asc = np.arange(south + 0.5 * res, north, res, dtype=np.float32)
    if x_asc.size == 0 or y_asc.size == 0:
        raise RuntimeError(f"Empty projected grid for resolution_m={resolution_m}")
    return y_asc[::-1].copy(), x_asc.copy()


def projected_transform(y_desc: np.ndarray, x_asc: np.ndarray) -> rasterio.Affine:
    res_y = float(np.median(np.abs(np.diff(y_desc))))
    res_x = float(np.median(np.diff(x_asc)))
    west = float(x_asc[0] - 0.5 * res_x)
    east = float(x_asc[-1] + 0.5 * res_x)
    south = float(y_desc[-1] - 0.5 * res_y)
    north = float(y_desc[0] + 0.5 * res_y)
    return from_bounds(west, south, east, north, len(x_asc), len(y_desc))


def projected_edges(values_asc: np.ndarray) -> np.ndarray:
    step = float(np.median(np.diff(values_asc)))
    edges = np.empty(values_asc.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (values_asc[:-1] + values_asc[1:])
    edges[0] = values_asc[0] - 0.5 * step
    edges[-1] = values_asc[-1] + 0.5 * step
    return edges


def list_hrrr_issue_paths(roots: List[Path]) -> Dict[pd.Timestamp, Path]:
    out: Dict[pd.Timestamp, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.grib2"):
            m = HRRR_RE.match(path.name)
            if not m:
                continue
            date_part, hour_part = m.groups()
            ts = pd.to_datetime(f"{date_part}{hour_part}", format="%Y%m%d%H")
            out.setdefault(ts, path)
    return dict(sorted(out.items()))


def build_sample_records(ctx: RegionalContext) -> List[SampleRecord]:
    hours = {int(v) for v in ctx.config.get("issue_hours", [0, 6, 12, 18])}
    lead_hours = int(ctx.config["target_offset_hours"])
    hrrr_index = list_hrrr_issue_paths(ctx.hrrr_roots)
    records: List[SampleRecord] = []
    for item in ctx.config["input_ranges"]:
        for day in pd.date_range(item["start"], item["end"], freq="D"):
            for hour in sorted(hours):
                issue_ts = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hour)
                hrrr_path = hrrr_index.get(issue_ts)
                if hrrr_path is None:
                    continue
                target_ts = issue_ts + pd.Timedelta(hours=lead_hours)
                label_path = ctx.firms_dir / f"{target_ts.strftime('%Y%m%d_%H')}.csv"
                if not label_path.exists():
                    continue
                records.append(
                    SampleRecord(
                        issue_ts=issue_ts,
                        target_ts=target_ts,
                        hrrr_path=hrrr_path,
                        label_path=label_path,
                    )
                )
    if not records:
        raise RuntimeError("No regional HRRR/FIRMS sample records found.")
    return records


def open_hrrr_dataset(path: Path, group_name: str) -> xr.Dataset:
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Ignore shared cfgrib sidecar indexes under raw HRRR roots. Some existing
    # `.idx` files are truncated/corrupted and can poison concurrent regional
    # cache jobs. Using an empty indexpath forces a fresh in-process scan.
    kwargs = {
        "filter_by_keys": dict(VAR_GROUPS[group_name]["filter_by_keys"]),
        "indexpath": "",
    }
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs=kwargs)


def build_source_resampler(sample_grib: Path, bounds: Dict[str, float], target_crs: str, y_desc: np.ndarray, x_asc: np.ndarray) -> Dict[str, np.ndarray]:
    ds = open_hrrr_dataset(sample_grib, "level2")
    lat2d = np.asarray(ds["latitude"].values, dtype=np.float64)
    lon2d = np.asarray(ds["longitude"].values, dtype=np.float64)
    ds.close()
    lon2d = np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    buffer_deg = float(bounds.get("buffer_deg", 1.0))
    mask = (
        (lat2d >= float(bounds["lat_min"]) - buffer_deg)
        & (lat2d <= float(bounds["lat_max"]) + buffer_deg)
        & (lon2d >= float(bounds["lon_min"]) - buffer_deg)
        & (lon2d <= float(bounds["lon_max"]) + buffer_deg)
    )
    flat_idx = np.flatnonzero(mask.ravel())
    if flat_idx.size == 0:
        raise RuntimeError("No HRRR source points remain after the requested bbox crop.")
    src_lat = lat2d.ravel()[flat_idx]
    src_lon = lon2d.ravel()[flat_idx]
    src_x, src_y = transform("EPSG:4326", target_crs, src_lon.tolist(), src_lat.tolist())
    src_xy = np.column_stack([np.asarray(src_x, dtype=np.float32), np.asarray(src_y, dtype=np.float32)])

    yy, xx = np.meshgrid(y_desc, x_asc, indexing="ij")
    tgt_xy = np.column_stack([xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)])
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nn.fit(src_xy)
    _, gather = nn.kneighbors(tgt_xy, return_distance=True)
    return {
        "flat_idx": flat_idx.astype(np.int64),
        "gather_idx": gather[:, 0].astype(np.int64),
        "target_h": np.array([len(y_desc)], dtype=np.int64),
        "target_w": np.array([len(x_asc)], dtype=np.int64),
    }


def resample_field(field: np.ndarray, resampler: Dict[str, np.ndarray]) -> np.ndarray:
    flat = np.asarray(field, dtype=np.float32).ravel()[resampler["flat_idx"]]
    gathered = flat[resampler["gather_idx"]]
    h = int(resampler["target_h"][0])
    w = int(resampler["target_w"][0])
    return np.nan_to_num(gathered.reshape(h, w), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_hrrr_cube(ctx: RegionalContext, records: List[SampleRecord], y_desc: np.ndarray, x_asc: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    weather_names = list(ctx.config["hrrr_vars"])
    weather = np.zeros((len(records), len(weather_names), len(y_desc), len(x_asc)), dtype=np.float32)
    resampler = build_source_resampler(records[0].hrrr_path, ctx.config, str(ctx.config["target_crs"]), y_desc, x_asc)

    var_to_group = {}
    for group_name, spec in VAR_GROUPS.items():
        for name in spec["vars"]:
            var_to_group[name] = group_name

    for i, record in enumerate(records):
        opened: Dict[str, xr.Dataset] = {}
        try:
            for j, name in enumerate(weather_names):
                group = var_to_group.get(name)
                if group is None:
                    raise KeyError(f"No HRRR group mapping for variable '{name}'")
                ds = opened.get(group)
                if ds is None:
                    ds = open_hrrr_dataset(record.hrrr_path, group)
                    opened[group] = ds
                if name not in ds.data_vars:
                    raise KeyError(f"Variable '{name}' missing from {record.hrrr_path} in group '{group}'")
                weather[i, j] = resample_field(np.asarray(ds[name].values, dtype=np.float32), resampler)
        finally:
            for ds in opened.values():
                ds.close()
        if (i + 1) % 20 == 0 or (i + 1) == len(records):
            print(
                json.dumps(
                    {
                        "stage": "load_hrrr_cube_progress",
                        "loaded_samples": i + 1,
                        "total_samples": len(records),
                        "latest_issue_timestamp": record.issue_ts.isoformat(),
                    }
                ),
                flush=True,
            )
    return weather, weather_names


def load_static_cube_projected(ctx: RegionalContext, y_desc: np.ndarray, x_asc: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
    transform_dst = projected_transform(y_desc, x_asc)
    shape = (len(y_desc), len(x_asc))
    res_y = float(np.median(np.abs(np.diff(y_desc))))
    res_x = float(np.median(np.diff(x_asc)))
    west = float(x_asc[0] - 0.5 * res_x)
    east = float(x_asc[-1] + 0.5 * res_x)
    south = float(y_desc[-1] - 0.5 * res_y)
    north = float(y_desc[0] + 0.5 * res_y)
    target_crs = str(ctx.config["target_crs"])

    arrays: List[np.ndarray] = []
    valid_arrays: List[np.ndarray] = []
    names: List[str] = []
    for name, spec in ctx.static_layers.items():
        names.append(name)
        dst = np.full(shape, np.nan, dtype=np.float32)
        with rasterio.open(spec["path"]) as src:
            src_bounds = transform_bounds(target_crs, src.crs, west, south, east, north, densify_pts=21)
            window = src.window(*src_bounds).round_offsets().round_lengths()
            source = src.read(1, window=window)
            source_transform = src.window_transform(window)
            reproject(
                source=source,
                destination=dst,
                src_transform=source_transform,
                src_crs=src.crs,
                dst_transform=transform_dst,
                dst_crs=target_crs,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                resampling=resampling_from_name(spec["resampling"]),
            )
        valid = np.isfinite(dst) & (dst > -9000.0)
        valid_arrays.append(valid.astype(np.float32))
        dst = np.where(dst <= -9000.0, np.nan, dst)
        dst = np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0)
        arrays.append(dst.astype(np.float32))
    valid_fraction = np.stack(valid_arrays, axis=0).mean(axis=0, keepdims=True).astype(np.float32)
    return np.stack(arrays, axis=0), names, valid_fraction


def rasterize_firms_counts_projected(firms_path: Path, y_desc: np.ndarray, x_asc: np.ndarray, target_crs: str, bounds: Dict[str, float]) -> np.ndarray:
    if not firms_path.exists():
        return np.zeros((len(y_desc), len(x_asc)), dtype=np.float32)
    try:
        df = pd.read_csv(firms_path, usecols=["latitude", "longitude", "type"])
    except ValueError:
        df = pd.read_csv(firms_path)
    if "type" in df.columns:
        df = df[df["type"] == 0]
    if df.empty:
        return np.zeros((len(y_desc), len(x_asc)), dtype=np.float32)
    df = df[
        (df["latitude"] >= float(bounds["lat_min"]))
        & (df["latitude"] <= float(bounds["lat_max"]))
        & (df["longitude"] >= float(bounds["lon_min"]))
        & (df["longitude"] <= float(bounds["lon_max"]))
    ]
    if df.empty:
        return np.zeros((len(y_desc), len(x_asc)), dtype=np.float32)

    xs, ys = transform(
        "EPSG:4326",
        target_crs,
        df["longitude"].astype(float).tolist(),
        df["latitude"].astype(float).tolist(),
    )
    y_asc = y_desc[::-1]
    y_edges = projected_edges(y_asc)
    x_edges = projected_edges(x_asc)
    counts, _, _ = np.histogram2d(np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64), bins=[y_edges, x_edges])
    return counts.astype(np.float32)[::-1, :]


def write_cache(
    ctx: RegionalContext,
    records: List[SampleRecord],
    weather: np.ndarray,
    static: np.ndarray,
    static_valid: np.ndarray,
    y_desc: np.ndarray,
    x_asc: np.ndarray,
    weather_names: List[str],
    static_names: List[str],
) -> pd.DataFrame:
    static_path = ctx.static_dir / "static_regional_phase1_v1.npz"
    np.savez_compressed(
        static_path,
        static=static,
        static_valid=static_valid,
        lat=y_desc,
        lon=x_asc,
        coord_crs=np.array([str(ctx.config["target_crs"])], dtype=object),
        static_names=np.array(static_names, dtype=object),
    )

    h = int(len(y_desc))
    w = int(len(x_asc))
    empty_firewx = np.zeros((0, h, w), dtype=np.float32)
    firewx_valid = np.ones((1, h, w), dtype=np.float32)
    rows: List[Dict[str, object]] = []
    target_crs = str(ctx.config["target_crs"])

    for i, record in enumerate(records):
        y_count = rasterize_firms_counts_projected(record.label_path, y_desc, x_asc, target_crs, ctx.config)
        y_occ = (y_count > 0).astype(np.float32)
        sample_id = record.issue_ts.strftime("%Y%m%d_%H")
        sample_path = ctx.sample_dir / f"phase1_regional_{sample_id}.npz"
        np.savez_compressed(
            sample_path,
            weather=weather[i],
            firewx=empty_firewx,
            firewx_valid=firewx_valid,
            y_count=y_count[None, ...],
            y_occ=y_occ[None, ...],
            lat=y_desc,
            lon=x_asc,
            coord_crs=np.array([target_crs], dtype=object),
            weather_names=np.array(weather_names, dtype=object),
            firewx_names=np.array([], dtype=object),
        )
        split = split_for_timestamp(record.issue_ts, ctx.config)
        rows.append(
            {
                "sample_id": sample_id,
                "input_date": str(record.issue_ts.date()),
                "target_date": str(record.target_ts.date()),
                "input_timestamp": record.issue_ts.isoformat(),
                "target_timestamp": record.target_ts.isoformat(),
                "split": split,
                "sample_path": str(sample_path),
                "static_path": str(static_path),
                "pos_cells": int(y_occ.sum()),
                "fire_points": float(y_count.sum()),
            }
        )
        if (i + 1) % 20 == 0 or (i + 1) == len(records):
            print(
                json.dumps(
                    {
                        "stage": "write_cache_progress",
                        "written_samples": i + 1,
                        "total_samples": len(records),
                        "latest_sample_id": sample_id,
                    }
                ),
                flush=True,
            )

    manifest = pd.DataFrame(rows).sort_values("input_timestamp").reset_index(drop=True)
    manifest.to_csv(ctx.manifest_dir / "phase1_manifest.csv", index=False)
    for split in ["train", "val", "test"]:
        manifest[manifest["split"] == split].copy().to_csv(ctx.split_dir / f"{split}.csv", index=False)

    summary = {
        "num_samples": int(len(manifest)),
        "weather_shape": list(weather.shape),
        "firewx_shape": [len(records), 0, h, w],
        "firewx_valid_shape": [len(records), 1, h, w],
        "static_shape": list(static.shape),
        "static_valid_shape": list(static_valid.shape),
        "splits": {k: int((manifest["split"] == k).sum()) for k in ["train", "val", "test"]},
        "total_positive_cells": int(manifest["pos_cells"].sum()),
        "total_fire_points": float(manifest["fire_points"].sum()),
        "grid_height": h,
        "grid_width": w,
        "target_crs": str(ctx.config["target_crs"]),
        "target_resolution_m": int(ctx.config["target_resolution_m"]),
        "target_offset_hours": int(ctx.config["target_offset_hours"]),
        "weather_names": weather_names,
        "static_names": static_names,
    }
    (ctx.manifest_dir / "phase1_cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    ctx = build_context(config)
    records = build_sample_records(ctx)
    y_desc, x_asc = build_projected_grid(ctx.config, str(ctx.config["target_crs"]), float(ctx.config["target_resolution_m"]))

    print(
        f"[stage] regional_samples={len(records)} first={records[0].issue_ts.isoformat()} last={records[-1].issue_ts.isoformat()}",
        flush=True,
    )
    print(
        f"[stage] grid_crs={ctx.config['target_crs']} resolution_m={ctx.config['target_resolution_m']} "
        f"shape=({len(y_desc)}, {len(x_asc)})",
        flush=True,
    )
    print("[stage] load_hrrr_cube", flush=True)
    weather, weather_names = load_hrrr_cube(ctx, records, y_desc, x_asc)
    print("[stage] load_static_cube", flush=True)
    static, static_names, static_valid = load_static_cube_projected(ctx, y_desc, x_asc)
    print("[stage] write_cache", flush=True)
    manifest = write_cache(
        ctx=ctx,
        records=records,
        weather=weather,
        static=static,
        static_valid=static_valid,
        y_desc=y_desc,
        x_asc=x_asc,
        weather_names=weather_names,
        static_names=static_names,
    )
    print("[stage] done", flush=True)
    print(f"Built regional HRRR cache with {len(manifest)} samples.")


if __name__ == "__main__":
    raise SystemExit(main())
