from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class PolygonAggregationConfig:
    """Configuration for mapping a gridded probability field to polygons."""

    probability_path: str
    polygons_path: str
    output_path: str
    value_column: str = "firewx_prob_mean"
    id_column: str | None = None
    threshold: float = 0.5
    latitude_name: str = "lat"
    longitude_name: str = "lon"
    probability_name: str = "prob"
    crs: str = "EPSG:5070"
    all_touched: bool = True


def _load_probability_npz(path: Path, probability_name: str, latitude_name: str, longitude_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    prob = np.asarray(data[probability_name], dtype=np.float32)
    if prob.ndim == 3:
        if prob.shape[0] != 1:
            raise ValueError("3D probability arrays must have shape [1, y, x].")
        prob = prob[0]
    if prob.ndim != 2:
        raise ValueError("Probability array must have shape [y, x] or [1, y, x].")
    y = np.asarray(data[latitude_name], dtype=np.float64)
    x = np.asarray(data[longitude_name], dtype=np.float64)
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("This adapter expects 1D grid coordinate arrays.")
    if prob.shape != (y.size, x.size):
        raise ValueError(f"Probability shape {prob.shape} does not match coordinates {(y.size, x.size)}.")
    return prob, y, x


def _edges_from_centers(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        raise ValueError("At least two coordinate values are required.")
    mid = 0.5 * (values[:-1] + values[1:])
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = mid
    edges[0] = values[0] - (mid[0] - values[0])
    edges[-1] = values[-1] + (values[-1] - mid[-1])
    return edges


def _transform_from_centers(y: np.ndarray, x: np.ndarray):
    from rasterio.transform import from_bounds

    x_edges = _edges_from_centers(x)
    y_asc = y[::-1] if y[0] > y[-1] else y
    y_edges = _edges_from_centers(y_asc)
    west = float(x_edges[0])
    east = float(x_edges[-1])
    south = float(y_edges[0])
    north = float(y_edges[-1])
    return from_bounds(west, south, east, north, x.size, y.size)


def _stats(values: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    valid = np.isfinite(values)
    if not np.any(valid):
        return {
            "cell_count": 0,
            "firewx_prob_mean": None,
            "firewx_prob_max": None,
            "firewx_prob_p90": None,
            "firewx_area_fraction_ge_threshold": None,
        }
    vals = values[valid]
    return {
        "cell_count": int(vals.size),
        "firewx_prob_mean": float(np.mean(vals)),
        "firewx_prob_max": float(np.max(vals)),
        "firewx_prob_p90": float(np.percentile(vals, 90)),
        "firewx_area_fraction_ge_threshold": float(np.mean(vals >= threshold)),
    }


def _iter_feature_masks(polygons, out_shape: tuple[int, int], transform, all_touched: bool) -> Iterable[tuple[int, np.ndarray]]:
    from rasterio.features import geometry_mask

    for idx, geom in enumerate(polygons.geometry):
        if geom is None or geom.is_empty:
            yield idx, np.zeros(out_shape, dtype=bool)
            continue
        mask = geometry_mask(
            [geom],
            out_shape=out_shape,
            transform=transform,
            invert=True,
            all_touched=all_touched,
        )
        yield idx, mask


def aggregate_grid_to_polygons(config: PolygonAggregationConfig) -> dict[str, object]:
    """Aggregate one FireWx-FM probability grid to user-defined polygon units.

    The model remains gridded. This adapter changes only the serving unit by
    area-overlaying the grid with user-provided polygons.
    """
    import geopandas as gpd

    prob, y, x = _load_probability_npz(
        Path(config.probability_path),
        probability_name=config.probability_name,
        latitude_name=config.latitude_name,
        longitude_name=config.longitude_name,
    )
    polygons = gpd.read_file(config.polygons_path)
    if polygons.empty:
        raise ValueError("No polygons found.")
    polygons = polygons.to_crs(config.crs)
    transform = _transform_from_centers(y, x)

    rows = []
    for idx, mask in _iter_feature_masks(polygons, prob.shape, transform, config.all_touched):
        record = _stats(prob[mask], config.threshold)
        if config.id_column and config.id_column in polygons.columns:
            record[config.id_column] = polygons.iloc[idx][config.id_column]
        rows.append(record)

    out = polygons.copy()
    for key in rows[0].keys() if rows else []:
        out[key] = [row.get(key) for row in rows]
    out[config.value_column] = out["firewx_prob_mean"]

    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".geojson", ".json"}:
        out.to_file(output, driver="GeoJSON")
    elif output.suffix.lower() in {".gpkg"}:
        out.to_file(output, driver="GPKG")
    else:
        out.drop(columns="geometry").to_csv(output, index=False)

    metadata = {
        "status": "ok",
        "output_path": str(output),
        "num_polygons": int(len(out)),
        "config": asdict(config),
        "native_grid_shape": list(prob.shape),
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate FireWx-FM grid probabilities to user-defined polygon granularities.")
    parser.add_argument("--probability-npz", required=True, help="NPZ containing prob, lat, and lon arrays.")
    parser.add_argument("--polygons", required=True, help="Polygon file readable by GeoPandas.")
    parser.add_argument("--output", required=True, help="Output .geojson, .gpkg, or .csv path.")
    parser.add_argument("--id-column", help="Optional polygon id column to preserve in output.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for area-fraction summary.")
    parser.add_argument("--crs", default="EPSG:5070", help="Grid CRS used by FireWx-FM probability coordinates.")
    parser.add_argument("--probability-name", default="prob")
    parser.add_argument("--latitude-name", default="lat")
    parser.add_argument("--longitude-name", default="lon")
    parser.add_argument("--strict-centers", action="store_true", help="Use center-in-polygon behavior instead of all touched cells.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = PolygonAggregationConfig(
        probability_path=args.probability_npz,
        polygons_path=args.polygons,
        output_path=args.output,
        id_column=args.id_column,
        threshold=args.threshold,
        crs=args.crs,
        probability_name=args.probability_name,
        latitude_name=args.latitude_name,
        longitude_name=args.longitude_name,
        all_touched=not args.strict_centers,
    )
    print(json.dumps(aggregate_grid_to_polygons(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
