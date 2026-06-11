from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import datetime as dt
import json
import os

from data_downloader.catalog import DATASETS
from data_downloader import sources


@dataclass
class DownloadRequest:
    datasets: Sequence[str]
    output_root: str
    temporal_window: Optional[Tuple[str, str]] = None
    area_of_interest_bbox: Optional[Tuple[float, float, float, float]] = None
    years: Optional[Tuple[int, int]] = None
    dry_run: bool = False
    overwrite: bool = False
    max_files: Optional[int] = None
    extra_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def list_datasets() -> List[Dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "family": spec.family,
            "access": spec.access,
            "time_mode": spec.time_mode,
            "aoi_mode": spec.aoi_mode,
            "default_years": spec.default_years,
            "notes": spec.notes,
            "provider_urls": spec.provider_urls,
        }
        for spec in DATASETS.values()
    ]


def save_download_report(results: Sequence[Dict[str, Any]], output_path: str) -> str:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _years_for(name: str, request: DownloadRequest) -> tuple[int, int]:
    spec = DATASETS[name]
    opts = request.extra_options.get(name, {})
    if "years" in opts:
        y = opts["years"]
        return (int(y[0]), int(y[1]))
    if request.years is not None:
        return (int(request.years[0]), int(request.years[1]))
    if request.temporal_window is not None:
        return (int(request.temporal_window[0][:4]), int(request.temporal_window[1][:4]))
    if spec.default_years is not None:
        return spec.default_years
    year = dt.date.today().year
    return (year, year)


def _limit_rows(rows: list[tuple[str, str]], request: DownloadRequest, opts: dict) -> list[tuple[str, str]]:
    max_files = opts.get("max_files", request.max_files)
    if max_files is None:
        return rows
    return rows[: int(max_files)]


def _finish(dataset: str, out_dir: Path, spec_meta: dict, file_results: list[dict], warnings: list[str]) -> dict:
    counts: Dict[str, int] = {}
    for item in file_results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    status = "pass"
    if any(item["status"] in {"error", "url_error"} for item in file_results):
        status = "fail"
    if any(item["status"] == "http_error" for item in file_results):
        status = "partial"
    result = {
        "dataset": dataset,
        "status": status,
        "output_dir": str(out_dir),
        "counts": counts,
        "warnings": warnings,
        "meta": spec_meta,
        "files": file_results,
    }
    sources.write_json(out_dir / "download_manifest.json", result)
    return result


def _download_direct_rows(dataset: str, request: DownloadRequest, rows: list[tuple[str, str]], meta: dict, warnings: list[str]) -> dict:
    out_dir = Path(request.output_root).expanduser().resolve() / dataset
    rows = _limit_rows(rows, request, request.extra_options.get(dataset, {}))
    file_results = sources.download_many(rows, out_dir, overwrite=request.overwrite, dry_run=request.dry_run)
    return _finish(dataset, out_dir, meta, file_results, warnings)


def _download_wfigs_current(request: DownloadRequest) -> dict:
    dataset = "wfigs_current"
    out_dir = Path(request.output_root).expanduser().resolve() / dataset
    service_root = (
        "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
        "WFIGS_Interagency_Perimeters_Current/FeatureServer/0"
    )
    rows = [
        sources.download_file(f"{service_root}?f=pjson", out_dir / "service_metadata.json", request.overwrite, request.dry_run),
        sources.download_arcgis_query(
            f"{service_root}/query",
            out_dir / "wfigs_current.geojson",
            params={"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson"},
            overwrite=request.overwrite,
            dry_run=request.dry_run,
        ),
    ]
    return _finish(dataset, out_dir, {"service_root": service_root}, rows, [])


def _download_firms_area(request: DownloadRequest) -> dict:
    dataset = "firms_area"
    out_dir = Path(request.output_root).expanduser().resolve() / dataset
    opts = request.extra_options.get(dataset, {})
    map_key = opts.get("map_key") or os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        instruction = sources.write_instruction_manifest(
            dataset,
            out_dir,
            "NASA FIRMS area API requires a MAP_KEY. Set FIRMS_MAP_KEY or pass extra_options={'firms_area': {'map_key': '...'}}.",
            DATASETS[dataset].provider_urls,
        )
        return {
            "dataset": dataset,
            "status": "auth_required",
            "output_dir": str(out_dir),
            "counts": {},
            "warnings": [instruction["reason"]],
            "meta": instruction,
            "files": [],
        }
    if request.temporal_window is None:
        raise ValueError("firms_area requires temporal_window=('YYYY-MM-DD', 'YYYY-MM-DD').")
    if request.area_of_interest_bbox is None:
        raise ValueError("firms_area requires area_of_interest_bbox=(min_lon,min_lat,max_lon,max_lat).")
    source = opts.get("source", "VIIRS_SNPP_NRT")
    day_range = int(opts.get("day_range", 1))
    start = request.temporal_window[0]
    bbox = ",".join(str(x) for x in request.area_of_interest_bbox)
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{bbox}/{day_range}/{start}"
    rows = sources.download_many([(url, f"{source}_{start}_{day_range}d.csv")], out_dir, request.overwrite, request.dry_run)
    return _finish(dataset, out_dir, {"source": source, "start": start, "day_range": day_range}, rows, [])


def _download_instruction_only(dataset: str, request: DownloadRequest, reason: str) -> dict:
    out_dir = Path(request.output_root).expanduser().resolve() / dataset
    payload = sources.write_instruction_manifest(dataset, out_dir, reason, DATASETS[dataset].provider_urls)
    return {
        "dataset": dataset,
        "status": "instructions_only",
        "output_dir": str(out_dir),
        "counts": {},
        "warnings": [reason],
        "meta": payload,
        "files": [],
    }


def _download_one(name: str, request: DownloadRequest) -> dict:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Use list_datasets().")
    spec = DATASETS[name]
    opts = request.extra_options.get(name, {})
    years = _years_for(name, request)
    warnings: list[str] = []

    if name == "aqs_pm25":
        return _download_direct_rows(name, request, sources.build_aqs_pm25(years), {"years": years}, warnings)
    if name == "hms_smoke":
        return _download_direct_rows(name, request, sources.build_hms_smoke(years), {"years": years}, warnings)
    if name == "ibtracs":
        return _download_direct_rows(name, request, sources.build_ibtracs(), {"version": "v04r01"}, warnings)
    if name == "hurdat2":
        rows, meta = sources.build_hurdat2()
        return _download_direct_rows(name, request, rows, meta, warnings)
    if name == "usdm":
        include_current = bool(opts.get("include_current", True))
        return _download_direct_rows(
            name,
            request,
            sources.build_usdm(years, include_current=include_current),
            {"years": years, "include_current": include_current},
            warnings,
        )
    if name == "hrrr_fireseason":
        hours = str(opts.get("hours", "00,06,12,18")).split(",")
        product = str(opts.get("product", "wrfsfcf00"))
        season_start = str(opts.get("season_start", "06-01"))
        season_end = str(opts.get("season_end", "10-30"))
        rows = sources.build_hrrr_fireseason(years, season_start, season_end, hours, product)
        return _download_direct_rows(
            name,
            request,
            rows,
            {"years": years, "season_start": season_start, "season_end": season_end, "hours": hours, "product": product},
            warnings,
        )
    if name == "wfigs_current":
        return _download_wfigs_current(request)
    if name == "firms_area":
        return _download_firms_area(request)
    if name == "landfire_static":
        products = opts.get("products", ["fbfm40", "cc"])
        version = str(opts.get("version", "LF2024"))
        rows, landfire_warnings = sources.build_landfire_static(list(products), version=version)
        warnings.extend(landfire_warnings)
        return _download_direct_rows(name, request, rows, {"version": version, "products": products}, warnings)
    if name == "wrc_housing":
        url = opts.get("url")
        if not url:
            return _download_instruction_only(
                name,
                request,
                "WRC housing download URLs can redirect or change. Use the provider page, or pass extra_options={'wrc_housing': {'url': '...'}}.",
            )
        return _download_direct_rows(name, request, [(str(url), Path(str(url)).name or "wrc_housing.zip")], {"url": url}, warnings)
    if name == "landscan":
        return _download_instruction_only(
            name,
            request,
            "LandScan is governed by provider-specific access terms; obtain data from ORNL and place it under the configured raw-data root.",
        )
    if name == "merra2":
        return _download_instruction_only(
            name,
            request,
            "MERRA-2 requires NASA Earthdata/GES DISC credentials. Use provider tooling or configure authenticated access before automated download.",
        )
    if name == "mtbs_perimeters":
        return _download_instruction_only(
            name,
            request,
            "MTBS ArcGIS chunked download is supported by the internal Slurm scripts; client package currently records provider instructions to avoid heavy unaudited pulls.",
        )

    return _download_instruction_only(name, request, f"{spec.name} is cataloged but not implemented yet.")


def download_data(request: DownloadRequest) -> List[Dict[str, Any]]:
    results = []
    for raw_name in request.datasets:
        name = str(raw_name).strip().lower()
        results.append(_download_one(name, request))
    return results
