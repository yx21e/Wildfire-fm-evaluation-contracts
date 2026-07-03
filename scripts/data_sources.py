from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


USER_AGENT = "wildfire-fm-data-downloader/0.1"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_text(url: str, timeout: int = 120) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def download_file(url: str, out_path: Path, overwrite: bool = False, dry_run: bool = False) -> dict:
    ensure_dir(out_path.parent)
    if out_path.exists() and not overwrite:
        return {
            "url": url,
            "path": str(out_path),
            "status": "exists",
            "bytes": out_path.stat().st_size,
        }
    if dry_run:
        return {
            "url": url,
            "path": str(out_path),
            "status": "dry_run",
            "bytes": None,
        }

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    bytes_written = 0
    try:
        with urlopen(req, timeout=180) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
        os.replace(tmp_path, out_path)
        return {
            "url": url,
            "path": str(out_path),
            "status": "downloaded",
            "bytes": bytes_written,
        }
    except HTTPError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return {
            "url": url,
            "path": str(out_path),
            "status": "http_error",
            "http_status": exc.code,
            "error": str(exc),
        }
    except URLError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return {
            "url": url,
            "path": str(out_path),
            "status": "url_error",
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover
        if tmp_path.exists():
            tmp_path.unlink()
        return {
            "url": url,
            "path": str(out_path),
            "status": "error",
            "error": repr(exc),
        }


def iter_dates(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def iter_tuesdays(start_year: int, end_year: int) -> Iterable[dt.date]:
    start = dt.date(start_year, 1, 1)
    while start.weekday() != 1:
        start += dt.timedelta(days=1)
    end = dt.date(end_year, 12, 31)
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=7)


def build_aqs_pm25(years: tuple[int, int]) -> list[tuple[str, str]]:
    start_year, end_year = years
    base = "https://aqs.epa.gov/aqsweb/airdata"
    rows = [
        (f"{base}/aqs_sites.zip", "metadata/aqs_sites.zip"),
        (f"{base}/aqs_monitors.zip", "metadata/aqs_monitors.zip"),
    ]
    for year in range(start_year, end_year + 1):
        for code in ("88101", "88502"):
            name = f"hourly_{code}_{year}.zip"
            rows.append((f"{base}/{name}", f"hourly/{name}"))
    return rows


def build_hms_smoke(years: tuple[int, int]) -> list[tuple[str, str]]:
    start_year, end_year = years
    smoke_base = (
        "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/"
        "Shapefile/Annual_Bundles"
    )
    fire_base = (
        "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Fire_Points/"
        "Shapefile/Annual_Bundles"
    )
    rows = []
    for year in range(start_year, end_year + 1):
        rows.append((f"{smoke_base}/hms_smoke{year}.zip", f"smoke/hms_smoke{year}.zip"))
        rows.append((f"{fire_base}/hms_fire{year}.zip", f"fire/hms_fire{year}.zip"))
    return rows


def build_ibtracs() -> list[tuple[str, str]]:
    base = (
        "https://www.ncei.noaa.gov/data/"
        "international-best-track-archive-for-climate-stewardship-ibtracs/v04r01"
    )
    return [
        (
            f"{base}/access/csv/ibtracs.since1980.list.v04r01.csv",
            "csv/ibtracs.since1980.list.v04r01.csv",
        ),
        (f"{base}/access/csv/ibtracs.NA.list.v04r01.csv", "csv/ibtracs.NA.list.v04r01.csv"),
        (f"{base}/access/csv/ibtracs.EP.list.v04r01.csv", "csv/ibtracs.EP.list.v04r01.csv"),
        (f"{base}/doc/IBTrACS_v04r01_change_log.txt", "doc/IBTrACS_v04r01_change_log.txt"),
        (
            "https://www.ncei.noaa.gov/sites/default/files/2025-09/"
            "IBTrACS_v04r01_column_documentation.pdf",
            "doc/IBTrACS_v04r01_column_documentation.pdf",
        ),
        (
            "https://www.ncei.noaa.gov/sites/default/files/2025-04/"
            "IBTrACS_version4r01_Technical_Details.pdf",
            "doc/IBTrACS_version4r01_Technical_Details.pdf",
        ),
    ]


def build_hurdat2() -> tuple[list[tuple[str, str]], dict]:
    index_url = "https://www.nhc.noaa.gov/data/hurdat/"
    html = fetch_text(index_url)

    def latest(pattern: str) -> str:
        matches = re.findall(pattern, html)
        if not matches:
            raise RuntimeError(f"No HURDAT2 match for pattern: {pattern}")
        return sorted(set(matches))[-1]

    files = {
        "combined": latest(r'href="(hurdat2-1851-[^"]+?\.txt)"'),
        "atl": latest(r'href="(hurdat2-atl-[^"]+?\.txt)"'),
        "nepac": latest(r'href="(hurdat2-nepac-[^"]+?\.txt)"'),
        "fmt_atl": "hurdat2-format-atl-1851-2021.pdf",
        "fmt_nepac": "hurdat2-format-nencpac-1949-2021.pdf",
    }
    rows = [
        (f"{index_url}{files['combined']}", f"text/{files['combined']}"),
        (f"{index_url}{files['atl']}", f"text/{files['atl']}"),
        (f"{index_url}{files['nepac']}", f"text/{files['nepac']}"),
        (f"{index_url}{files['fmt_atl']}", f"doc/{files['fmt_atl']}"),
        (f"{index_url}{files['fmt_nepac']}", f"doc/{files['fmt_nepac']}"),
    ]
    return rows, files


def build_usdm(years: tuple[int, int], include_current: bool = True) -> list[tuple[str, str]]:
    start_year, end_year = years
    base = "https://droughtmonitor.unl.edu/data/shapefiles_m"
    rows = []
    for day in iter_tuesdays(start_year, end_year):
        name = f"USDM_{day.strftime('%Y%m%d')}_M.zip"
        rows.append((f"{base}/{name}", f"weekly/{day.year}/{name}"))
    if include_current:
        rows.append((f"{base}/USDM_current_M.zip", "weekly/current/USDM_current_M.zip"))
    return rows


def build_hrrr_fireseason(
    years: tuple[int, int],
    season_start: str,
    season_end: str,
    hours: list[str],
    product: str,
) -> list[tuple[str, str]]:
    start_year, end_year = years
    start_month, start_day = [int(x) for x in season_start.split("-")]
    end_month, end_day = [int(x) for x in season_end.split("-")]
    rows = []
    for year in range(start_year, end_year + 1):
        start = dt.date(year, start_month, start_day)
        end = dt.date(year, end_month, end_day)
        for day in iter_dates(start, end):
            ymd = day.strftime("%Y%m%d")
            for hour in hours:
                hour = str(hour).zfill(2)
                name = f"hrrr.t{hour}z.{product}.grib2"
                url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{ymd}/conus/{name}"
                rel = f"{day.year}/{ymd}/{name}"
                rows.append((url, rel))
                rows.append((f"{url}.idx", f"{rel}.idx"))
    return rows


def build_landfire_static(products: list[str], version: str = "LF2024") -> tuple[list[tuple[str, str]], list[str]]:
    known = {
        "fbfm40": f"https://landfire.gov/data-downloads/CONUS_{version}/{version}_FBFM40_CONUS.zip",
        "cc": f"https://landfire.gov/data-downloads/CONUS_{version}/{version}_CC_CONUS.zip",
    }
    rows = []
    warnings = []
    for product in products:
        key = product.lower()
        if key in known:
            rows.append((known[key], f"{version}_{key}.zip"))
        else:
            warnings.append(
                f"No built-in direct URL for LANDFIRE product {product!r}; use provider portal or pass an explicit URL in extra_options."
            )
    return rows, warnings


def write_instruction_manifest(dataset: str, out_dir: Path, reason: str, provider_urls: list[str]) -> dict:
    payload = {
        "dataset": dataset,
        "status": "instructions_only",
        "reason": reason,
        "provider_urls": provider_urls,
    }
    write_json(out_dir / "INSTRUCTIONS.json", payload)
    return payload


def download_many(rows: list[tuple[str, str]], out_dir: Path, overwrite: bool, dry_run: bool) -> list[dict]:
    results = []
    for url, rel_path in rows:
        results.append(download_file(url, out_dir / rel_path, overwrite=overwrite, dry_run=dry_run))
    return results


def download_arcgis_query(
    service_url: str,
    out_path: Path,
    params: dict,
    overwrite: bool,
    dry_run: bool,
    max_attempts: int = 5,
) -> dict:
    ensure_dir(out_path.parent)
    if out_path.exists() and not overwrite:
        return {"url": service_url, "path": str(out_path), "status": "exists", "bytes": out_path.stat().st_size}
    if dry_run:
        return {"url": service_url, "path": str(out_path), "status": "dry_run", "params": params}
    import urllib.parse

    encoded = urllib.parse.urlencode(params).encode("utf-8")
    for attempt in range(1, max_attempts + 1):
        req = Request(service_url, data=encoded, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(req, timeout=180) as response:
                body = response.read()
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            tmp.write_bytes(body)
            tmp.replace(out_path)
            return {
                "url": service_url,
                "path": str(out_path),
                "status": "downloaded",
                "bytes": len(body),
                "attempt": attempt,
            }
        except Exception as exc:
            if attempt == max_attempts:
                return {"url": service_url, "path": str(out_path), "status": "error", "error": repr(exc)}
            time.sleep(min(60, 2**attempt + random.random()))
    return {"url": service_url, "path": str(out_path), "status": "error", "error": "unreachable"}
