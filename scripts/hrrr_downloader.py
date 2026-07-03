from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


USER_AGENT = "wildfire-fm-hrrr-downloader/0.1"
AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def iter_dates(start: dt.date, end: dt.date):
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def parse_csv_ints(value: str, width: int = 2) -> list[str]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(str(int(item)).zfill(width))
    return out


def hrrr_file_name(hour: str, product: str, forecast_hour: str) -> str:
    return f"hrrr.t{hour}z.{product}f{forecast_hour}.grib2"


def hrrr_url(day: dt.date, hour: str, product: str, forecast_hour: str, domain: str = "conus") -> str:
    ymd = day.strftime("%Y%m%d")
    name = hrrr_file_name(hour, product, forecast_hour)
    return f"{AWS_BASE}/hrrr.{ymd}/{domain}/{name}"


def hrrr_relative_path(day: dt.date, hour: str, product: str, forecast_hour: str, domain: str = "conus") -> str:
    ymd = day.strftime("%Y%m%d")
    name = hrrr_file_name(hour, product, forecast_hour)
    return f"hrrr/{day.year}/{ymd}/{domain}/{name}"


def build_hrrr_rows(
    start: dt.date,
    end: dt.date,
    hours: list[str],
    forecast_hours: list[str],
    product: str,
    domain: str,
    include_idx: bool,
    idx_only: bool = False,
) -> list[tuple[str, str]]:
    rows = []
    for day in iter_dates(start, end):
        for hour in hours:
            for forecast_hour in forecast_hours:
                url = hrrr_url(day, hour, product, forecast_hour, domain)
                rel = hrrr_relative_path(day, hour, product, forecast_hour, domain)
                if not idx_only:
                    rows.append((url, rel))
                if include_idx or idx_only:
                    rows.append((f"{url}.idx", f"{rel}.idx"))
    return rows


def download_file(url: str, out_path: Path, overwrite: bool = False, dry_run: bool = False) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return {"url": url, "path": str(out_path), "status": "exists", "bytes": out_path.stat().st_size}
    if dry_run:
        return {"url": url, "path": str(out_path), "status": "dry_run", "bytes": None}

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
        return {"url": url, "path": str(out_path), "status": "downloaded", "bytes": bytes_written}
    except HTTPError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return {"url": url, "path": str(out_path), "status": "http_error", "http_status": exc.code, "error": str(exc)}
    except URLError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return {"url": url, "path": str(out_path), "status": "url_error", "error": str(exc)}
    except Exception as exc:  # pragma: no cover
        if tmp_path.exists():
            tmp_path.unlink()
        return {"url": url, "path": str(out_path), "status": "error", "error": repr(exc)}


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NOAA HRRR files from the public AWS archive.")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD.")
    parser.add_argument("--hours", default="00,06,12,18", help="Comma-separated cycle hours, e.g. 00,06,12,18.")
    parser.add_argument("--forecast-hours", default="00", help="Comma-separated forecast hours, e.g. 00 or 00,01,02.")
    parser.add_argument("--product", default="wrfsfc", help="HRRR product prefix without forecast hour, usually wrfsfc.")
    parser.add_argument("--domain", default="conus", help="HRRR archive domain, usually conus.")
    parser.add_argument("--output-root", default="./downloads/hrrr", help="Directory for downloaded files and manifest.")
    parser.add_argument("--include-idx", action="store_true", help="Also download .idx sidecar files.")
    parser.add_argument("--idx-only", action="store_true", help="Download only .idx sidecar files; useful for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest without downloading file bytes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--max-files", type=int, help="Limit number of files for a smoke test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end < start:
        raise SystemExit("--end-date must be on or after --start-date")

    rows = build_hrrr_rows(
        start=start,
        end=end,
        hours=parse_csv_ints(args.hours),
        forecast_hours=parse_csv_ints(args.forecast_hours, width=2),
        product=args.product,
        domain=args.domain,
        include_idx=args.include_idx,
        idx_only=args.idx_only,
    )
    if args.max_files is not None:
        rows = rows[: int(args.max_files)]

    output_root = Path(args.output_root).expanduser().resolve()
    results = [download_file(url, output_root / rel, overwrite=args.overwrite, dry_run=args.dry_run) for url, rel in rows]
    counts: dict[str, int] = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1

    manifest = {
        "dataset": "noaa_hrrr",
        "archive": "noaa-hrrr-bdp-pds",
        "status_counts": counts,
        "request": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "hours": args.hours,
            "forecast_hours": args.forecast_hours,
            "product": args.product,
            "domain": args.domain,
            "include_idx": bool(args.include_idx),
            "idx_only": bool(args.idx_only),
            "dry_run": bool(args.dry_run),
        },
        "files": results,
    }
    manifest_path = output_root / "hrrr_download_manifest.json"
    write_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "status_counts": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
