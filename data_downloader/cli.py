from __future__ import annotations

import argparse
import json

from data_downloader import DownloadRequest, download_data, list_datasets, save_download_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Client-facing downloader for FireWx-FM and general hazard data sources.")
    parser.add_argument("--list", action="store_true", help="List known datasets and exit.")
    parser.add_argument("--datasets", nargs="*", default=[], help="Dataset ids to download.")
    parser.add_argument("--output-root", default="./downloads", help="Output root for downloaded raw files and manifests.")
    parser.add_argument("--start-date", help="Optional start date YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Optional end date YYYY-MM-DD.")
    parser.add_argument("--start-year", type=int, help="Optional start year.")
    parser.add_argument("--end-year", type=int, help="Optional end year.")
    parser.add_argument("--bbox", help="Optional bbox as min_lon,min_lat,max_lon,max_lat.")
    parser.add_argument("--dry-run", action="store_true", help="Build manifests without downloading file bytes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--max-files", type=int, help="Limit files per dataset; useful for smoke tests.")
    parser.add_argument("--extra-json", default="{}", help="JSON object of per-dataset extra options.")
    parser.add_argument("--report", help="Optional report JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print(json.dumps(list_datasets(), indent=2, sort_keys=True))
        return 0
    if not args.datasets:
        raise SystemExit("--datasets is required unless --list is used")

    temporal_window = None
    if args.start_date or args.end_date:
        if not (args.start_date and args.end_date):
            raise SystemExit("--start-date and --end-date must be provided together")
        temporal_window = (args.start_date, args.end_date)

    years = None
    if args.start_year or args.end_year:
        if not (args.start_year and args.end_year):
            raise SystemExit("--start-year and --end-year must be provided together")
        years = (args.start_year, args.end_year)

    bbox = None
    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            raise SystemExit("--bbox must be min_lon,min_lat,max_lon,max_lat")
        bbox = tuple(parts)  # type: ignore[assignment]

    request = DownloadRequest(
        datasets=args.datasets,
        output_root=args.output_root,
        temporal_window=temporal_window,
        area_of_interest_bbox=bbox,
        years=years,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        max_files=args.max_files,
        extra_options=json.loads(args.extra_json),
    )
    results = download_data(request)
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.report:
        save_download_report(results, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
