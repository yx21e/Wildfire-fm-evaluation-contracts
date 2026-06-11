#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_downloader import DownloadRequest, download_data, list_datasets, save_download_report


def main() -> None:
    print("Available datasets:")
    for row in list_datasets():
        print(f"- {row['name']}: access={row['access']}, time={row['time_mode']}")

    request = DownloadRequest(
        datasets=["aqs_pm25", "hms_smoke", "ibtracs"],
        output_root="./downloads_demo",
        years=(2024, 2024),
        dry_run=True,
        max_files=3,
    )
    results = download_data(request)
    for item in results:
        print(item["dataset"], item["status"], item["counts"], item["warnings"])
    save_download_report(results, "./downloads_demo/report.json")


if __name__ == "__main__":
    main()
