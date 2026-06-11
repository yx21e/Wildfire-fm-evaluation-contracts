# Data Downloader

This directory contains a lightweight, client-facing downloader for data sources
used by FireWx-FM and planned general hazard-model work. It is intentionally a
raw-data downloader, not a model-ready dataloader. Dataset adapters and aligned
caches should be built after raw files are obtained under each provider's terms.

## Design

The downloader follows the older `Pyhazard_data_downloader` API pattern:

```python
from data_downloader import DownloadRequest, download_data, list_datasets
```

Each dataset declares:

- `access`: whether it is public direct download, ArcGIS REST, auth-required, or instructions-only.
- `time_mode`: how time selection works.
- `aoi_mode`: whether bounding-box selection is native, post-processing only, or unsupported.

This avoids pretending that every source supports the same query semantics.

## Quick Start

List datasets:

```bash
python3 -m data_downloader.cli --list
```

Dry-run a small client request:

```bash
python3 -m data_downloader.cli \
  --datasets aqs_pm25 hms_smoke ibtracs \
  --output-root ./downloads_demo \
  --start-year 2024 \
  --end-year 2024 \
  --dry-run \
  --max-files 3 \
  --report ./downloads_demo/report.json
```

For datasets that use a bounding box, pass negative longitudes with the equals
form so the shell/argparse does not parse them as flags:

```bash
python3 -m data_downloader.cli \
  --datasets firms_area \
  --output-root ./downloads_firms \
  --start-date 2024-06-01 \
  --end-date 2024-06-02 \
  --bbox=-125,32,-114,42 \
  --dry-run
```

Python API:

```python
from data_downloader import DownloadRequest, download_data

request = DownloadRequest(
    datasets=["aqs_pm25", "hms_smoke"],
    output_root="./downloads",
    years=(2024, 2025),
    dry_run=False,
)
results = download_data(request)
```

## Implemented Direct Downloads

These sources are implemented as direct public downloads:

- `aqs_pm25`: EPA AirData hourly PM2.5 files and site/monitor metadata.
- `hms_smoke`: NOAA HMS annual smoke-polygon and fire-point bundles.
- `ibtracs`: NOAA/NCEI IBTrACS v04r01 CSV tracks and documentation.
- `hurdat2`: NOAA/NHC HURDAT2 latest best-track text files and format docs.
- `usdm`: U.S. Drought Monitor weekly GIS shapefile archive.
- `hrrr_fireseason`: NOAA HRRR public S3 files for selected years, fire-season dates, hours, and product.
- `wfigs_current`: current WFIGS perimeter snapshot through ArcGIS REST.
- `landfire_static`: known LF2024 CONUS `FBFM40` and `CC` static products.

## Credential or Terms-Limited Sources

These are intentionally not silently downloaded:

- `firms_area`: requires a NASA FIRMS `MAP_KEY`; set `FIRMS_MAP_KEY` or pass it in `extra_options`.
- `merra2`: requires NASA Earthdata/GES DISC authenticated access.
- `landscan`: governed by ORNL/LandScan provider-specific access terms.
- `wrc_housing`: provider links can redirect/change; pass an explicit URL or use the provider portal.
- `mtbs_perimeters`: heavy ArcGIS chunk pulls are available in internal Slurm scripts, but this public client records instructions first.

## Notes for Clients

- Raw source files are not redistributed by this repository.
- The downloader writes `download_manifest.json` under each dataset output folder.
- Use `--dry-run` and `--max-files` before large requests.
- For large HRRR, GOES, MERRA-2, NWM, or full MTBS/RAVG pulls, run downloads on compute/storage infrastructure rather than a login node.
