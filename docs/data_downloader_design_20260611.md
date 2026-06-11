# Data Downloader Design Notes

Date: 2026-06-11

## Goal

Provide an easy client-facing way to obtain source data used by FireWx-FM and future general hazard-model work, while respecting provider access rules. The downloader should make raw-data acquisition reproducible where possible and give explicit instructions where automated download is not appropriate.

## Reused Ideas

From `Pyhazard_data_downloader`:

- `DownloadRequest`
- `download_data`
- `list_datasets`
- dataset capability metadata
- per-dataset warnings and structured result reports

From `Rai_dataloader_h5`:

- unified sample/schema thinking
- separation between raw source adapters and model-ready samples
- HDF5/cache handoff idea for later aligned caches

## Current Implementation

The minimum implementation is in:

- `data_downloader/catalog.py`
- `data_downloader/client.py`
- `data_downloader/sources.py`
- `data_downloader/cli.py`
- `data_downloader/README.md`
- `examples/download_data_client.py`

Implemented public-direct or ArcGIS downloaders:

- AQS PM2.5
- HMS smoke/fire
- IBTrACS
- HURDAT2
- USDM
- HRRR fire-season subset
- WFIGS current perimeter snapshot
- LF2024 LANDFIRE FBFM40 and canopy cover

Credential or terms-limited entries are cataloged but not silently downloaded:

- FIRMS area API
- MERRA-2
- LandScan
- WRC housing unless an explicit URL is provided
- heavy MTBS ArcGIS chunk pulls

## Recommended Next Step

Keep this downloader as the raw-data layer. Build a separate adapter layer that turns raw data into the `SampleRecord` proposed in `phase1_run/docs/general_fm_data_audit_and_retraining_plan_20260611.md`.
