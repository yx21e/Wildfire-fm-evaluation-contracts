# FireWx-FM

FireWx-FM is a wildfire-specialized gridded model for short-lead **active-fire occupancy prediction** over the Lower 48 United States. The released checkpoint consumes a fixed 16-channel tensor on a 5 km EPSG:5070 grid and predicts the probability that each grid cell contains active fire at a 12-hour lead.

<p align="center">
  <img src="examples/final_prediction/firewxfm_conus_20240601_heatmap.png" width="760" alt="FireWx-FM CONUS active-fire occupancy heatmap for 2024-06-01">
</p>

<p align="center">
  <b>Task:</b> active-fire occupancy &nbsp; | &nbsp;
  <b>Domain:</b> Lower-48 CONUS &nbsp; | &nbsp;
  <b>Grid:</b> EPSG:5070, 5 km &nbsp; | &nbsp;
  <b>Lead:</b> 12 hours
</p>

## Release Contents

| Component | Path |
|---|---|
| Model checkpoint | [`models/checkpoints/firewxfm_conus_lowposw_noexposure_seed42.pt`](models/checkpoints/firewxfm_conus_lowposw_noexposure_seed42.pt) |
| Input channel contract | [`models/metadata/input_channels.json`](models/metadata/input_channels.json) |
| Normalization statistics | [`models/metadata/input_normalization_stats.json`](models/metadata/input_normalization_stats.json) |
| Inference code | [`firewxfm/`](firewxfm/) |
| Inference contract | [`docs/inference_contract.md`](docs/inference_contract.md) |
| Serving notes | [`docs/serving_notes.md`](docs/serving_notes.md) |
| Data source notes | [`data_sources/DATA_SOURCES.md`](data_sources/DATA_SOURCES.md) |
| HRRR downloader | [`scripts/hrrr_downloader.py`](scripts/hrrr_downloader.py) |
| Example output | [`examples/final_prediction/`](examples/final_prediction/) |
| File checksums | [`release.sha256`](release.sha256) |

Raw source datasets are not redistributed. Users should obtain the required NOAA HRRR, NASA FIRMS, LANDFIRE, Wildfire Risk to Communities, and LandScan resources from the original providers.

## Input Contract

The model expects a tensor with shape `[16, H, W]`. Channel order is fixed.

| Channels | Source | Role |
|---|---|---|
| `0-9` | NOAA HRRR | Dynamic weather fields, including CAPE from the 0-3000 m above-ground layer. |
| `10-11` | Input builder | Dynamic-weather and static-layer validity masks. |
| `12` | LANDFIRE | Fire-behavior fuel model categorical code. |
| `13` | LANDFIRE | Canopy cover. |
| `14-15` | Wildfire Risk to Communities and LandScan | Housing density and population. These channels are zeroed in the final no-exposure serving run. |

Continuous channels use train-split z-score statistics from [`models/metadata/input_normalization_stats.json`](models/metadata/input_normalization_stats.json). The final serving script applies normalization first, then zeros channels `14` and `15`.

## Quick Start

Install the lightweight inference dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run occupancy inference from a prebuilt 16-channel NumPy stack:

```bash
python -m firewxfm.serve_conus \
  --input-npy /path/to/input_stack_16chw.npy \
  --checkpoint models/checkpoints/firewxfm_conus_lowposw_noexposure_seed42.pt \
  --normalization-stats models/metadata/input_normalization_stats.json \
  --output-npy outputs/firewxfm_occupancy_probability.npy
```

The output is a probability map in `[H, W]` order with values in `[0, 1]`.

## Serving Guidance

Use the default overlap-window inference settings for CONUS-scale maps:

| Setting | Value |
|---|---:|
| Window | `256` |
| Stride | `64` |
| Halo crop | `32` |
| Phase shifts | `0,4,8,12` along both axes |

The 32 by 32 crop size used during training is not the serving tile size. For full-map inference, use overlap windows and phase averaging rather than stitching independent 32 by 32 predictions.

## Example Output

The example output is a Lower-48 CONUS active-fire occupancy prediction for `2024-06-01` using the final no-exposure serving contract.

| File | Use |
|---|---|
| [`firewxfm_conus_20240601_probability_5km_lower48.tif`](examples/final_prediction/firewxfm_conus_20240601_probability_5km_lower48.tif) | Quantitative probability GeoTIFF. |
| [`firewxfm_conus_20240601_heatmap.png`](examples/final_prediction/firewxfm_conus_20240601_heatmap.png) | Visual preview. |
| [`firewxfm_conus_20240601_heatmap_rgb.tif`](examples/final_prediction/firewxfm_conus_20240601_heatmap_rgb.tif) | Georeferenced RGB heatmap. |

Run the release check from the repository root:

```bash
python scripts/check_release_files.py
```

## Spatial Aggregation

The native model output is a 5 km probability grid. County or sub-county summaries should be produced by aggregating the gridded probabilities after inference.

```bash
python -m spatial_serving.grid_to_polygons \
  --probability-npz outputs/firewxfm_probability_with_coords.npz \
  --polygons /path/to/counties_or_subcounty_units.geojson \
  --id-column GEOID \
  --output outputs/firewxfm_polygon_summary.geojson
```

The polygon adapter reports mean, maximum, 90th percentile, and area fraction above a chosen probability threshold.

## Data Access

The HRRR downloader can fetch public NOAA HRRR files or write a dry-run manifest:

```bash
python scripts/hrrr_downloader.py \
  --start-date 2024-06-01 \
  --end-date 2024-06-01 \
  --hours 00,06,12,18 \
  --forecast-hours 00 \
  --include-idx \
  --output-root downloads/hrrr
```

The full 16-channel stack also requires FIRMS-derived labels for training and static rasters from LANDFIRE, Wildfire Risk to Communities, and LandScan. See [`data_sources/DATA_SOURCES.md`](data_sources/DATA_SOURCES.md).

## Citation

```bibtex
@misc{firewxfm2026,
  title = {A Wildfire Foundation Model},
  author = {Yang, X. and collaborators},
  year = {2026},
  note = {FireWx-FM active-fire occupancy model release},
  url = {https://github.com/yx21e/Wildfire-fm-evaluation-contracts}
}
```

## License and Data Boundary

Code is released for research use under the license in this repository. Raw source data are not redistributed. Users are responsible for obtaining source data and complying with provider terms.
