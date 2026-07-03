# Training Pipeline

The training code in this directory is included so the released checkpoint can be inspected and adapted. Raw data and intermediate caches are not redistributed.

## Main Files

| File | Purpose |
|---|---|
| `build_phase1_cache_regional_hrrr.py` | Builds gridded weather/static/target caches from source datasets. |
| `train_cold_tiled_mainline.py` | Trains the compact U-Net on cached time maps using sparse-label tile sampling. |
| `eval_metrics.py` | Metric utilities used during model development. |
| `configs/stage1_cache_conus_hrrr_us_5km_l12_template.json` | Template for a CONUS 5 km, 12-hour-lead cache. |
| `configs/train_firewxfm_conus_noexposure_template.json` | Path-free training template with the main final-run settings. |

## Training Strategy

The model is trained from chronological time maps. Training crops 32 by 32 tiles from the maps to increase exposure to sparse active-fire cells while retaining non-fire context. The final model uses per-channel z-score normalization for continuous inputs, class-weighted loss terms for sparse occupancy labels, and an auxiliary spatial-support head.

The 32 by 32 crop size is a training choice, not the serving contract. Final CONUS inference should use `firewxfm.serve_conus` with larger overlapping windows and phase averaging.

Example command:

```bash
python training/train_cold_tiled_mainline.py \
  --config training/configs/train_firewxfm_conus_noexposure_template.json \
  --run-name firewxfm_conus_noexposure_seed42
```

## Data Boundary

The scripts expect local copies of HRRR, FIRMS, LANDFIRE, Wildfire Risk to Communities, and LandScan resources. See [`../data_sources/DATA_SOURCES.md`](../data_sources/DATA_SOURCES.md) for source links and access notes.
