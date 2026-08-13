# FireWx-FM Inference Contract

This document defines the final serving contract for the released FireWx-FM CONUS model.

## Native Grid

- Domain: Lower-48 CONUS.
- CRS: EPSG:5070.
- Native resolution: 5 km grid cells.
- Prediction target: 12-hour active-fire occupancy probability.
- Tensor order: `[channel, y, x]`.
- Input shape: `[16, H, W]`.

The model remains gridded. County or sub-county outputs are produced by post-inference aggregation from the native 5 km probability grid.

## Channel Order

| Index | Name | Source | Description | Unit |
|---:|---|---|---|---|
| 0 | `t2m` | NOAA HRRR | 2 m air temperature | K |
| 1 | `d2m` | NOAA HRRR | 2 m dew point temperature | K |
| 2 | `u10` | NOAA HRRR | 10 m eastward wind component | m s-1 |
| 3 | `v10` | NOAA HRRR | 10 m northward wind component | m s-1 |
| 4 | `cape` | NOAA HRRR | CAPE, 0-3000 m above-ground layer | J kg-1 |
| 5 | `sp` | NOAA HRRR | Surface pressure | Pa |
| 6 | `blh` | NOAA HRRR | Boundary-layer height | m |
| 7 | `vis` | NOAA HRRR | Visibility | m |
| 8 | `prate` | NOAA HRRR | Precipitation rate | kg m-2 s-1 |
| 9 | `tp` | NOAA HRRR | Accumulated precipitation | kg m-2 |
| 10 | `dynamic_valid` | Input builder | Dynamic weather validity fraction | 0-1 |
| 11 | `static_valid` | Input builder | Static-layer validity fraction after reprojection | 0-1 |
| 12 | `fuel_fbfm40` | LANDFIRE | Fire-behavior fuel model | category code |
| 13 | `canopy_cover` | LANDFIRE | Canopy cover | percent |
| 14 | `housing_density` | Wildfire Risk to Communities | Housing-unit density | provider native |
| 15 | `population` | LandScan | Population | persons |

Machine-readable metadata is stored in [`models/metadata/input_channels.json`](../models/metadata/input_channels.json).

## Normalization

The final model uses train-split z-score normalization for continuous channels. In the released training template this covers HRRR channels `0-9`, canopy cover channel `13`, and exposure channels `14-15`. The inference script applies the statistics from:

```text
models/metadata/input_normalization_stats.json
```

The expected statistics format is the one written by `training/train_cold_tiled_mainline.py`: a JSON object with `enabled`, `method`, `eps`, and a `channels` list containing `index`, `mean`, and `std`.

## Validity Masks

The two mask channels are part of the input contract.

- `dynamic_valid` records dynamic weather-input validity after the cache-building step.
- `static_valid` records the fraction of static layers with valid values after reprojection.

These masks are not model outputs. They are input features that let the model distinguish true zeros from sanitized missing values.

## Final Serving Settings

The released serving run uses:

- Checkpoint: `models/checkpoints/firewxfm_2024_staticfix_region_balanced_bce_seed42.pt`.
- Window: `256`.
- Stride: `64`.
- Halo: `32`.
- Phase shifts: all combinations of `0,4,8,12` along y and x.
- Zeroed channels after normalization: `14,15`.

Production live preprocessing should use `scripts/build_5km_input_stack.py` to
repair the 16-channel stack with the corrected static cache before inference.

The phase ensemble averages shifted overlapping-window predictions. This reduces dependence on tile origin during full-domain inference.

## Output

The model output is a probability map in `[H, W]` order with values in `[0, 1]`. GeoTIFF writing is intentionally kept outside `firewxfm.serve_conus` so users can preserve their own transform, CRS, and mask handling.
