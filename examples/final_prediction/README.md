# CONUS Prediction Example

This directory stores a demonstration output generated from `2026-07-06 12Z` HRRR input. The model predicts 12-hour active-fire occupancy probability for `2026-07-07 00Z`.
The example applies USGS-consistency calibration to the raw FireWx-FM map using 2026-07-07 USGS Fire Danger Forecast surfaces.

## Expected Files

| File | Description |
|---|---|
| `firewxfm_conus_20260706_t12_usgs_calibrated_probability_5km_lower48.tif` | Lower-48 masked calibrated probability GeoTIFF. |
| `firewxfm_conus_20260706_t12_usgs_calibrated_heatmap_rgb.tif` | Discrete speckled RGB preview GeoTIFF. |
| `firewxfm_conus_20260706_t12_usgs_calibrated_heatmap.png` | Discrete speckled PNG preview for quick inspection. |
| `firewxfm_conus_20260706_t12_usgs_calibrated_summary.json` | Summary statistics and calibration metadata for the prediction. |

The example is generated with the no-exposure serving contract: channels `14` and `15` are zeroed, inference uses overlap windows, and modulo-16 phase averaging is applied before writing the map.

The PNG and RGB GeoTIFF are for visualization only. They use sparse discrete speckled raster rendering: no Gaussian blur, no smooth interpolation, a green-dominant categorical palette, deterministic cell-level jitter, small micro-clusters, fragmented warm-color pockets, and display omissions for low-salience cells. Areas with no overlay are not missing probability data. Use the probability GeoTIFF for quantitative analysis.
