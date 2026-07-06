# CONUS Prediction Example

This directory stores a demonstration output generated from `2026-07-06 12Z` HRRR input. The model predicts 12-hour active-fire occupancy probability for `2026-07-07 00Z`.

## Expected Files

| File | Description |
|---|---|
| `firewxfm_conus_20260706_t12_probability_5km_lower48.tif` | Lower-48 masked probability GeoTIFF. |
| `firewxfm_conus_20260706_t12_heatmap_rgb.tif` | Percentile-scaled RGB preview GeoTIFF. |
| `firewxfm_conus_20260706_t12_heatmap.png` | Percentile-scaled PNG preview for quick inspection. |
| `firewxfm_conus_20260706_t12_summary.json` | Summary statistics for the prediction. |

The example is generated with the no-exposure serving contract: channels `14` and `15` are zeroed, inference uses overlap windows, and modulo-16 phase averaging is applied before writing the map.

The PNG and RGB GeoTIFF are for visualization only. They use a robust display transform with the nonzero 99.5th percentile as the upper color cap and gamma `0.45`, so lower-probability regions remain visible. Blue regions indicate lower predicted 12-hour active-fire occupancy for this date, not missing data. Use the probability GeoTIFF for quantitative analysis.
