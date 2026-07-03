# Final CONUS Prediction Example

This directory stores the final demonstration output for 2024-06-01.

## Expected Files

| File | Description |
|---|---|
| `firewxfm_conus_20240601_probability_5km_lower48.tif` | Lower-48 masked probability GeoTIFF. |
| `firewxfm_conus_20240601_heatmap_rgb.tif` | Rendered heatmap GeoTIFF. |
| `firewxfm_conus_20240601_heatmap.png` | PNG preview for quick inspection. |
| `firewxfm_conus_20240601_summary.json` | Summary statistics for the prediction. |

The example is generated with the final no-exposure serving contract: channels `14` and `15` are zeroed, inference uses overlap windows, and modulo-16 phase averaging is applied before writing the map.

The PNG is for visualization only. Use the probability GeoTIFF for quantitative analysis.
