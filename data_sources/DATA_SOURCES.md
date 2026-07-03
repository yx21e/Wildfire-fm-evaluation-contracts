# Data Sources

This release documents the source datasets needed for the FireWx-FM active-fire occupancy model. Raw source data are not redistributed. Users should obtain each resource from the original provider and follow the provider terms.

## Source Inventory

| Source | Role | Access |
|---|---|---|
| NOAA High-Resolution Rapid Refresh (HRRR) | Dynamic weather input channels on the 5 km grid. | NOAA/NCEI product page: <https://www.ncei.noaa.gov/products/weather-climate-models/high-resolution-rapid-refresh>; AWS Open Data archive: <https://registry.opendata.aws/noaa-hrrr-pds/>. |
| NASA FIRMS active-fire detections | Active-fire detections used to derive the occupancy target. | FIRMS download and API services: <https://firms.modaps.eosdis.nasa.gov/download/> and <https://firms.modaps.eosdis.nasa.gov/api/>. |
| LANDFIRE 40 Fire Behavior Fuel Models | Static fuel-model input channel. | LANDFIRE data portal: <https://landfire.gov/data>. |
| LANDFIRE canopy cover | Static canopy-cover input channel. | LANDFIRE data portal: <https://landfire.gov/data>. |
| Wildfire Risk to Communities housing-unit density | Static exposure input channel; zeroed in final no-exposure serving. | Wildfire Risk to Communities downloads: <https://wildfirerisk.org/download/>. |
| LandScan Global 2024 | Static population input channel; zeroed in final no-exposure serving. | ORNL LandScan access: <https://landscan.ornl.gov/>. |

## Input Channels

The released checkpoint expects a fixed 16-channel tensor in `[channel, y, x]` order.

| Channel | Name | Dataset/source | Level or selection | Units | Serving treatment |
|---:|---|---|---|---|---|
| 0 | `t2m` | NOAA HRRR | 2 m above ground | K | z-score normalized |
| 1 | `d2m` | NOAA HRRR | 2 m above ground | K | z-score normalized |
| 2 | `u10` | NOAA HRRR | 10 m above ground | m s-1 | z-score normalized |
| 3 | `v10` | NOAA HRRR | 10 m above ground | m s-1 | z-score normalized |
| 4 | `cape` | NOAA HRRR | 0-3000 m above-ground layer | J kg-1 | z-score normalized |
| 5 | `sp` | NOAA HRRR | surface | Pa | z-score normalized |
| 6 | `blh` | NOAA HRRR | boundary-layer diagnostic | m | z-score normalized |
| 7 | `vis` | NOAA HRRR | visibility diagnostic | m | z-score normalized |
| 8 | `prate` | NOAA HRRR | surface precipitation rate | kg m-2 s-1 | z-score normalized |
| 9 | `tp` | NOAA HRRR | accumulated precipitation | kg m-2 | z-score normalized |
| 10 | `dynamic_valid` | input builder | dynamic weather validity | 0-1 | passed through |
| 11 | `static_valid` | input builder | static-layer validity fraction | 0-1 | passed through |
| 12 | `fuel_fbfm40` | LANDFIRE | fire-behavior fuel model | category code | passed through |
| 13 | `canopy_cover` | LANDFIRE | canopy cover | percent | z-score normalized |
| 14 | `housing_density` | Wildfire Risk to Communities | static exposure | provider native | z-score normalized, then zeroed |
| 15 | `population` | LandScan Global 2024 | static exposure | persons | z-score normalized, then zeroed |

FIRMS detections define the occupancy target and are not input channels.

## Static Resampling

| Static layer | Resampling |
|---|---|
| LANDFIRE fire-behavior fuel model | nearest |
| LANDFIRE canopy cover | nearest |
| Wildfire Risk to Communities housing density | bilinear |
| LandScan population | bilinear |

## Notes

- Native grid: Lower-48 CONUS, EPSG:5070, 5 km.
- Prediction target: 12-hour active-fire occupancy probability.
- County and sub-county products are post-inference aggregations of the 5 km probability grid.
