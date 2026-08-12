# Northeast Feathered Candidate

This directory contains a standalone FireWx-FM 12-hour active-fire occupancy
probability candidate for handoff to the `wildfire_interactive_map` website
pipeline. It is not deployed automatically.

## Files

| File | Purpose |
|---|---|
| `firewxfm_20260707_t12_staticfix_ne_feathered_medium_probability_lower48.tif` | Lower-48 masked probability GeoTIFF for website ETL handoff. |
| `firewxfm_20260707_t12_staticfix_ne_feathered_medium_quicklook.png` | Side-by-side quicklook: baseline, candidate, and candidate-minus-baseline. |
| `firewxfm_20260707_t12_staticfix_ne_feathered_medium_audit.json` | Region summaries, seam checks, input-channel summaries, and generation metadata. |

## Handoff

For the website ETL, point `FIREWXFM_PROBABILITY_TIF` to the probability GeoTIFF
above and regenerate `current_firewxfm_forecast.mbtiles` using the existing
`ETLS/WILDFIRE_FM_DAILY` workflow.

This candidate keeps Central and Western US probabilities unchanged. It applies
a feathered Northeast/coastal-East logit calibration using same-grid USGS
surfaces as a soft spatial prior only; those USGS rasters are not treated as
ground-truth active-fire labels.
