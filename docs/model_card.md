# FireWx-FM Model Card

## Summary

FireWx-FM is a compact U-Net model trained for 12-hour active-fire occupancy prediction on a 5 km EPSG:5070 grid over the Lower 48 United States. The final serving checkpoint is intended for gridded wildfire-risk screening and for downstream aggregation to county or sub-county units.

## Inputs

The model uses a fixed 16-channel tensor containing HRRR weather variables, validity masks, LANDFIRE fuel and canopy layers, Wildfire Risk to Communities housing density, and LandScan population. The final no-exposure serving run zeros the housing and population channels.

## Output

The output is a probability map with the same spatial grid as the input. Each value represents the model-estimated probability of active-fire occupancy at a 12-hour lead.

## Training Scope

The released final model family was trained with chronological splitting over 2024 fire-season time maps. Training uses tile sampling to reduce empty-cell dominance, per-channel input normalization for continuous variables, sparse-label-aware loss weighting, and an auxiliary spatial-support objective.

## Intended Use

- Research evaluation of wildfire-focused gridded prediction.
- CONUS-scale active-fire occupancy screening.
- County or sub-county summaries produced from the native 5 km grid by transparent aggregation.

## Not Intended For

- Emergency response decisions without expert review.
- Direct replacement of operational fire-weather or incident-management products.
- Interpretation as a calibrated forecast without local validation.
- Inference from inputs that do not follow the documented channel order and normalization contract.

## Data Boundary

This repository does not redistribute raw source data. Users must obtain NOAA HRRR, NASA FIRMS, LANDFIRE, Wildfire Risk to Communities, and LandScan resources from their original providers.

## Known Limitations

- The native model is gridded at 5 km resolution; finer administrative summaries are aggregations, not new fine-scale predictions.
- Output quality depends on correct reprojection, normalization, mask handling, and source-variable selection.
