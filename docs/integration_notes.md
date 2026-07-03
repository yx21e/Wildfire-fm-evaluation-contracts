# Integration Notes

This note summarizes the final FireWx-FM serving pipeline and the main changes relative to the first CONUS adaptation attempt.

## Main Cause of the Center-Dot Artifact

The visible center-dot pattern came from using the 32 by 32 training crop as the CONUS inference unit. Training uses 32 by 32 tiles to balance sparse fire labels. That crop size is not the native serving resolution and should not be stitched across CONUS as independent tiles.

When the map is reconstructed from non-overlapping or weakly overlapping 32 by 32 patches, the U-Net response is repeatedly written from the same patch-relative locations. The result looks like fixed points or small regular structures inside each box, even when the underlying weather and static fields are spatially smoother.

## Final Serving Fix

The final serving path uses larger overlapping windows and phase averaging:

- Window: `256`.
- Stride: `64`.
- Halo crop: `32`.
- Phase shifts: `0,4,8,12` along both axes.
- Aggregation: average the shifted probability maps after shifting them back.

This keeps the model architecture unchanged. The fix is in the reconstruction contract: evaluate context-rich windows, crop unreliable interior seams, and average multiple tile-origin phases.

## Input Contract

The model consumes a 16-channel `[channel, y, x]` tensor. Channel order is fixed and documented in:

```text
models/metadata/input_channels.json
docs/inference_contract.md
```

The two points most likely to cause mismatches are:

- CAPE is the NOAA HRRR 0-3000 m above-ground layer.
- Continuous channels are z-score normalized with the train-split statistics in `models/metadata/input_normalization_stats.json`. The released template normalizes channels `0-9`, `13`, `14`, and `15`.

## Exposure Channels

The final no-exposure serving checkpoint zeros channels `14` and `15` during inference:

- `14`: housing density.
- `15`: population.

This prevents the demonstration map from being driven by population or housing patterns when the desired output is a physical active-fire occupancy field.

## Masks

The masks are numeric input features:

- `dynamic_valid`: dynamic weather-input validity after cache construction.
- `static_valid`: fraction of static layers valid after reprojection.

They are not output masks. They should be supplied alongside the physical channels so missing or invalid source cells are not confused with genuine zero values.

## Static Resampling

The cache template records the intended static-layer resampling:

- LANDFIRE categorical fuel model: nearest neighbor.
- LANDFIRE canopy cover: nearest neighbor in the current cache template.
- Housing density: bilinear.
- Population: bilinear.

The final example output is additionally clipped to the Lower-48 land mask before visualization.

## Final Files To Use

Use these files when the repository artifacts are present:

```text
models/checkpoints/firewxfm_conus_lowposw_noexposure_seed42.pt
models/metadata/input_normalization_stats.json
models/metadata/input_channels.json
firewxfm/serve_conus.py
firewxfm/tiled_inference.py
```

Run `python scripts/check_release_files.py` to verify that the local checkout has the checkpoint, metadata, and final example outputs.
