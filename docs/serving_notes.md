# Serving Notes

FireWx-FM predicts 12-hour active-fire occupancy on a native 5 km grid. This page records the serving choices needed for full-map inference.

## Use The Fixed Channel Contract

The model consumes a 16-channel tensor in `[channel, y, x]` order. The channel list and units are defined in:

```text
models/metadata/input_channels.json
docs/inference_contract.md
```

Two details are especially important:

- The CAPE channel is NOAA HRRR CAPE for the 0-3000 m above-ground layer.
- Continuous channels are z-score normalized with `models/metadata/input_normalization_stats.json`.

## Apply The No-Exposure Serving Contract

The released checkpoint is served with channels `14` and `15` zeroed after normalization:

- `14`: housing density.
- `15`: population.

The output should therefore be interpreted as active-fire occupancy probability under the released no-exposure serving contract.

## Use Overlap-Window Inference

Training uses 32 by 32 crops to sample sparse active-fire labels. That crop size is not the full-map serving tile size. For CONUS-scale inference, use the default serving settings:

| Setting | Value |
|---|---:|
| Window | `256` |
| Stride | `64` |
| Halo crop | `32` |
| Phase shifts | `0,4,8,12` along y and x |

The phase ensemble evaluates shifted versions of the same input and averages the shifted-back probability maps. This reduces dependence on tile origin and produces a smoother full-domain probability field.

## Preserve Validity Masks

The validity masks are numeric input features:

- `dynamic_valid`: dynamic weather-input validity after cache construction.
- `static_valid`: fraction of static layers valid after reprojection.

They are not output masks. They should be supplied with the physical channels so sanitized missing values are not confused with true zeros.
