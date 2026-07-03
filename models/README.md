# Models

This directory stores the released FireWx-FM model artifacts.

## Expected Files

| Path | Purpose |
|---|---|
| `checkpoints/firewxfm_conus_lowposw_noexposure_seed42.pt` | Final CONUS no-exposure serving checkpoint. |
| `metadata/input_channels.json` | Machine-readable input-channel contract. |
| `metadata/input_normalization_stats.json` | Train-split z-score statistics used at inference. |
| `metadata/final_model_manifest.json` | Serving manifest for the final checkpoint. |
| `metadata/run_summary.json` | Training/evaluation summary for the final run. |
| `metadata/tile_summary.json` | Tile construction and sparse-label summary. |

Run the release check from the repository root:

```bash
python scripts/check_release_files.py
```

If the checkpoint is absent, inference code can still be inspected, but the final prediction example cannot be reproduced from this checkout.
