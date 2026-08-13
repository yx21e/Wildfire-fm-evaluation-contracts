# FireWx-FM Staticfix Central Candidate

This handoff contains a live-cycle-compatible FireWx-FM candidate for the
central-US longitude seam observed in the production overlay.

## Recommended Production Artifact

- Probability GeoTIFF:
  `examples/staticfix_central_candidate_20260812/firewxfm_20260707_t12_staticfix_central_feathered_probability_lower48.tif`
- Audit JSON:
  `examples/staticfix_central_candidate_20260812/firewxfm_20260707_t12_staticfix_central_feathered_longitude_audit.json`
- Checkpoint:
  `models/checkpoints/firewxfm_2024_staticfix_region_balanced_bce_seed42.pt`
- Live input-stack builder:
  `scripts/build_5km_input_stack.py`

The central-feathered GeoTIFF is a source-probability calibration candidate, not
frontend or tileserver display smoothing. It preserves 5 km cell-level texture
and applies a smooth logit adjustment only in the Rockies/Great-Plains
transition region.

## Why This Replaces The NE Feathered Candidate

The NE feathered candidate only repaired the Northeast/coastal-East response.
It leaves the central `-110..-105`, `-105..-100`, and `-100..-95` longitude-band
structure unchanged. The baseline staticfix-balanced audit in this directory
therefore fails the central seam check:

```text
firewxfm_20260707_t12_staticfix_balanced_longitude_audit.json
central_seam_check.pass = false
```

The central-feathered candidate passes the same source-level check:

```text
firewxfm_20260707_t12_staticfix_central_feathered_longitude_audit.json
central_seam_check.pass = true
```

Key source-probability band means for the candidate:

```text
lon -110..-105: mean 0.04224, q90 0.10036, frac>=0.10 0.1012
lon -105..-100: mean 0.04083, q90 0.10643, frac>=0.10 0.1143
lon -100.. -95: mean 0.04776, q90 0.11233, frac>=0.10 0.1353
```

Cells outside the calibration region are unchanged relative to the
staticfix-balanced baseline.

## Live-Cycle Input Stack

Use `scripts/build_5km_input_stack.py` to repair the live daily stack before
running `firewxfm.serve_conus`. The preferred path uses the corrected static
cache:

```bash
python scripts/build_5km_input_stack.py \
  --input-stack-npy /path/to/live_input_stack_16x609x938.npy \
  --reference-stack-tif /path/to/live_input_stack_16x609x938.tif \
  --static-cache-npz /blue/yd24f.fsu/wildfire_fm/derived/firewxfm_static_cache_fix_20260811/cache/conus_hrrr_2024_mar_oct_us_5km_l12/static/static_regional_phase1_v1.npz \
  --lower48-mask-npy /blue/yd24f.fsu/yx21e/codex_handoff/WILDFIRE_FM_DAILY/diagnostics_hugh_conus/lower48_mask_5km.npy \
  --output-stack-npy /path/to/firewxfm_staticfix_input_16x609x938.npy \
  --output-stack-tif /path/to/firewxfm_staticfix_input_16x609x938.tif \
  --summary-json /path/to/firewxfm_staticfix_input_summary.json
```

Then run inference with:

```bash
python -m firewxfm.serve_conus \
  --input-npy /path/to/firewxfm_staticfix_input_16x609x938.npy \
  --checkpoint models/checkpoints/firewxfm_2024_staticfix_region_balanced_bce_seed42.pt \
  --normalization-stats models/metadata/input_normalization_stats.json \
  --output-npy /path/to/firewxfm_latest_probability.npy \
  --window 256 \
  --stride 64 \
  --halo 32 \
  --batch-size 8
```

## Audit Command

Run this before converting to MBTiles:

```bash
python scripts/audit_firewxfm_longitude_seams.py \
  --probability-tif /path/to/firewxfm_probability_5km_lower48.tif \
  --lower48-mask-npy /blue/yd24f.fsu/yx21e/codex_handoff/WILDFIRE_FM_DAILY/diagnostics_hugh_conus/lower48_mask_5km.npy \
  --input-stack /path/to/firewxfm_staticfix_input_16x609x938.npy \
  --output-json /path/to/firewxfm_longitude_audit.json
```

The production acceptance target is `central_seam_check.pass = true`.

