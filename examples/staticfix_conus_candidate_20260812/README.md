# FireWx-FM Staticfix CONUS Candidate

This handoff contains the all-US FireWx-FM repair candidate we want to push
upstream. It is CONUS-wide, not Northeast-only.

## Recommended Production Artifact

- Probability GeoTIFF:
  `examples/staticfix_conus_candidate_20260812/firewxfm_20260707_t12_staticfix_conus_probability_lower48.tif`
- Audit JSON:
  `examples/staticfix_conus_candidate_20260812/firewxfm_20260707_t12_staticfix_conus_longitude_audit.json`
- Checkpoint:
  `models/checkpoints/firewxfm_2024_staticfix_region_balanced_bce_seed42.pt`
- Live input-stack builder:
  `scripts/build_5km_input_stack.py`

The CONUS candidate is a source-probability calibration product, not frontend
or tileserver display smoothing. It preserves 5 km cell-level texture and
applies a smooth logit adjustment across the repaired CONUS seam region.

## Why This Replaces The NE Feathered Candidate

The NE feathered candidate only repaired the Northeast/coastal-East response.
The CONUS candidate keeps that fix but also removes the visible central
longitude-band seam. The baseline staticfix-balanced audit in this directory
therefore fails the seam check:

```text
firewxfm_20260707_t12_staticfix_balanced_longitude_audit.json
central_seam_check.pass = false
```

The CONUS candidate passes the same source-level check:

```text
firewxfm_20260707_t12_staticfix_conus_longitude_audit.json
central_seam_check.pass = true
```

Key source-probability band means for the candidate:

```text
lon -110..-105: mean 0.04224, q90 0.10036, frac>=0.10 0.1012
lon -105..-100: mean 0.04083, q90 0.10643, frac>=0.10 0.1143
lon -100.. -95: mean 0.04776, q90 0.11233, frac>=0.10 0.1353
lon  -95.. -90: mean 0.04597, q90 0.09843, frac>=0.10 0.0960
lon  -90.. -85: mean 0.04689, q90 0.09101, frac>=0.10 0.0684
lon  -85.. -75: mean 0.06279, q90 0.12858, frac>=0.10 0.1968
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

