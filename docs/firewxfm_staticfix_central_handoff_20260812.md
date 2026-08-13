# FireWx-FM Central Seam Handoff

This handoff addresses the longitude-direction seam observed in the production
FireWx-FM source probability raster around the Rockies / Great Plains
transition.

## What Changed

- Added `scripts/build_5km_input_stack.py` as the live-cycle entry point for the
  corrected 16-channel 5 km stack.
- Added `scripts/audit_firewxfm_longitude_seams.py` for source-level seam
  inspection.
- Added `scripts/make_firewxfm_central_feathered_candidate.py` for a
  source-probability calibration candidate.
- Added a production-ready checkpoint reference:
  `models/checkpoints/firewxfm_2024_staticfix_region_balanced_bce_seed42.pt`
- Added a central-feathered GeoTIFF candidate and matching audit outputs under
  `examples/staticfix_central_candidate_20260812/`.

## Current Recommendation

Use the corrected live input-stack builder together with the staticfix-balanced
checkpoint. If the current production probability raster still shows the central
seam after that fix, use the central-feathered source candidate as the
intermediate published artifact while investigating any remaining region-specific
calibration issue.

## Verification

The central-feathered candidate passes the seam audit:

```text
central_seam_check.pass = true
```

The baseline staticfix-balanced raster fails the same audit:

```text
central_seam_check.pass = false
```

