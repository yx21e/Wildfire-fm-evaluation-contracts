# FireWx-FM CONUS Staticfix Handoff

This handoff packages the all-US FireWx-FM repair we want to push upstream.
It is not a Northeast-only tweak. It is the CONUS-serving version built on the
staticfix-balanced checkpoint and a source-level probability candidate that
removes the visible longitude seam while keeping the East and West response
regions intact.

## Main Artifacts

- Live input-stack builder: `scripts/build_5km_input_stack.py`
- CONUS candidate script: `scripts/make_firewxfm_conus_feathered_candidate.py`
- Probability GeoTIFF:
  `examples/staticfix_conus_candidate_20260812/firewxfm_20260707_t12_staticfix_conus_probability_lower48.tif`
- Audit JSON:
  `examples/staticfix_conus_candidate_20260812/firewxfm_20260707_t12_staticfix_conus_longitude_audit.json`
- Checkpoint:
  `models/checkpoints/firewxfm_2024_staticfix_region_balanced_bce_seed42.pt`

## What This Fix Covers

- All Lower-48 CONUS serving, not just the Northeast.
- The Rockies / Great Plains longitude seam.
- Eastern response that was previously too weak or too flat.
- West and Central bands, which remain part of the same calibrated field.

## Verification

The CONUS candidate passes the source-level seam audit:

```text
central_seam_check.pass = true
```

Representative longitude-band means:

```text
lon -110..-105: mean 0.04224
lon -105..-100: mean 0.04083
lon -100.. -95: mean 0.04776
lon  -95.. -90: mean 0.04597
lon  -90.. -85: mean 0.04689
lon  -85.. -75: mean 0.06279
```

## Production Use

1. Rebuild the live 16-channel stack with `scripts/build_5km_input_stack.py`.
2. Run `firewxfm.serve_conus` with the staticfix-balanced checkpoint.
3. Publish the repaired CONUS probability GeoTIFF to MBTiles.
4. Keep the old NE handoff only as history.

