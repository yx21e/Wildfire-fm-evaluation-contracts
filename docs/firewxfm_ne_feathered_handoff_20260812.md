# FireWx-FM Northeast Candidate Handoff

This is a minimal handoff for testing a corrected FireWx-FM probability map in
the `wildfire_interactive_map` website pipeline.

## Candidate

Use this repo artifact:

```text
examples/ne_feathered_candidate_20260812/firewxfm_20260707_t12_staticfix_ne_feathered_medium_probability_lower48.tif
```

Server-side source path used to generate the repo artifact:

```text
/blue/yd24f.fsu/wildfire_fm/derived/firewxfm_post_inspection_20260812/ne_feathered_candidate/firewxfm_20260707_t12_staticfix_ne_feathered_medium_probability_lower48.tif
```

## What Changed

The candidate starts from the staticfix-balanced FireWx-FM output and applies a
smooth Northeast/coastal-East calibration. The correction is feathered, not a
hard longitude cutoff, and uses same-grid 2026-07-07 USGS fire-danger rasters as
a soft spatial prior.

Important guardrails:

- Central and Western US probabilities are unchanged.
- The `lon=-90` seam check is unchanged from the baseline, so no new vertical
  boundary was introduced there.
- The correction is standalone and has not been deployed to `rai-fire.com`.

Key audit values:

- Northeast `p >= 0.10`: baseline `0.0143`, candidate `0.0775`.
- Mid-Atlantic `p >= 0.10`: baseline `0.1527`, candidate `0.2467`.
- West mean probability: baseline `0.07359`, candidate `0.07359`.
- Central mean probability: baseline `0.03585`, candidate `0.03585`.

Full audit:

```text
examples/ne_feathered_candidate_20260812/firewxfm_20260707_t12_staticfix_ne_feathered_medium_audit.json
```

Quicklook:

```text
examples/ne_feathered_candidate_20260812/firewxfm_20260707_t12_staticfix_ne_feathered_medium_quicklook.png
```

## Website Integration

On the website host, update:

```text
ETLS/WILDFIRE_FM_DAILY/.env
```

Set:

```text
FIREWXFM_PROBABILITY_TIF=/absolute/path/to/firewxfm_20260707_t12_staticfix_ne_feathered_medium_probability_lower48.tif
```

Then run the existing website ETL/deploy flow. For tile-only testing, the usual
path is:

```bash
cd ETLS/WILDFIRE_FM_DAILY
FIREWXFM_SKIP_DB=1 ./job.sh
```

For production, follow the existing `HANDOFF_FIREWXFM_PRODUCTION_DEPLOY.md` in
the website repo and rebuild/restart the Docker services after regenerating the
MBTiles.

## Caveat

No local 2026-07 FIRMS active-fire target was found during this pass. Historical
FIRMS evaluation supports the staticfix-balanced model as the stronger base
model, while the 2026-07 USGS rasters are used only as a same-day spatial prior
for this handoff candidate.
