# External Fire-Danger Comparison

This directory stores a same-scale sanity check against public USGS fire-danger products.

## Expected Files

| File | Description |
|---|---|
| `rank_heatmaps_firewxfm_and_usgs_fdf.png` | Same-scale rank heatmaps for FireWx-FM and external products. |
| `rank_difference_usgs_minus_firewxfm.png` | Rank-difference maps between external products and FireWx-FM. |
| `external_fire_danger_summary.csv` | Summary correlations and overlap statistics. |
| `external_fire_danger_comparison.json` | Machine-readable comparison details with public source labels. |

The comparison is intentionally rank-based and same-scale. Public fire-danger products and FireWx-FM do not predict exactly the same target: the external products describe fire danger, spread potential, or large-fire probability, while FireWx-FM predicts 12-hour active-fire occupancy.

In the final comparison run, Spearman rank correlations with FireWx-FM were approximately `0.570` for USGS WFSP, `0.474` for USGS WLFP, and `0.206` for USGS WFPI. These values support a broad spatial sanity check while preserving the target mismatch.
