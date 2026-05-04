# Wildfire FM Evaluation Contracts

Code and release artifacts for reproducing the tables and figures in the wildfire Earth-FM evaluation-contract paper.

## Quick Reproduction

The paper-output reproduction path uses only small released result artifacts. It does not require raw wildfire data, backbone checkpoints, GPUs, or Slurm.

```bash
python scripts/reproduce_all.py
```

This rebuilds:

- `paper_outputs/tables/table_primary_results.tex`
- `paper_outputs/tables/table_supporting_results.tex`
- `paper_outputs/tables/table_fireprone_contract_progression.tex`
- `paper_outputs/tables/table_appendix_additional_values.tex`
- `paper_outputs/figures/fig_fireprone_contract_progression_compact.pdf`
- `paper_outputs/figures/fig_comparator_heatmap_dense.pdf`

## Repository Layout

- `artifacts/results/`: frozen CSV/JSON result artifacts used to rebuild paper outputs.
- `artifacts/manifests/`: table/figure-to-script mapping.
- `paper_outputs/`: generated TeX tables and PDF figures.
- `scripts/`: standard-library Python builders and release audit.
- `experiments/`: Slurm templates and raw-rerun notes.
- `docs/table_code_map.md`: concise map from paper item to code and artifact.

## Audit

Run:

```bash
python scripts/audit_release.py
```

The audit checks that required outputs exist, no local machine paths are exposed, stale backbone labels are absent, and appendix value tables do not contain displayed zero standard deviations or duplicate displayed cells.

## Raw Reruns

Raw-data reruns require wildfire inputs and backbone feature caches that are not bundled in this public release. See `experiments/README.md` and the Slurm templates under `experiments/slurm/`.
