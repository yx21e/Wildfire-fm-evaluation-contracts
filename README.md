# Wildfire FM Evaluation Contracts

Code and release artifacts for reproducing the tables and figures in the wildfire Earth-FM evaluation-contract paper.

This repository also includes a lightweight raw-data downloader and registry-backed data interfaces for local FireWx-FM reconstruction and future general hazard-model adapters. Raw provider data are not redistributed here.

## Quick Reproduction

The paper-output reproduction path uses only small released result artifacts. It does not require raw wildfire data, backbone checkpoints, GPUs, or Slurm.

```bash
python scripts/reproduce_all.py
```

This rebuilds:

- `paper_outputs/tables/table_primary_results.tex`
- `paper_outputs/tables/table_supporting_results.tex`
- `paper_outputs/tables/table_fireprone_contract_progression.tex`
- `paper_outputs/tables/table_appendix_additional_values.tex`, including occupancy PPR, spread AP by \(\Omega\), and cross-task appendix value tables
- `paper_outputs/figures/fig_fireprone_contract_progression_compact.pdf`
- `paper_outputs/figures/fig_comparator_heatmap_dense.pdf`

## Repository Layout

- `artifacts/results/`: frozen CSV/JSON result artifacts used to rebuild paper outputs.
- `artifacts/manifests/`: table/figure-to-script mapping.
- `data_downloader/`: client-facing raw-data downloader for public and credentialed provider sources.
- `registries/`: source, variable, grid, task, and split registries for FireWx-FM compatibility and future adapters.
- `src/wildfire_eval_contracts/data/`: minimal `SampleRecord` and `DatasetAdapter` interfaces.
- `examples/`: small client examples for downloader usage.
- `paper_outputs/`: generated TeX tables and PDF figures.
- `scripts/`: standard-library Python builders and release audit.
- `experiments/`: Slurm templates and raw-rerun notes.
- `docs/table_code_map.md`: concise map from paper item to code and artifact.

## Data Downloader and Registries

List supported raw-data sources:

```bash
python3 -m data_downloader.cli --list
```

Dry-run a small download request:

```bash
python3 -m data_downloader.cli \
  --datasets aqs_pm25 hms_smoke ibtracs \
  --output-root ./downloads_demo \
  --start-year 2024 \
  --end-year 2024 \
  --dry-run \
  --max-files 3
```

Validate the public data registries:

```bash
python3 scripts/validate_registries.py
```

The downloader records raw provider files and manifests only. Model-ready sample construction should be implemented through the registry-backed adapter interfaces in `src/wildfire_eval_contracts/data/`.

## Audit

Run:

```bash
python scripts/audit_release.py
```

The audit checks that required outputs exist, no local machine paths are exposed, stale backbone labels are absent, and appendix value tables use four decimals, no displayed zero standard deviations, no duplicate displayed values, and plain captions with the paper notation.

## Raw Reruns

Raw-data reruns require wildfire inputs and backbone feature caches that are not bundled in this public release. See `experiments/README.md` and the Slurm templates under `experiments/slurm/`.
