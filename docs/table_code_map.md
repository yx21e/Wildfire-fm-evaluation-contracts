# Table and Figure Reproduction Map

This repository is organized around fixed paper outputs. Each table or figure is regenerated from release artifacts under `artifacts/results/`.

| Paper item | Output | Rebuild command | Main input |
|---|---|---|---|
| Primary fixed-contract table | `paper_outputs/tables/table_primary_results.tex` | `python scripts/build_tables.py` | `artifacts/results/release_table_values.json` |
| Supporting task matrix | `paper_outputs/tables/table_supporting_results.tex` | `python scripts/build_tables.py` | `artifacts/results/release_table_values.json` |
| Fire-prone contract progression table | `paper_outputs/tables/table_fireprone_contract_progression.tex` | `python scripts/build_tables.py` | `artifacts/results/fireprone_contract_progression_summary.raw.json` |
| Appendix occupancy PPR table | `paper_outputs/tables/table_appendix_additional_values.tex` | `python scripts/build_tables.py` | `artifacts/results/fireprone_contract_progression_summary.raw.json` |
| Appendix burned-area error-shape table | `paper_outputs/tables/table_appendix_additional_values.tex` | `python scripts/build_tables.py` | `artifacts/results/cross_task_appendix_supplements.json` |
| Appendix analog retrieval diagnostic table | `paper_outputs/tables/table_appendix_additional_values.tex` | `python scripts/build_tables.py` | `artifacts/results/cross_task_appendix_supplements.json` |
| Appendix smoke high-event table | `paper_outputs/tables/table_appendix_additional_values.tex` | `python scripts/build_tables.py` | `artifacts/results/cross_task_appendix_supplements.json` |
| Appendix heat exceedance-decomposition table | `paper_outputs/tables/table_appendix_additional_values.tex` | `python scripts/build_tables.py` | `artifacts/results/cross_task_appendix_supplements.json` |
| Figure 3 fire-prone bar plot | `paper_outputs/figures/fig_fireprone_contract_progression_compact.pdf` | `python scripts/build_figures.py` | `artifacts/results/fireprone_contract_progression_summary.raw.json` |
| Task-comparator heatmap | `paper_outputs/figures/fig_comparator_heatmap_dense.pdf` | `python scripts/build_figures.py` | `artifacts/results/release_table_values.json` |
| Selection-regret CSVs | `artifacts/results/selection_regret_*.csv` | archived result artifact | fixed-feature head-selection runs |

Run all reproducible paper-output steps:

```bash
python scripts/reproduce_all.py
```

The release audit checks for forbidden local paths, stale labels, missing outputs, and appendix table cells with displayed zero standard deviation.
