#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".py", ".sh", ".md", ".tex", ".csv", ".json", ".txt", ".yml", ".yaml", ".toml"}
FORBIDDEN = [
    "/home/yx21e",
    "/blue/",
    "/orange/",
    "yd24f",
    "fsu-compsci",
    "Pangu-Weather 6h",
    "TBD",
    "pending",
    "N/A",
]


def iter_text_files() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.name == "audit_release.py":
            continue
        if path.suffix in TEXT_SUFFIXES:
            out.append(path)
    return sorted(out)


def audit_forbidden() -> list[str]:
    issues: list[str] = []
    for path in iter_text_files():
        text = path.read_text(errors="ignore")
        for token in FORBIDDEN:
            if token in text:
                issues.append(f"{path.relative_to(ROOT)} contains forbidden token {token!r}")
    return issues


def audit_tex_cells() -> list[str]:
    issues: list[str] = []
    for path in sorted((ROOT / "paper_outputs" / "tables").glob("*.tex")):
        text = path.read_text()
        cells = re.findall(r"\\ms\{([^}]*)\}\{([^}]*)\}", text)
        by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, (mean, std) in enumerate(cells, start=1):
            by_cell[(mean, std)].append(idx)
            if std == "0.0000":
                issues.append(f"{path.relative_to(ROOT)} cell {idx} displays zero std: {mean}+/-{std}")
        # Identical cells are allowed in the primary/supporting headline tables only when
        # they are documented same-run rows. The appendix value tables are stricter.
        if path.name.startswith("table_appendix"):
            for cell, idxs in by_cell.items():
                if len(idxs) > 1:
                    issues.append(f"{path.relative_to(ROOT)} repeats displayed cell {cell} at positions {idxs}")
    return issues


def audit_required_outputs() -> list[str]:
    required = [
        "paper_outputs/tables/table_primary_results.tex",
        "paper_outputs/tables/table_supporting_results.tex",
        "paper_outputs/tables/table_fireprone_contract_progression.tex",
        "paper_outputs/tables/table_appendix_additional_values.tex",
        "paper_outputs/figures/fig_fireprone_contract_progression_compact.pdf",
        "paper_outputs/figures/fig_comparator_heatmap_dense.pdf",
        "artifacts/results/release_table_values.json",
        "artifacts/results/fireprone_contract_progression_summary.raw.json",
        "artifacts/results/selection_regret_per_seed.csv",
        "artifacts/results/selection_regret_summary.csv",
    ]
    return [f"missing required output {name}" for name in required if not (ROOT / name).exists()]


def main() -> None:
    issues = audit_required_outputs() + audit_forbidden() + audit_tex_cells()
    if issues:
        print("Release audit failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Release audit passed.")


if __name__ == "__main__":
    main()
