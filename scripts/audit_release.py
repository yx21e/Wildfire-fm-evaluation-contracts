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
        for table_idx, block in enumerate(re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, flags=re.S), start=1):
            cells = re.findall(r"\\ms\{([^}]*)\}\{([^}]*)\}", block)
            by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
            by_mean: dict[str, list[int]] = defaultdict(list)
            for idx, (mean, std) in enumerate(cells, start=1):
                by_cell[(mean, std)].append(idx)
                by_mean[mean].append(idx)
                if not re.fullmatch(r"-?\d+\.\d{4}", mean):
                    issues.append(f"{path.relative_to(ROOT)} table {table_idx} cell {idx} mean is not four decimals: {mean}")
                if not re.fullmatch(r"-?\d+\.\d{4}", std):
                    issues.append(f"{path.relative_to(ROOT)} table {table_idx} cell {idx} std is not four decimals: {std}")
                if std == "0.0000":
                    issues.append(f"{path.relative_to(ROOT)} table {table_idx} cell {idx} displays zero std: {mean}+/-{std}")
            for cell, idxs in by_cell.items():
                if len(idxs) > 1:
                    issues.append(f"{path.relative_to(ROOT)} table {table_idx} repeats displayed cell {cell} at positions {idxs}")
    return issues


def audit_cross_task_supplement_artifact() -> list[str]:
    issues: list[str] = []
    path = ROOT / "artifacts" / "results" / "cross_task_appendix_supplements.json"
    if not path.exists():
        return issues
    data = __import__("json").loads(path.read_text())
    for table_name in data.get("accepted_tables", []):
        table = data["tables"][table_name]
        for metric in table["metrics"]:
            by_cell: dict[str, list[str]] = defaultdict(list)
            by_mean: dict[str, list[str]] = defaultdict(list)
            for label, row in table["summary"].items():
                node = row[metric]
                mean = f"{float(node['mean']):.4f}"
                std = f"{float(node['std']):.4f}"
                if std == "0.0000":
                    issues.append(f"{path.relative_to(ROOT)} {table_name} {label} {metric} displays zero std")
                by_cell[f"{mean}+/-{std}"].append(label)
                by_mean[mean].append(label)
            for cell, labels in by_cell.items():
                if len(labels) > 1:
                    issues.append(f"{path.relative_to(ROOT)} {table_name} {metric} repeats displayed cell {cell} across {labels}")
            for mean, labels in by_mean.items():
                if len(labels) > 1:
                    issues.append(f"{path.relative_to(ROOT)} {table_name} {metric} repeats displayed mean {mean} across {labels}")
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
    issues = audit_required_outputs() + audit_forbidden() + audit_tex_cells() + audit_cross_task_supplement_artifact()
    if issues:
        print("Release audit failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Release audit passed.")


if __name__ == "__main__":
    main()
