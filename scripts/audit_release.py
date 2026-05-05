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
    "diagnostic",
    "decomposition",
    "error-shape",
    "supporting evidence",
    "high-smoke bias",
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
            label_match = re.search(r"\\label\{([^}]*)\}", block)
            label = label_match.group(1) if label_match else f"table {table_idx}"
            is_appendix_value = label.startswith("tab:app_") and label not in {"tab:app_occupancy_ppr_scope", "tab:app_head_architectures"}
            if is_appendix_value:
                if r"\mathcal{T}" not in block:
                    issues.append(f"{path.relative_to(ROOT)} {label} caption does not use \\mathcal{{T}}")
                if r"\Omega" not in block:
                    issues.append(f"{path.relative_to(ROOT)} {label} caption does not use \\Omega")
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
            for mean, idxs in by_mean.items():
                if is_appendix_value and len(idxs) > 1:
                    issues.append(f"{path.relative_to(ROOT)} table {table_idx} repeats displayed mean {mean} at positions {idxs}")
    return issues


def audit_summary_artifact(path: Path, table_name: str, table: dict, scale: float = 1.0) -> list[str]:
    issues: list[str] = []
    for metric in table["metrics"]:
        by_cell: dict[str, list[str]] = defaultdict(list)
        by_mean: dict[str, list[str]] = defaultdict(list)
        for label, row in table["summary"].items():
            node = row[metric]
            mean = f"{float(node['mean']) * scale:.4f}"
            std = f"{float(node['std']) * scale:.4f}"
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


def audit_appendix_artifacts() -> list[str]:
    issues: list[str] = []
    cross_path = ROOT / "artifacts" / "results" / "cross_task_appendix_supplements.json"
    if cross_path.exists():
        data = __import__("json").loads(cross_path.read_text())
        for table_name in data.get("accepted_tables", []):
            issues.extend(audit_summary_artifact(cross_path, table_name, data["tables"][table_name]))
    spread_path = ROOT / "artifacts" / "results" / "spread_appendix_ap_by_scope.json"
    if spread_path.exists():
        data = __import__("json").loads(spread_path.read_text())
        issues.extend(
            audit_summary_artifact(
                spread_path,
                "spread_ap_by_scope",
                {"metrics": data["metrics"], "summary": data["summary"]},
                scale=100.0,
            )
        )
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
        "artifacts/results/spread_appendix_ap_by_scope.json",
        "artifacts/results/selection_regret_per_seed.csv",
        "artifacts/results/selection_regret_summary.csv",
    ]
    return [f"missing required output {name}" for name in required if not (ROOT / name).exists()]


def main() -> None:
    issues = audit_required_outputs() + audit_forbidden() + audit_tex_cells() + audit_appendix_artifacts()
    if issues:
        print("Release audit failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Release audit passed.")


if __name__ == "__main__":
    main()
