#!/usr/bin/env python3
from __future__ import annotations

import json

from release_utils import ARTIFACTS, OUT_TABLES, ROW_ORDER, ensure_dirs, ms, ms_from_summary


MODEL_TAG_ORDER = [
    ("reference", "Reference"),
    ("prithvi_wxc", "Prithvi-WxC"),
    ("aurora", "Aurora"),
    ("climax", "ClimaX"),
    ("stormcast", "StormCast"),
    ("pangu24", "Pangu24"),
    ("dlwp", "DLWP"),
    ("fcn", "FCN"),
    ("fengwu", "FengWu"),
    ("fuxi", "FuXi"),
    ("pangu6", "Pangu-Weather"),
    ("alphaearth", "AlphaEarth"),
]

SCOPE_ORDER = [
    ("full_domain", "global"),
    ("train_fire_top05pct", "top 5\\%"),
    ("train_fire_top10pct", "top 10\\%"),
    ("train_fire_top20pct", "top 20\\%"),
]

def load_fireprone_summary() -> dict[tuple[str, str], dict]:
    data = json.loads((ARTIFACTS / "fireprone_contract_progression_summary.raw.json").read_text())
    return {(row["model_tag"], row["scope"]): row for row in data["summary"]}


def build_fireprone_contract_progression() -> None:
    by_key = load_fireprone_summary()
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \caption{Occupancy \(F_1\) scores across global and fire-prone scopes. Global uses the full validation/test domain; top-\(k\) rows use train-defined fire-prone masks from historical fire frequency. Values are percentages from the same validation-selected strict threshold. Tolerance is spatial-only; union adds temporal and spatial matching. \(\Delta\) is union minus strict. Cells report five-seed mean with std in small type.}",
        r"    \label{tab:fireprone_contract_progression}",
        r"    \begin{adjustbox}{max width=\textwidth}",
        r"    \begin{tabular}{@{}llcccc@{}}",
        r"        \toprule",
        r"        Backbone & \(\Omega\) & Strict \(F_1\uparrow\) & Tol.\ \(F_1\uparrow\) & Union \(F_1\uparrow\) & \(\Delta\) \(\uparrow\) \\",
        r"        \midrule",
    ]
    for model_tag, label in MODEL_TAG_ORDER:
        for idx, (scope, scope_label) in enumerate(SCOPE_ORDER):
            row = by_key[(model_tag, scope)]
            name = label if idx == 0 else ""
            cells = [
                ms_from_summary(row["strict_f1"], scale=100.0),
                ms_from_summary(row["tolerance_f1"], scale=100.0),
                ms_from_summary(row["union_f1"], scale=100.0),
                ms_from_summary(row["difference"], scale=100.0),
            ]
            lines.append("        " + name + " & " + scope_label + " & " + " & ".join(cells) + r" \\")
        if model_tag != MODEL_TAG_ORDER[-1][0]:
            lines.append(r"        \addlinespace[1pt]")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \end{adjustbox}",
            r"\end{table*}",
            "",
        ]
    )
    (OUT_TABLES / "table_fireprone_contract_progression.tex").write_text("\n".join(lines))


def build_occupancy_ppr() -> str:
    by_key = load_fireprone_summary()
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\caption{Occupancy predicted-positive rate.",
        r"Values are percentages under the same validation-selected strict threshold.",
        r"Scopes \(\Omega\) are fixed before test scoring; cells report five-seed mean with std in small type.}",
        r"\label{tab:app_occupancy_ppr_scope}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Backbone} & \textbf{\(\Omega=\)global} & \textbf{\(\Omega=\)top 5\%} & \textbf{\(\Omega=\)top 10\%} & \textbf{\(\Omega=\)top 20\%} \\",
        r"\midrule",
    ]
    for model_tag, label in MODEL_TAG_ORDER:
        row_cells = []
        for scope, _scope_label in SCOPE_ORDER:
            row_cells.append(ms_from_summary(by_key[(model_tag, scope)]["predicted_positive_rate"], scale=100.0))
        lines.append(label + " & " + " & ".join(row_cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_appendix_additional_values() -> None:
    parts = [build_occupancy_ppr()]
    supplement_path = ARTIFACTS / "cross_task_appendix_supplements.json"
    if supplement_path.exists():
        payload = json.loads(supplement_path.read_text())
        for name in payload["accepted_tables"]:
            parts.append(build_cross_task_supplement_table(name, payload["tables"][name]))
    (OUT_TABLES / "table_appendix_additional_values.tex").write_text("\n".join(parts))


SUPPLEMENT_META = {
    "burned_area_error_shape": {
        "headers": ["log median AE", "acre median AE", "acre MAPE"],
        "caption": "Final burned-area error-shape check. This table adds median and acre-scale errors beyond the main log-RMSE/log-MAE/Spearman columns.",
        "label": "tab:app_burned_area_error_shape",
    },
    "analog_rank_depth": {
        "headers": ["nDCG@5", "best log gap", r"rank $\rho$"],
        "caption": "Analog retrieval diagnostic check. This table adds nDCG@5, best log gap, and rank correlation beyond the main nDCG@10/log-RMSE/log-MAE columns.",
        "label": "tab:app_analog_rank_depth",
    },
    "smoke_high_event": {
        "headers": ["high-smoke RMSE", "high-smoke MAE", "high-smoke bias"],
        "caption": r"Smoke PM$_{2.5}$ high-event check. Metrics use test rows with observed PM$_{2.5}\ge35$ and bootstrap those rows.",
        "label": "tab:app_smoke_high_event",
    },
    "heat_event_decomposition": {
        "headers": ["precision", "recall"],
        "caption": "Extreme-heat exceedance decomposition. This table reports precision and recall behind the main exceedance-F1 column.",
        "label": "tab:app_heat_event_decomposition",
    },
}


def build_cross_task_supplement_table(name: str, table: dict) -> str:
    meta = SUPPLEMENT_META[name]
    metrics = table["metrics"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{meta['caption']} Cells report mean with small std.}}",
        rf"\label{{{meta['label']}}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{l" + "c" * len(metrics) + r"}",
        r"\toprule",
        "Backbone & " + " & ".join(meta["headers"]) + r" \\",
        r"\midrule",
    ]
    summary = table["summary"]
    for label in ROW_ORDER:
        if label not in summary:
            continue
        row = summary[label]
        cells = [ms_from_summary(row[metric]) for metric in metrics]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_primary_results() -> None:
    manifest = json.loads((ARTIFACTS / "release_table_values.json").read_text())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.20}",
        r"\caption{\textbf{Primary occupancy and fire-spread results (RQ3).} Occupancy metrics: exact, tolerated, and union $F_1$ (\%). Fire spread metrics: exact $F_1$, spatial $F_1$, and AP (\%). Each block uses the same $\mathcal{T}$, $\Lambda$, $\Omega$, and $\mathcal{A}$ choices.}",
        r"\label{tab:primary_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Occupancy}} & \multicolumn{3}{c}{\textbf{Fire spread}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"\textbf{Comparator} & \textbf{Exact $F_1\uparrow$} & \textbf{Tol.\ $F_1\uparrow$} & \textbf{Union $F_1\uparrow$} & \textbf{Exact $F_1\uparrow$} & \textbf{Spatial $F_1\uparrow$} & \textbf{AP$\uparrow$} \\",
        r"\midrule",
    ]
    for idx, row in enumerate(manifest["primary"]["rows"]):
        label = row["name"]
        cells = [ms(cell["mean"], cell["std"]) for cell in row["cells"]]
        display = "Wildfire ref." if label == "Reference" else label
        lines.append(display + "\n& " + " & ".join(cells) + r" \\")
        if idx == 0:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    (OUT_TABLES / "table_primary_results.tex").write_text("\n".join(lines))


def build_supporting_results() -> None:
    manifest = json.loads((ARTIFACTS / "release_table_values.json").read_text())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\caption{\textbf{Supporting task results (RQ4).} Top block: final burned area and analog retrieval. Bottom block: smoke PM$_{2.5}$ and extreme heat. Each block uses the same $\mathcal{T}$, $\Lambda$, and $\Omega$ choices.}",
        r"\label{tab:supporting_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Burned area}} & \multicolumn{3}{c}{\textbf{Analog retrieval}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"\textbf{Backbone} & \textbf{log-RMSE$\downarrow$} & \textbf{log-MAE$\downarrow$} & \textbf{Spearman$\uparrow$} & \textbf{nDCG@10$\uparrow$} & \textbf{log-RMSE$\downarrow$} & \textbf{log-MAE$\downarrow$} \\",
        r"\midrule",
    ]
    for idx, row in enumerate(manifest["supporting_top"]["rows"]):
        label = row["name"]
        display = "Wildfire ref." if label == "Reference" else label
        cells = [ms(cell["mean"], cell["std"]) for cell in row["cells"]]
        lines.append(display + "\n& " + " & ".join(cells) + r" \\")
        if idx == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            "",
            r"\vspace{4pt}",
            "",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"& \multicolumn{3}{c}{\textbf{Smoke PM$_{2.5}$}} & \multicolumn{3}{c}{\textbf{Extreme heat}} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"\textbf{Backbone} & \textbf{RMSE$\downarrow$} & \textbf{MAE$\downarrow$} & \textbf{Pearson $r\uparrow$} & \textbf{RMSE-C$\downarrow$} & \textbf{MAE-C$\downarrow$} & \textbf{Exceed.\ $F_1\uparrow$} \\",
            r"\midrule",
        ]
    )
    for idx, row in enumerate(manifest["supporting_bottom"]["rows"]):
        label = row["name"]
        display = "Wildfire ref." if label == "Reference" else label
        cells = [ms(cell["mean"], cell["std"]) for cell in row["cells"]]
        lines.append(display + "\n& " + " & ".join(cells) + r" \\")
        if idx == 0:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    (OUT_TABLES / "table_supporting_results.tex").write_text("\n".join(lines))


def build_all() -> None:
    ensure_dirs()
    build_primary_results()
    build_supporting_results()
    build_fireprone_contract_progression()
    build_appendix_additional_values()


if __name__ == "__main__":
    build_all()
