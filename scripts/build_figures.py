#!/usr/bin/env python3
from __future__ import annotations

import json

from release_utils import ARTIFACTS, OUT_FIGURES, ROW_ORDER, ensure_dirs
from simple_pdf import PdfCanvas, clamp, draw_axes, mix


MODEL_ORDER = [
    ("reference", "Reference", "Ref."),
    ("prithvi_wxc", "Prithvi-WxC", "WxC"),
    ("aurora", "Aurora", "Aurora"),
    ("climax", "ClimaX", "ClimaX"),
    ("stormcast", "StormCast", "Storm"),
    ("pangu24", "Pangu24", "Pangu24"),
    ("alphaearth", "AlphaEarth", "Alpha"),
    ("dlwp", "DLWP", "DLWP"),
    ("fcn", "FCN", "FCN"),
    ("fengwu", "FengWu", "FengWu"),
    ("fuxi", "FuXi", "FuXi"),
    ("pangu6", "Pangu-Weather", "Pangu-W"),
]
SCOPE_ORDER = [
    ("full_domain", "global"),
    ("train_fire_top05pct", "top 5%"),
    ("train_fire_top10pct", "top 10%"),
    ("train_fire_top20pct", "top 20%"),
]


def heat_color(value: float) -> tuple[float, float, float]:
    if value <= 0.5:
        return mix((0.94, 0.95, 0.94), (0.54, 0.79, 0.76), value / 0.5)
    return mix((0.54, 0.79, 0.76), (0.05, 0.38, 0.42), (value - 0.5) / 0.5)


def luminance(color: tuple[float, float, float]) -> float:
    r, g, b = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def dashed_vline(c: PdfCanvas, x: float, y_start: float, y_end: float) -> None:
    dash, gap = 7.0, 5.0
    y = y_start
    while y < y_end:
        c.line([(x, y), (x, min(y + dash, y_end))], color=(0.42, 0.44, 0.46), lw=0.75)
        y += dash + gap


def build_fireprone_contract_progression() -> None:
    data = json.loads((ARTIFACTS / "fireprone_contract_progression_summary.raw.json").read_text())
    by_key = {(row["model_tag"], row["scope"]): row for row in data["summary"]}
    c = PdfCanvas(width=1320, height=470)
    c.rect(0, 0, c.width, c.height, fill=(1, 1, 1))

    x0, y0, plot_w, plot_h, ymax = 72, 132, 1194, 268, 80.0
    draw_axes(c, x0, y0, plot_w, plot_h, ymax, [0, 20, 40, 60, 80])
    c.text(x0 - 38, y0 + plot_h + 8, "F1 (%)", size=8, color=(0.15, 0.15, 0.15), bold=True)

    colors = {
        "strict": (0.09, 0.22, 0.37),
        "tolerance": (0.31, 0.55, 0.80),
        "union": (0.75, 0.84, 0.94),
    }
    block_gap = 10.0
    fire_gap = 28.0
    block_w = (plot_w - fire_gap - 2 * block_gap) / len(SCOPE_ORDER)
    bar_step = block_w / len(MODEL_ORDER)
    bar_w = min(18.0, bar_step * 0.80)

    scope_lefts: dict[str, float] = {}
    current_x = x0 + 8.0
    for scope_idx, (scope, scope_label) in enumerate(SCOPE_ORDER):
        if scope_idx == 1:
            dashed_vline(c, current_x - fire_gap / 2.0, y0 - 6, y0 + plot_h + 16)
        scope_lefts[scope] = current_x
        c.text(current_x + block_w / 2.0, y0 + plot_h + 17, scope_label, size=15.0, align="center", bold=True)
        if scope_idx < len(SCOPE_ORDER) - 1:
            current_x += block_w + (fire_gap if scope_idx == 0 else block_gap)

    for scope, _scope_label in SCOPE_ORDER:
        block_x = scope_lefts[scope]
        for idx, (model_tag, _label, short) in enumerate(MODEL_ORDER):
            row = by_key[(model_tag, scope)]
            strict = row["strict_f1"]["mean"] * 100.0
            tolerance = row["tolerance_f1"]["mean"] * 100.0
            union = row["union_f1"]["mean"] * 100.0
            bx = block_x + idx * bar_step + (bar_step - bar_w) / 2.0
            base = y0
            for segment, value in [
                ("strict", max(0.0, strict)),
                ("tolerance", max(0.0, tolerance - strict)),
                ("union", max(0.0, union - tolerance)),
            ]:
                height = plot_h * value / ymax
                if height <= 0:
                    continue
                c.rect(bx, base, bar_w, height, fill=colors[segment], stroke=(1, 1, 1), lw=0.35)
                base += height
            c.text_rotated(bx + bar_w / 2.0 - 3.0, y0 - 76, short, angle_deg=-45.0, size=10.0, align="right")

    legend_x, legend_y = x0 + 18, y0 + plot_h - 26
    c.rect(legend_x - 13, legend_y - 12, 304, 23, fill=(0.98, 0.98, 0.96), stroke=(0.78, 0.80, 0.78), lw=0.45)
    for idx, (label, color) in enumerate([("Strict", colors["strict"]), ("Tolerance", colors["tolerance"]), ("Union", colors["union"])]):
        x = legend_x + idx * 98
        c.rect(x, legend_y - 3, 24, 9, fill=color, stroke=(1, 1, 1), lw=0.35)
        c.text(x + 31, legend_y - 1, label, size=8.0)

    c.save(OUT_FIGURES / "fig_fireprone_contract_progression_compact.pdf")


def row_values(row: dict) -> list[float]:
    return [cell["mean"] for cell in row["cells"]]


def build_task_comparator_heatmap() -> None:
    manifest = json.loads((ARTIFACTS / "release_table_values.json").read_text())
    primary = {row["name"]: row_values(row) for row in manifest["primary"]["rows"]}
    support_top = {row["name"]: row_values(row) for row in manifest["supporting_top"]["rows"]}
    support_bottom = {row["name"]: row_values(row) for row in manifest["supporting_bottom"]["rows"]}

    rows = [
        ("Occupancy", "Union F1 (%)", "higher", [primary[label][2] for label in ROW_ORDER]),
        ("Fire spread", "AP (%)", "higher", [primary[label][5] for label in ROW_ORDER]),
        ("Burned area", "log-RMSE", "lower", [support_top[label][0] for label in ROW_ORDER]),
        ("Analog retrieval", "nDCG@10", "higher", [support_top[label][3] for label in ROW_ORDER]),
        ("Smoke PM2.5", "RMSE", "lower", [support_bottom[label][0] for label in ROW_ORDER]),
        ("Extreme heat", "RMSE-C", "lower", [support_bottom[label][3] for label in ROW_ORDER]),
    ]
    c = PdfCanvas(width=1010, height=410)
    c.rect(0, 0, c.width, c.height, fill=(1, 1, 1))

    x0, y0 = 100, 82
    cell_w, cell_h = 72, 38
    grid_h = cell_h * len(rows)
    grid_top = y0 + grid_h

    for j, entry in enumerate(ROW_ORDER):
        x = x0 + j * cell_w + cell_w / 2
        c.text_rotated(x - 6, grid_top + 23, entry, angle_deg=45, size=8.6, align="left", bold=True)

    for i, (task, metric, direction, values) in enumerate(rows):
        y = grid_top - (i + 1) * cell_h
        c.text(12, y + 24, task, size=7.7, bold=True)
        c.text(12, y + 10, metric, size=6.9)
        ref = values[0]
        for j, value in enumerate(values):
            x = x0 + j * cell_w
            score = value / ref if direction == "higher" else ref / value
            score = clamp(score, 0, 1)
            color = heat_color(score)
            text_color = (1, 1, 1) if luminance(color) < 0.45 else (0.08, 0.10, 0.11)
            c.rect(x, y, cell_w, cell_h, fill=color, stroke=(1, 1, 1), lw=1.0)
            c.text(x + cell_w / 2, y + 22, f"{score:.2f}", size=10.2, align="center", bold=True, color=text_color)
            c.text(x + cell_w / 2, y + 10, f"{value:.2f}" if abs(value) >= 1 else f"{value:.3f}", size=6.6, align="center", color=text_color)

    c.rect(x0, y0, cell_w * len(ROW_ORDER), grid_h, stroke=(0.25, 0.25, 0.25), lw=0.8)
    c.save(OUT_FIGURES / "fig_comparator_heatmap_dense.pdf")


def build_all() -> None:
    ensure_dirs()
    build_fireprone_contract_progression()
    build_task_comparator_heatmap()


if __name__ == "__main__":
    build_all()
