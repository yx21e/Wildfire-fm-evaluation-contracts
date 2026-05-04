from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "results"
OUT_TABLES = ROOT / "paper_outputs" / "tables"
OUT_FIGURES = ROOT / "paper_outputs" / "figures"


ROW_ORDER = [
    "Reference",
    "Prithvi-WxC",
    "Aurora",
    "ClimaX",
    "StormCast",
    "Pangu24",
    "DLWP",
    "FCN",
    "FengWu",
    "FuXi",
    "Pangu-Weather",
    "AlphaEarth",
]


def finite(values: Iterable[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def stat(values: Iterable[float]) -> dict[str, float | int]:
    vals = finite(values)
    if not vals:
        return {"n": 0, "mean": math.nan, "std": math.nan}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def ms(mean: float, std: float, digits: int = 4) -> str:
    return rf"\ms{{{mean:.{digits}f}}}{{{std:.{digits}f}}}"


def ms_from_summary(summary: dict[str, float | int], scale: float = 1.0, digits: int = 4) -> str:
    return ms(float(summary["mean"]) * scale, float(summary["std"]) * scale, digits=digits)


def ensure_dirs() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
