from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from rasterio.enums import Resampling


def ensure_dirs(paths: List[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def resampling_from_name(name: str) -> Resampling:
    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "average": Resampling.average,
    }
    return mapping[name]


def split_for_timestamp(ts: pd.Timestamp, config: Dict[str, object]) -> str:
    if "split_years" in config:
        for split, years in config["split_years"].items():
            if int(ts.year) in [int(v) for v in years]:
                return split
        raise RuntimeError(f"Year {ts.year} does not map to any split.")

    split_months = config["split_months"]
    for split, months in split_months.items():
        if int(ts.month) in months:
            return split
    raise RuntimeError(f"Month {ts.month} does not map to any split.")
