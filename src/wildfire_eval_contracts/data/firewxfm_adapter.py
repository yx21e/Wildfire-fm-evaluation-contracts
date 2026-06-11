"""Adapter skeleton for the released FireWx-FM occupancy task.

The public release does not redistribute raw provider data or local feature
caches. This class defines the contract that a local reconstruction must
satisfy before it can feed the released model or retraining scripts.
"""

from __future__ import annotations

from typing import Iterator

from .adapter import DatasetAdapter
from .sample import SampleRecord


class FireWxFMOccupancyAdapter(DatasetAdapter):
    """Contract for 12-hour wildfire occupancy samples on the California grid."""

    task_id = "wildfire_occupancy_12h"
    split_id = "firewxfm_released_2024"

    required_inputs = (
        "weather",
        "weather_valid_mask",
        "static_context",
        "static_valid_mask",
    )
    required_targets = ("wildfire_active_fire_occupancy",)

    def iter_samples(self, partition: str) -> Iterator[SampleRecord]:
        if partition not in self.partitions():
            raise ValueError(f"Unknown partition {partition!r}; expected one of {self.partitions()}")
        raise NotImplementedError(
            "FireWxFMOccupancyAdapter documents the sample contract only. "
            "Implement this method against locally prepared provider data."
        )

    @classmethod
    def validate_sample(cls, sample: SampleRecord) -> None:
        """Check that a local sample exposes the released occupancy contract."""

        sample.require(inputs=list(cls.required_inputs), targets=list(cls.required_targets))
