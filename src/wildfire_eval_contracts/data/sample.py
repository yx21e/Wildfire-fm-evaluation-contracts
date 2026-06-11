"""Task-native sample container used by dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SampleRecord:
    """One task sample after provider data have been adapted locally.

    The container intentionally accepts generic Python objects so adapters can
    return tensors, arrays, tables, or geometries without forcing a single
    runtime stack on every task.
    """

    sample_id: str
    task_id: str
    split: str
    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    masks: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def require(self, *, inputs: list[str] | None = None, targets: list[str] | None = None) -> None:
        """Raise a clear error when required fields are missing."""

        missing_inputs = [name for name in inputs or [] if name not in self.inputs]
        missing_targets = [name for name in targets or [] if name not in self.targets]
        if missing_inputs or missing_targets:
            pieces = []
            if missing_inputs:
                pieces.append(f"inputs={missing_inputs}")
            if missing_targets:
                pieces.append(f"targets={missing_targets}")
            raise KeyError(f"SampleRecord {self.sample_id!r} is missing " + ", ".join(pieces))
