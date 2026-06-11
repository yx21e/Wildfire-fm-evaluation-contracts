"""Base classes for registry-backed dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from .sample import SampleRecord


@dataclass(frozen=True)
class RegistryBundle:
    """Loaded registry files used to configure an adapter."""

    sources: Mapping[str, Any]
    variables: Mapping[str, Any]
    grids: Mapping[str, Any]
    tasks: Mapping[str, Any]
    splits: Mapping[str, Any]

    def task(self, task_id: str) -> Mapping[str, Any]:
        tasks = self.tasks.get("tasks", {})
        if task_id not in tasks:
            raise KeyError(f"Unknown task_id {task_id!r}")
        return tasks[task_id]

    def split(self, split_id: str) -> Mapping[str, Any]:
        splits = self.splits.get("splits", {})
        if split_id not in splits:
            raise KeyError(f"Unknown split_id {split_id!r}")
        return splits[split_id]


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not load as a mapping")
    return data


def load_registry_bundle(registry_dir: str | Path = "registries") -> RegistryBundle:
    """Load the public registry files from a repository checkout."""

    root = Path(registry_dir)
    return RegistryBundle(
        sources=_load_yaml(root / "sources.yml"),
        variables=_load_yaml(root / "variables.yml"),
        grids=_load_yaml(root / "grids.yml"),
        tasks=_load_yaml(root / "tasks.yml"),
        splits=_load_yaml(root / "splits.yml"),
    )


class DatasetAdapter(ABC):
    """Abstract adapter from provider-prepared data to task samples."""

    task_id: str
    split_id: str

    def __init__(self, registry: RegistryBundle, data_root: str | Path):
        self.registry = registry
        self.data_root = Path(data_root)
        self.task_spec = registry.task(self.task_id)
        self.split_spec = registry.split(self.split_id)

    @abstractmethod
    def iter_samples(self, partition: str) -> Iterator[SampleRecord]:
        """Yield samples for ``train``, ``validation``, or ``test``."""

    def partitions(self) -> tuple[str, ...]:
        """Return the canonical split partitions expected by the registry."""

        return tuple(name for name in ("train", "validation", "test") if name in self.split_spec)

    def source_ids(self) -> tuple[str, ...]:
        """Return source ids declared by the task contract."""

        dynamic = tuple(self.task_spec.get("dynamic_sources", ()))
        static = tuple(self.task_spec.get("static_sources", ()))
        return dynamic + static
