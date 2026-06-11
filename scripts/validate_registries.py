#!/usr/bin/env python3
"""Validate public data registries for the WildFIRE-FM release.

The registries are meant to be read by downstream data adapters. This script
keeps references explicit so future task additions do not silently introduce
unregistered sources, targets, grids, or masks.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml


REGISTRY_FILES = {
    "sources": "sources.yml",
    "variables": "variables.yml",
    "grids": "grids.yml",
    "tasks": "tasks.yml",
    "splits": "splits.yml",
}

FORBIDDEN_PATTERNS = [
    re.compile(r"/home/"),
    re.compile("/" + "blue" + "/"),
    re.compile("/" + "orange" + "/"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)password\s*[:=]"),
    re.compile(r"(?i)secret\s*[:=]"),
    re.compile(r"(?i)token\s*[:=]"),
]

ALLOWED_OBSERVATION_MASK_VALUES = {
    "required",
    "event_label_available",
    "station_observation_available",
    "hms_product_available",
    "weather_truth_available",
    "track_observation_available",
    "usdm_week_available",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not load as a mapping")
    return data


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def collect_variable_refs(variables: dict[str, Any]) -> tuple[set[str], set[str], set[str], set[str]]:
    dynamic = set((variables.get("dynamic_weather") or {}).keys())
    static = set((variables.get("static_context") or {}).keys())
    masks = set((variables.get("masks") or {}).keys())
    targets = set((variables.get("targets") or {}).keys())
    return dynamic, static, masks, targets


def check_source_ref(errors: list[str], ref: str, sources: set[str], location: str) -> None:
    if ref not in sources:
        errors.append(f"{location}: unknown source '{ref}'")


def check_forbidden_text(errors: list[str], registry_dir: Path) -> None:
    for path in sorted(registry_dir.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                errors.append(f"{path.name}: forbidden local path or credential-like value matched {pattern.pattern!r}")


def validate_variables(errors: list[str], variables: dict[str, Any], source_ids: set[str]) -> None:
    for group_name in ("dynamic_weather", "static_context", "targets"):
        group = variables.get(group_name) or {}
        if not isinstance(group, dict):
            errors.append(f"variables.yml:{group_name} must be a mapping")
            continue
        for name, spec in group.items():
            if not isinstance(spec, dict):
                errors.append(f"variables.yml:{group_name}.{name} must be a mapping")
                continue
            refs = []
            refs.extend(as_list(spec.get("source")))
            refs.extend(as_list(spec.get("source_candidates")))
            for ref in refs:
                check_source_ref(errors, str(ref), source_ids, f"variables.yml:{group_name}.{name}")


def validate_tasks(
    errors: list[str],
    tasks: dict[str, Any],
    source_ids: set[str],
    grid_ids: set[str],
    target_ids: set[str],
    mask_ids: set[str],
) -> None:
    task_specs = tasks.get("tasks") or {}
    if not isinstance(task_specs, dict):
        errors.append("tasks.yml:tasks must be a mapping")
        return
    for task_id, spec in task_specs.items():
        if not isinstance(spec, dict):
            errors.append(f"tasks.yml:{task_id} must be a mapping")
            continue
        grid = spec.get("input_grid")
        if grid not in grid_ids:
            errors.append(f"tasks.yml:{task_id}.input_grid unknown grid '{grid}'")
        target = spec.get("target")
        if target not in target_ids:
            errors.append(f"tasks.yml:{task_id}.target unknown target '{target}'")
        for ref in as_list(spec.get("dynamic_sources")) + as_list(spec.get("static_sources")):
            check_source_ref(errors, str(ref), source_ids, f"tasks.yml:{task_id}")
        observation_mask = spec.get("observation_mask")
        if (
            observation_mask
            and observation_mask not in mask_ids
            and observation_mask not in ALLOWED_OBSERVATION_MASK_VALUES
        ):
            errors.append(f"tasks.yml:{task_id}.observation_mask unknown mask or policy '{observation_mask}'")


def validate_splits(errors: list[str], splits: dict[str, Any], grid_ids: set[str]) -> None:
    split_specs = splits.get("splits") or {}
    if not isinstance(split_specs, dict):
        errors.append("splits.yml:splits must be a mapping")
        return
    for split_id, spec in split_specs.items():
        if not isinstance(spec, dict):
            errors.append(f"splits.yml:{split_id} must be a mapping")
            continue
        grid = spec.get("grid")
        if grid not in grid_ids:
            errors.append(f"splits.yml:{split_id}.grid unknown grid '{grid}'")


def validate(registry_dir: Path) -> list[str]:
    errors: list[str] = []
    loaded: dict[str, dict[str, Any]] = {}
    for key, filename in REGISTRY_FILES.items():
        path = registry_dir / filename
        if not path.exists():
            errors.append(f"missing registry file: {path}")
            continue
        try:
            loaded[key] = load_yaml(path)
        except Exception as exc:  # Keep CLI messages concise.
            errors.append(f"{filename}: failed to parse YAML: {exc}")
    if errors:
        return errors

    source_ids = set((loaded["sources"].get("sources") or {}).keys())
    grid_ids = set((loaded["grids"].get("grids") or {}).keys())
    _, _, mask_ids, target_ids = collect_variable_refs(loaded["variables"])

    if not source_ids:
        errors.append("sources.yml contains no sources")
    if not grid_ids:
        errors.append("grids.yml contains no grids")
    if not target_ids:
        errors.append("variables.yml contains no targets")

    check_forbidden_text(errors, registry_dir)
    validate_variables(errors, loaded["variables"], source_ids)
    validate_tasks(errors, loaded["tasks"], source_ids, grid_ids, target_ids, mask_ids)
    validate_splits(errors, loaded["splits"], grid_ids)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir",
        default="registries",
        type=Path,
        help="Directory containing sources.yml, variables.yml, grids.yml, tasks.yml, and splits.yml.",
    )
    args = parser.parse_args(argv)

    errors = validate(args.registry_dir)
    if errors:
        print("Registry validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"Registry validation passed: {args.registry_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
