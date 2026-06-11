# Data Registries

These YAML files define the data contract for FireWx-FM compatibility and the
planned general hazard backbone.

## Files

- `sources.yml`: provider-level source metadata, access mode, downloader id,
  citation key, and source role.
- `variables.yml`: canonical dynamic variables, static variables, masks, and
  targets.
- `grids.yml`: canonical grids and geometry types.
- `tasks.yml`: task definitions, target types, head families, metrics, and
  observation-mask rules.
- `splits.yml`: chronological/event split definitions.

## Rules

- Do not place local storage paths, tokens, usernames, or provider credentials
  in these files.
- A dataset adapter must read from these registries rather than hard-coding
  source names, variable names, or split boundaries.
- Missing observation and observed negative are different states. Any sparse
  source such as WFIGS, HMS, station data, or track data must carry an
  observation mask.
- The current FireWx-FM split and 16-channel input contract remain fixed for
  compatibility. General HazardWx-FM adapters can add new sources and targets,
  but should not silently redefine the released wildfire task.

## Intended Next Step

Implement `SampleRecord` and a base `DatasetAdapter` that use these registries.
The first adapter should rebuild the current FireWx-FM samples before adding
new smoke, drought, cyclone, or heat adapters.

The initial interface lives in `src/wildfire_eval_contracts/data/`. It defines:

- `SampleRecord`: a task-native sample container with inputs, targets, masks,
  and metadata.
- `DatasetAdapter`: a registry-backed base class for local provider-data
  adapters.
- `FireWxFMOccupancyAdapter`: the released 12-hour occupancy contract that a
  local FireWx-FM reconstruction must satisfy.

## Validation

Run this check before adding a new source, target, grid, task, or split:

```bash
python3 scripts/validate_registries.py
```

The validator checks that registry YAML files parse cleanly, that task
references resolve to registered sources, targets, grids, and masks, and that
the public registry files do not contain local paths or credential-like values.
