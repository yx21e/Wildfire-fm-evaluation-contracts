#!/usr/bin/env python3
"""Check that the FireWx-FM release artifacts are present."""

from __future__ import annotations

from pathlib import Path


REQUIRED = [
    "firewxfm/modeling_unet.py",
    "firewxfm/tiled_inference.py",
    "firewxfm/serve_conus.py",
    "models/metadata/input_channels.json",
    "models/metadata/final_model_manifest.json",
    "models/metadata/input_normalization_stats.json",
    "models/checkpoints/firewxfm_conus_lowposw_noexposure_seed42.pt",
    "models/checkpoints/firewxfm_2024_staticfix_region_balanced_bce_seed42.pt",
    "examples/final_prediction/firewxfm_conus_20260706_t12_usgs_calibrated_probability_5km_lower48.tif",
    "examples/final_prediction/firewxfm_conus_20260706_t12_usgs_calibrated_heatmap.png",
    "examples/final_prediction/firewxfm_conus_20260706_t12_usgs_calibrated_heatmap_rgb.tif",
    "examples/final_prediction/firewxfm_conus_20260706_t12_usgs_calibrated_summary.json",
]


OPTIONAL = [
    "models/metadata/run_summary.json",
    "models/metadata/tile_summary.json",
    "docs/fm_refinement_experiment_plan.md",
    "docs/split_design.md",
    "docs/reproducibility_manifest.md",
    "scripts/audit_firewxfm_phase0.py",
    "scripts/prepare_firms_target_hour_csvs.py",
    "slurm/README.md",
    "slurm/firewxfm_2024_pilot_smoke.sbatch",
    "slurm/submit_firewxfm_2024_pilot_smoke.sh",
    "training/configs/stage1_cache_conus_multiyear_template.json",
    "training/configs/train_firewxfm_conus_multiyear_template.json",
    "training/configs/train_firewxfm_multitask_pretrain_template.json",
    "training/configs/eval_region_disjoint_template.json",
    "training/configs/baseline_suite_template.json",
    "scripts/build_5km_input_stack.py",
    "scripts/audit_firewxfm_longitude_seams.py",
    "scripts/make_firewxfm_central_feathered_candidate.py",
    "scripts/make_firewxfm_conus_feathered_candidate.py",
    "docs/firewxfm_staticfix_conus_handoff_20260812.md",
    "examples/staticfix_conus_candidate_20260812/README.md",
    "examples/staticfix_conus_candidate_20260812/firewxfm_20260707_t12_staticfix_conus_probability_lower48.tif",
    "examples/staticfix_conus_candidate_20260812/firewxfm_20260707_t12_staticfix_conus_longitude_audit.json",
    "examples/staticfix_conus_candidate_20260812/firewxfm_20260707_t12_staticfix_conus_baseline_probability_lower48.tif",
    "examples/staticfix_conus_candidate_20260812/firewxfm_20260707_t12_staticfix_conus_summary.json",
]


def describe(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        return f"ok ({size_mb:.2f} MB)"
    return "ok"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing = []
    print("Required artifacts")
    for rel in REQUIRED:
        status = describe(root / rel)
        print(f"  {rel}: {status}")
        if status == "missing":
            missing.append(rel)
    print("\nOptional artifacts")
    for rel in OPTIONAL:
        print(f"  {rel}: {describe(root / rel)}")
    if missing:
        print("\nRelease check failed. Missing required files:")
        for rel in missing:
            print(f"  - {rel}")
        return 1
    print("\nRelease check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
