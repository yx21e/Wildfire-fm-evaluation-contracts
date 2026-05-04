#!/usr/bin/env bash
#SBATCH --job-name=wildfire-raw-rerun
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

: "${WILDFIRE_DATA_ROOT:?Set WILDFIRE_DATA_ROOT to the raw wildfire input directory.}"
: "${WILDFIRE_FEATURE_ROOT:?Set WILDFIRE_FEATURE_ROOT to the frozen-backbone feature directory.}"
: "${WILDFIRE_OUTPUT_ROOT:?Set WILDFIRE_OUTPUT_ROOT to the desired result directory.}"

echo "This is a template for raw-data reruns."
echo "Use scripts from the private/raw-data workflow to create per-task summaries, then rebuild paper outputs with:"
echo "  python scripts/reproduce_all.py"
