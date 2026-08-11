#!/bin/bash
#SBATCH --job-name=datalenzai_characterize
#SBATCH --output=characterize-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#
# FairXAI dataset characterization SLURM job.
#
# The WebApp cluster_gateway submits this with `sbatch --export=ALL,VAR=...`
# and usually overrides --job-name / --output on the command line, e.g.:
#
#   sbatch --job-name=datalenzai_<job_id> \
#          --output=<RESULTS_DIR>/<job_id>.slurm.log \
#          --export=ALL,DATASET_PATH=...,RESULTS_DIR=...,TARGET_COLUMN=... \
#          characterize.slurm
#
# Required env: DATASET_PATH, RESULTS_DIR
# Optional env: FAIRXAI_VENV, HPC_MODULES, TARGET_COLUMN, INDEX_COLUMN
#
# Characterization only. Triage is a separate job — see analysis.slurm
# (ANALYSIS_TYPE=triage), submitted independently by the WebApp gateway,
# which owns triage_status.
#
# Output: <RESULTS_DIR>/<stem-of-DATASET_PATH>.json
set -euo pipefail

: "${DATASET_PATH:?DATASET_PATH is required}"
: "${RESULTS_DIR:?RESULTS_DIR is required}"

# SLURM jobs start a fresh shell, so the venv is activated here, not just at setup.
# No module by default: the venv brings its own interpreter, and the old
# `python/3.11.7` default died in the 2026 Spack rebuild. 
# Set HPC_MODULES only if a job needs something the venv lacks.
HPC_MODULES="${HPC_MODULES:-}"
if [ -n "${HPC_MODULES//[[:space:]]/}" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $HPC_MODULES
fi

FAIRXAI_VENV="${FAIRXAI_VENV:-}"
if [ -n "$FAIRXAI_VENV" ]; then
    # shellcheck disable=SC1091
    source "$FAIRXAI_VENV/bin/activate"
fi

mkdir -p "$RESULTS_DIR"

DATASETS_DIR="$(dirname "$DATASET_PATH")"
FILENAME="$(basename "$DATASET_PATH")"

cmd=(fairxai characterize
    --filename "$FILENAME"
    --datasets-dir "$DATASETS_DIR"
    --output-dir "$RESULTS_DIR")

[ -n "${TARGET_COLUMN:-}" ] && cmd+=(--target-column "$TARGET_COLUMN")
[ -n "${INDEX_COLUMN:-}" ] && cmd+=(--index-column "$INDEX_COLUMN")

echo "Running: ${cmd[*]}"
"${cmd[@]}"
