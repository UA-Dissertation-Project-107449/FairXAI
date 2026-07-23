#!/bin/bash
#SBATCH --job-name=datalenzai_analysis
#SBATCH --output=analysis-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#
# FairXAI optional / post-hoc analysis SLURM job: binning, clustering, or triage.
#
# Runs the matching `fairxai` CLI and writes a single JSON file to RESULT_FILE
# (stdout of the CLI is captured into RESULT_FILE because sbatch is async —
# the cluster_gateway scp's RESULT_FILE back, it never sees stdout).
#
# Submitted by cluster_gateway with `sbatch --export=ALL,VAR=...`.
#
# Required env: ANALYSIS_TYPE (binning|clustering|triage), DATASET_PATH,
#               RESULT_FILE, TARGET_COLUMN
# binning   also needs: ATTRIBUTE, STRATEGY
# clustering also needs: METHOD;  optional: PCA2D_FILE
# Optional env: FAIRXAI_VENV, HPC_MODULES, INDEX_COLUMN, SENSITIVE_COLUMNS_JSON
#
# SENSITIVE_COLUMNS_JSON is a JSON array, not a space-separated list: a column
# named "race group" is one argument and word-splitting would make it two.
set -euo pipefail

: "${ANALYSIS_TYPE:?ANALYSIS_TYPE is required}"
: "${DATASET_PATH:?DATASET_PATH is required}"
: "${RESULT_FILE:?RESULT_FILE is required}"
: "${TARGET_COLUMN:?TARGET_COLUMN is required}"

HPC_MODULES="${HPC_MODULES:-python/3.11.7}"
if command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $HPC_MODULES
fi

FAIRXAI_VENV="${FAIRXAI_VENV:-}"
if [ -n "$FAIRXAI_VENV" ]; then
    # shellcheck disable=SC1091
    source "$FAIRXAI_VENV/bin/activate"
fi

mkdir -p "$(dirname "$RESULT_FILE")"

# Decode SENSITIVE_COLUMNS_JSON into the `sensitive` array. NUL-delimited so a
# column name may contain spaces, newlines, or quotes and still stay one word.
read_sensitive_columns() {
    sensitive=()
    [ -n "${SENSITIVE_COLUMNS_JSON:-}" ] || return 0
    while IFS= read -r -d '' column; do
        sensitive+=("$column")
    done < <(python3 -c '
import json, os, sys
for column in json.loads(os.environ["SENSITIVE_COLUMNS_JSON"]):
    sys.stdout.write(column + "\0")
')
}

case "$ANALYSIS_TYPE" in
    binning)
        : "${ATTRIBUTE:?ATTRIBUTE is required for binning}"
        : "${STRATEGY:?STRATEGY is required for binning}"
        fairxai binning \
            --filename "$DATASET_PATH" \
            --target-column "$TARGET_COLUMN" \
            --attribute "$ATTRIBUTE" \
            --strategy "$STRATEGY" >"$RESULT_FILE"
        ;;
    clustering)
        : "${METHOD:?METHOD is required for clustering}"
        cmd=(fairxai clustering
            --filename "$DATASET_PATH"
            --target-column "$TARGET_COLUMN"
            --method "$METHOD")
        [ -n "${PCA2D_FILE:-}" ] && cmd+=(--pca2d-file "$PCA2D_FILE")
        [ -n "${INDEX_COLUMN:-}" ] && cmd+=(--index-column "$INDEX_COLUMN")
        read_sensitive_columns
        [ "${#sensitive[@]}" -gt 0 ] && cmd+=(--sensitive-columns "${sensitive[@]}")
        "${cmd[@]}" >"$RESULT_FILE"
        ;;
    triage)
        DATASETS_DIR="$(dirname "$DATASET_PATH")"
        FILENAME="$(basename "$DATASET_PATH")"
        cmd=(fairxai triage
            --filename "$FILENAME"
            --datasets-dir "$DATASETS_DIR"
            --target-column "$TARGET_COLUMN")
        [ -n "${INDEX_COLUMN:-}" ] && cmd+=(--index-column "$INDEX_COLUMN")
        read_sensitive_columns
        [ "${#sensitive[@]}" -gt 0 ] && cmd+=(--sensitive-columns "${sensitive[@]}")
        "${cmd[@]}" >"$RESULT_FILE"
        ;;
    *)
        echo "Unknown ANALYSIS_TYPE: $ANALYSIS_TYPE" >&2
        exit 2
        ;;
esac

echo "Wrote $RESULT_FILE"
