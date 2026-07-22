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
# Optional env: FAIRXAI_VENV, HPC_MODULES, TARGET_COLUMN, INDEX_COLUMN,
#               INCLUDE_TRIAGE (0/1), SENSITIVE_COLUMNS (space-separated)
#
# Output: <RESULTS_DIR>/<stem-of-DATASET_PATH>.json
set -euo pipefail

: "${DATASET_PATH:?DATASET_PATH is required}"
: "${RESULTS_DIR:?RESULTS_DIR is required}"

# SLURM jobs start a fresh shell — load modules + venv here, not just at setup.
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

mkdir -p "$RESULTS_DIR"

DATASETS_DIR="$(dirname "$DATASET_PATH")"
FILENAME="$(basename "$DATASET_PATH")"

STEM="${FILENAME%.*}"
RESULT_JSON="$RESULTS_DIR/$STEM.json"

cmd=(fairxai characterize
    --filename "$FILENAME"
    --datasets-dir "$DATASETS_DIR"
    --output-dir "$RESULTS_DIR")

[ -n "${TARGET_COLUMN:-}" ] && cmd+=(--target-column "$TARGET_COLUMN")
[ -n "${INDEX_COLUMN:-}" ] && cmd+=(--index-column "$INDEX_COLUMN")

echo "Running: ${cmd[*]}"
"${cmd[@]}"

# Triage is a separate operation. Its report is merged into the characterization
# JSON only because the WebApp still reads a single file; once the gateway calls
# `fairxai triage` itself, this whole block goes away.
if [ "${INCLUDE_TRIAGE:-0}" = "1" ]; then
    # TARGET_COLUMN is optional for characterization but required by triage, so
    # use the column characterization actually resolved. Referencing the raw
    # TARGET_COLUMN here would abort the whole job under `set -u` when unset,
    # losing the characterization result that triage is supposed to preserve.
    TRIAGE_TARGET="$(RESULT_JSON="$RESULT_JSON" python3 - <<'PY'
import json
import os

with open(os.environ["RESULT_JSON"], encoding="utf-8") as fp:
    print(json.load(fp)["target_column"])
PY
)"

    triage_cmd=(fairxai triage
        --filename "$FILENAME"
        --datasets-dir "$DATASETS_DIR"
        --target-column "$TRIAGE_TARGET")
    [ -n "${INDEX_COLUMN:-}" ] && triage_cmd+=(--index-column "$INDEX_COLUMN")
    if [ -n "${SENSITIVE_COLUMNS:-}" ]; then
        # shellcheck disable=SC2206
        sensitive=($SENSITIVE_COLUMNS)
        triage_cmd+=(--sensitive-columns "${sensitive[@]}")
    fi

    TRIAGE_FILE="$RESULTS_DIR/$STEM.triage.json"
    TRIAGE_STATUS=success
    TRIAGE_ERROR=""
    echo "Running: ${triage_cmd[*]}"
    if ! TRIAGE_ERROR="$("${triage_cmd[@]}" 2>&1 >"$TRIAGE_FILE")"; then
        TRIAGE_STATUS=failed
        echo "Triage failed (characterization result retained): $TRIAGE_ERROR" >&2
    fi

    RESULT_JSON="$RESULT_JSON" TRIAGE_FILE="$TRIAGE_FILE" \
    TRIAGE_STATUS="$TRIAGE_STATUS" TRIAGE_ERROR="$TRIAGE_ERROR" python3 - <<'PY'
import json
import os

result_path = os.environ["RESULT_JSON"]
status = os.environ["TRIAGE_STATUS"]

with open(result_path, encoding="utf-8") as fp:
    result = json.load(fp)

result["triage_status"] = status
if status == "success":
    with open(os.environ["TRIAGE_FILE"], encoding="utf-8") as fp:
        result["triage_report"] = json.load(fp)
else:
    result["triage_error"] = os.environ.get("TRIAGE_ERROR", "")

with open(result_path, "w", encoding="utf-8") as fp:
    json.dump(result, fp, indent=4)
PY
    rm -f "$TRIAGE_FILE"
else
    # Keep the HPC result contract identical to the local shim's, which always
    # reports a triage_status. Consumers can then branch on one key regardless
    # of where the job ran.
    RESULT_JSON="$RESULT_JSON" python3 - <<'PY'
import json
import os

path = os.environ["RESULT_JSON"]
with open(path, encoding="utf-8") as fp:
    result = json.load(fp)

result["triage_status"] = "not_requested"

with open(path, "w", encoding="utf-8") as fp:
    json.dump(result, fp, indent=4)
PY
fi
