#!/usr/bin/env bash
set -euo pipefail

# Dermatology baseline pipeline: load -> profile -> recommend -> preprocess ->
# train -> assess -> compare -> explain -> mitigate. All stages run by default
# (like cardiac); scope a subset with --go-until <stage> / --resume-from <stage>
# (or RESUME_FROM=... / GO_UNTIL=... env vars) over an existing run.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PYTHON=${PYTHON:-python3}
RUN_RECOMMENDATIONS=${RUN_RECOMMENDATIONS:-true}
VERBOSE=${VERBOSE:-0}
RESUME_FROM=${RESUME_FROM:-""}
GO_UNTIL=${GO_UNTIL:-""}
DATASETS=()
MODEL_TYPES=()
DEVICE=""
EPOCHS=""
BATCH_SIZE=""
PRETRAINED_ARGS=()
AUGMENTATION_ARGS=()
FIGURE_ARGS=()
GROUP_VIEW_ARGS=()

# Stage mapping — loaded from the Python catalog, never declared here.
# shellcheck source=../common/stage_registry.sh
source "$ROOT_DIR/scripts/common/stage_registry.sh"
load_stage_registry dermatology "$ROOT_DIR" "$PYTHON" || {
    echo "ERROR: Could not load the dermatology stage registry." >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do DATASETS+=("$1"); shift; done
            continue
            ;;
        --model-types)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do MODEL_TYPES+=("$1"); shift; done
            continue
            ;;
        --resume-from)
            shift; [[ $# -gt 0 ]] || { echo "ERROR: --resume-from requires a value" >&2; exit 1; }
            RESUME_FROM="$1"
            ;;
        --go-until)
            shift; [[ $# -gt 0 ]] || { echo "ERROR: --go-until requires a value" >&2; exit 1; }
            GO_UNTIL="$1"
            ;;
        --run-id)
            shift; [[ $# -gt 0 ]] || { echo "ERROR: --run-id requires a value" >&2; exit 1; }
            RUN_ID="$1"
            ;;
        --device)
            shift; [[ $# -gt 0 ]] || { echo "ERROR: --device requires a value" >&2; exit 1; }
            DEVICE="$1"
            ;;
        --epochs)
            shift; [[ $# -gt 0 ]] || { echo "ERROR: --epochs requires a value" >&2; exit 1; }
            EPOCHS="$1"
            ;;
        --batch-size)
            shift; [[ $# -gt 0 ]] || { echo "ERROR: --batch-size requires a value" >&2; exit 1; }
            BATCH_SIZE="$1"
            ;;
        --pretrained)
            PRETRAINED_ARGS=(--pretrained)
            ;;
        --no-pretrained)
            PRETRAINED_ARGS=(--no-pretrained)
            ;;
        --augmentation)
            AUGMENTATION_ARGS=(--augmentation)
            ;;
        --no-augmentation)
            AUGMENTATION_ARGS=(--no-augmentation)
            ;;
        --figures)
            FIGURE_ARGS=(--figures)
            ;;
        --no-figures)
            FIGURE_ARGS=(--no-figures)
            ;;
        --group-views)
            GROUP_VIEW_ARGS=(--group-views)
            ;;
        --no-group-views)
            GROUP_VIEW_ARGS=(--no-group-views)
            ;;
        --no-recommendations)
            RUN_RECOMMENDATIONS=false
            ;;
        --explain)
            RUN_EXPLAIN=true
            ;;
        --no-explain)
            RUN_EXPLAIN=false
            ;;
        -v) VERBOSE=1 ;;
        -vv) VERBOSE=2 ;;
        *)
            echo "ERROR: Unknown argument '$1'" >&2
            exit 1
            ;;
    esac
    shift
done

START_NUM=$STAGE_FIRST
END_NUM=$STAGE_LAST
[[ -n "$RESUME_FROM" ]] && START_NUM=$(resolve_stage "$RESUME_FROM")
[[ -n "$GO_UNTIL" ]] && END_NUM=$(resolve_stage "$GO_UNTIL")
if (( START_NUM > END_NUM )); then
    echo "ERROR: --resume-from (${STAGE_NAME[$START_NUM]}, #$START_NUM) is after --go-until (${STAGE_NAME[$END_NUM]}, #$END_NUM)." >&2
    exit 1
fi

should_run() {
    local num=$1
    (( num >= START_NUM && num <= END_NUM ))
}

BASE_RESULTS="$ROOT_DIR/output/dermatology"
if [[ -n "$RESUME_FROM" && -z "${RUN_ID:-}" ]]; then
    if [[ -L "$BASE_RESULTS/latest_run" ]]; then
        RUN_ID=$(basename "$(readlink -f "$BASE_RESULTS/latest_run")")
    elif [[ -f "$BASE_RESULTS/latest_run.txt" ]]; then
        RUN_ID=$(tr -d '[:space:]' < "$BASE_RESULTS/latest_run.txt")
    else
        echo "ERROR: No RUN_ID provided and no latest dermatology run found." >&2
        exit 1
    fi
fi
RUN_ID=${RUN_ID:-$("$PYTHON" - <<'PY'
import os
import uuid
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
print(f"run_{ts}_{os.getpid()}_{uuid.uuid4().hex[:6]}")
PY
)}
export RUN_ID

RUN_ROOT="$BASE_RESULTS/runs/$RUN_ID"
CHECKPOINT_DIR="$RUN_ROOT/.checkpoints"
mkdir -p "$CHECKPOINT_DIR" "$BASE_RESULTS" "$ROOT_DIR/logs/dermatology"
ln -sfn "runs/$RUN_ID" "$BASE_RESULTS/latest_run" 2>/dev/null || true
echo "$RUN_ID" > "$BASE_RESULTS/latest_run.txt"

mark_done() {
    # Always writes the canonical marker name from the registry, so markers
    # stay interchangeable with the ones the Python helpers write.
    local num=$1
    local name="${STAGE_NAME[$num]}"
    mkdir -p "$CHECKPOINT_DIR"
    "$PYTHON" - "$CHECKPOINT_DIR/${num}_${name}.done" "$num" "$name" <<'PY'
import json, os, socket, sys
from datetime import datetime
path, number, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
payload = {
    "stage": name,
    "number": number,
    "completed_at": datetime.now().isoformat(),
    "hostname": socket.gethostname(),
    "pid": os.getpid(),
}
open(path, "w").write(json.dumps(payload, indent=2) + "\n")
PY
}

case "$VERBOSE" in true) VERBOSE=1 ;; false) VERBOSE=0 ;; esac
VERBOSE_FLAG=""
(( VERBOSE >= 2 )) && VERBOSE_FLAG="-vv"
(( VERBOSE == 1 )) && VERBOSE_FLAG="-v"

DATASET_ARGS=()
(( ${#DATASETS[@]} > 0 )) && DATASET_ARGS=(--datasets "${DATASETS[@]}")
MODEL_TYPE_ARGS=()
(( ${#MODEL_TYPES[@]} > 0 )) && MODEL_TYPE_ARGS=(--model-types "${MODEL_TYPES[@]}")
DEVICE_ARGS=()
[[ -n "$DEVICE" ]] && DEVICE_ARGS=(--device "$DEVICE")
EPOCH_ARGS=()
[[ -n "$EPOCHS" ]] && EPOCH_ARGS=(--epochs "$EPOCHS")
BATCH_ARGS=()
[[ -n "$BATCH_SIZE" ]] && BATCH_ARGS=(--batch-size "$BATCH_SIZE")

# Stage 10 (explain) reloads every trained model and renders SHAP/LIME/Grad-CAM
# overlays. On a cached-frozen-features run it dominates wall-clock while adding
# nothing to the fairness numbers, so it is skippable without editing the config.
# Precedence: --explain / --no-explain > RUN_EXPLAIN > xai.enabled in the config.
XAI_ENABLED=$("$PYTHON" - "$ROOT_DIR" <<'XAI_PY'
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load((Path(sys.argv[1]) / "configs/pipelines/dermatology.yaml").read_text()) or {}
print("true" if (cfg.get("xai", {}) or {}).get("enabled", False) else "false")
XAI_PY
)
RUN_EXPLAIN=${RUN_EXPLAIN:-$XAI_ENABLED}
# Normalize truthy values (1/true/yes/on, any case) -> "true".
shopt -s nocasematch
[[ "$RUN_EXPLAIN" =~ ^(1|true|yes|on)$ ]] && RUN_EXPLAIN=true || RUN_EXPLAIN=false
shopt -u nocasematch

echo "======================================================================"
echo "DERMATOLOGY BASELINE PIPELINE"
echo "======================================================================"
echo "Working directory: $ROOT_DIR"
echo "Run ID:           $RUN_ID"
echo "Stages:           $START_NUM..$END_NUM"
echo "Datasets:         ${DATASETS[*]:-config/default}"
echo "Model types:      ${MODEL_TYPES[*]:-config/default}"
echo "Device:           ${DEVICE:-config/default}"
echo "Explain (XAI):    $RUN_EXPLAIN"
echo ""

if should_run 1; then
    echo "[PHASE 1] Loading dermatology datasets"
    "$PYTHON" "$ROOT_DIR/scripts/dermatology/load_data.py" "${DATASET_ARGS[@]}" $VERBOSE_FLAG
    mark_done 1
else
    echo "[1] load - SKIPPED"
fi

if should_run 2; then
    echo "[PHASE 2] Profiling dermatology datasets"
    "$PYTHON" "$ROOT_DIR/scripts/dermatology/profile_data.py" "${DATASET_ARGS[@]}" --run-id "$RUN_ID" $VERBOSE_FLAG
    mark_done 2
else
    echo "[2] profile - SKIPPED"
fi

if should_run 3; then
    if [[ "$RUN_RECOMMENDATIONS" == "true" ]]; then
        echo "[PHASE 3] Generating recommendations"
        "$PYTHON" "$ROOT_DIR/scripts/dermatology/generate_recommendations.py" "${DATASET_ARGS[@]}" --run-id "$RUN_ID" $VERBOSE_FLAG
    else
        echo "[3] recommend - SKIPPED (disabled)"
    fi
    mark_done 3
else
    echo "[3] recommend - SKIPPED"
fi

if should_run 4; then
    echo "[PHASE 4] Preprocessing dermatology datasets"
    "$PYTHON" "$ROOT_DIR/scripts/dermatology/preprocess.py" "${DATASET_ARGS[@]}" "${FIGURE_ARGS[@]}" $VERBOSE_FLAG
    mark_done 4
else
    echo "[4] preprocess - SKIPPED"
fi

if should_run 7; then
    echo "[PHASE 7] Training image baseline"
    "$PYTHON" "$ROOT_DIR/scripts/dermatology/train_baseline.py" "${DATASET_ARGS[@]}" "${MODEL_TYPE_ARGS[@]}" "${DEVICE_ARGS[@]}" "${EPOCH_ARGS[@]}" "${BATCH_ARGS[@]}" "${PRETRAINED_ARGS[@]}" "${AUGMENTATION_ARGS[@]}" $VERBOSE_FLAG
    mark_done 7
else
    echo "[7] train - SKIPPED"
fi

if should_run 8; then
    echo "[PHASE 8] Assessing post-prediction fairness"
    "$PYTHON" "$ROOT_DIR/scripts/dermatology/assess_predictions.py" "${DATASET_ARGS[@]}" "${MODEL_TYPE_ARGS[@]}" "${GROUP_VIEW_ARGS[@]}" "${FIGURE_ARGS[@]}" $VERBOSE_FLAG
    mark_done 8
else
    echo "[8] assess - SKIPPED"
fi

if should_run 9; then
    echo "[PHASE 9] Comparing baseline models"
    "$PYTHON" "$ROOT_DIR/scripts/dermatology/compare.py" "${DATASET_ARGS[@]}" "${MODEL_TYPE_ARGS[@]}" "${FIGURE_ARGS[@]}" $VERBOSE_FLAG
    mark_done 9
else
    echo "[9] compare - SKIPPED"
fi

if should_run 10; then
    if [[ "$RUN_EXPLAIN" == "true" ]]; then
        echo "[PHASE 10] Explaining baseline models (XAI)"
        "$PYTHON" "$ROOT_DIR/scripts/dermatology/explain.py" "${DATASET_ARGS[@]}" "${MODEL_TYPE_ARGS[@]}" $VERBOSE_FLAG
    else
        echo "[10] explain - SKIPPED (disabled)"
    fi
    mark_done 10
else
    echo "[10] explain - SKIPPED"
fi

if should_run 11; then
    echo "[PHASE 11] Post-processing fairness mitigation"
    "$PYTHON" "$ROOT_DIR/scripts/dermatology/mitigate.py" "${DATASET_ARGS[@]}" "${MODEL_TYPE_ARGS[@]}" "${FIGURE_ARGS[@]}" $VERBOSE_FLAG
    mark_done 11
else
    echo "[11] mitigate - SKIPPED"
fi

echo "Dermatology baseline complete: $RUN_ROOT"
