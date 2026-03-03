#!/usr/bin/env bash
set -euo pipefail

# Cardiac end-to-end pipeline: load -> preprocess -> train -> fairness -> experiments
# Supports --resume-from / --go-until via env vars RESUME_FROM / GO_UNTIL.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

# ======================================================================
# Stage mapping (name -> number, number -> name)
# ======================================================================
declare -A STAGE_NUM=(
    [load]=1 [profile]=2 [profiling]=2 [recommend]=3 [recommendations]=3
    [triage]=3 [preprocess]=4 [preprocessing]=4 [train]=5 [baseline]=5
    [training]=5 [assess]=6 [fairness]=6 [assessment]=6 [age_binning]=7
    [mitigation]=8 [combinatorial]=9 [combo]=9 [compare]=10 [comparison]=10
)
declare -A STAGE_NAME=(
    [1]=load [2]=profile [3]=recommend [4]=preprocess [5]=train
    [6]=assess [7]=age_binning [8]=mitigation [9]=combinatorial [10]=compare
)

resolve_stage() {
    local input="${1,,}"  # lowercase
    # Plain number
    if [[ "$input" =~ ^[0-9]+$ ]]; then
        [[ -n "${STAGE_NAME[$input]+x}" ]] && echo "$input" && return
    fi
    # Strip common prefixes (phase3, stage3, step3)
    local stripped="${input#phase}"
    stripped="${stripped#stage}"
    stripped="${stripped#step}"
    if [[ "$stripped" =~ ^[0-9]+$ ]] && [[ -n "${STAGE_NAME[$stripped]+x}" ]]; then
        echo "$stripped" && return
    fi
    # Name / alias
    if [[ -n "${STAGE_NUM[$input]+x}" ]]; then
        echo "${STAGE_NUM[$input]}" && return
    fi
    echo "ERROR: Unknown stage '$1'. Valid: load(1) profile(2) recommend(3) preprocess(4) train(5) assess(6) age_binning(7) mitigation(8) combinatorial(9) compare(10)" >&2
    exit 1
}

# ======================================================================
# Configuration
# ======================================================================
RUN_AGE_BINNING=${RUN_AGE_BINNING:-true}
RUN_MITIGATION=${RUN_MITIGATION:-true}
RUN_COMBINATORIAL=${RUN_COMBINATORIAL:-true}
RUN_COMPARISON=${RUN_COMPARISON:-true}
RUN_RECOMMENDATIONS=${RUN_RECOMMENDATIONS:-true}
VERBOSE=${VERBOSE:-0}
RESUME_FROM=${RESUME_FROM:-""}
GO_UNTIL=${GO_UNTIL:-""}
AGE_BINNING_CONFIG="$ROOT_DIR/configs/experiments/age_binning.yaml"
MITIGATION_CONFIG="$ROOT_DIR/configs/experiments/mitigation.yaml"
COMBINATORIAL_CONFIG="$ROOT_DIR/configs/experiments/combinatorial.yaml"

# ======================================================================
# Resolve stage range
# ======================================================================
START_NUM=1
END_NUM=10

if [[ -n "$RESUME_FROM" ]]; then
    START_NUM=$(resolve_stage "$RESUME_FROM")
fi
if [[ -n "$GO_UNTIL" ]]; then
    END_NUM=$(resolve_stage "$GO_UNTIL")
fi

if (( START_NUM > END_NUM )); then
    echo "ERROR: --resume-from (${STAGE_NAME[$START_NUM]}, #$START_NUM) is after --go-until (${STAGE_NAME[$END_NUM]}, #$END_NUM)." >&2
    exit 1
fi

should_run() {
    local num=$1
    (( num >= START_NUM && num <= END_NUM ))
}

# ======================================================================
# Resolve run ID
# ======================================================================
BASE_RESULTS="$ROOT_DIR/output/cardiac"

if [[ -n "$RESUME_FROM" ]] && [[ -z "${RUN_ID:-}" ]]; then
    # Auto-resolve from latest_run pointer
    if [[ -L "$BASE_RESULTS/latest_run" ]]; then
        RUN_ID=$(basename "$(readlink -f "$BASE_RESULTS/latest_run")")
        echo "Auto-resolved run ID from latest_run symlink: $RUN_ID"
    elif [[ -f "$BASE_RESULTS/latest_run.txt" ]]; then
        RUN_ID=$(cat "$BASE_RESULTS/latest_run.txt" | tr -d '[:space:]')
        echo "Auto-resolved run ID from latest_run.txt: $RUN_ID"
    else
        echo "ERROR: No RUN_ID provided and no latest run found under $BASE_RESULTS. Cannot resume." >&2
        exit 1
    fi
fi

RUN_ID=${RUN_ID:-$(python3 - <<'PY'
import os
import uuid
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
pid = os.getpid()
suffix = uuid.uuid4().hex[:6]
print(f"run_{ts}_{pid}_{suffix}")
PY
)}
export RUN_ID

RUN_ROOT="$BASE_RESULTS/runs/$RUN_ID"
CHECKPOINT_DIR="$RUN_ROOT/.checkpoints"

# ======================================================================
# Checkpoint helpers
# ======================================================================
mark_done() {
    local num=$1
    local name=$2
    mkdir -p "$CHECKPOINT_DIR"
    cat > "$CHECKPOINT_DIR/${num}_${name}.done" <<EOF
{
  "stage": "$name",
  "number": $num,
  "completed_at": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "pid": $$
}
EOF
}

check_marker() {
    local num=$1
    local name=$2
    [[ -f "$CHECKPOINT_DIR/${num}_${name}.done" ]]
}

# ======================================================================
# Validate prior stages on resume
# ======================================================================
if [[ -n "$RESUME_FROM" ]] && (( START_NUM > 1 )); then
    echo "Validating prior stages for resume..."
    MISSING=""
    for (( i=1; i<START_NUM; i++ )); do
        sname="${STAGE_NAME[$i]}"
        if ! check_marker "$i" "$sname"; then
            MISSING="$MISSING  Stage $i ($sname): no checkpoint marker at $CHECKPOINT_DIR/${i}_${sname}.done\n"
        fi
    done
    if [[ -n "$MISSING" ]]; then
        echo -e "ERROR: Cannot resume from '${STAGE_NAME[$START_NUM]}' (stage $START_NUM)." >&2
        echo -e "Missing completion markers:\n$MISSING" >&2
        echo "Hint: re-run the full pipeline or an earlier RESUME_FROM to generate them." >&2
        exit 1
    fi
    echo "Resume validation passed — stages 1..$((START_NUM - 1)) are complete."
fi

# ======================================================================
# Banner
# ======================================================================
echo "======================================================================"
echo "CARDIAC FAIRNESS PIPELINE"
echo "======================================================================"
echo "Working directory: $ROOT_DIR"
echo "Run ID:           $RUN_ID"
echo "Stages:           $START_NUM..${END_NUM}  (${STAGE_NAME[$START_NUM]} → ${STAGE_NAME[$END_NUM]})"
echo "Age binning:      $RUN_AGE_BINNING"
echo "Mitigation:       $RUN_MITIGATION"
echo "Combinatorial:    $RUN_COMBINATORIAL"
echo "Comparison:       $RUN_COMPARISON"
echo "Recommendations:  $RUN_RECOMMENDATIONS"
echo ""

# Normalise VERBOSE: accept legacy true/false or numeric 0/1/2
case "$VERBOSE" in
    true)  VERBOSE=1 ;;
    false) VERBOSE=0 ;;
esac
VERBOSE=${VERBOSE:-0}

VERBOSE_FLAG=""
if (( VERBOSE >= 2 )); then
    VERBOSE_FLAG="-vv"
elif (( VERBOSE >= 1 )); then
    VERBOSE_FLAG="-v"
fi

# ======================================================================
# Pipeline stages
# ======================================================================

# Stage 1 — Load
if should_run 1; then
    echo "[PHASE 1/10] Loading cardiac datasets (standardization)"
    python3 "$ROOT_DIR/scripts/cardiac/load_data.py" $VERBOSE_FLAG
    mark_done 1 "load"
    echo ""
else
    echo "[1/10] load — SKIPPED (outside active range)"
fi

# Stage 2 — Profile
if should_run 2; then
    echo "[PHASE 2/10] Profiling datasets (complexity + fairness)"
    python3 "$ROOT_DIR/scripts/cardiac/profile_data.py" $VERBOSE_FLAG
    mark_done 2 "profile"
    echo ""
else
    echo "[2/10] profile — SKIPPED (outside active range)"
fi

# Stage 3 — Recommendations
if should_run 3; then
    if [[ "$RUN_RECOMMENDATIONS" == "true" ]]; then
        echo "[PHASE 3/10] Generating fairness triage recommendations"
        python3 "$ROOT_DIR/scripts/cardiac/generate_recommendations.py" --run-id "$RUN_ID" $VERBOSE_FLAG
        mark_done 3 "recommend"
        echo ""
    else
        echo "[3/10] Recommendations SKIPPED (disabled)"
    fi
else
    echo "[3/10] recommend — SKIPPED (outside active range)"
fi

# Stage 4 — Preprocess
if should_run 4; then
    echo "[PHASE 4/10] Preprocessing datasets (split + scale + fairness profiles)"
    PREPROCESS_ARGS=""
    if [[ "$RUN_COMBINATORIAL" == "true" ]]; then
        PREPROCESS_ARGS="--all-binnings"
    fi
    python3 "$ROOT_DIR/scripts/cardiac/preprocess.py" $PREPROCESS_ARGS $VERBOSE_FLAG
    mark_done 4 "preprocess"
    echo ""
else
    echo "[4/10] preprocess — SKIPPED (outside active range)"
fi

# Stage 5 — Train baseline
if should_run 5; then
    echo "[PHASE 5/10] Training baseline model(s)"
    python3 "$ROOT_DIR/scripts/cardiac/train_baseline.py" $VERBOSE_FLAG
    mark_done 5 "train"
    echo ""
else
    echo "[5/10] train — SKIPPED (outside active range)"
fi

# Stage 6 — Assess fairness
if should_run 6; then
    echo "[PHASE 6/10] Assessing post-prediction fairness"
    python3 "$ROOT_DIR/scripts/cardiac/assess_predictions.py" $VERBOSE_FLAG
    mark_done 6 "assess"
    echo ""
else
    echo "[6/10] assess — SKIPPED (outside active range)"
fi

# Stage 7 — Age binning (optional)
ARCHIVE_EXPERIMENTS=true
if should_run 7; then
    if [[ "$RUN_AGE_BINNING" == "true" ]]; then
        echo "[PHASE 7/10] Age binning strategies analysis"
        EXPERIMENT_RUN_MODE=full ARCHIVE_PREVIOUS=$ARCHIVE_EXPERIMENTS python3 "$ROOT_DIR/scripts/cardiac/age_binning.py" \
            --config "$AGE_BINNING_CONFIG" --run-id "$RUN_ID" $VERBOSE_FLAG
        ARCHIVE_EXPERIMENTS=false
        mark_done 7 "age_binning"
        echo ""
    else
        echo "[7/10] Age binning SKIPPED (disabled)"
    fi
else
    echo "[7/10] age_binning — SKIPPED (outside active range)"
fi

# Stage 8 — Mitigation (optional)
if should_run 8; then
    if [[ "$RUN_MITIGATION" == "true" ]]; then
        echo "[PHASE 8/10] Mitigation techniques comparison"
        EXPERIMENT_RUN_MODE=full ARCHIVE_PREVIOUS=$ARCHIVE_EXPERIMENTS python3 "$ROOT_DIR/scripts/cardiac/mitigation.py" \
            --config "$MITIGATION_CONFIG" --run-id "$RUN_ID" $VERBOSE_FLAG
        ARCHIVE_EXPERIMENTS=false
        mark_done 8 "mitigation"
        echo ""
    else
        echo "[8/10] Mitigation SKIPPED (disabled)"
    fi
else
    echo "[8/10] mitigation — SKIPPED (outside active range)"
fi

# Stage 9 — Combinatorial (optional)
if should_run 9; then
    if [[ "$RUN_COMBINATORIAL" == "true" ]]; then
        echo "[PHASE 9/10] Combinatorial experiments"
        python3 "$ROOT_DIR/scripts/cardiac/combinatorial.py" \
            --config "$COMBINATORIAL_CONFIG" --run-id "$RUN_ID" $VERBOSE_FLAG
        mark_done 9 "combinatorial"
        echo ""
    else
        echo "[9/10] Combinatorial SKIPPED (disabled)"
    fi
else
    echo "[9/10] combinatorial — SKIPPED (outside active range)"
fi

# Stage 10 — Compare (optional)
if should_run 10; then
    if [[ "$RUN_COMPARISON" == "true" ]]; then
        echo "[PHASE 10/10] Experiment comparison"
        python3 "$ROOT_DIR/scripts/cardiac/compare.py" --run-id "$RUN_ID" $VERBOSE_FLAG
        mark_done 10 "compare"
        echo ""
    else
        echo "[10/10] Comparison SKIPPED (disabled)"
    fi
else
    echo "[10/10] compare — SKIPPED (outside active range)"
fi

# ======================================================================
# Summary
# ======================================================================
echo "======================================================================"
echo "PIPELINE COMPLETE"
echo "======================================================================"
echo "Stages executed: ${STAGE_NAME[$START_NUM]} → ${STAGE_NAME[$END_NUM]}"
echo "Results saved to:"
echo "  - Run root:           $RUN_ROOT"
should_run 1 && echo "  - Raw data:           $ROOT_DIR/data/raw/cardiac"
should_run 4 && echo "  - Processed data:     $ROOT_DIR/data/processed/cardiac"
should_run 2 && echo "  - Profiling:          $RUN_ROOT/profiling"
should_run 3 && [[ "$RUN_RECOMMENDATIONS" == "true" ]] && echo "  - Recommendations:    $RUN_ROOT/recommendations"
should_run 5 && echo "  - Baseline:           $RUN_ROOT/baseline"
should_run 7 && [[ "$RUN_AGE_BINNING" == "true" ]] && echo "  - Age binning:        $RUN_ROOT/experiments/full/age_binning"
should_run 8 && [[ "$RUN_MITIGATION" == "true" ]] && echo "  - Mitigation:         $RUN_ROOT/experiments/full/mitigation"
should_run 9 && [[ "$RUN_COMBINATORIAL" == "true" ]] && echo "  - Combinatorial:      $RUN_ROOT/experiments/full"
should_run 10 && [[ "$RUN_COMPARISON" == "true" ]] && echo "  - Comparison:         $RUN_ROOT/experiments/full/comparisons"
echo ""
