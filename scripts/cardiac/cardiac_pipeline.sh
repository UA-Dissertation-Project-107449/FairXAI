#!/usr/bin/env bash
set -euo pipefail

# Cardiac end-to-end pipeline: load -> preprocess -> studies -> baseline -> experiments
# Supports --resume-from / --go-until via env vars RESUME_FROM / GO_UNTIL.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

# ======================================================================
# Stage mapping (name -> number, number -> name)
# ======================================================================
declare -A STAGE_NUM=(
    [load]=1 [profile]=2 [profiling]=2 [recommend]=3 [recommendations]=3
    [triage]=3 [preprocess]=4 [preprocessing]=4
    [hpo_study]=5 [hpo]=5 [feature_selection_study]=6 [feature_selection]=6 [fs_study]=6
    [train]=7 [baseline]=7 [training]=7 [assess]=8 [fairness]=8 [assessment]=8
    [attribute_binning]=9 [age_binning]=9 [mitigation]=10
    [combinatorial]=11 [combo]=11 [compare]=12 [comparison]=12
)
declare -A STAGE_NAME=(
    [1]=load [2]=profile [3]=recommend [4]=preprocess [5]=hpo_study
    [6]=feature_selection_study [7]=train [8]=assess [9]=attribute_binning
    [10]=mitigation [11]=combinatorial [12]=compare
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
    echo "ERROR: Unknown stage '$1'. Valid: load(1) profile(2) recommend(3) preprocess(4) hpo_study(5) feature_selection_study(6) train(7) assess(8) attribute_binning(9) mitigation(10) combinatorial(11) compare(12)" >&2
    exit 1
}

# ======================================================================
# Configuration
# ======================================================================
RUN_ATTRIBUTE_BINNING=${RUN_ATTRIBUTE_BINNING:-${RUN_AGE_BINNING:-true}}
RUN_HPO_STUDY=${RUN_HPO_STUDY:-true}
RUN_FEATURE_SELECTION_STUDY=${RUN_FEATURE_SELECTION_STUDY:-true}
RUN_MITIGATION=${RUN_MITIGATION:-true}
RUN_COMBINATORIAL=${RUN_COMBINATORIAL:-true}
RUN_COMPARISON=${RUN_COMPARISON:-true}
RUN_RECOMMENDATIONS=${RUN_RECOMMENDATIONS:-true}
VERBOSE=${VERBOSE:-0}
RESUME_FROM=${RESUME_FROM:-""}
GO_UNTIL=${GO_UNTIL:-""}
ATTRIBUTE_BINNING_CONFIG="$ROOT_DIR/configs/experiments/age_binning.yaml"
GROUPING_CONFIG="$ROOT_DIR/configs/experiments/clustering.yaml"
MITIGATION_CONFIG="$ROOT_DIR/configs/experiments/mitigation.yaml"
COMBINATORIAL_CONFIG="$ROOT_DIR/configs/experiments/combinatorial.yaml"
HPO_CONFIG="$ROOT_DIR/configs/experiments/hpo.yaml"
FEATURE_SELECTION_STUDY_CONFIG="$ROOT_DIR/configs/experiments/feature_selection_study.yaml"

DATASETS=()
MODEL_TYPES=()

# Optional CLI overrides for dataset/model scope.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASETS+=("$1")
                shift
            done
            continue
            ;;
        --model-types)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODEL_TYPES+=("$1")
                shift
            done
            continue
            ;;
        --resume-from)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --resume-from requires a value" >&2; exit 1; }
            RESUME_FROM="$1"
            ;;
        --go-until)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --go-until requires a value" >&2; exit 1; }
            GO_UNTIL="$1"
            ;;
        --run-id)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --run-id requires a value" >&2; exit 1; }
            RUN_ID="$1"
            ;;
        -v)
            VERBOSE=1
            ;;
        -vv)
            VERBOSE=2
            ;;
        --no-hpo-study)
            RUN_HPO_STUDY=false
            ;;
        --no-feature-selection-study)
            RUN_FEATURE_SELECTION_STUDY=false
            ;;
        *)
            echo "ERROR: Unknown argument '$1'" >&2
            echo "Supported: --datasets ... --model-types ... --resume-from STAGE --go-until STAGE --run-id ID --no-hpo-study --no-feature-selection-study -v -vv" >&2
            exit 1
            ;;
    esac
    shift
done

DATASET_ARGS=()
if (( ${#DATASETS[@]} > 0 )); then
    DATASET_ARGS=(--datasets "${DATASETS[@]}")
fi

MODEL_TYPE_ARGS=()
if (( ${#MODEL_TYPES[@]} > 0 )); then
    MODEL_TYPE_ARGS=(--model-types "${MODEL_TYPES[@]}")
fi

# ======================================================================
# Resolve stage range
# ======================================================================
START_NUM=1
END_NUM=12

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
SELECTOR_CONTRACT_PATH="$RUN_ROOT/recommendations/selector_contract.json"

# Point logs/cardiac/latest_run at this run
LOG_RUN_DIR="$ROOT_DIR/logs/cardiac/runs/$RUN_ID"
mkdir -p "$LOG_RUN_DIR"
_LOG_BASE="$ROOT_DIR/logs/cardiac"
_LOG_LINK="$_LOG_BASE/latest_run"
_LOG_TXT="$_LOG_BASE/latest_run.txt"
[[ -L "$_LOG_LINK" ]] && rm -f "$_LOG_LINK"
ln -s "runs/$RUN_ID" "$_LOG_LINK" 2>/dev/null || true
echo "$RUN_ID" > "$_LOG_TXT"

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
echo "HPO study:        $RUN_HPO_STUDY"
echo "Feature study:    $RUN_FEATURE_SELECTION_STUDY"
echo "Attr binning:     $RUN_ATTRIBUTE_BINNING"
echo "Mitigation:       $RUN_MITIGATION"
echo "Combinatorial:    $RUN_COMBINATORIAL"
echo "Comparison:       $RUN_COMPARISON"
echo "Recommendations:  $RUN_RECOMMENDATIONS"
echo "Datasets:         ${DATASETS[*]:-config/default}"
echo "Model types:      ${MODEL_TYPES[*]:-config/default}"
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

STUDY_VERBOSE_FLAG=""
if (( VERBOSE >= 1 )); then
    STUDY_VERBOSE_FLAG="-v"
fi

# ======================================================================
# Pipeline stages
# ======================================================================

# Stage 1 — Load
if should_run 1; then
    echo "[PHASE 1/12] Loading cardiac datasets (standardization)"
    python3 "$ROOT_DIR/scripts/cardiac/load_data.py" "${DATASET_ARGS[@]}" $VERBOSE_FLAG
    mark_done 1 "load"
    echo ""
else
    echo "[1/12] load — SKIPPED (outside active range)"
fi

# Stage 2 — Profile
if should_run 2; then
    echo "[PHASE 2/12] Profiling datasets (complexity + fairness)"
    python3 "$ROOT_DIR/scripts/cardiac/profile_data.py" "${DATASET_ARGS[@]}" $VERBOSE_FLAG
    mark_done 2 "profile"
    echo ""
else
    echo "[2/12] profile — SKIPPED (outside active range)"
fi

# Stage 3 — Recommendations
if should_run 3; then
    if [[ "$RUN_RECOMMENDATIONS" == "true" ]]; then
        echo "[PHASE 3/12] Generating fairness triage recommendations"
        python3 "$ROOT_DIR/scripts/cardiac/generate_recommendations.py" --run-id "$RUN_ID" $VERBOSE_FLAG
        mark_done 3 "recommend"
        echo ""
    else
        echo "[3/12] Recommendations SKIPPED (disabled)"
        mark_done 3 "recommend"
        echo "[3/12] recommend - checkpointed as skipped (disabled)"
    fi
else
    echo "[3/12] recommend — SKIPPED (outside active range)"
fi

# Stage 4 — Preprocess
if should_run 4; then
    echo "[PHASE 4/12] Preprocessing datasets (split + scale + fairness profiles)"
    PREPROCESS_ARGS=""
    if [[ "$RUN_COMBINATORIAL" == "true" ]]; then
        PREPROCESS_ARGS="--all-binnings"
    fi
    python3 "$ROOT_DIR/scripts/cardiac/preprocess.py" $PREPROCESS_ARGS "${DATASET_ARGS[@]}" $VERBOSE_FLAG
    mark_done 4 "preprocess"
    echo ""
else
    echo "[4/12] preprocess — SKIPPED (outside active range)"
fi

# Stage 5 — HPO study (optional)
if should_run 5; then
    if [[ "$RUN_HPO_STUDY" == "true" ]]; then
        echo "[PHASE 5/12] Hyperparameter optimisation study"
        python3 "$ROOT_DIR/scripts/studies/run_hpo.py" \
            --pipeline cardiac --config "$HPO_CONFIG" \
            "${DATASET_ARGS[@]}" "${MODEL_TYPE_ARGS[@]}" $STUDY_VERBOSE_FLAG
        mark_done 5 "hpo_study"
        echo ""
    else
        echo "[5/12] HPO study SKIPPED (disabled)"
        mark_done 5 "hpo_study"
        echo "[5/12] hpo_study - checkpointed as skipped (disabled)"
    fi
else
    echo "[5/12] hpo_study — SKIPPED (outside active range)"
fi

# Stage 6 — Feature-selection study (optional)
if should_run 6; then
    if [[ "$RUN_FEATURE_SELECTION_STUDY" == "true" ]]; then
        echo "[PHASE 6/12] Feature-selection ablation study"
        python3 "$ROOT_DIR/scripts/studies/run_feature_selection_study.py" \
            --pipeline cardiac --config "$FEATURE_SELECTION_STUDY_CONFIG" \
            "${DATASET_ARGS[@]}" "${MODEL_TYPE_ARGS[@]}" $STUDY_VERBOSE_FLAG
        mark_done 6 "feature_selection_study"
        echo ""
    else
        echo "[6/12] Feature-selection study SKIPPED (disabled)"
        mark_done 6 "feature_selection_study"
        echo "[6/12] feature_selection_study - checkpointed as skipped (disabled)"
    fi
else
    echo "[6/12] feature_selection_study — SKIPPED (outside active range)"
fi

# Wiring helper for stages that consume study recommendations
if should_run 7 || { should_run 11 && [[ "$RUN_COMBINATORIAL" == "true" ]]; }; then
    echo "[WIRING] Building selector contract from study artifacts"
    python3 "$ROOT_DIR/scripts/studies/build_selector_contract.py" \
        --pipeline cardiac --run-id "$RUN_ID" \
        "${DATASET_ARGS[@]}" "${MODEL_TYPE_ARGS[@]}" $STUDY_VERBOSE_FLAG
    echo ""
else
    echo "[WIRING] selector_contract — SKIPPED (downstream stages not active)"
fi

# Stage 7 — Train baseline
if should_run 7; then
    echo "[PHASE 7/12] Training baseline model(s)"
    python3 "$ROOT_DIR/scripts/cardiac/train_baseline.py" \
        --selector-contract "$SELECTOR_CONTRACT_PATH" \
        "${DATASET_ARGS[@]}" "${MODEL_TYPE_ARGS[@]}" $VERBOSE_FLAG
    mark_done 7 "train"
    echo ""
else
    echo "[7/12] train — SKIPPED (outside active range)"
fi

# Stage 8 — Assess fairness
if should_run 8; then
    echo "[PHASE 8/12] Assessing post-prediction fairness"
    python3 "$ROOT_DIR/scripts/cardiac/assess_predictions.py" "${DATASET_ARGS[@]}" "${MODEL_TYPE_ARGS[@]}" $VERBOSE_FLAG
    mark_done 8 "assess"
    echo ""
else
    echo "[8/12] assess — SKIPPED (outside active range)"
fi

# Stage 9 — Attribute binning (optional)
ARCHIVE_EXPERIMENTS=true
if should_run 9; then
    if [[ "$RUN_ATTRIBUTE_BINNING" == "true" ]]; then
        echo "[PHASE 9/12] Attribute binning strategies analysis"
        EXPERIMENT_RUN_MODE=full ARCHIVE_PREVIOUS=$ARCHIVE_EXPERIMENTS python3 "$ROOT_DIR/scripts/experiments/run_attribute_binning_analysis.py" \
            --config "$ATTRIBUTE_BINNING_CONFIG" --run-id "$RUN_ID" --pipeline cardiac "${DATASET_ARGS[@]}" $VERBOSE_FLAG
        ARCHIVE_EXPERIMENTS=false
        mark_done 9 "attribute_binning"
        echo ""
    else
        echo "[9/12] Attribute binning SKIPPED (disabled)"
        mark_done 9 "attribute_binning"
        echo "[9/12] attribute_binning - checkpointed as skipped (disabled)"
    fi
else
    echo "[9/12] attribute_binning — SKIPPED (outside active range)"
fi

# Stage 10 — Mitigation (optional)
if should_run 10; then
    if [[ "$RUN_MITIGATION" == "true" ]]; then
        echo "[PHASE 10/12] Mitigation techniques comparison"
        EXPERIMENT_RUN_MODE=full ARCHIVE_PREVIOUS=$ARCHIVE_EXPERIMENTS python3 "$ROOT_DIR/scripts/cardiac/mitigation.py" \
            --config "$MITIGATION_CONFIG" --run-id "$RUN_ID" "${DATASET_ARGS[@]}" $VERBOSE_FLAG
        ARCHIVE_EXPERIMENTS=false
        mark_done 10 "mitigation"
        echo ""
    else
        echo "[10/12] Mitigation SKIPPED (disabled)"
        mark_done 10 "mitigation"
        echo "[10/12] mitigation - checkpointed as skipped (disabled)"
    fi
else
    echo "[10/12] mitigation — SKIPPED (outside active range)"
fi

# Stage 11 — Combinatorial (optional)
if should_run 11; then
    if [[ "$RUN_COMBINATORIAL" == "true" ]]; then
        echo "[PHASE 11/12] Combinatorial experiments"
        python3 "$ROOT_DIR/scripts/cardiac/combinatorial.py" \
            --config "$COMBINATORIAL_CONFIG" --run-id "$RUN_ID" \
            --selector-contract "$SELECTOR_CONTRACT_PATH" \
            "${DATASET_ARGS[@]}" "${MODEL_TYPE_ARGS[@]}" $VERBOSE_FLAG
        mark_done 11 "combinatorial"
        echo ""
    else
        echo "[11/12] Combinatorial SKIPPED (disabled)"
        mark_done 11 "combinatorial"
        echo "[11/12] combinatorial - checkpointed as skipped (disabled)"
    fi
else
    echo "[11/12] combinatorial — SKIPPED (outside active range)"
fi

# Stage 12 — Compare (optional)
if should_run 12; then
    if [[ "$RUN_COMPARISON" == "true" ]]; then
        echo "[PHASE 12/12] Experiment comparison and dissertation plots"
        python3 "$ROOT_DIR/scripts/cardiac/compare.py" --run-id "$RUN_ID" $VERBOSE_FLAG
        python3 "$ROOT_DIR/scripts/studies/generate_dissertation_plots.py" --run-id "$RUN_ID"
        mark_done 12 "compare"
        echo ""
    else
        echo "[12/12] Comparison SKIPPED (disabled)"
        mark_done 12 "compare"
        echo "[12/12] compare - checkpointed as skipped (disabled)"
    fi
else
    echo "[12/12] compare — SKIPPED (outside active range)"
fi

# ======================================================================
# Log summary
# ======================================================================
python3 -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('$ROOT_DIR') / 'src'))
from fairxai.utils.logging_utils import summarize_run_logs
s = summarize_run_logs(pathlib.Path('$LOG_RUN_DIR'))
tw, te = s['total_warnings'], s['total_errors']
if tw or te:
    print(f'Log summary: {tw} warning(s), {te} error(s) — see $LOG_RUN_DIR/run_summary.json')
else:
    print('Log summary: no warnings or errors recorded.')
"

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
should_run 5 && [[ "$RUN_HPO_STUDY" == "true" ]] && echo "  - HPO study:          $ROOT_DIR/output/cardiac/studies/hpo"
should_run 6 && [[ "$RUN_FEATURE_SELECTION_STUDY" == "true" ]] && echo "  - FS study:           $ROOT_DIR/output/cardiac/studies/feature_selection"
should_run 2 && echo "  - Profiling:          $RUN_ROOT/profiling"
should_run 3 && [[ "$RUN_RECOMMENDATIONS" == "true" ]] && echo "  - Recommendations:    $RUN_ROOT/recommendations"
[[ -f "$SELECTOR_CONTRACT_PATH" ]] && echo "  - Selector contract:  $SELECTOR_CONTRACT_PATH"
should_run 7 && echo "  - Baseline:           $RUN_ROOT/baseline"
should_run 9 && [[ "$RUN_ATTRIBUTE_BINNING" == "true" ]] && echo "  - Attr binning:       $RUN_ROOT/experiments/attribute_binning"
should_run 10 && [[ "$RUN_MITIGATION" == "true" ]] && echo "  - Mitigation:         $RUN_ROOT/experiments/mitigation"
should_run 11 && [[ "$RUN_COMBINATORIAL" == "true" ]] && echo "  - Combinatorial:      $RUN_ROOT/experiments"
should_run 12 && [[ "$RUN_COMPARISON" == "true" ]] && echo "  - Comparison:         $RUN_ROOT/experiments/comparisons"
echo ""
