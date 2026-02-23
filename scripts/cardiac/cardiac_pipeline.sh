#!/usr/bin/env bash
set -euo pipefail

# Cardiac end-to-end pipeline: load -> preprocess -> train -> fairness -> experiments
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

# Configuration
RUN_AGE_BINNING=${RUN_AGE_BINNING:-true}
RUN_MITIGATION=${RUN_MITIGATION:-true}
RUN_COMBINATORIAL=${RUN_COMBINATORIAL:-true}
RUN_COMPARISON=${RUN_COMPARISON:-true}
RUN_RECOMMENDATIONS=${RUN_RECOMMENDATIONS:-true}
VERBOSE=${VERBOSE:-false}
AGE_BINNING_CONFIG="$ROOT_DIR/configs/experiments/age_binning.yaml"
MITIGATION_CONFIG="$ROOT_DIR/configs/experiments/mitigation.yaml"
COMBINATORIAL_CONFIG="$ROOT_DIR/configs/experiments/combinatorial.yaml"
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

echo "======================================================================"
echo "CARDIAC FAIRNESS PIPELINE"
echo "======================================================================"
echo "Working directory: $ROOT_DIR"
echo "Age binning: $RUN_AGE_BINNING"
echo "Mitigation: $RUN_MITIGATION"
echo "Combinatorial: $RUN_COMBINATORIAL"
echo "Comparison: $RUN_COMPARISON"
echo "Recommendations: $RUN_RECOMMENDATIONS"
echo "Run ID: $RUN_ID"
echo ""

VERBOSE_FLAG=""
if [[ "$VERBOSE" == "true" ]]; then
    VERBOSE_FLAG="-v"
fi

echo "[PHASE 1/10] Loading cardiac datasets (standardization)"
python3 "$ROOT_DIR/scripts/cardiac/load_data.py" $VERBOSE_FLAG
echo ""

echo "[PHASE 2/10] Profiling datasets (complexity + fairness)"
python3 "$ROOT_DIR/scripts/cardiac/profile_data.py" $VERBOSE_FLAG
echo ""

if [[ "$RUN_RECOMMENDATIONS" == "true" ]]; then
    echo "[PHASE 3/10] Generating fairness triage recommendations"
    python3 "$ROOT_DIR/scripts/cardiac/generate_recommendations.py" --run-id "$RUN_ID" $VERBOSE_FLAG
    echo ""
else
    echo "[3/10] Recommendations SKIPPED"
fi

echo "[PHASE 4/10] Preprocessing datasets (split + scale + fairness profiles)"
PREPROCESS_ARGS=""
if [[ "$RUN_COMBINATORIAL" == "true" ]]; then
    PREPROCESS_ARGS="--all-binnings"
fi
python3 "$ROOT_DIR/scripts/cardiac/preprocess.py" $PREPROCESS_ARGS $VERBOSE_FLAG
echo ""

echo "[PHASE 5/10] Training baseline model(s)"
python3 "$ROOT_DIR/scripts/cardiac/train_baseline.py" $VERBOSE_FLAG
echo ""

echo "[PHASE 6/10] Assessing post-prediction fairness"
python3 "$ROOT_DIR/scripts/cardiac/assess_predictions.py" $VERBOSE_FLAG
echo ""

ARCHIVE_EXPERIMENTS=true
if [[ "$RUN_AGE_BINNING" == "true" ]]; then
    echo "[PHASE 7/10] Age binning strategies analysis"
    EXPERIMENT_RUN_MODE=full ARCHIVE_PREVIOUS=$ARCHIVE_EXPERIMENTS python3 "$ROOT_DIR/scripts/cardiac/age_binning.py" \
        --config "$AGE_BINNING_CONFIG" --run-id "$RUN_ID" $VERBOSE_FLAG
    ARCHIVE_EXPERIMENTS=false
    echo ""
else
    echo "[7/10] Age binning analysis SKIPPED"
fi

if [[ "$RUN_MITIGATION" == "true" ]]; then
    echo "[PHASE 8/10] Mitigation techniques comparison"
    EXPERIMENT_RUN_MODE=full ARCHIVE_PREVIOUS=$ARCHIVE_EXPERIMENTS python3 "$ROOT_DIR/scripts/cardiac/mitigation.py" \
        --config "$MITIGATION_CONFIG" --run-id "$RUN_ID" $VERBOSE_FLAG
    ARCHIVE_EXPERIMENTS=false
    echo ""
else
    echo "[8/10] Mitigation comparison SKIPPED"
fi

if [[ "$RUN_COMBINATORIAL" == "true" ]]; then
    echo "[PHASE 9/10] Combinatorial experiments"
    python3 "$ROOT_DIR/scripts/cardiac/combinatorial.py" \
        --config "$COMBINATORIAL_CONFIG" --run-id "$RUN_ID" $VERBOSE_FLAG
    echo ""
else
    echo "[9/10] Combinatorial experiments SKIPPED"
fi

if [[ "$RUN_COMPARISON" == "true" ]]; then
    echo "[PHASE 10/10] Experiment comparison"
    python3 "$ROOT_DIR/scripts/cardiac/compare.py" --run-id "$RUN_ID" $VERBOSE_FLAG
    echo ""
else
    echo "[10/10] Experiment comparison SKIPPED"
fi

echo "======================================================================"
echo "PIPELINE COMPLETE"
echo "======================================================================"
echo "Results saved to:"
echo "  - Raw data:           $ROOT_DIR/data/raw/cardiac"
echo "  - Processed data:     $ROOT_DIR/data/processed/cardiac"
echo "  - Baseline results:   $ROOT_DIR/results/cardiac/baseline"
echo "  - Baseline models:    $ROOT_DIR/results/cardiac/baseline/models"
echo "  - Profiling results:  $ROOT_DIR/results/cardiac/profiling"
if [[ "$RUN_RECOMMENDATIONS" == "true" ]]; then
    echo "  - Recommendations:    $ROOT_DIR/results/cardiac/recommendations"
fi
if [[ "$RUN_AGE_BINNING" == "true" ]]; then
    echo "  - Age binning:        $ROOT_DIR/results/cardiac/runs/$RUN_ID/experiments/full/age_binning"
fi
if [[ "$RUN_MITIGATION" == "true" ]]; then
    echo "  - Mitigation:         $ROOT_DIR/results/cardiac/runs/$RUN_ID/experiments/full/mitigation"
fi
if [[ "$RUN_COMBINATORIAL" == "true" ]]; then
    echo "  - Combinatorial:      $ROOT_DIR/results/cardiac/runs/$RUN_ID/experiments/full"
fi
if [[ "$RUN_COMPARISON" == "true" ]]; then
    echo "  - Comparison:         $ROOT_DIR/results/cardiac/runs/$RUN_ID/experiments/full/comparisons"
fi
