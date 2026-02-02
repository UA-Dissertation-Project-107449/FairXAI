#!/usr/bin/env bash
set -euo pipefail

# Cardiac end-to-end pipeline: load -> preprocess -> train -> fairness -> experiments
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

# Configuration
RUN_AGE_BINNING=${RUN_AGE_BINNING:-true}
RUN_MITIGATION=${RUN_MITIGATION:-true}
RUN_COMBINATORIAL=${RUN_COMBINATORIAL:-true}
RUN_COMPARISON=${RUN_COMPARISON:-true}
VERBOSE=${VERBOSE:-false}
AGE_BINNING_CONFIG="$ROOT_DIR/configs/experiments/age_binning.yaml"
MITIGATION_CONFIG="$ROOT_DIR/configs/experiments/mitigation.yaml"
COMBINATORIAL_CONFIG="$ROOT_DIR/configs/experiments/combinatorial.yaml"

echo "======================================================================"
echo "CARDIAC FAIRNESS PIPELINE"
echo "======================================================================"
echo "Working directory: $ROOT_DIR"
echo "Age binning: $RUN_AGE_BINNING"
echo "Mitigation: $RUN_MITIGATION"
echo "Combinatorial: $RUN_COMBINATORIAL"
echo "Comparison: $RUN_COMPARISON"
echo ""

VERBOSE_FLAG=""
if [[ "$VERBOSE" == "true" ]]; then
    VERBOSE_FLAG="-v"
fi

echo "[PHASE 1/8] Loading cardiac datasets (standardization + profiling)"
python3 "$ROOT_DIR/scripts/cardiac/load_data.py" $VERBOSE_FLAG
echo ""

echo "[PHASE 2/8] Preprocessing datasets (split + scale + fairness profiles)"
python3 "$ROOT_DIR/scripts/cardiac/preprocess.py" $VERBOSE_FLAG
echo ""

echo "[PHASE 3/8] Training baseline model(s)"
python3 "$ROOT_DIR/scripts/cardiac/train_baseline.py" $VERBOSE_FLAG
echo ""

echo "[PHASE 4/8] Assessing post-prediction fairness"
python3 "$ROOT_DIR/scripts/cardiac/assess_predictions.py" $VERBOSE_FLAG
echo ""

ARCHIVE_EXPERIMENTS=true
if [[ "$RUN_AGE_BINNING" == "true" ]]; then
    echo "[PHASE 5/8] Age binning strategies analysis"
    EXPERIMENT_RUN_MODE=full ARCHIVE_PREVIOUS=$ARCHIVE_EXPERIMENTS python3 "$ROOT_DIR/scripts/cardiac/age_binning.py" \
        --config "$AGE_BINNING_CONFIG" $VERBOSE_FLAG
    ARCHIVE_EXPERIMENTS=false
    echo ""
else
    echo "[5/8] Age binning analysis SKIPPED"
fi

if [[ "$RUN_MITIGATION" == "true" ]]; then
    echo "[PHASE 6/8] Mitigation techniques comparison"
    EXPERIMENT_RUN_MODE=full ARCHIVE_PREVIOUS=$ARCHIVE_EXPERIMENTS python3 "$ROOT_DIR/scripts/cardiac/mitigation.py" \
        --config "$MITIGATION_CONFIG" $VERBOSE_FLAG
    ARCHIVE_EXPERIMENTS=false
    echo ""
else
    echo "[6/8] Mitigation comparison SKIPPED"
fi

if [[ "$RUN_COMBINATORIAL" == "true" ]]; then
    echo "[PHASE 7/8] Combinatorial experiments"
    python3 "$ROOT_DIR/scripts/cardiac/combinatorial.py" \
        --config "$COMBINATORIAL_CONFIG" --archive-previous $VERBOSE_FLAG
    echo ""
else
    echo "[7/8] Combinatorial experiments SKIPPED"
fi

if [[ "$RUN_COMPARISON" == "true" ]]; then
    echo "[PHASE 8/8] Experiment comparison"
    python3 "$ROOT_DIR/scripts/cardiac/compare.py" $VERBOSE_FLAG
    echo ""
else
    echo "[8/8] Experiment comparison SKIPPED"
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
if [[ "$RUN_AGE_BINNING" == "true" ]]; then
    echo "  - Age binning:        $ROOT_DIR/results/cardiac/experiments/age_binning"
fi
if [[ "$RUN_MITIGATION" == "true" ]]; then
    echo "  - Mitigation:         $ROOT_DIR/results/cardiac/experiments/mitigation"
fi
if [[ "$RUN_COMBINATORIAL" == "true" ]]; then
    echo "  - Combinatorial:      $ROOT_DIR/results/cardiac/experiments/full/latest_run"
fi
if [[ "$RUN_COMPARISON" == "true" ]]; then
    echo "  - Comparison:         $ROOT_DIR/results/cardiac/experiments/full/latest_run/comparisons"
fi
