# Testing Guide

## Structure

```
tests/
├── conftest.py                          # Shared fixtures (synthetic data, tmp dirs, fairness dicts)
├── unit/
│   ├── test_combinatorial_runner.py     # Mitigation filtering logic, coerce helpers
│   ├── test_experiment_comparison.py    # Per-group fairness extraction, baseline lookup key
│   ├── test_fairness_metrics.py         # Demographic parity and equalized odds formulas
│   └── test_gpu_detection.py            # detect_accelerator() always returns a valid device
└── integration/
    ├── test_combinatorial_multi_model.py  # YAML config, mitigation filter across all 4 models
    ├── test_compare_multi_model.py        # cross_model_summary, per_group_comparison, delta calc
    └── test_multi_model_baseline.py       # [slow] subprocess: train_baseline.py → 4 model CSVs
```

## Running Tests

```bash
cd Code/FairXAI

# Fast (unit + integration, no subprocess) — < 5 seconds
python3 -m pytest tests/unit/ tests/integration/ -m "not slow"

# All tests including slow subprocess test — up to 5 minutes
python3 -m pytest tests/

# Unit only
python3 -m pytest tests/unit/ -v

# Integration only (no slow)
python3 -m pytest tests/integration/ -m "not slow" -v

# Integration including the slow baseline subprocess test
python3 -m pytest tests/integration/ -v

# Slow baseline test only (serial)
python3 -m pytest tests/integration/test_multi_model_baseline.py -v

# Slow baseline test only (parallel subsets via xdist)
python3 -m pytest tests/integration/test_multi_model_baseline.py -n auto -v

# Single file
python3 -m pytest tests/unit/test_experiment_comparison.py -v
```

## Marks

| Mark | Meaning | Deselect |
|------|---------|----------|
| `slow` | Runs a real subprocess (`train_baseline.py`). | `-m "not slow"` |
| `xgboost_model` | Includes XGBoost training paths (used in baseline subset case `svm_xgb`). | `-m "not xgboost_model"` |

Register custom marks is handled in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Why The Slow Baseline Test Became Fast

The large speedup is expected and is not caused by filesystem search time.

- Before the fix, the baseline script resolved project root from the script location, so the temporary test config could be ignored and the run could execute against heavier repo-level paths/configs.
- After the fix, the test passes an explicit `--project-root` pointing to the isolated temp directory, so it always trains on tiny synthetic data and writes outputs there.
- The test now runs in two parameterized subsets (`lr_rf` and `svm_xgb`), which can run concurrently with `-n auto`.
- Test model configs are lightweight (fewer trees, linear SVM, constrained jobs), reducing per-model training cost.

## Shared Fixtures (`conftest.py`)

| Fixture | What it provides |
|---------|-----------------|
| `synthetic_cardiac_df` | 50-row cardiac-like DataFrame (age_group, sex, clinical cols, heart_disease) |
| `synthetic_predictions_df` | Same + y_true, y_pred, y_proba columns |
| `minimal_fairness_metrics_dict` | Realistic nested fairness dict (demographic_parity + equalized_odds for 2 age groups) |
| `tmp_run_root` | Temp run directory tree mirroring `output/cardiac/runs/<id>/` layout |
| `sample_baseline_fairness_json` | Writes a stage-6 fairness assessment JSON into `tmp_run_root` |

## What Is and Isn't Tested

**Tested:**
- Mitigation filtering logic (baseline always runs; non-LR models skip non-baseline mitigations)
- Config-driven `mitigation_supported_model_types` (extending the set enables a new model)
- Baseline lookup key includes `model_type` — regression guard against cross-model contamination
- Per-group fairness extraction from nested experiment JSONs
- `_build_per_group_comparison` writes `per_group_comparison.csv` with correct columns and delta
- `cross_model_summary.csv` has exactly one row per model type
- `detect_accelerator()` never raises and returns a valid device string
- `_coerce_probability_vector` and `_coerce_label_vector` handle all array shapes
- `FairnessMetrics.demographic_parity` produces correct max_difference for known inputs
- YAML config has all 4 model types and `mitigation_supported_model_types` key

**Not tested (by design):**
- Pipeline stage ordering — covered by running the bash script end-to-end
- SHAP/LIME output correctness — too slow; covered when XAI is enabled in a full run
- Numerical fairness precision beyond 1e-9 — not meaningful for clinical tabular data
- Full combinatorial sweep — runtime would be minutes; test the logic, not the volume
