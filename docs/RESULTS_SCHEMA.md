# Experiment Results Schema

Documents the structure of JSON result files produced by the combinatorial experiment runner
(`scripts/experiments/run_combinatorial_experiments.py`).

Each experiment writes one file: `results/{dataset}/{holdout|cv}/results_{exp_id}.json`

---

## Top-Level Structure

```json
{
  "experiment_id": "a3f2b1...",
  "dataset": "cleveland",
  "configuration": { ... },
  "test_metrics": { ... },
  "fairness_metrics": { ... },
  "cv_results": { ... },
  "gate_recall_passed": true,
  "gate_fairness_passed": false,
  "fairness_gap": 0.12,
  "execution": { ... }
}
```

---

## `configuration`

```json
{
  "dataset": "cleveland",
  "binning_strategy": "fixed_10yr",
  "mitigation_technique": "reweighting",
  "training_method": "single_split",
  "model_type": "logistic_regression",
  "model_variant": "c_1_0",
  "sensitive_attributes": ["age_group", "sex", "ethnicity", "group_cluster"],
  "feature_selection_mode": "exclude_sensitive",
  "n_features": 13,
  "n_train": 212,
  "random_seed": 42,
  "cv_folds": 5,
  "hpo_params_used": true
}
```

---

## `test_metrics`

Standard classification metrics on the holdout test set.

```json
{
  "accuracy": 0.832,
  "precision": 0.801,
  "recall": 0.875,
  "f1_score": 0.836,
  "auc_roc": 0.901,
  "threshold": 0.5,
  "confusion_matrix": {
    "tn": 42, "fp": 9, "fn": 7, "tp": 31
  }
}
```

`threshold_analysis` (from baseline training results only): list of the same dict at each
threshold in `[0.3, 0.4, 0.5, 0.6, 0.7]`.

---

## `fairness_metrics`

Nested by fairness concept, then by sensitive attribute.

### Group fairness

```json
"group_fairness": {
  "age_group": {
    "demographic_parity": {
      "group_rates": {"<40": 0.31, "40-49": 0.44, "50-59": 0.55, "60-69": 0.62, "70+": 0.71},
      "overall_rate": 0.52,
      "max_difference": 0.40,
      "is_fair": false
    },
    "equalized_odds": {
      "group_metrics": {
        "<40":  {"tpr": 0.60, "fpr": 0.10},
        "40-49": {"tpr": 0.72, "fpr": 0.18}
      },
      "tpr_max_difference": 0.24,
      "fpr_max_difference": 0.15,
      "is_fair": false
    },
    "equal_opportunity": {
      "group_tpr": {"<40": 0.60, "40-49": 0.72},
      "max_difference": 0.24,
      "is_fair": false
    },
    "predictive_parity": {
      "group_precision": {"<40": 0.70, "40-49": 0.75},
      "max_difference": 0.12,
      "is_fair": false
    }
  },
  "sex": { ... },
  "ethnicity": { ... },
  "group_cluster": { ... }
}
```

`is_fair` uses a 10% max-difference threshold (configurable in `configs/recommendations/thresholds.yaml`).

### Calibration

```json
"calibration": {
  "age_group": {
    "calibration_by_group": {
      "<40": {
        "ece": 0.042,
        "bins": [
          {"mean_predicted": 0.12, "mean_true": 0.10, "n": 8},
          ...
        ]
      }
    },
    "max_ece_difference": 0.031,
    "is_fair": true
  }
}
```

### Individual fairness

```json
"individual_fairness": {
  "mean_consistency": 0.874,
  "std_consistency": 0.092,
  "min_consistency": 0.600,
  "median_consistency": 0.900,
  "k": 5
}
```

Consistency = fraction of k nearest neighbors that receive the same prediction.
Higher = more individually fair.

---

## `cv_results`

Present only when `training_method = "kfold_cv"`.

```json
"cv_results": {
  "fold_metrics": [
    {"fold": 0, "accuracy": 0.81, "f1_score": 0.83, "recall": 0.86, "auc_roc": 0.90},
    {"fold": 1, ...},
    ...
  ],
  "mean": {"accuracy": 0.824, "f1_score": 0.831, "recall": 0.861, "auc_roc": 0.897},
  "std":  {"accuracy": 0.018, "f1_score": 0.021, "recall": 0.025, "auc_roc": 0.014},
  "effective_folds": 5
}
```

`effective_folds` may be less than 5 on small datasets (e.g. cleveland) when per-group
stratification reduces available folds. This is logged as a warning during execution.

---

## `execution`

```json
"execution": {
  "status": "success",
  "timestamp": "2026-04-07T14:23:01Z",
  "duration_seconds": 12.4,
  "error": null
}
```

`status` is `"failed"` and `"error"` is non-null when an exception occurred.

---

## Gates

```json
"gate_recall_passed": true,
"gate_fairness_passed": false,
"fairness_gap": 0.12
```

- `gate_recall_passed`: `recall >= min_recall` (threshold from `thresholds.yaml`)
- `gate_fairness_passed`: `fairness_gap < max_fairness_violation`
- `fairness_gap`: composite gap score (max across all group fairness metric × attribute combos)

Only experiments where both gates pass are included in the Pareto frontier analysis.

---

## Loading Results into a DataFrame

```python
import json
from pathlib import Path
import pandas as pd

results_dir = Path("output/cardiac/runs/latest_run/experiments/results/cleveland/holdout")
rows = []
for f in results_dir.glob("results_*.json"):
    with open(f) as fh:
        r = json.load(fh)
    row = {
        "exp_id": r["experiment_id"],
        "dataset": r["dataset"],
        "binning": r["configuration"]["binning_strategy"],
        "mitigation": r["configuration"]["mitigation_technique"],
        "model": r["configuration"]["model_type"],
        "variant": r["configuration"]["model_variant"],
        "training_method": r["configuration"]["training_method"],
        "accuracy": r["test_metrics"]["accuracy"],
        "f1": r["test_metrics"]["f1_score"],
        "recall": r["test_metrics"]["recall"],
        "auc_roc": r["test_metrics"]["auc_roc"],
        "dem_parity_sex_gap": r["fairness_metrics"]["group_fairness"]["sex"]["demographic_parity"]["max_difference"],
        "dem_parity_age_gap": r["fairness_metrics"]["group_fairness"]["age_group"]["demographic_parity"]["max_difference"],
        "eq_odds_sex_tpr": r["fairness_metrics"]["group_fairness"]["sex"]["equalized_odds"]["tpr_max_difference"],
        "individual_consistency": r["fairness_metrics"]["individual_fairness"]["mean_consistency"],
        "gate_recall": r["gate_recall_passed"],
        "gate_fairness": r["gate_fairness_passed"],
        "fairness_gap": r["fairness_gap"],
        "duration_s": r["execution"]["duration_seconds"],
    }
    rows.append(row)

df = pd.DataFrame(rows)
```

`run_experiment_comparison.py` already does a version of this — extend it rather than writing
a new loader from scratch.

---

## Baseline Training Results

`output/cardiac/baseline/latest_run/results/training_results.json` has a different (flatter) schema:

```json
{
  "cleveland": {
    "logistic_regression": {
      "status": "success",
      "model_params": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
      "n_features": 13,
      "n_train": 212,
      "train_metrics": {"accuracy": 0.89, "f1_score": 0.88, ...},
      "test_metrics": {"accuracy": 0.83, "f1_score": 0.84, "threshold": 0.5, "confusion_matrix": {...}},
      "threshold_analysis": [
        {"threshold": 0.3, "accuracy": ..., "recall": ..., ...},
        {"threshold": 0.4, ...},
        ...
      ]
    },
    "random_forest": { ... },
    "svm": { ... },
    "xgboost": { ... }
  },
  "kaggle_heart": { ... }
}
```
