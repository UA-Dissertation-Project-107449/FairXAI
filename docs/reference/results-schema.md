# Experiment Results Schema

Schema reference for result artifacts produced by baseline, mitigation,
combinatorial, and comparison stages, plus the evidence layer and thesis outputs.

## Run Roots

Run-scoped artifacts live under:

```text
output/cardiac/runs/<run_id>/
```

Pointers to the latest run:

```text
output/cardiac/latest_run
output/cardiac/latest_run.txt
```

Do not place a `latest_run` directory under `runs/`; `latest_run` is a pointer
at `output/cardiac/latest_run`.

Dermatology runs use the same layout under `output/dermatology/runs/<run_id>/`,
with `output/dermatology/latest_run` as the pointer.

## Combinatorial Result JSON

Each experiment writes:

```text
output/cardiac/runs/<run_id>/experiments/results/<dataset>/<holdout|cv>/results_<exp_id>.json
```

Top-level shape:

```json
{
  "experiment_id": "a3f2b1...",
  "dataset": "cleveland_uci",
  "configuration": {},
  "test_metrics": {},
  "fairness_metrics": {},
  "cv_results": {},
  "gate_recall_passed": true,
  "gate_fairness_passed": false,
  "fairness_gap": 0.12,
  "execution": {}
}
```

## `configuration`

```json
{
  "dataset": "cleveland_uci",
  "binning_strategy": "fixed_10yr",
  "mitigation_technique": "reweighting",
  "training_method": "single_split",
  "model_type": "logistic_regression",
  "model_variant": "c_1_0",
  "sensitive_attributes": ["age_group", "sex", "group_cluster"],
  "feature_selection_mode": "exclude_sensitive",
  "n_features": 13,
  "n_train": 212,
  "random_seed": 42,
  "cv_folds": 5,
  "hpo_params_used": true
}
```

## `test_metrics`

```json
{
  "accuracy": 0.832,
  "precision": 0.801,
  "recall": 0.875,
  "f1_score": 0.836,
  "auc_roc": 0.901,
  "threshold": 0.5,
  "confusion_matrix": {"tn": 42, "fp": 9, "fn": 7, "tp": 31}
}
```

Baseline training results may also include `threshold_analysis` for thresholds
`0.3` through `0.7`.

## `fairness_metrics`

Group fairness is nested by sensitive attribute:

```json
{
  "group_fairness": {
    "sex": {
      "demographic_parity": {
        "group_rates": {"Female": 0.31, "Male": 0.44},
        "overall_rate": 0.38,
        "max_difference": 0.13,
        "is_fair": false
      },
      "equalized_odds": {
        "group_metrics": {
          "Female": {"tpr": 0.60, "fpr": 0.10},
          "Male": {"tpr": 0.72, "fpr": 0.18}
        },
        "tpr_max_difference": 0.12,
        "fpr_max_difference": 0.08,
        "is_fair": false
      }
    }
  },
  "calibration": {},
  "individual_fairness": {
    "mean_consistency": 0.874,
    "std_consistency": 0.092,
    "min_consistency": 0.600,
    "median_consistency": 0.900,
    "k": 5
  }
}
```

`is_fair` uses the configured max-difference threshold from
`configs/recommendations/thresholds.yaml`.

## `cv_results`

Present for `training_method = "kfold_cv"`:

```json
{
  "fold_metrics": [
    {"fold": 0, "accuracy": 0.81, "f1_score": 0.83, "recall": 0.86, "auc_roc": 0.90}
  ],
  "mean": {"accuracy": 0.824, "f1_score": 0.831, "recall": 0.861, "auc_roc": 0.897},
  "std": {"accuracy": 0.018, "f1_score": 0.021, "recall": 0.025, "auc_roc": 0.014},
  "effective_folds": 5
}
```

## Gates

```json
{
  "gate_recall_passed": true,
  "gate_fairness_passed": false,
  "fairness_gap": 0.12
}
```

- `gate_recall_passed`: `recall >= min_recall`.
- `gate_fairness_passed`: `fairness_gap < max_fairness_violation`.
- `fairness_gap`: max/composite gap used by the comparison stage.

## Canonical Comparison Tables

`scripts/cardiac/compare.py` and `fairxai.comparison.write_canonical_comparison_outputs`
produce tables under:

```text
output/cardiac/runs/<run_id>/experiments/comparisons/data/
```

Key tables:

| File | Purpose |
|------|---------|
| `full_comparison.csv` | Flattened experiment-level performance and fairness metrics |
| `per_group_comparison.csv` | Group-level before/after metrics and deltas |
| `cross_model_summary.csv` | Best/config summary by model type |
| `metric_values.csv` | Long-form metric values |
| `metric_deltas.csv` | Long-form experiment minus baseline deltas |
| `group_metric_values.csv` | Long-form group metric values |
| `group_metric_deltas.csv` | Long-form group deltas |
| `fairness_evidence_summary.csv` | Ranked fairness evidence rows |

## Baseline Training Results

Baseline results are run-scoped:

```text
output/cardiac/runs/<run_id>/baseline/results/training_results.json
```

The shape is flatter than combinatorial results:

```json
{
  "cleveland_uci": {
    "logistic_regression": {
      "status": "success",
      "model_params": {},
      "n_features": 13,
      "n_train": 212,
      "train_metrics": {},
      "test_metrics": {},
      "threshold_analysis": []
    }
  }
}
```

## Evidence Layer

Bootstrap intervals and subgroup attributions sit beside the point estimates.
A max-gap interval is descriptive only. The test is the pairwise group
difference, Benjamini-Hochberg adjusted. See
[decisions.md](../architecture/decisions.md#fairness-intervals-max-gap-cis-are-descriptive-pairwise-differences-are-the-test).

`<stem>` is the prediction file stem, for example `<dataset>_<model>`. CV
predictions use `<stem>_cv`.

| Stage | File | Location under `runs/<run_id>/` |
|-------|------|---------------------------------|
| 7 train | `subgroup_summary.csv`, `subgroup_disparity.csv`, `subgroup_agreement.csv` | `baseline/xai/<dataset>/holdout/shap/` |
| 8 assess | `<stem>_ci.csv` (per-group and max-gap intervals) | `baseline/prediction_fairness/` |
| 8 assess | `<stem>_pairwise_ci.csv` (pairwise differences, adjusted p-values) | `baseline/prediction_fairness/` |
| 8 assess | `<stem>_performance_ci.csv` (F1, accuracy, precision, recall, AUC) | `baseline/prediction_fairness/` |
| 10 mitigate | `paired_effects.csv`, `paired_effects_cv.csv` (each arm against its own baseline, paired on shared rows) | `experiments/mitigation/` |
| 10 mitigate | the three subgroup SHAP tables, one folder per arm | `experiments/mitigation/subgroup_shap/<dataset>_<model>_<technique>_<attribute>/` |
| after 10 | `<arm>_ci.csv`, `<arm>_pairwise_ci.csv` per arm; `arms_ci.csv`, `arms_pairwise_ci.csv`, `arms_summary.csv` combined | `experiments/mitigation/prediction_fairness/` |

The post-stage-10 tables come from `scripts/common/assess_mitigated_predictions.py`
and follow `fairness.uncertainty.enabled` (override with `RUN_MITIGATION_INTERVALS`).

## Thesis Outputs

`scripts/thesis/` reads finished runs named in `scripts/thesis/runs.yaml` and
writes under `output/thesis/`:

| File | Writer |
|------|--------|
| `baseline_performance.csv`, `ablation_paired.csv` | `performance_intervals.py` |
| `chapter6_ledger.md` | `build_ledger.py` |
| `figures/` | `make_figures.py` |
| `similarity_heldout.json` | `similarity_heldout.py` |
