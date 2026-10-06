# Fairness Module

Fairness evaluation and mitigation for model predictions in FairXAI.

This module provides both post-prediction fairness metrics and mitigation
techniques spanning pre-processing, in-processing, and post-processing stages.

## Files

| File | Purpose |
|------|---------|
| `metrics.py` | Group and individual fairness metrics computation (tabular) |
| `mitigation.py` | Mitigation techniques and orchestration engine |
| `uncertainty.py` | Bootstrap intervals for fairness and performance metrics, pairwise group tests, paired arm effects |
| `image_assessment.py` | Post-prediction fairness for dermatology image baselines + post-hoc group views |
| `image_mitigation.py` | Dermatology post-processing: group thresholds (`ThresholdOptimizer`) fit on saved train predictions |
| `image_feature_mitigation.py` | Dermatology pre/in-processing: rebuilds frozen-backbone features, runs the cardiac technique catalog on the head |
| `__init__.py` | Public re-exports for fairness APIs (tabular surface) |

## Public API

- `FairnessMetrics`
  - Computes demographic parity, equalized odds, and individual fairness metrics.
  - `individual_fairness_consistency(..., standardize=True)` - k-NN prediction
    consistency. Features are **z-scored before distance** (default) so
    high-magnitude columns (e.g. `chol`) don't dominate the neighbourhood vs
    `sex ∈ {0,1}`; pass `standardize=False` for raw distance.
  - `individual_fairness_by_group(df, feature_cols, group_col, ...)` - the same
    consistency aggregated **per sensitive-group** (`mean/std/min/max/n`); a low
    score for a group flags an individual-fairness gap for that subgroup.

- `PreProcessingMitigation`
  - Data-level techniques (e.g., reweighting, SMOTE, ROS/RUS/ADASYN).

- `InProcessingMitigation`
  - Fairness-aware training with fairlearn reductions.

- `PostProcessingMitigation`
  - Decision-threshold optimization for fairness constraints.

- `MitigationEngine`
  - Unified orchestration to apply techniques by stage.

- Bootstrap intervals (`uncertainty.py`)
  - `bootstrap_fairness_metrics`, `bootstrap_performance_metrics`,
    `paired_arm_differences`, `adaptive_bootstrap_replicates`,
    `BootstrapResult`, `PairedEffectResult`.

## Bootstrap Intervals

`uncertainty.py` resamples the prediction frame and re-runs the whole
`FairnessMetrics` computation per replicate, so every reported scalar gets a
percentile interval. Resampling is stratified by group × outcome and falls back
(group, then unstratified) when a stratum is too small; the scheme that ran is
recorded.

- **Max-gap intervals are descriptive only.** `max(rate) - min(rate)` is never
  below zero and biased upward, so its interval almost never contains zero, even
  on null data.
- **The pairwise table is the test.** Signed differences between two named
  groups, each with an interval, a two-sided bootstrap p-value and a
  Benjamini-Hochberg adjusted p-value across the run.
- `paired_arm_differences` compares two arms (e.g. baseline vs mitigated) on the
  same test rows.
- `adaptive_bootstrap_replicates(n)` picks the replicate count from cohort size.
- Individual fairness is excluded: O(n²) per replicate, and duplicate rows
  inflate k-NN consistency.

Stage 8 (`assess_predictions.py`) and stage 10 (`assess_mitigated_predictions.py`)
write the `*_ci.csv` / `*_pairwise_ci.csv` tables; see
[results-schema.md](../../../docs/reference/results-schema.md).

## Image Fairness Assessment

`image_assessment.py` is the dermatology counterpart to the tabular assessment —
it scores **post-prediction** fairness from a saved test-predictions CSV
(`y_true`, `y_pred`, sensitive columns), with **no retraining**. Functions are
script-facing (`scripts/dermatology/`), not re-exported from `__init__.py`;
import from `fairxai.fairness.image_assessment` directly.

- `assess_predictions_frame(df, sensitive_attrs, min_group_samples=50)` —
  per-attribute group performance + fairness deltas (dp/tpr/fpr/eo). Groups
  below `min_group_samples` are dropped from metrics but reported as skipped, so
  small subgroups never silently inflate a delta.
- **Post-hoc group views** (the "binning" without retraining): recompute the
  same fairness under alternate subgroup definitions from the *same* CSV.
  - `derive_group_view_columns(df, views)` — derives view columns
    (`age_coarse`, `sex`, `fitzpatrick_group`, intersectional `sex_x_fitzpatrick`)
    and metadata flagging exploratory/intersectional views.
  - `assess_group_views_frame(df, views, min_group_samples=50,
    intersection_min_group_samples=30)` — loops views; intersectional views use
    the stricter exploratory support gate (30) vs the main gate (50).
  - `render_group_view_markdown` / `_flatten_group_views_for_csv` — Markdown + CSV
    renderers.
- `assess_run(...)` — discovers per-model predictions under a run and emits the
  fairness JSON + Markdown.

Default support gates: `DEFAULT_MIN_GROUP_SAMPLES = 50`,
`DEFAULT_INTERSECTION_MIN_GROUP_SAMPLES = 30`.

## Image Mitigation

Dermatology stage 11 matches cardiac's mitigation matrix technique for technique.
Both modules are script-facing (`scripts/dermatology/mitigate.py`).

- `image_mitigation.py` never loads a model. Per sensitive attribute and
  constraint, it fits `ThresholdOptimizer` on train predictions, applies it to
  test predictions and recomputes the stage-8 report for before/after deltas.
  Entry point: `mitigate_run`.
- `image_feature_mitigation.py` runs the frozen backbone once to rebuild the
  feature matrix, standardises it, then runs `MitigationEngine` on the linear
  head. Entry point: `mitigate_run_features`. The intervention is on the head,
  not the representation; every report carries that caveat in its `scope` field.

## Fairness Concepts

- **Demographic parity**: predicted positive rate parity across groups
- **Equalized odds**: parity of TPR/FPR across groups
- **Individual fairness**: similar individuals receive similar predictions

## Mitigation Stages

| Stage | Typical techniques | Dependencies |
|-------|--------------------|--------------|
| Pre-processing | Reweighting, SMOTE, ROS, RUS, ADASYN | `imbalanced-learn`, `scikit-learn` |
| In-processing | Exponentiated Gradient, Grid Search | `fairlearn` |
| Post-processing | Threshold Optimizer | `fairlearn` |

## Usage Example

```python
from fairxai.fairness import FairnessMetrics, MitigationEngine

metrics = FairnessMetrics(sensitive_attributes=["age_group", "sex"])
results = metrics.calculate_all_metrics(predictions_df)

engine = MitigationEngine()
mitigated = engine.apply_technique(
    technique_name="exponentiated_gradient",
    stage="in-processing",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    sensitive_train=sensitive_train,
    sensitive_test=sensitive_test,
    sensitive_attr="sex",
)
```

## Dependencies

- `numpy`
- `pandas`
- `scikit-learn`
- `imbalanced-learn`
- `fairlearn`

## Related

- Results schema: [../../../docs/reference/results-schema.md](../../../docs/reference/results-schema.md)
- Dissertation evidence: [../../../docs/research/dissertation-evidence-check.md](../../../docs/research/dissertation-evidence-check.md)
