# Fairness Module

Fairness evaluation and mitigation for model predictions in FairXAI.

This module provides both post-prediction fairness metrics and mitigation
techniques spanning pre-processing, in-processing, and post-processing stages.

## Files

| File | Purpose |
|------|---------|
| `metrics.py` | Group and individual fairness metrics computation |
| `mitigation.py` | Mitigation techniques and orchestration engine |
| `__init__.py` | Public re-exports for fairness APIs |

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
