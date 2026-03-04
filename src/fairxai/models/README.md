# Models Module

Model training abstractions used by FairXAI experiments and baseline pipelines.

This module provides a baseline sklearn wrapper and a cross-validation trainer
that supports fairness-aware stratification and optional per-fold XAI.

## Files

| File | Purpose |
|------|---------|
| `baseline.py` | Baseline logistic regression wrapper + prediction helpers |
| `cv_trainer.py` | Cross-validation training orchestration with fold-level metrics and XAI hooks |
| `__init__.py` | Public re-exports for model APIs |

## Key Classes and Functions

- `BaselineLogisticRegression`
  - Thin wrapper around `sklearn.linear_model.LogisticRegression`
  - Methods: `train`, `predict`, `predict_proba`, `evaluate`, `save_model`, `load_model`

- `generate_predictions_with_metadata(...)`
  - Builds enriched prediction tables used in downstream analysis.

- `CVTrainer`
  - Creates stratified folds with sensitive/group-aware keys
  - Runs full CV experiments and aggregates fold metrics
  - Supports per-fold XAI execution (`xai_enabled=True`) and SHAP/LIME aggregation

## XAI Integration

`CVTrainer` integrates with `fairxai.explainability`:

- Per-fold SHAP: global (train) and local (validation)
- Per-fold LIME: tracked validation instances
- Aggregation helpers:
  - `aggregate_cv_shap(..., scope='global'|'local')`
  - `aggregate_cv_lime(...)`

## Usage Example

```python
from fairxai.models import BaselineLogisticRegression, CVTrainer

model = BaselineLogisticRegression(max_iter=1000, class_weight='balanced')
trainer = CVTrainer(n_folds=5, random_state=42)

cv_results = trainer.run_cv_experiment(
    model_class=BaselineLogisticRegression,
    X=X_full,
    y=y_full,
    sensitive_attrs=sensitive_full,
    model_params={"max_iter": 1000, "class_weight": "balanced"},
    xai_enabled=True,
)
```

## Dependencies

- `numpy`
- `pandas`
- `scikit-learn`
- `joblib`
