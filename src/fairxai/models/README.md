# Models Module

Model training abstractions used by FairXAI experiments and baseline pipelines.

This module provides sklearn-compatible wrappers for four model types and a
cross-validation trainer that supports fairness-aware stratification and
optional per-fold XAI.

## Model Registry

| Key | Class | Notes |
|-----|-------|-------|
| `logistic_regression` | `BaselineLogisticRegression` | Primary mitigation-supported model; interpretable via coefficients |
| `random_forest` | `RandomForestModel` | Supports cuML GPU path when `use_gpu=True` (CUDA required) |
| `svm` | `SVMModel` | Requires standard scaling; SHAP skipped by default (`skip_shap_model_types: [svm]` in config) |
| `xgboost` | `XGBoostModel` | Accepts `device='cuda'`/`'cpu'`; uses `tree_method='hist'` on CUDA |

Model hyperparameter defaults live in `configs/models/<type>.yaml`. Runtime overrides
(device, n_jobs) are injected by the combinatorial runner.

## Files

| File | Purpose |
|------|---------|
| `baseline.py` | `BaselineLogisticRegression` wrapper + prediction helpers |
| `random_forest.py` | `RandomForestModel` wrapper with optional cuML GPU path |
| `svm.py` | `SVMModel` wrapper (linear kernel default; RBF for small datasets) |
| `xgboost_model.py` | `XGBoostModel` wrapper with `device` parameter for CUDA training |
| `sklearn_wrapper.py` | Base class shared by all four wrappers; `get_feature_importance()` |
| `cv_trainer.py` | Cross-validation orchestration with fold-level metrics and XAI hooks |
| `__init__.py` | Public re-exports and `MODEL_REGISTRY` dict |

## Key Classes and Functions

- `BaselineLogisticRegression` / `RandomForestModel` / `SVMModel` / `XGBoostModel`
  - Common interface: `train`, `predict`, `predict_proba`, `evaluate`, `save_model`, `load_model`
  - `get_feature_importance()` — returns feature-weight dict (coef_ for LR/SVM, feature_importances_ for RF/XGB)

- `generate_predictions_with_metadata(...)`
  - Builds enriched prediction tables with `y_true`, `y_pred`, `y_proba`, `confidence`, `near_threshold`.

- `CVTrainer`
  - Creates stratified folds with sensitive/group-aware keys
  - Runs full CV experiments and aggregates fold metrics
  - Supports per-fold XAI execution (`xai_enabled=True`) and SHAP/LIME aggregation
  - Note: on small datasets (e.g. cleveland n≈300), effective folds may be reduced from 5 to 3
    when per-group sizes fall below the stratification threshold

## XAI Integration

`CVTrainer` integrates with `fairxai.explainability`:

- Per-fold SHAP: global (train) and local (validation)
- Per-fold LIME: tracked validation instances (near-threshold)
- Aggregation helpers:
  - `aggregate_cv_shap(..., scope='global'|'local')`
  - `aggregate_cv_lime(...)`
- SHAP is skipped for SVM by default; override via `allow_svm_shap=True` or remove `svm` from
  `skip_shap_model_types` in `configs/pipelines/cardiac.yaml`

## GPU / Accelerator Notes

- **XGBoost CUDA**: pass `device='cuda'` to `XGBoostModel` (injected automatically by the
  combinatorial runner when `detect_accelerator()` returns `'cuda'`).
- **cuML Random Forest**: pass `use_gpu=True` to `RandomForestModel`. Falls back to sklearn
  silently if `cuml` is not installed. cuml-cu12 is not in pyproject.toml — install manually
  after `module load cuda/12.4.0` on HPC: `pip install cuml-cu12==25.2.1`.

## Usage Example

```python
from fairxai.models import BaselineLogisticRegression, XGBoostModel, CVTrainer

model = XGBoostModel(n_estimators=200, device='cuda')
trainer = CVTrainer(n_folds=5, random_state=42)

cv_results = trainer.run_cv_experiment(
    model_class=XGBoostModel,
    X=X_full,
    y=y_full,
    sensitive_attrs=sensitive_full,
    model_params={"n_estimators": 200, "device": "cuda"},
    xai_enabled=True,
)
```

## Dependencies

- `numpy`, `pandas`, `scikit-learn`, `joblib`
- `xgboost` (for XGBoostModel)
- `cuml-cu12` (optional, CUDA only — for RandomForestModel GPU path)
