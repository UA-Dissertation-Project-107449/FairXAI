# Models Module

Model registry, sklearn-compatible wrappers, baseline model training support,
and cross-validation trainer utilities.

## Files

| File | Purpose |
|------|---------|
| `baseline.py` | Logistic regression baseline and prediction metadata export |
| `random_forest.py` | Random forest wrapper |
| `svm.py` | SVM wrapper |
| `xgboost_model.py` | XGBoost wrapper with accelerator-aware settings |
| `sklearn_wrapper.py` | Shared classifier wrapper behavior |
| `cv_trainer.py` | Cross-validation orchestration |
| `__init__.py` | Model registry and public exports |

## Public API

- `BaselineLogisticRegression`
- `RandomForestModel`
- `SVMModel`
- `XGBoostModel`
- `CVTrainer`
- `MODEL_REGISTRY`
- `get_model_class`
- `generate_predictions_with_metadata`

## Registry Keys

| Key | Class |
|-----|-------|
| `logistic_regression` | `BaselineLogisticRegression` |
| `random_forest` | `RandomForestModel` |
| `svm` | `SVMModel` |
| `xgboost` | `XGBoostModel` |

## Config And Artifacts

- Model defaults: `configs/models/*.yaml`
- Baseline experiment defaults: `configs/experiments/baseline.yaml`
- HPO outputs: `output/cardiac/studies/hpo/best_params_<dataset>_<model>.json`
- Baseline run artifacts: `output/cardiac/runs/<run_id>/baseline/`

## Usage

```python
from fairxai.models import get_model_class

ModelClass = get_model_class("xgboost")
model = ModelClass()
```

## Related

- Training helpers: [../training/README.md](../training/README.md)
- Results schema: [../../../docs/reference/results-schema.md](../../../docs/reference/results-schema.md)
