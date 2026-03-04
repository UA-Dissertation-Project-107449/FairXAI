# Explainability Module

Model-agnostic explainability helpers for tabular models in FairXAI.
The module wraps SHAP and LIME with a stable API and standardized return
objects so scripts can consume explanations consistently.

## Files

| File | Purpose |
|------|---------|
| `tabular.py` | SHAP and LIME explainers for tabular models, plus counterfactual placeholder |
| `__init__.py` | Public re-exports for explainability APIs |

## Public API

- `shap_explain_tabular(model, data, feature_names=None, max_samples=1000)`
  - Computes SHAP values on tabular data.
- `lime_explain_instance(model, data_row, training_data, feature_names=None, class_names=None, num_features=10)`
  - Computes a LIME explanation for one instance.
- `counterfactual_stub(*args, **kwargs)`
  - Placeholder for future counterfactual support.

## Data Classes

- `ShapExplanation`
  - `shap_values`, `base_values`, `expected_value`, `feature_names`, `data`
- `LimeExplanation`
  - `weights`, `intercept`, `score`, `local_pred`

## Configuration

Explainability behavior is configured from YAML in caller scripts, not from
module-level environment variables.

Typical pipeline config:

```yaml
xai:
  enabled: true
  cv_enabled: true
  lime_instances: 3
  cv_lime_instances: 3
  global_max_samples: 1000
```

Typical combinatorial config:

```yaml
xai:
  enabled: true
  max_samples: 200
  lime_instances: 2
```

## Usage Example

```python
from fairxai.explainability import shap_explain_tabular, lime_explain_instance

shap_exp = shap_explain_tabular(model, X_train, max_samples=500)
lime_exp = lime_explain_instance(model, X_test.iloc[0], X_train)
```

## Dependencies

- `shap`
- `lime`
- `numpy`
- `pandas`

## Roadmap

- Counterfactual explanations are intentionally not implemented yet.
- Planned implementation target: Q2 2026 (`counterfactual_stub`).
