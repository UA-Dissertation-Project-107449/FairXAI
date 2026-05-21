# Explainability Module

Tabular SHAP and LIME wrappers used by baseline and combinatorial training
scripts. Counterfactual support remains an explicit placeholder.

## Files

| File | Purpose |
|------|---------|
| `tabular.py` | SHAP/LIME dataclasses and helper functions |
| `__init__.py` | Public exports |

## Public API

- `ShapExplanation`
- `LimeExplanation`
- `shap_explain_tabular`
- `lime_explain_instance`
- `counterfactual_stub`

## Config And Artifacts

XAI is enabled/configured by caller-level YAML:

- `configs/pipelines/cardiac.yaml`
- `configs/experiments/combinatorial.yaml`

Typical baseline output:

```text
output/cardiac/runs/<run_id>/baseline/xai/<dataset>/
├── holdout/
└── cv/
```

## Usage

```python
from fairxai.explainability import shap_explain_tabular

explanation = shap_explain_tabular(
    model=model,
    data=X_test,
    feature_names=list(X_test.columns),
)
```

## Current Limit

`counterfactual_stub` is intentionally present and unimplemented. The
counterfactual workstream is deferred rather than silently absent.

## Related

- Plots: [../../../docs/reference/plots.md](../../../docs/reference/plots.md)
- Roadmap: [../../../docs/planning/roadmap.md](../../../docs/planning/roadmap.md)
