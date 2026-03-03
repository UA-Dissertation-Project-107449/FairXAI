# Profiling Module

Complexity and overlap profiling utilities used to characterize datasets before
or alongside model training.

This module computes class-overlap, neighborhood, linearity, and imbalance
signals that feed downstream fairness analysis and recommendation workflows.

## Files

| File | Purpose |
|------|---------|
| `complexity.py` | Complexity metric implementations and metric registry helpers |
| `__init__.py` | Public re-exports for profiling APIs |

## Public API

- `compute_complexity_metrics(df, target='heart_disease', ...)`
  - Computes supported metrics from numeric features.
- `get_supported_complexity_metrics(include_aliases=False)`
  - Returns canonical metric names (and optional imbalance aliases).
- `is_complexity_metric_key(metric_name)`
  - Validates whether a key is a primary metric or alias.
- `is_primary_complexity_metric(metric_name)`
  - Checks canonical metric membership.

## Supported Metrics

Primary metric keys exposed today:

- `F2`, `F3`, `F4`
- `N2`, `N3`, `N4`
- `Raug`
- `L1`, `L2`, `L3`
- `T1`
- `BayesImbalance`

Imbalance aliases are also exposed (e.g., `F2Imbalance`, `N3Imbalance`).

## Design Notes

- Numeric-only feature selection is handled internally.
- Binary-target requirements are validated before metric computation.
- Optional sklearn dependency (`LogisticRegression`) is guarded for robustness.

## Usage Example

```python
from fairxai.profiling import compute_complexity_metrics

profile = compute_complexity_metrics(df, target="heart_disease")
print(profile.get("F2"), profile.get("N3"))
```

## Roadmap

- Move tunable thresholds/constants to YAML configuration once profiling
  parameterization is formalized (planned module-hardening follow-up).
- Keep metric keys backward-compatible for existing result consumers.

## Dependencies

- `numpy`
- `pandas`
- `scikit-learn` (optional for specific metrics)
