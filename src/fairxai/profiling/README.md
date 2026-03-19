# Profiling Module

Complexity and overlap profiling utilities used to characterize datasets before
or alongside model training.

This module computes class-overlap, neighborhood, linearity, and imbalance
signals that feed downstream fairness analysis and recommendation workflows.

## Files

| File | Purpose |
|------|---------|
| `complexity.py` | Complexity metric implementations and metric registry helpers |
| `config.py` | `ComplexityConfig` dataclass and `load_complexity_config()` loader (reads `configs/profiling/complexity.yaml`) |
| `domain_characterization.py` | WebApp-compatible characterization API and EBM difficulty scoring |
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
- `ComplexityConfig`
  - Typed configuration for tunables (max samples, random seed, solver, etc.).
- `load_complexity_config(path=None)`
  - Loads `ComplexityConfig` from YAML (defaults to `configs/profiling/complexity.yaml`).
- `characterize_dataset(filename, output_dir, ...)`
  - Computes compatibility metrics and writes `<jobId>.json` for WebApp consumption.

## Domain Characterization Output

`characterize_dataset` writes JSON in the shape:

```json
{
  "jobId": "<file_stem>",
  "metrics": {
    "nSamples": 0,
    "nFeatures": 0,
    "nClasses": 0,
    "F2Imbalance": 0.0,
    "F3Imbalance": 0.0,
    "F4Imbalance": 0.0,
    "L1Imbalance": 0.0,
    "L2Imbalance": 0.0,
    "L3Imbalance": 0.0,
    "N2Imbalance": 0.0,
    "N3Imbalance": 0.0,
    "N4Imbalance": 0.0,
    "T1Imbalance": 0.0,
    "RaugImbalance": 0.0,
    "BayesImbalance": 0.0,
    "ebmDifficulty": 0.0
  }
}
```

## EBM Runtime Requirement

EBM inference is loaded from `src/fairxai/profiling/models/ebm_model.joblib`.

Runtime requirement:
- `interpret` must be available in the active Python environment (included in `.[experiment]` / `.[hpc]`).

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

## Dependencies

- `numpy`
- `pandas`
- `scikit-learn` (optional for specific metrics)
