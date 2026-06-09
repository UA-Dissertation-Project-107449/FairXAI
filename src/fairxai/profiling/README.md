# Profiling Module

Complexity metrics, profiling configuration, and WebApp-compatible dataset
characterization.

## Files

| File | Purpose |
|------|---------|
| `complexity.py` | Complexity metric implementations and registry helpers |
| `config.py` | `ComplexityConfig` and YAML loader |
| `domain_characterization.py` | CSV-to-JSON characterization API and EBM difficulty |
| `METRIC_NOTES.md` | Metric behavior and reproducibility notes |
| `__init__.py` | Public exports |

## Public API

- `compute_complexity_metrics`
- `get_supported_complexity_metrics`
- `is_complexity_metric_key`
- `is_primary_complexity_metric`
- `ComplexityConfig`
- `load_complexity_config`
- `characterize_dataset`

## Metrics

Primary keys include:

- `F2`, `F3`, `F4`
- `N2`, `N3`, `N4`
- `Raug`
- `L1`, `L2`, `L3`
- `T1`
- `BayesImbalance`

Compatibility aliases such as `F2Imbalance` are also emitted for WebApp
contracts and legacy consumers.

## Config And Artifacts

- Config: `configs/profiling/complexity.yaml`
- Optional EBM model: `src/fairxai/profiling/models/ebm_model.joblib`
- Profiling outputs: `output/cardiac/runs/<run_id>/profiling/`
- Standalone fallback: `output/cardiac/profiling/`

## Usage

```python
from fairxai.profiling import compute_complexity_metrics

metrics = compute_complexity_metrics(df, target="heart_disease")
```

```python
from fairxai.profiling import characterize_dataset

payload = characterize_dataset(
    filename="cleveland_standardized.csv",
    datasets_dir="data/raw/cardiac",
    output_dir="/tmp/fairxai_characterize",
)
```

## Related

- Metric notes: [METRIC_NOTES.md](METRIC_NOTES.md)
- Notebook inputs: [../../../notebooks/README.md](../../../notebooks/README.md)
