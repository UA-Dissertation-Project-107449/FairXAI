# Integration Module

WebApp-oriented adapters that wrap FairXAI internals and return
JSON-serializable dictionaries.

## Files

| File | Purpose |
|------|---------|
| `characterize.py` | Dataset characterization adapter |
| `profile.py` | Upload-time lightweight profiling (re-exports `profiling.domain_characterization.profile_dataset`) |
| `triage.py` | Standalone fairness triage report; writes nothing to disk |
| `binning.py` | Attribute binning adapter and bin-stat summaries |
| `clustering.py` | Clustering adapter, PCA points, cluster summaries |
| `__init__.py` | Public exports |

## Public API

- `characterize_dataset`
- `profile_dataset`
- `run_binning`
- `run_clustering`
- `triage_dataset`

Characterization no longer runs triage. Callers that need both run
`fairxai characterize` then `fairxai triage`.

## Contracts

- Inputs are paths or tabular payloads prepared by callers.
- Outputs are JSON-serializable dicts intended for WebApp consumption.
- Heavy pipeline orchestration is not duplicated here; adapters call package internals.

## Usage

```python
from fairxai.integration import (
    characterize_dataset,
    run_binning,
    run_clustering,
    triage_dataset,
)

profile = characterize_dataset(filename="data.csv", output_dir="/tmp/profile")
binning = run_binning("data.csv", target_column="heart_disease", attribute="age", strategy="quantile_5")
clusters = run_clustering("data.csv", target_column="heart_disease")
report = triage_dataset("data.csv", target_column="heart_disease", sensitive_columns=["sex"])
```

## Related

- Profiling module: [../profiling/README.md](../profiling/README.md)
- Clustering module: [../clustering/README.md](../clustering/README.md)
- Root WebApp notes: [../../../README.md](../../../README.md)
