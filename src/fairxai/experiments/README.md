# Experiments Module

Experiment utilities for FairXAI: artifact versioning, binning analysis, and
shared data I/O helpers used across experiment scripts.

## Files

| File | Purpose |
|------|---------|
| `versioning.py` | Experiment artifact management (manifests, results, predictions, models, XAI) |
| `age_binning.py` | Config-driven binning strategy analysis and fairness impact scoring |
| `data_io.py` | Shared schema loading and exclude-column helpers for experiment scripts |
| `__init__.py` | Public re-exports for experiment APIs |

## Key Components

- `ExperimentVersioning`
  - Generates experiment IDs
  - Saves and loads manifests/results/predictions/models
  - Supports split-aware organization (`holdout` / `cv`) under dataset folders
  - Creates summaries and supports run archiving patterns used by scripts

- `age_binning` API
  - `create_binning_strategy`
  - `apply_binning`
  - `analyze_strategy_comprehensive`
  - `compare_strategies`
  - `generate_summary_report`

- `data_io` helpers
  - `load_schema_config`
  - `build_schema_excludes`
  - `default_exclude_columns`

## Output Organization

Experiment artifacts are expected under run directories such as:

```text
output/{pipeline}/{latest_run|runs/{run_id}}/experiments/full/
├── manifests/{dataset}/{holdout|cv}/
├── results/{dataset}/{holdout|cv}/
├── predictions/{dataset}/{holdout|cv}/
├── models/{dataset}/
└── xai/{dataset}/{holdout|cv}/{shap|lime}/
```

## Configuration Inputs

Typical config sources consumed by scripts using this module:

- Experiment configs: `configs/experiments/*.yaml`
- Pipeline configs: `configs/pipelines/*.yaml`
- Schema mapping: `configs/schema/{pipeline}.json`

## Usage Example

```python
from pathlib import Path
from fairxai.experiments import ExperimentVersioning, create_binning_strategy

versioning = ExperimentVersioning(Path("output/cardiac/latest_run/experiments/full"))
exp_id = versioning.generate_experiment_id()

bins, labels = create_binning_strategy(df, "quantile_5", age_col="age_raw")
```

## Notes

- `age_binning.py` is already attribute-agnostic internally; the filename is
  historical and may be renamed in a future sprint.
- Versioning methods are designed to be backward-compatible with recursive
  loaders for nested output layouts.
