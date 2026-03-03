# Data Module

Data loading, schema harmonization, preprocessing, and profiling support for
FairXAI workflows.

This module standardizes heterogeneous cardiac datasets into a unified format
used by modeling, fairness, and explainability pipelines.

## Files

| File | Purpose |
|------|---------|
| `loaders.py` | Dataset loading, schema-aware column harmonization, feature mapping |
| `preprocessors.py` | Missing-value handling, feature prep, scaling, stratified train/test split |
| `profilers.py` | Dataset profiling for fairness and complexity analysis |
| `schemas.py` | Canonical sensitive attributes and schema harmonization helpers |
| `__init__.py` | Package entrypoint |

## Key Components

- `CardiacDataLoader`
  - Loads datasets from configured paths
  - Harmonizes target/sensitive columns
  - Applies optional feature map aliases

- `CardiacPreprocessor`
  - Handles missing values
  - Builds model feature matrices and targets
  - Applies scaling and stratified splitting

- `DataProfiler`
  - Computes profile outputs for downstream fairness triage and reporting

## Schema Conventions

The module uses canonical columns when possible:

- `heart_disease` — binary target
- `age_raw` — numeric age source column
- `age_group` — binned age category
- `sex` — normalized sex value for model-safe processing
- `sex_extended` / `sex_bin` — additional sex representations for analysis

Preferred sensitive/group columns are defined in `schemas.py`:
- `age_group`
- `sex`
- `ethnicity`
- `group_cluster`

## Configuration Inputs

Common config sources consumed by scripts using this module:

- Schema metadata: `configs/schema/{pipeline}.json`
- Dataset-level loading settings: pipeline-specific loader configs
- Optional feature alias map: YAML passed to `CardiacDataLoader`

## Usage Example

```python
from fairxai.data.loaders import CardiacDataLoader
from fairxai.data.preprocessors import CardiacPreprocessor

loader = CardiacDataLoader("configs/schema/cardiac.json")
df = loader.load_dataset("cleveland", "data/external")

prep = CardiacPreprocessor(sensitive_attrs=["age_group", "sex"])
train_df, test_df = prep.stratified_split(df)
```

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `pyyaml`
