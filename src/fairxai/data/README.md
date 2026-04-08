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
| `feature_selection.py` | Feature set construction with sensitive-attribute mode control |
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

## Feature Selection Modes

`feature_selection.py` exposes `build_feature_set()` — called by `train_baseline.py` before
model training to control which columns are visible to the model:

```python
from fairxai.data.feature_selection import build_feature_set

X_train, feature_cols = build_feature_set(
    X_train_full,
    sensitive_attrs=["age_group", "sex", "ethnicity"],
    mode="exclude_sensitive",   # see modes below
    top_k=10,                   # only used by rfe_top_k mode
)
X_test = X_test_full[feature_cols]
```

| Mode | Description |
|------|-------------|
| `exclude_sensitive` | **Default (model-blind)**. Drops all sensitive attribute columns. Privacy-preserving baseline. |
| `include_all_sensitive` | All sensitive attributes visible to model. Research target for fairness argument. |
| `include_sex_only` | Only `sex` included; other sensitive attrs excluded. Ablation study. |
| `include_age_only` | Only `age_group` included. Ablation study. |
| `include_ethnicity_only` | Only `ethnicity` included (where present). Ablation study. |
| `rfe_top_k` | Keeps top-k features by importance score. Requires a fitted `trained_model`. Falls back to permutation importance for non-linear SVM; skips permutation on large datasets (>5000 rows or >50 features). |

The feature selection study config is at `configs/experiments/feature_selection_study.yaml`.
Run via: `python scripts/common/train_baseline.py --pipeline cardiac --feature-selection-mode <mode>`

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `pyyaml`
