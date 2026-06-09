# Notebook Utilities Module

Notebook-facing helpers for resolving project context, loading data, applying
schema-aware labels, and reusing plotting palettes.

## Files

| File | Purpose |
|------|---------|
| `context.py` | Root resolution, domain config loading, figure path builders |
| `data.py` | External/raw/processed dataset loading and stage summaries |
| `profiling.py` | Notebook helpers for profiling artifacts |
| `__init__.py` | Re-exports plus label and annotation helpers |

## Public API Areas

- Context: `resolve_root_dir`, `load_domain_config`, `get_relevant_datasets`, `make_figure_path_builder`
- Data: `load_external_datasets`, `load_raw_datasets`, `load_processed_scaled_datasets`, `summarize_stage`, `canonical_features_for_columns`
- Labels: `dataset_age_unit`, `age_group_order`, `apply_age_group_order`, `age_to_years`, `resolve_sex_series`
- Plot annotation: `add_bar_labels`, `add_bar_labels_with_counts`, `add_grouped_bar_labels`, `add_point_labels`
- Palettes: `PALETTE_DATASET`, `PALETTE_SEX`, `PALETTE_TARGET`, `UNITS`

## Inputs And Outputs

- Reads profiling/run artifacts from `output/<pipeline>/runs/<run_id>/` and latest-run pointers.
- Reads raw/processed data from `data/raw/` and `data/processed/`.
- Writes notebook exports to `notebooks/tables/<pipeline>/` and `notebooks/figures/<pipeline>/`.

## Usage

```python
from fairxai.notebook_utils import load_raw_datasets, resolve_root_dir

root = resolve_root_dir()
datasets = load_raw_datasets(root / "data/raw/cardiac", ["cleveland"])
```

## Related

- Notebook folder: [../../../notebooks/README.md](../../../notebooks/README.md)
- Plot guide: [../../../docs/reference/plots.md](../../../docs/reference/plots.md)
