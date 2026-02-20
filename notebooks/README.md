# Notebook Inputs And Expected Formats

This folder contains exploratory and profiling notebooks for the FairXAI cardiac pipeline. The notebooks expect profiling artifacts produced by `scripts/common/profile_data.py` (via `scripts/cardiac/profile_data.py`).

## Expected Profiling Outputs

Primary artifacts are written under:

- `results/<pipeline>/profiling/` (standalone, no `--run-id`)
- `results/<pipeline>/runs/<run_id>/profiling/` (when `--run-id` is passed or `RUN_ID` env is set)

Files:

- `<dataset>_data_profile.json`
- `<dataset>_complexity.json` (extra artifact for complexity metrics)
- `dataset_comparison.json`

## Profile JSON Schema (High-Level)

Each `<dataset>_data_profile.json` contains:

- `dataset_name`: string
- `basic_stats`:
  - `n_samples`: int
  - `n_features`: int
  - `target_name`: string
  - `target_prevalence`: float
- `sensitive_attr_distribution` (per attribute):
  - `counts`: {group: count}
  - `proportions`: {group: proportion}
- `target_distribution`:
  - `counts`: {label: count}
  - `proportions`: {label: proportion}
  - `imbalance_ratio`: float
- `group_statistics` (per attribute, per group):
  - `n_samples`: int
  - `proportion_of_total`: float
  - `target_prevalence`: float
  - `target_counts`: {label: count}
- `representation_balance` (per attribute):
  - `coefficient_of_variation`: float
  - `min_group_size`: int
  - `max_group_size`: int
  - `size_ratio`: float
  - `counts`: {group: count}
- `label_imbalance_by_group` (per attribute):
  - `positive_rates`: {group: rate}
  - `statistical_parity_difference`:
    - `max_difference`: float
    - `max_ratio`: float
- `missing_value_analysis`:
  - `total_missing`: int
  - `columns_with_missing`: {column: count}
  - `missing_by_group`: {attribute: {column: {group: count}}}
- `complexity_metrics`:
  - Primary metrics: `F2`, `F3`, `F4`, `N2`, `N3`, `N4`, `Raug`, `L1`, `L2`, `L3`, `T1`, `BayesImbalance`: float | null
  - Alias metrics: `F2Imbalance`, `F3Imbalance`, `F4Imbalance`, `N2Imbalance`, `N3Imbalance`, `N4Imbalance`, `RaugImbalance`, `L1Imbalance`, `L2Imbalance`, `L3Imbalance`, `T1Imbalance`: float | null
  - `max_samples`: int
- `group_complexity_metrics`:
  - Per sensitive attribute, per group: `n_samples`, `status`, `complexity_metrics`
- `intersection_complexity_metrics`:
  - Per attribute pair (e.g., `age_group__sex`), per subgroup combination: `n_samples`, `status`, `complexity_metrics`

## Complexity Metrics Artifact

Each `<dataset>_complexity.json` contains the `complexity_metrics` block only, to support lightweight consumption in notebooks or downstream scripts.

## Notes

- Sensitive attributes are configured via `configs/pipelines/<pipeline>.yaml` under `fairness.sensitive_attributes`.
- Profiling relies on standardized raw datasets (`*_standardized.csv`).
- If profile files are missing, rerun the profiling step or the full pipeline.
- Profiling notebook loaders prefer `results/<pipeline>/latest_run/profiling/` when available, and fall back to `results/<pipeline>/profiling/`.
- Notebook helpers prefer `<dataset>_data_profile.json` as source of truth and fall back to `<dataset>_complexity.json` only when primary profile complexity fields are missing.
- Notebook-exported tables are saved to `notebooks/tables/<pipeline>/` with a `profiling_` prefix (e.g., `profiling_dataset_overview.csv`).
