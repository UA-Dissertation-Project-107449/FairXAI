# Data Module

Data loading, schema harmonization, preprocessing, profiling input generation,
and feature-selection helpers. Loaders cover both domains; the `synthetic/`
subpackage generates datasets for the profiling-sensitivity study.

## Files

| File | Purpose |
|------|---------|
| `loaders.py` | Cardiac dataset loaders, standardized raw loading, processed split loading |
| `preprocessors.py` | Cleaning, clinical constraints, splits, scaling, fairness profiles |
| `profilers.py` | Dataset-level profile dictionaries used by recommendations/notebooks |
| `schemas.py` | Cardiac schema, sex/age mappings, sensitive attribute helpers |
| `feature_selection.py` | Feature-selection mode helpers |
| `synthetic/generators.py` | `generate_abstract` (sklearn core + injected sensitive columns) and `generate_healthcare` (cardiac-shaped frame) |
| `synthetic/missingness.py` | `inject_missingness`: MCAR or MAR NaNs; target, sensitive and index columns are never nulled |
| `synthetic/duplication.py` | `inject_duplicates`: copies a fraction of rows verbatim, then shuffles |
| `synthetic/grid.py` | `build_grid` / `build_smoke_grid`: one-knob-at-a-time sweep with paired seeds |
| `synthetic/manifest.py` | `write_dataset` (CSV + `<id>.meta.json`) and `write_grid_manifest` |
| `synthetic/config.py` | `SyntheticConfig` and `GroundTruthColumn` (the designed type of each column) |
| `synthetic/__init__.py` | `generate` dispatcher + re-exports |
| `__init__.py` | Public submodule exports |

## Public API

`fairxai.data` exports submodules:

- `loaders`
- `preprocessors`
- `profilers`
- `schemas`
- `feature_selection`

Important concrete classes/functions include `CardiacDataLoader`,
`CardiacPreprocessor`, `DataProfiler`, `load_standardized_raw`,
`load_processed_splits`, `harmonize_cardiac_schema`,
`available_sensitive`, and `preferred_sensitive`.

`fairxai.data.synthetic` re-exports its whole surface; the study driver is
`scripts/studies/run_profiling_sensitivity_study.py`.

## Config And Artifacts

- Domain metadata: `configs/domain/cardiac.yaml`
- Dataset schema: `configs/schema/cardiac.json`
- Pipeline defaults: `configs/pipelines/cardiac.yaml`
- Standardized raw data: `data/raw/cardiac/*_standardized.csv`
- Processed splits: `data/processed/cardiac/<dataset>_<binning>/`

The default processed-data layout is per-dataset/per-binning subdirectories,
not flat files under `data/processed/cardiac/`.

## Usage

```python
from fairxai.data.loaders import load_processed_dataset

train_df, test_df = load_processed_dataset(
    dataset="cleveland_uci",
    root=".",
    area="cardiac",
    binning="fixed_10yr",
    scaled=True,
)
```

## Feature Selection Modes

Configured experiments may use feature modes such as excluding sensitive
attributes, clinical-only features, or ablation variants. See
`configs/experiments/feature_selection_study.yaml` for active study settings.

## Related

- Data directory: [../../../data/README.md](../../../data/README.md)
- Configs: [../../../configs/README.md](../../../configs/README.md)
- Decisions: [../../../docs/architecture/decisions.md](../../../docs/architecture/decisions.md)
