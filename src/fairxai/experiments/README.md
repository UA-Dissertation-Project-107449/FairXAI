# Experiments Module

Reusable experiment helpers for attribute binning, versioning, and processed
data resolution used by scripts under `scripts/experiments/` and `scripts/studies/`.

## Files

| File | Purpose |
|------|---------|
| `attribute_binning.py` | Binning strategies, repair, scoring, summary reports |
| `versioning.py` | Experiment output version/run directory helpers |
| `data_io.py` | Processed dataset directory and default-binning resolution |
| `__init__.py` | Public exports |

## Public API

- `create_binning_strategy`
- `apply_binning`
- `sensitive_attribute_distribution`
- `compute_fairness_metrics`
- `analyze_strategy_comprehensive`
- `compare_strategies`
- `compute_strategy_score`
- `generate_summary_report`
- `validate_and_repair`
- `ExperimentVersioning`

## Config And Artifacts

- Attribute binning config: `configs/experiments/age_binning.yaml`
- Combinatorial config: `configs/experiments/combinatorial.yaml`
- Processed data: `data/processed/cardiac/<dataset>_<binning>/`
- Run-scoped experiments: `output/cardiac/runs/<run_id>/experiments/`
- Standalone studies: `output/cardiac/studies/<study_type>/`

## Usage

```python
from fairxai.experiments import apply_binning, create_binning_strategy

bins, labels = create_binning_strategy(df, "fixed_10yr", col="age")
df_binned = apply_binning(df, bins, labels, col="age", output_col="age_group_exp")
```

## Related

- Attribute binning reference: [../../../docs/reference/attribute-binning.md](../../../docs/reference/attribute-binning.md)
- Results schema: [../../../docs/reference/results-schema.md](../../../docs/reference/results-schema.md)
