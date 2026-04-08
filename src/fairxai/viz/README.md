# Visualization Module (`viz`)

Visualization toolkit for EDA, drift analysis, and experiment comparison in
FairXAI.

The module centralizes plotting APIs and style constants so notebook and script
outputs remain consistent.

## Files

| File | Purpose | Status |
|------|---------|--------|
| `distributions.py` | Distribution-focused plots (categorical/numeric/target/missing/outliers) | ✅ Implemented |
| `comparisons.py` | Dataset comparison plots (correlation/PCA/KS/drift) | ✅ Implemented |
| `experiment_plots.py` | Experiment comparison outputs (heatmap/scatter/pareto) | ✅ Implemented |
| `style.py` | Shared palettes, labels, units, style conventions | ✅ Implemented |
| `constants.py` | Canonical value mappings and display order helpers | ✅ Implemented |
| `utils.py` | Shared plotting helper utilities | ✅ Implemented |
| `fairness.py` | Fairness-specific plotting functions | ⚠️ Scaffolded (planned) |
| `transformations.py` | Transformation impact plotting functions | ⚠️ Scaffolded (planned) |
| `__init__.py` | Public API exports | ✅ Implemented |

## Public API Categories

- **Distributions**: group distributions, target-by-group, missingness, outliers
- **Comparisons**: two-dataset comparisons, drift, and statistical summaries
- **Experiment Plots**: comparison heatmaps, trade-off scatter, pareto frontier
- **Style/Constants**: palettes and canonical category mappings

## Style Contract

All plotting functions should:
- Use palette and style constants from `style.py`
- Return matplotlib figure/axes objects where practical
- Avoid hard-coded colors in feature-level plotting code
- Keep labels/titles consistent with analysis terminology

## Stubbed APIs and Roadmap

The following functions are intentionally scaffolded and currently raise
`NotImplementedError`:

- `plot_fairness_metric_heatmap`
- `plot_group_performance_gaps`
- `plot_bias_amplification_waterfall`
- `plot_transformation_impact`
- `plot_before_after_distributions`
- `plot_scaling_effects`

These functions will raise `NotImplementedError` at runtime. Implementation is deferred
(see `docs/ROADMAP.md` for current project status).

## Usage Example

```python
from fairxai.viz import (
    plot_numeric_distribution_comparison,
    save_tradeoff_scatter,
)

fig, ax = plot_numeric_distribution_comparison(df_a, df_b, feature="cholesterol")
save_tradeoff_scatter(tradeoff_df, "Fairness vs Performance", output_file)
```
