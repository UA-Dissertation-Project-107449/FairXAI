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
| `experiment_plots.py` | Experiment comparison outputs (heatmap/scatter/pareto/radar/intersectional/effectiveness) | ✅ Implemented |
| `fairness.py` | Fairness-specific plots (metric heatmap, group gaps, waterfall) | ✅ Implemented |
| `transformations.py` | Transformation impact plots (before/after, distributions, scaling) | ✅ Implemented |
| `style.py` | Shared palettes, labels, units, style conventions | ✅ Implemented |
| `constants.py` | Canonical value mappings and display order helpers | ✅ Implemented |
| `utils.py` | Shared plotting helper utilities | ✅ Implemented |
| `__init__.py` | Public API exports | ✅ Implemented |

## Public API Categories

- **Distributions**: group distributions, target-by-group, missingness, outliers
- **Comparisons**: two-dataset comparisons, drift, and statistical summaries
- **Experiment Plots**: comparison heatmaps, trade-off scatter, pareto frontier,
  cross-model radar, intersectional fairness, mitigation effectiveness matrix,
  all-model pareto
- **Fairness**: fairness metric heatmap, per-group performance gaps, bias amplification waterfall
- **Transformations**: transformation impact, before/after distributions, scaling effects
- **Style/Constants**: palettes (`PALETTE_DATASET`, `PALETTE_SEX`, `PALETTE_TARGET`, `PALETTE_MODEL`)
  and canonical category mappings

## Style Contract

All plotting functions should:
- Use palette and style constants from `style.py` / `PALETTE_MODEL` in `experiment_plots.py`
- Return the output path on success, `None` on empty/invalid input
- Avoid hard-coded colors in feature-level plotting code
- Keep labels/titles consistent with analysis terminology

## Usage Example

```python
from fairxai.viz import (
    plot_fairness_metric_heatmap,
    plot_group_performance_gaps,
    plot_bias_amplification_waterfall,
    plot_transformation_impact,
    plot_before_after_distributions,
    plot_scaling_effects,
    save_intersectional_heatmap,
    save_cross_model_radar,
    save_mitigation_effectiveness_matrix,
    save_pareto_all_models,
)

# Fairness metric heatmap (from full_comparison.csv)
plot_fairness_metric_heatmap(full_df, "age_group_cat", "fairness_heatmap_age.png")

# Cross-model radar chart (from cross_model_summary.csv)
save_cross_model_radar(summary_df, "radar.png")
```

## Generating All Dissertation Plots

Use the delivery script to generate all figures from a completed pipeline run:

```bash
python scripts/generate_dissertation_plots.py --run-id latest
# → output/cardiac/dissertation_figures/<run_id>/{fairness,transformations,cross_model}/
```
