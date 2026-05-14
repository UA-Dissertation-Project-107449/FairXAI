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
| `fairness_comparison.py` | Metric-level fairness comparison plots (radar, deltas, subgroup bars, intersectional heatmap) | ✅ Implemented |
| `fairness.py` | Fairness-specific plots (metric heatmap, group gaps, waterfall) | ✅ Implemented |
| `transformations.py` | Transformation impact plots (before/after, distributions, scaling) | ✅ Implemented |
| `labels.py` | Shared display labels for mitigation and subgroup names | ✅ Implemented |
| `save_utils.py` | Shared save/layout helpers for publication figures | ✅ Implemented |
| `style.py` | Shared palettes, labels, units, style conventions | ✅ Implemented |
| `constants.py` | Canonical value mappings and display order helpers | ✅ Implemented |
| `utils.py` | Shared plotting helper utilities | ✅ Implemented |
| `__init__.py` | Public API exports | ✅ Implemented |

## Public API Categories

- **Distributions**: group distributions, target-by-group, missingness, outliers
- **Comparisons**: two-dataset comparisons, drift, and statistical summaries
- **Fairness Comparison**: metric-level before/after radar, mitigation delta
  matrix, subgroup bars, baseline cross-model radar, intersectional heatmap
- **Fairness**: fairness metric heatmap, per-group performance gaps, bias amplification waterfall
- **Transformations**: transformation impact, before/after distributions, scaling effects
- **Style/Constants**: palettes (`PALETTE_DATASET`, `PALETTE_SEX`, `PALETTE_TARGET`, `PALETTE_MODEL`)
  and canonical category mappings

## Style Contract

All plotting functions should:
- Use palette and style constants from `style.py` and shared labels from `labels.py`
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
    save_group_performance_gap_bars,
    save_intersectional_heatmap,
    save_cross_model_baseline_radar,
    save_mitigation_delta_matrix,
)

# Fairness metric heatmap (from full_comparison.csv)
plot_fairness_metric_heatmap(full_df, "age_group_cat", "fairness_heatmap_age.png")

# Baseline-only cross-model radar chart (from canonical comparison data)
save_cross_model_baseline_radar(full_df, "cleveland_baseline_cross_model_radar.png")

# Per-group before/after performance gaps (from canonical paired group data)
save_group_performance_gap_bars(
    group_metric_deltas,
    "cleveland_lr_primary_sex_performance_gaps.png",
    "sex",
    selected_row,
)
```

## Generating All Dissertation Plots

Use the delivery script to generate all figures from a completed pipeline run:

```bash
python scripts/studies/generate_dissertation_plots.py --run-id latest
# output/cardiac/studies/dissertation_figures/<run_id>/
```
