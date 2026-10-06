# Visualization Module

Plotting APIs for EDA, dataset comparison, fairness analysis, clustering, and
dissertation figures.

## Files

| File | Purpose |
|------|---------|
| `distributions.py` | Feature distributions, missingness, outliers, target-by-group |
| `comparisons.py` | Correlations, PCA/KMeans, dataset drift |
| `fairness.py` | Fairness heatmaps, group gaps, bias waterfall |
| `transformations.py` | Pre/post transformation and scaling plots |
| `fairness_comparison.py` | Dissertation comparison figures from canonical tables |
| `clustering.py` | Cluster profile and fairness figures |
| `dermatology_readiness.py` | Stage 4: pre-model split fairness profiles (`render_readiness_figures`) |
| `dermatology_fairness.py` | Stage 8: subgroup heatmaps and group-view figures (`render_subgroup_heatmaps`, `render_group_view_figures`) |
| `dermatology_comparison.py` | Stage 9: learning curves and model comparison (`render_learning_curves`, `render_comparison_figures`) |
| `dermatology_mitigation.py` | Stage 11: baseline vs mitigated gaps per attribute × constraint (`render_mitigation_figures`) |
| `constants.py` | Cardiac category normalization/display order |
| `labels.py` | Display label helpers |
| `style.py` | Shared palettes and units |
| `save_utils.py` | Figure saving and sizing helpers |
| `utils.py` | Misc visualization helpers |
| `__init__.py` | Public exports |

## Public API Categories

- EDA/distributions: `plot_categorical_distribution_grid`, `plot_numeric_distribution_comparison`, `plot_target_distribution_by_group`, `plot_missing_data_patterns`
- Dataset comparison: `plot_correlation_heatmap_grid`, `plot_pca_kmeans_scatter_grid`, `plot_drift_heatmap`
- Fairness: `plot_fairness_metric_heatmap`, `plot_group_performance_gaps`, `plot_bias_amplification_waterfall`
- Dissertation comparisons: `save_mitigation_delta_matrix`, `save_before_after_metric_radar`, `save_cross_model_best_available_radar`, `save_intersectional_heatmap`
- Clustering: `save_cluster_profile_bars`, `save_cluster_fairness_heatmap`
- Dermatology: the four `dermatology_*` modules. Not re-exported; their
  callers import them only when figures are requested, so the CSV/JSON paths
  stay matplotlib-free.

Current cross-experiment/dissertation figures live in
`fairness_comparison.py`.

## Inputs And Outputs

- Reads canonical comparison tables from `output/cardiac/runs/<run_id>/experiments/comparisons/data/`.
- Dissertation batch figures go to `output/cardiac/studies/dissertation_figures/<run_id>/`.
- Notebook figures may be written under `notebooks/figures/<pipeline>/`.
- Dermatology figures go under `output/dermatology/runs/<run_id>/`, next to the
  stage output they plot.
- Thesis figures come from `scripts/thesis/make_figures.py`, not this package's
  batch script; see [plots.md](../../../docs/reference/plots.md).

## Usage

```python
from fairxai.viz.fairness_comparison import save_mitigation_delta_matrix

save_mitigation_delta_matrix(full_df, "mitigation_delta_matrix.png")
```

## Related

- Plot reference: [../../../docs/reference/plots.md](../../../docs/reference/plots.md)
- Results schema: [../../../docs/reference/results-schema.md](../../../docs/reference/results-schema.md)
