# FairXAI Plots Guide

Reference for plotting APIs in `src/fairxai/viz/` and figure-generation scripts.

## Module Map

| File | Purpose |
|------|---------|
| `distributions.py` | EDA distributions, missingness, outliers, target-by-group plots |
| `comparisons.py` | Dataset comparison plots: correlations, PCA/KMeans, drift heatmap |
| `fairness.py` | Fairness metric heatmap, before/after group gaps, bias waterfall |
| `transformations.py` | Before/after transformations, scaling effects |
| `fairness_comparison.py` | Dissertation comparison figures from canonical experiment tables |
| `clustering.py` | Cluster profile bars and cluster fairness heatmaps |
| `dermatology_readiness.py` | Stage 4: pre-model split fairness profiles |
| `dermatology_fairness.py` | Stage 8: subgroup heatmaps and group-view figures |
| `dermatology_comparison.py` | Stage 9: learning curves and model comparison |
| `dermatology_mitigation.py` | Stage 11: baseline vs mitigated gaps per attribute × constraint |
| `constants.py`, `labels.py`, `style.py`, `save_utils.py` | Shared labels, palettes, normalization, and figure saving |
| `utils.py` | Placeholder; helpers are still notebook-local |

Older docs used a non-existent experiment-plot module name. Current
cross-experiment/dissertation figures live in `viz/fairness_comparison.py`.

## Dissertation Comparison Plots

`fairness_comparison.py` consumes canonical tables produced by the comparison
stage, especially:

- `full_comparison.csv`
- `per_group_comparison.csv`
- `cross_model_summary.csv`
- `metric_values.csv`
- `metric_deltas.csv`
- `group_metric_values.csv`
- `group_metric_deltas.csv`
- `fairness_evidence_summary.csv`

Important exports:

| Function | Use |
|----------|-----|
| `save_mitigation_delta_matrix` | Compare fairness movement and score cost by mitigation |
| `save_before_after_metric_radar` | Show baseline vs selected mitigation across metrics |
| `save_group_before_after_bars` | Compare group-level rates before/after mitigation |
| `save_group_delta_bars` | Show group-level improvements or harm |
| `save_group_performance_gap_bars` | Show TPR/FPR/precision gaps |
| `save_group_error_consequence_bars` | Frame group-level error consequences |
| `save_cross_model_baseline_radar` | Baseline model comparison |
| `save_cross_model_best_available_radar` | Best available model/config comparison |
| `save_intersectional_heatmap` | Intersectional subgroup metric deltas |
| `save_binning_strategy_delta_matrix` | Binning-strategy sensitivity |
| `save_model_overfit_gap_bars` | Train/test overfit diagnostics |
| `save_top_n_binning_strategy_summary` | Top binning strategies summary |
| `save_top_n_binning_strategy_age_group_small_multiples` | Age-group binning sensitivity detail |

## General Plot APIs

| Area | Functions |
|------|-----------|
| Distributions | `plot_categorical_distribution_grid`, `plot_numeric_distribution_comparison`, `plot_target_distribution_by_group`, `plot_stacked_group_distribution_grid`, `plot_missing_data_patterns`, `plot_outlier_analysis`, `plot_mixed_feature_batches`, `plot_bmi_and_bp_relationship` |
| Dataset comparison | `plot_correlation_heatmap_grid`, `plot_pca_kmeans_scatter_grid`, `plot_two_dataset_feature_distributions`, `summarize_ks_test_between_datasets`, `plot_drift_heatmap` |
| Fairness | `plot_fairness_metric_heatmap`, `plot_group_performance_gaps`, `plot_bias_amplification_waterfall` |
| Transformations | `plot_transformation_impact`, `plot_before_after_distributions`, `plot_scaling_effects` |
| Clustering | `save_cluster_profile_bars`, `save_cluster_fairness_heatmap` |
| Dermatology | `render_readiness_figures`, `render_subgroup_heatmaps`, `render_group_view_figures`, `render_learning_curves`, `render_comparison_figures`, `render_mitigation_figures` |

The dermatology modules are not re-exported from `fairxai.viz`. Their callers
import them only when figures are requested, so the CSV/JSON paths stay
matplotlib-free. The dermatology stage scripts take `--figures` /
`--no-figures`; without a flag they follow the matching section of
`configs/pipelines/dermatology.yaml` (`readiness_figures`, `fairness.figures`,
`comparison.figures`, `mitigation.figures`).

## Figure Outputs

Pipeline comparison and dissertation plot stages write under:

```text
output/cardiac/runs/<run_id>/experiments/comparisons/
output/cardiac/studies/dissertation_figures/<run_id>/
```

Dermatology figures land under `output/dermatology/runs/<run_id>/`: stage 4
writes `profiling/readiness_figures/<dataset>/`, and stages 8, 9 and 11 write a
`figures/` folder beside their report, for example `baseline/mitigation/figures/`.

Standalone notebooks may write under:

```text
notebooks/figures/<pipeline>/
notebooks/tables/<pipeline>/
```

## Thesis Figures

`scripts/thesis/make_figures.py` builds the Chapter 6 and 7 figures from the
runs named in `scripts/thesis/runs.yaml`. It reads finished run outputs and
does not use the `fairness_comparison.py` exports. Output goes to
`output/thesis/figures/`, or to `--out`.

`FIGURES` lists the builders in run order:

| Function | Figure |
|----------|--------|
| `fig_evidence` | CV pairwise tests (untestable / tested / significant), held-out significant count marked |
| `fig_foursite_age` | Forest plot of all ten age-group pairs on four-site pooled CV predictions |
| `fig_binning` | Prediction-level age parity gap against the label base-rate gap, per binning scheme |
| `fig_binning_grid` | Appendix C: age parity gap under every binning scheme, per cohort and model family |
| `fig_shap_share` | Per-group attribution share by age group (four-site and Cardio70k logistic regression) |
| `fig_gradcam` | Four stratified Grad-CAM cases from the augmented dermatology run |
| `fig_frontier` | F1 against sex parity for every CV sweep cell, coloured by constraint attribute |
| `fig_mitigation_deltas` | Change in the targeted parity gap per mitigation arm |
| `fig_cluster_gaps` | Cardio70k held-out age and sex parity gaps inside each cluster, against the outcome gap |

Cluster figures show where gaps sit inside discovered clusters. Clusters are
exploratory subgroups, not protected groups, so a cluster gap is not a
fairness result on its own.

## Minimal Usage

```python
from pathlib import Path
import pandas as pd

from fairxai.viz.fairness_comparison import save_mitigation_delta_matrix

run_id = "run_YYYYMMDD_HHMMSS_xxx"
base = Path("output/cardiac/runs") / run_id / "experiments/comparisons/data"
full_df = pd.read_csv(base / "full_comparison.csv")

save_mitigation_delta_matrix(
    full_df,
    Path("output/cardiac/studies/dissertation_figures") / run_id / "mitigation_delta_matrix.png",
)
```
