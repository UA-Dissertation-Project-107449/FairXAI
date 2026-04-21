# FairXAI Plots Guide

Reference for all plotting functions in `src/fairxai/viz/`. Describes what each plot shows,
what research question it answers, and where to find the inputs.

---

## Experiment Comparison Plots (`viz/experiment_plots.py`)

These three plots are the primary dissertation visualizations. They work on the flat DataFrame
produced by `run_experiment_comparison.py`.

### `save_comparison_heatmap(df, title, output_file)`

**What it shows:** A heatmap where rows are experiment configurations (mitigation technique ×
binning strategy) and columns are fairness metrics × sensitive attributes. Cell color = metric
gap magnitude (darker = worse fairness).

**Research question:** *Which combinations of mitigation and binning strategy produce the
most uniformly fair outcomes across all sensitive attributes?*

**How to read it:**
- Dark cells in a row → that config has a fairness problem for that metric/attribute
- A row that is uniformly light → this config achieves broad fairness across all dimensions
- Compare rows to identify which mitigation techniques consistently reduce gaps

**Input:** DataFrame with columns including `mitigation_technique`, `binning_strategy`,
and one column per fairness metric (e.g. `dem_parity_sex_gap`, `eq_odds_age_tpr_gap`).

**Typical call:**
```python
from fairxai.viz.experiment_plots import save_comparison_heatmap
save_comparison_heatmap(results_df, "Fairness Metric Heatmap — Cleveland LR", output_path)
```

---

### `save_tradeoff_scatter(df, title, output_file)`

**What it shows:** A scatter plot where each dot is one experiment. X-axis = F1 score
(performance), Y-axis = demographic parity gap (lower = fairer). Color or shape can encode
mitigation technique.

**Research question:** *Which mitigation strategies achieve both high accuracy AND low
fairness gap (upper-right area is bad; lower-right is ideal)?*

**How to read it:**
- **Bottom-right** → accurate AND fair (ideal)
- **Top-right** → accurate but unfair
- **Bottom-left** → fair but poor accuracy
- Clusters of dots from the same mitigation technique reveal its characteristic trade-off

**Input:** DataFrame with columns `f1_score` and `dem_parity_gap` (or another fairness metric).

---

### `save_pareto_frontier(df, title, output_file)`

**What it shows:** Same scatter as the trade-off plot, but dots on the Pareto frontier are
highlighted. A dot is Pareto-optimal if no other experiment is strictly better on both axes
simultaneously.

**Research question:** *Which experiment configurations are non-dominated — i.e., the best
candidates to recommend for deployment?*

**How to read it:**
- Highlighted dots = Pareto-optimal configs: these are your "best trade-off" candidates
- Non-highlighted dots are dominated — at least one other config beats them on both axes
- Use the highlighted set to select the final model/mitigation recommendation for the dissertation

**Input:** Same as trade-off scatter.

---

## Distribution Plots (`viz/distributions.py`)

Used during EDA and preprocessing validation. All implemented.

| Function | What it shows | When to use |
|----------|--------------|-------------|
| `plot_target_distribution_by_group(df, sensitive_attr, target)` | Bar chart: positive rate per group for a sensitive attribute | Check for label imbalance across demographic groups |
| `plot_categorical_distribution_grid(df, cols)` | Grid of bar charts per categorical feature | EDA — understand class frequencies |
| `plot_numeric_distribution_comparison(df_a, df_b, feature)` | Histogram + KDE for the same feature across two datasets | Check distributional shift between cleveland and kaggle_heart |
| `plot_stacked_group_distribution_grid(df, sensitive_attr)` | Stacked bars per feature, colored by group | Check whether features differ across demographic groups |
| `plot_missing_data_patterns(df)` | Heatmap of missing values by column | Data quality check before preprocessing |
| `plot_outlier_analysis(df, col)` | Scatter + box plot for outlier detection | Validate clinical constraint thresholds |
| `plot_mixed_feature_batches(df, n_per_batch)` | Multi-subplot batches of features | Compact EDA over many features |
| `plot_bmi_and_bp_relationship(df)` | BMI vs blood pressure scatter | Cardiac-specific: validate physiological relationship |

---

## Comparison Plots (`viz/comparisons.py`)

Used to compare two datasets side by side. All implemented.

| Function | What it shows | When to use |
|----------|--------------|-------------|
| `plot_correlation_heatmap_grid(df_a, df_b)` | Correlation matrices for two datasets | Compare feature relationships across datasets |
| `plot_pca_kmeans_scatter_grid(df_a, df_b)` | PCA 2D scatter with KMeans cluster coloring | Visualize dataset structure and separability |
| `plot_two_dataset_feature_distributions(df_a, df_b)` | Numeric + categorical distributions side by side | Holistic dataset comparison |
| `plot_drift_heatmap(df_a, df_b)` | KS statistic per feature as a heatmap | Identify features with distributional shift between datasets |

---

## Fairness Plots (`viz/fairness.py`)

All three functions are implemented.

### `plot_fairness_metric_heatmap(df, sensitive_attr, output_file)`

**What it shows:** rows = mitigation technique (non-baseline), cols = three fairness metrics
for `sensitive_attr` (`dem_parity_{attr}_max_diff`, `eq_odds_{attr}_tpr_diff`,
`eq_odds_{attr}_fpr_diff`); cell = mean over binning strategies; diverging colormap
anchored at 0.10 threshold.

**Input:** `full_comparison.csv`, a sensitive attribute name (e.g. `"age_group_cat"`).

**Research question:** *Which mitigation technique reduces which fairness metric, and by how much?*

---

### `plot_group_performance_gaps(before_json, after_json, sensitive_attr, output_file)`

**What it shows:** grouped bar chart — x-axis = demographic groups, three subplots for
TPR, FPR, precision; baseline (grey) vs after-mitigation (blue) side by side.

**Input:** Two fairness assessment JSON dicts or paths (stage-6 baseline + experiment).

**Research question:** *Which demographic groups benefit most from mitigation?*

---

### `plot_bias_amplification_waterfall(stages_dict, output_file)`

**What it shows:** bar chart of fairness gap at each pipeline stage; green = improved
(gap decreased), red = worsened; orange dashed line at threshold 0.10.

**Input:** Ordered dict `{stage_name: fairness_gap_value}` — assembled manually from
run outputs and placed in `{run_dir}/stage_gaps.json` for the delivery script.

**Research question:** *At which pipeline stage does bias originate or worsen?*

---

## Transformation Plots (`viz/transformations.py`)

All three functions are implemented.

### `plot_transformation_impact(before_dict, after_dict, output_file)`

**What it shows:** side-by-side bars for each metric (f1, recall, precision, auc_roc,
fairness_gap); delta annotation above each pair.

**Input:** Two `{metric: value}` dicts (before/after).

---

### `plot_before_after_distributions(X_before, X_after, feature_cols, output_file)`

**What it shows:** per-feature KDE overlay (before=grey, after=blue); subplots sorted by
KS statistic descending; KS value in subtitle.

**Input:** Two DataFrames with the same numeric feature columns.

---

### `plot_scaling_effects(X_raw, X_scaled, output_file)`

**What it shows:** box plots per feature — raw vs scaled side by side.

**Input:** Two DataFrames (before/after StandardScaler).

---

## Cross-Model & Intersectional Plots (`viz/experiment_plots.py`)

Four new functions added alongside the three existing ones.

### `save_intersectional_heatmap(per_group_df, metric, output_file)`

**What it shows:** rows = mitigation technique, cols = subgroup labels
(`sensitive_attr + group`); cell = mean delta (experiment − baseline); green = improvement.

**Input:** `per_group_comparison.csv`, a metric name (e.g. `"demographic_parity_rate"`).

**Research question:** *Which demographic subgroups benefit most from mitigation — and which are harmed?*

---

### `save_cross_model_radar(summary_df, output_file)`

**What it shows:** spider/radar chart with 5 axes (F1, Recall, Precision, AUC-ROC,
Fairness = 1−gap); one filled polygon per model type.

**Input:** `cross_model_summary.csv` with best config per model type.

**Research question:** *Which model type dominates across all performance + fairness dimensions?*

---

### `save_mitigation_effectiveness_matrix(full_df, output_file)`

**What it shows:** two side-by-side heatmaps — `fairness_gain_pct` (green) and
`score_drop_pct` (red) per mitigation technique.

**Input:** `full_comparison.csv`.

**Research question:** *Which mitigation gives the best fairness gain for the smallest accuracy cost?*

---

### `save_pareto_all_models(full_df, output_file, x_col, y_col)`

**What it shows:** single scatter with 4 coloured point clouds (one per model type);
Pareto frontier drawn per model type.

**Input:** `full_comparison.csv`; defaults `x_col="f1_value"`, `y_col="fairness_gap"`.

**Research question:** *Does any model type dominate the F1–fairness Pareto space?*

---

## Where to Find Plot Outputs

Pipeline plots are saved by the comparison script:

```
output/cardiac/runs/<run_id>/experiments/comparisons/plots/
├── comparison_heatmap.png
├── tradeoff_scatter.png
└── pareto_frontier.png
```

EDA plots are generated by the profiling notebook (`notebooks/cardiac_profilling.ipynb`) and
are not automatically saved by pipeline scripts.
