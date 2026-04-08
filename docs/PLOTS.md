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

## Not Yet Implemented (`viz/fairness.py`, `viz/transformations.py`)

The following functions are scaffolded but **raise `NotImplementedError` at runtime**:

| Function | File | Intended purpose |
|----------|------|-----------------|
| `plot_fairness_metric_heatmap` | `fairness.py` | Per-group fairness metric comparison heatmap |
| `plot_group_performance_gaps` | `fairness.py` | Bar chart of TPR/FPR/precision gaps by group |
| `plot_bias_amplification_waterfall` | `fairness.py` | Waterfall showing how bias changes across pipeline stages |
| `plot_transformation_impact` | `transformations.py` | Before/after comparison for a mitigation technique |
| `plot_before_after_distributions` | `transformations.py` | Feature distribution shift after preprocessing |
| `plot_scaling_effects` | `transformations.py` | Effect of scaling on feature distributions |

Do not call these — they will raise immediately. See `docs/ROADMAP.md` for status.

---

## Where to Find Plot Outputs

Pipeline plots are saved by the comparison script:

```
output/cardiac/runs/<run_id>/experiments/full/comparisons/
├── comparison_heatmap.png
├── tradeoff_scatter.png
└── pareto_frontier.png
```

EDA plots are generated by the profiling notebook (`notebooks/cardiac_profilling.ipynb`) and
are not automatically saved by pipeline scripts.
