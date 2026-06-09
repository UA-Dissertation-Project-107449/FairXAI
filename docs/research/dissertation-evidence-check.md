# Dissertation Evidence Check

Snapshot: latest cardiac run on 2026-05-21, `run_20260521_082158_311_40755_79f28a`.

This note is a quick interpretation checkpoint after the evidence-cleanup branch. It is not the final dissertation text, but it records what the latest artifacts support.

## Mitigation Evidence

The main conclusion is dataset-dependent.

For `kaggle_heart`, Exponentiated Gradient is the clear fairness winner. Averaged across logistic-regression configurations, the strongest fairness-gap reductions are:

- `adasyn+exponentiated_gradient`: mean `delta_fairness_gap = +0.2307`
- `smote+exponentiated_gradient`: mean `delta_fairness_gap = +0.1959`
- `exponentiated_gradient`: mean `delta_fairness_gap = +0.1719`

These gains are large, but they are not free. They come with substantial average F1/accuracy costs, especially for the pure or preprocessed Exponentiated Gradient variants.

Threshold Optimization appears to counter or dampen the large Exponentiated Gradient effect on `kaggle_heart`. The same family of methods drops to much smaller average fairness improvements after Threshold Optimization:

- `smote+exponentiated_gradient+threshold_optimizer`: mean `delta_fairness_gap = +0.0343`
- `adasyn+exponentiated_gradient+threshold_optimizer`: mean `delta_fairness_gap = +0.0279`

That should not be framed as a bug. It looks like postprocessing is pulling the model back toward a more conservative decision boundary: much smaller fairness movement, smaller performance damage, less dramatic redistribution.

For `cleveland`, the evidence is weaker and noisier. The best average fairness-gap reducer is `grid_search` (`+0.0364`), while Exponentiated Gradient variants show isolated strong configurations but worse averages. This is consistent with Cleveland being smaller and more sensitive to binning, split choice, and group counts.

The top evidence summary currently ranks Kaggle Heart first:

- `adasyn+exponentiated_gradient`, `clinical`, `single_split`, `c_1_0`: `delta_fairness_gap = +0.5127`
- `smote+exponentiated_gradient`, `clinical`, `kfold_cv`, `c_1_0`: `delta_fairness_gap = +0.4124`
- `exponentiated_gradient`, `clinical`, `single_split`, `c_0_5`: `delta_fairness_gap = +0.4021`

That supports a dissertation statement like: Exponentiated Gradient is the most aggressive fairness intervention in the current evidence, especially on Kaggle Heart, but its benefit is partly bought by redistributing or reducing predictive performance.

## AUC Sanity

The latest comparison tables no longer show the fake AUC collapse problem. `metric_deltas.csv` has no `auc_roc` rows with an experiment value of `0.0`. Missing probability-based AUC is now represented as unavailable instead of being turned into a false `-80 pp` or `-90 pp` drop.

This matters for figure selection: heatmaps and evidence rankings should no longer punish a mitigation because an unavailable score was silently converted to zero.

## Clustering Evidence

The clustering evidence is now more credible than before because the giant-cluster artifact is gone in the latest run.

Current selected cluster structures:

- `cleveland`: KMeans with 4 clusters, sizes `104`, `83`, `70`, `38`, silhouette `0.1432`
- `kaggle_heart`: hierarchical clustering with 3 clusters, sizes `321`, `302`, `120`, silhouette `0.1688`

These are not high-silhouette clusters, so they should be described as exploratory descriptive subgroups, not strong natural patient phenotypes.

Your age interpretation is broadly supported:

- In `cleveland`, cluster `0` is mostly younger patients (`<40` and `40-49`), cluster `2` is concentrated around `50-69`, and the remaining clusters split mixed middle/older groups with different sex balance.
- In `kaggle_heart`, cluster `0` is younger-heavy (`<40` and `40-49`), cluster `1` is mostly `50-69`, and cluster `2` is a smaller older/middle group with few extremes.

However, the clusters are not "fair" by the current fairness thresholds. `fairness_by_cluster.csv` marks all current clusters as `is_fair = False`. The useful claim is narrower: the clustering stage now produces balanced, interpretable subgroup evidence instead of one dominant cluster swallowing almost all rows.

## Suggested Figure Framing

Main-text candidates:

- Mitigation delta matrix: shows Exponentiated Gradient as aggressive fairness movement and Threshold Optimization as damping.
- Primary mitigation radar: good for showing the tradeoff between fairness axes and predictive metrics.
- Age/sex error consequence plots: best place to explain who gains and who loses.
- Top-5 binning summary: useful because Cleveland is sensitive to binning.
- Model overfit gap bars: necessary context for tree-model performance claims.

Appendix candidates:

- Cluster profile and cluster fairness plots, with a clear "exploratory subgroup evidence" caption.
- Cross-model baseline radar.
- Binning strategy delta matrix.

Do not overclaim cluster fairness. The current evidence supports subgroup interpretability and better diagnostics, not a fairness result by itself.
