# Clustering Module

Unsupervised discovery of latent patient subgroups, surfaced as a first-class
sensitive attribute (`group_cluster`) so downstream stages treat clusters like
age/sex/ethnicity.

Two distinct entry paths share the same engine:

- **Standalone study** - fit on the *concatenated* train+test frame for post-hoc
  reporting (`scripts/studies/run_grouping_analysis.py`).
- **Pre-train, leakage-safe** - fit on the **train split only**, assign test rows
  by nearest centroid, and write `group_cluster` into the split CSVs *before*
  training (`grouping_pipeline.cluster_and_persist`). This is what makes clusters
  affect CV stratification and per-cluster mitigation.

## Files

| File | Purpose |
|------|---------|
| `engine.py` | `ClusteringEngine` - fits KMeans/Hierarchical/DBSCAN/GMM, selects best by silhouette, with a validity gate |
| `grouping_pipeline.py` | Leakage-safe pre-train clustering + persistence of `group_cluster` into split variants |
| `profiles.py` | `ClusterProfiler` - per-cluster summaries / `subgroup_profiles.md` |
| `fairness.py` | `FairnessPerCluster` - per-cluster fairness + Cramér's V association matrix |
| `models.py` | `ClusterResult`, `ClusterDiagnostics`, `ClusterReport` dataclasses |
| `__init__.py` | Public re-exports |

## Public API

- `ClusteringEngine(config=None, feature_exclude=None, min_clusters=2, min_cluster_size_abs=1, min_cluster_size_frac=0.0)`
  - `fit(df, feature_cols=None) -> ClusterResult` - z-scores features, fits every
    configured method, picks the highest-silhouette **valid** solution.
- `ClusterProfiler(target_col).compute(df, cluster_col="group_cluster")`
- `FairnessPerCluster(sensitive_attrs).compute(...)` / `.cramers_v_matrix(...)`
- `cluster_and_persist(...)` (in `grouping_pipeline`) and
  `assign_clusters_nearest_centroid(...)` - pre-train path; not re-exported from
  the package root (import from `fairxai.clustering.grouping_pipeline`).

## Validity gate (degenerate-solution guard)

Every method still runs, but a solution is **disqualified as a whole** when it has
fewer than `min_clusters` or **any** cluster below the effective minimum size:

```
min_size = max(min_cluster_size_abs, ceil(min_cluster_size_frac * n_samples))
```

The best silhouette is then chosen **among survivors**, so a lopsided DBSCAN
(e.g. 88 % of rows in one cluster + a few tiny clusters) cannot win over a
balanced KMeans. If **no** solution qualifies, `fit` raises `ClusteringError`;
`cluster_and_persist` catches it and simply **does not inject** `group_cluster`
(the pipeline proceeds without clusters).

**Defaults are a no-op** (`min_cluster_size_abs=1`, `min_cluster_size_frac=0.0`,
`min_clusters=2`) → the WebApp adapter (`integration/clustering.py`) is byte-for-byte
unchanged. The cardiac pipeline opts in via `grouping.{min_clusters,
min_cluster_size_abs, min_cluster_size_frac}` in `configs/pipelines/cardiac.yaml`
(cleveland keeps a floor of 20, scaling up by 5 % of train size).

## Leakage guard (pre-train path)

`cluster_and_persist` fits the engine on the **train split only**, then labels
test rows by nearest train-cluster centroid in the train-scaled feature space
(method-agnostic, works for KMeans/Hierarchical/DBSCAN/GMM, which lack a
`predict`). It writes `group_cluster` into **every** split variant (plain +
`_scaled`, the file the trainer reads) and is **idempotent**: if both splits
already carry `group_cluster` it skips, so a later study pass can't overwrite the
train-only labels with leaky ones.

## DBSCAN noise

DBSCAN noise points (label `-1`) are mapped to the most common cluster
(`engine.py`). As a *training* sensitive attribute this folds outliers into a real
subgroup, flag in analysis. The validity gate above prevents the worst case
(noise-dominated lopsided solutions winning selection).

## Usage Example

```python
from fairxai.clustering import ClusteringEngine

engine = ClusteringEngine(
    config={"kmeans": {"parameters": {"n_clusters": [2, 3, 4]}}},
    feature_exclude=["heart_disease", "age_group", "sex", "group_cluster"],
    min_cluster_size_abs=20,
    min_cluster_size_frac=0.05,
)
result = engine.fit(train_df)            # train-only for the pre-train path
print(result.method, result.n_clusters, result.silhouette)
```

## Dependencies

- `numpy`
- `pandas`
- `scikit-learn`
