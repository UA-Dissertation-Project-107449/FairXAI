# Clustering Module

Unsupervised subgroup discovery for fairness analysis. This module finds latent
patient clusters, profiles them, and evaluates per-cluster fairness.

## Files

| File | Purpose |
|------|---------|
| `engine.py` | Runs clustering algorithms and selects/validates results |
| `models.py` | `ClusterResult`, `ClusterDiagnostics`, `ClusterReport` dataclasses |
| `profiles.py` | Cluster profile summaries and dominant feature descriptions |
| `fairness.py` | Per-cluster fairness calculations |
| `__init__.py` | Public exports |

## Public API

- `ClusteringEngine`
- `ClusteringError`
- `FairnessPerCluster`
- `ClusterProfiler`
- `ClusterResult`
- `ClusterDiagnostics`
- `ClusterReport`

## Config And Artifacts

- Config: `configs/experiments/clustering.yaml`
- Study script: `scripts/studies/run_grouping_analysis.py`
- Outputs: `output/cardiac/studies/grouping/` and run-linked grouping artifacts.
- Downstream column: `group_cluster` can be used as a sensitive/group attribute in later analysis.

## Usage

```python
from fairxai.clustering import ClusteringEngine

engine = ClusteringEngine()
result = engine.fit(df, feature_cols=feature_cols)
```

## Related

- Similarity module: [../similarity/README.md](../similarity/README.md)
- Plot guide: [../../../docs/reference/plots.md](../../../docs/reference/plots.md)
- Roadmap/status: [../../../docs/planning/roadmap.md](../../../docs/planning/roadmap.md)
