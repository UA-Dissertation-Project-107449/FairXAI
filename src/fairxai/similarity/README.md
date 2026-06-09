# Similarity Module

Individual-fairness analysis: *similar patients should receive similar
predictions*. Measured as **k-NN prediction consistency** - the fraction of a
patient's k nearest neighbours (in feature space) sharing their prediction.

This is **analysis only** - it needs predictions and never affects training
(unlike the clustering module, whose `group_cluster` feeds CV stratification and
mitigation). It runs **post-assess**, per model.

## Files

| File | Purpose |
|------|---------|
| `engine.py` | `SimilarityEngine` - multi-k consistency, per-group breakdown, per-sample scores |
| `density.py` | `ViolationDensityMapper` - PCA-2D scatter coloured by consistency (hotspot map) |
| `similarity_pipeline.py` | Post-assess orchestration: load all models' predictions → run per model |
| `models.py` | `SimilarityResult`, `SimilarityRow`, `ViolationMapResult` dataclasses |
| `__init__.py` | Public re-exports |

## Public API

- `SimilarityEngine(k_values=None, pred_col="y_pred", standardize=True)`
  - `compute(df, feature_cols) -> SimilarityResult` - consistency for each k.
  - `per_group_consistency(df, feature_cols, group_cols, k)` - consistency
    aggregated per sensitive group (`mean/std/min/max/n`), neighbours global.
  - `per_sample_consistency(df, feature_cols, k)` - per-row scores (used by the
    density mapper).
- `ViolationDensityMapper(k, sample_size, random_state).compute(...)` - writes the
  hotspot PNG; low-consistency regions (similar patients, inconsistent
  predictions) are individual-fairness violations.
- `run_similarity(run_root, dataset, sensitive_attrs, k_values, out_base, ...)` -
  loads **every** model's predictions and runs the full analysis per model.
- `run_similarity_for_predictions(pred_df, feature_cols, sensitive_attrs, k_values, out_dir, ...)`
  - the per-model core (scores CSV + `per_group_consistency.json` + density PNG).
- `load_all_model_predictions(run_root, dataset)` / `resolve_feature_cols(df, exclude)`.

## Feature scaling (important)

Distance is computed on **z-scored** features by default (`standardize=True`).
Raw Euclidean lets a high-magnitude column (`chol ~240`) dominate a binary
sensitive column (`sex ∈ {0,1}`), scrambling the neighbourhood. The engine and the
density mapper share one scaled distance path, so the scores CSV and the PNG agree.
`resolve_feature_cols` drops metadata, `*_cat` decoded columns, and sensitive
attributes from the feature set.

## Outputs (per model)

`<run>/baseline/individual_fairness/<dataset>/<model>/`:

| File | Contents |
|------|----------|
| `similarity_fairness_scores.csv` | per-k `mean/std/min/median` consistency |
| `per_group_consistency.json` | per sensitive attribute → per group `mean/std/min/max/n` |
| `violation_density_map.png` | PCA-2D consistency hotspot map |

## Pipeline wiring

Off by default. Enable with `similarity.enabled: true` in
`configs/pipelines/cardiac.yaml` or `RUN_SIMILARITY=1`. Runs after assess in both
orchestrators:

```bash
RUN_SIMILARITY=1 GO_UNTIL=assess bash scripts/cardiac/cardiac_pipeline.sh --datasets cleveland
python flows/cardiac_pipeline.py --similarity --go-until assess --datasets cleveland
```

The standalone study (`scripts/studies/run_grouping_analysis.py`) calls the same
shared core, so study and pipeline stay in lockstep.

## Usage Example

```python
from fairxai.similarity import SimilarityEngine

engine = SimilarityEngine(k_values=[5, 10, 20])
scores = engine.compute(pred_df, feature_cols=["chol", "thalach", "oldpeak"])
by_group = engine.per_group_consistency(
    pred_df, ["chol", "thalach", "oldpeak"], group_cols=["age_group", "sex"], k=5
)
```

## Dependencies

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib` (density map only; skipped gracefully if absent)
