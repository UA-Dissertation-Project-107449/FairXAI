# Similarity Module

Similarity-based individual fairness analysis. It checks whether similar
patients receive similar predictions and maps local violation density.

## Files

| File | Purpose |
|------|---------|
| `engine.py` | k-nearest-neighbor consistency analysis |
| `density.py` | Violation density mapping |
| `models.py` | `SimilarityResult`, `SimilarityRow`, `ViolationMapResult` dataclasses |
| `__init__.py` | Public exports |

## Public API

- `SimilarityEngine`
- `ViolationDensityMapper`
- `SimilarityResult`
- `SimilarityRow`
- `ViolationMapResult`

## Config And Artifacts

- Study script: `scripts/studies/run_grouping_analysis.py`
- Related config: `configs/experiments/clustering.yaml`
- Outputs: grouping/similarity evidence under `output/cardiac/studies/grouping/` and run-linked study outputs.

## Usage

```python
from fairxai.similarity import SimilarityEngine

engine = SimilarityEngine(k_values=[5])
result = engine.compute(df, feature_cols=feature_cols)
```

## Related

- Clustering module: [../clustering/README.md](../clustering/README.md)
- Fairness module: [../fairness/README.md](../fairness/README.md)
- Dissertation evidence: [../../../docs/research/dissertation-evidence-check.md](../../../docs/research/dissertation-evidence-check.md)
