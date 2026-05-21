# Fairness Module

Fairness metrics and mitigation engines used by baseline assessment,
mitigation comparison, combinatorial experiments, clustering, and similarity
analysis.

## Files

| File | Purpose |
|------|---------|
| `metrics.py` | Group fairness, calibration, individual fairness, summary helpers |
| `mitigation.py` | Pre-, in-, post-processing mitigation engines |
| `__init__.py` | Public exports |

## Public API

- `FairnessMetrics`
- `PreProcessingMitigation`
- `InProcessingMitigation`
- `PostProcessingMitigation`
- `MitigationEngine`

## Concepts

- Group fairness: demographic parity, equalized odds, equal opportunity, predictive parity.
- Calibration: expected calibration error by group.
- Individual fairness: k-nearest-neighbor prediction consistency.
- Mitigation: reweighting, resampling, fairlearn reductions, threshold optimization, and supported combinations.

## Config And Artifacts

- Thresholds: `configs/recommendations/thresholds.yaml`
- Mitigation config: `configs/experiments/mitigation.yaml`
- Experiment outputs: `output/cardiac/runs/<run_id>/experiments/`
- Baseline fairness outputs: `output/cardiac/runs/<run_id>/baseline/`

## Usage

```python
from fairxai.fairness import FairnessMetrics

metrics = FairnessMetrics(sensitive_attributes=["age_group", "sex"])
report = metrics.calculate_all_metrics(prediction_df, feature_cols=["age", "chol"])
```

## Related

- Results schema: [../../../docs/reference/results-schema.md](../../../docs/reference/results-schema.md)
- Dissertation evidence: [../../../docs/research/dissertation-evidence-check.md](../../../docs/research/dissertation-evidence-check.md)
