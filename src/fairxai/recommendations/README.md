# Recommendations Module

Pre-model fairness triage engine. It ingests a dataset/profile, checks data
readiness and fairness risk, and writes JSON/Markdown recommendations before
model training.

## Files

| File | Purpose |
|------|---------|
| `models.py` | Dataclasses and enums: ingestion metadata, recommendations, reports |
| `ingestion.py` | CSV/schema ingestion and role/type inference |
| `config.py` | Threshold YAML loading and defaults |
| `evidence.py` | Safe accessors for profiling evidence dictionaries |
| `history.py` | Historical run/reference distribution scanning |
| `rules.py` | Category A-F triage checks |
| `engine.py` | High-level ingestion/profile/check orchestration |
| `output.py` | JSON and Markdown serialization |
| `__init__.py` | Public exports |

## Public API

- `RecommendationEngine`
- `TriageReport`
- `Recommendation`
- `ReadinessStatus`
- `DatasetIngestion`

## Triage Categories

| Category | Focus |
|----------|-------|
| A | Task framing |
| B | Sensitive attribute adequacy |
| C | Representation risk |
| D | Overlap and ambiguity |
| E | Explainability suitability |
| F | Overall readiness |

## Config And Artifacts

- Config: `configs/recommendations/thresholds.yaml`
- Inputs: profiling artifacts under `output/cardiac/runs/<run_id>/profiling/`
- Outputs: `output/cardiac/runs/<run_id>/recommendations/*_triage.{json,md}`
- Selector contract: `output/cardiac/runs/<run_id>/recommendations/selector_contract.json`

Historical references are used when available; literature/default ranges remain
fallbacks when run history is sparse.

## Usage

```python
from fairxai.recommendations import RecommendationEngine

engine = RecommendationEngine(project_root=".")
report = engine.generate_from_csv(
    "data/raw/cardiac/cleveland_standardized.csv",
    label_column="heart_disease",
    sensitive_columns=["age_group", "sex"],
)
```

## Related

- Profiling: [../profiling/README.md](../profiling/README.md)
- Results schema: [../../../docs/reference/results-schema.md](../../../docs/reference/results-schema.md)
