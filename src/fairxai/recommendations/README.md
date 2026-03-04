# Recommendations Module

Fairness triage recommendation engine for FairXAI. Produces a
**pre-model triage report** that assesses whether a dataset is ready
for fair-ML experimentation — before any model is trained.

Justifications are backed by profiling metrics (complexity, imbalance,
representation) and, when available, by evidence from prior experiment
runs.

## File Overview

```
recommendations/
├── __init__.py        # Public re-exports
├── models.py          # Data classes: TriageReport, Recommendation, Priority, …
├── ingestion.py       # Auto-detection of column types & roles + schema fast-path
├── config.py          # Loads thresholds from YAML (see configs/recommendations/)
├── evidence.py        # Safe dict-key extractors for profiling dicts
├── history.py         # Scans prior runs for reference distributions
├── rules.py           # One function per triage category (A–F)
├── engine.py          # High-level orchestrator: ingest → profile → check → report
└── output.py          # Serialise TriageReport to JSON and/or Markdown
```

### models.py

Defines all data classes used across the module:

| Class | Purpose |
|-------|---------|
| `ColumnType` | Enum — `NUMERICAL`, `CATEGORICAL`, `BINARY`, `IDENTIFIER`, `DATETIME`, `TEXT`, `UNKNOWN` |
| `ColumnRole` | Enum — `FEATURE`, `LABEL`, `SENSITIVE`, `IDENTIFIER`, `EXCLUDE` |
| `ColumnMeta` | Per-column metadata (type, role, uniqueness, missingness, sample values) |
| `DatasetIngestion` | Full ingestion descriptor — list of `ColumnMeta`, dataset name, counts |
| `Priority` | P0 (Critical), P1 (High), P2 (Medium), P3 (Info) |
| `Confidence` | `HIGH`, `MEDIUM`, `LOW` |
| `TriageCategory` | A — Task Framing, B — Sensitive Adequacy, C — Representation Risk, D — Overlap/Ambiguity, E — Explainability Suitability, F — Overall Readiness |
| `Recommendation` | A single recommendation with category, priority, confidence, description, evidence dict, and suggested action |
| `ReadinessStatus` | `READY`, `READY_WITH_CAVEATS`, `NOT_READY` |
| `TriageReport` | Top-level container: dataset name, readiness status, list of recommendations, ingestion summary, timestamps |

Every data class exposes a `to_dict()` method for JSON-safe serialisation.

### ingestion.py

Two entry-points:

- **`DatasetIngestor.ingest(csv_path, ...)`** — reads a CSV, auto-detects
  separator/header, infers column types via heuristics (`_IDENTIFIER_UNIQUENESS_RATIO`,
  `_CATEGORICAL_MAX_CARDINALITY`), and assigns roles using name hints.
  Optional `label_column` / `sensitive_columns` args override the heuristics.

- **`ingestion_from_schema(schema_path, dataset_key, data_dir)`** — fast-path
  for already-registered datasets (reads column metadata straight from a
  `cardiac.json`-style schema file).

`confirm_ingestion(ingestion, overrides)` lets callers patch roles/types after
detection — the user-confirmation step.

### config.py

`TriageConfig` wraps the YAML dict loaded from
`configs/recommendations/thresholds.yaml` and exposes typed attributes.
`load_triage_config(path, project_root)` loads the file (or uses built-in
defaults if the file is missing).

### evidence.py

~25 pure helper functions that extract values from the profile dict produced
by `DataProfiler.profile_dataset()`. By centralising all nested key look-ups
here, the rule functions never hard-code dict paths.

Key functions: `get_n_samples`, `get_size_ratio`, `get_complexity_metric`,
`get_group_complexity`, `get_low_support_intersections`, `compare_to_reference`.

### history.py

`HistoricalReference` scans completed experiment runs under:
- `output/<pipeline>/runs/`
- `output/<pipeline>/archived_runs/`
- `output/<pipeline>/profiling/`
- `output/<pipeline>/baseline/fairness/`

It builds reference distributions (median, IQR, min, max) per complexity and
fairness metric, exposed through `ReferenceStats`.

When no history exists, it falls back to `_LITERATURE_DEFAULTS` — approximate
ranges from the data-complexity literature. **These defaults should be updated**
**with real experiment data as more cardiac runs accumulate** (see *Roadmap*
below).

### rules.py

One function per triage category, all with the same signature:

```python
def check_<category>(
    profile: Dict,
    config: TriageConfig,
    ref: Optional[HistoricalReference] = None,
    ingestion: Optional[DatasetIngestion] = None,
) -> List[Recommendation]:
```

| Function | Category | What it checks |
|----------|----------|----------------|
| `check_task_framing` | A | Multiclass support, subgroup class support, complexity warnings for multiclass instability |
| `check_sensitive_adequacy` | B | No sensitive attrs (P0), high nulls (P0), too few groups (P0) |
| `check_representation_risk` | C | Size ratio, statistical parity, small groups, intersectional low-support |
| `check_overlap_ambiguity` | D | Elevated complexity metrics vs reference, subgroup divergence |
| `check_explainability_suitability` | E | Linear complexity warnings (L1-L3, T1 above threshold) |
| `check_readiness` | F | Derives `ReadinessStatus` from P0/P1 counts |

`run_all_checks()` calls A–E, then F, and returns a sorted recommendation list.

### engine.py

`RecommendationEngine` is the public-facing orchestrator:

```python
engine = RecommendationEngine(project_root="/path/to/FairXAI")
ingestion = engine.ingest("data.csv", label_column="target",
                           sensitive_columns=["sex"])
report = engine.generate(ingestion)
```

Constructor params:
- `config_path` — path to thresholds YAML (defaults to `configs/recommendations/thresholds.yaml`)
- `project_root` — FairXAI root (for locating configs, history, data)
- `history_base_path` — custom base path for run history scan

Methods:
- `ingest(csv_path, ...) → DatasetIngestion`
- `ingest_from_schema(schema_path, dataset_key, data_dir) → DatasetIngestion`
- `generate(ingestion, profile=None) → TriageReport`  *(auto-profiles if no profile given)*
- `generate_from_csv(csv_path, ...) → TriageReport`  *(one-shot convenience)*

### output.py

- `to_json(report)` / `to_json_string(report)` — JSON serialisation
- `to_markdown(report)` — human-readable Markdown with readiness gauge,
  scorecard table, collapsible evidence blocks, and visual-panel references

## Configuration

All numeric thresholds live in
[configs/recommendations/thresholds.yaml](../../../configs/recommendations/thresholds.yaml).
The file has the following sections:

| Section | Purpose | Key values |
|---------|---------|------------|
| `representation` | Group size balance | `size_ratio_warning=3.0`, `min_group_samples=50`, `statistical_parity_warning=0.15` |
| `complexity` | Overlap / ambiguity | `high_overlap_percentile=75`, elevated metric list, `group_divergence_threshold=0.20` |
| `explainability` | Linear explanation stability | `high_threshold=0.5`, L1-L3, T1 metrics |
| `readiness` | Deriving overall status | `p0_makes_not_ready=true`, `p1_caveat_threshold=1` |
| `fairness` | Demographic constraints | `max_fairness_violation=0.10`, `min_recall=0.70` |
| `task_framing` | Multiclass stability | `multiclass_minority_support=20`, N3/N4/T1/F4 metrics |
| `sensitive_adequacy` | Column quality gating | `max_null_fraction=0.10`, `min_unique_groups=2` |
| `reference` | History / fallback behaviour | `use_historical=true`, `fallback_to_defaults=true` |

Override individual thresholds at runtime through the script `--overrides`
flag (JSON string), e.g.:

```bash
python scripts/cardiac/generate_recommendations.py \
    --overrides '{"representation.size_ratio_warning": 5.0}'
```

## Output Formats

The engine produces two output files per dataset:

- `<dataset>_triage.json` — machine-readable payload
- `<dataset>_triage.md` — human-readable report with:
  - Readiness gauge (🟢 / 🟡 / 🔴)
  - Recommendation scorecard table (category, priority, description)
  - Detailed sections per recommendation with collapsible evidence
  - Limitations note

Both are written to `output/<pipeline>/recommendations/`.

## Pipeline Integration

The recommendation engine runs as **Phase 3** (after profiling, before
preprocessing) in both the Prefect flow and the bash pipeline:

- **Prefect**: `flows/cardiac_pipeline.py` — `generate_recommendations` task
- **Bash**: `scripts/cardiac/cardiac_pipeline.sh` — controlled by
  `RUN_RECOMMENDATIONS` env var (default: `true`)

Enable/disable: `RUN_RECOMMENDATIONS=false ./scripts/cardiac/cardiac_pipeline.sh`

## Roadmap

- [ ] **Update literature defaults** — replace `_LITERATURE_DEFAULTS` in
  `history.py` with real reference distributions from accumulated cardiac
  experiment runs.
- [ ] **Dermatology support** — add `scripts/dermatology/generate_recommendations.py`
  wrapper once the dermatology pipeline is available.
- [ ] **Interactive confirmation UI** — the `confirm_ingestion()` path exists
  but is not yet wired into a CLI/TUI prompt; currently relies on programmatic
  overrides.
- [ ] **Rule expansion** — additional domain-specific rules (e.g., temporal
  drift detection, feature-level fairness proxies) as the triage framework
  matures.
