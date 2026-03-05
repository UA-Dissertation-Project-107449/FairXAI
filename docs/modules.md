# FairXAI Module Architecture

This document summarizes module responsibilities and cross-module dependencies
for `src/fairxai`.

## Module Overview

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `cli` | Script runner helpers, logging setup, run pointer/history utilities | `runner_base.py`, `runner_utils.py` |
| `data` | Data loading, schema harmonization, preprocessing, profiling inputs | `loaders.py`, `preprocessors.py`, `schemas.py`, `profilers.py` |
| `experiments` | Experiment versioning, attribute binning analysis, experiment data I/O | `versioning.py`, `attribute_binning.py`, `data_io.py` |
| `explainability` | SHAP/LIME wrappers and explainability dataclasses | `tabular.py` |
| `fairness` | Fairness metrics and mitigation techniques | `metrics.py`, `mitigation.py` |
| `models` | Baseline model wrapper and CV training orchestration | `baseline.py`, `cv_trainer.py` |
| `notebook_utils` | Notebook-facing context/data/profiling convenience helpers | `context.py`, `data.py`, `profiling.py` |
| `pipeline` | Stage definitions, checkpointing, flow-control helpers | `stages.py` |
| `profiling` | Dataset complexity metric computations | `complexity.py` |
| `recommendations` | Fairness triage recommendation engine | `engine.py`, `rules.py`, `models.py`, `output.py` |
| `utils` | Shared utility layer (config + logging) | `config.py`, `logging_utils.py` |
| `viz` | Visualization toolkit for EDA/comparisons/experiments | `distributions.py`, `comparisons.py`, `experiment_plots.py` |

## Dependency Notes

High-level dependency direction:

- `utils` supports most modules (config/logging)
- `data` feeds `models`, `fairness`, `profiling`, and `recommendations`
- `models` and `fairness` feed `experiments`
- `explainability` is consumed by `models`/scripts
- `pipeline` orchestrates staged script/module interactions
- `viz` is shared by scripts and notebooks

## Current Maturity Snapshot

- **Well-structured and documented**: `recommendations`, `models`, `fairness`, `experiments`
- **Now documented and standardized**: `explainability`, `data`, `cli`, `pipeline`, `profiling`, `utils`, `viz`, `notebook_utils`
- **Scaffolded plotting APIs with roadmap**: `viz/fairness.py`, `viz/transformations.py`

## Documentation Standard

Use `docs/STYLE_GUIDE.md` as the source of truth for:

- Module README requirements
- Docstring expectations
- Public API export style
- Stubs/roadmap conventions
- Docs-only safety checklist
