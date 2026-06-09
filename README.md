# FairXAI

FairXAI is the dissertation research repository for fairness-aware and
explainable healthcare decision-support experiments. The current complete
pipeline is cardiac-focused; dermatology is scaffolded but not an active
end-to-end pipeline.

## What This Repo Does

- Standardizes cardiac datasets into a shared schema.
- Profiles data complexity, group representation, and intersectional risk.
- Generates pre-model fairness triage recommendations.
- Trains baseline models and evaluates fairness.
- Runs HPO, feature-selection, age-binning, mitigation, combinatorial, clustering, similarity, and comparison studies.
- Produces dissertation-ready evidence tables and figures.
- Exposes WebApp-oriented characterization/binning/clustering adapters.

## Install

From `Code/FairXAI`:

```bash
# Core package
pip install -e .

# Pipeline and experiment dependencies
pip install -e ".[experiment]"

# Local research/dev tooling
pip install -e ".[dev]"

# Everything local except GPU-only packages
pip install -e ".[full]"
```

GPU acceleration is host-specific. Install RAPIDS/cuML only on compatible CUDA hosts after the environment is prepared.

## Run The Cardiac Pipeline

```bash
# Bash orchestrator
bash scripts/cardiac/cardiac_pipeline.sh

# Prefect orchestrator
python3 flows/cardiac_pipeline.py

# Cleveland-only smoke run
python3 flows/cardiac_pipeline.py --datasets cleveland

# Stop after recommendations
bash scripts/cardiac/cardiac_pipeline.sh --go-until recommend

# Resume latest run from training
bash scripts/cardiac/cardiac_pipeline.sh --resume-from train
```

Both orchestrators support `--datasets`, `--model-types`, `--resume-from`,
`--go-until`, `--run-id`, `-v`, and `-vv`.

## Pipeline Stages

| # | Stage | Purpose |
|---|-------|---------|
| 1 | `load` | Load and standardize raw cardiac datasets |
| 2 | `profile` | Build profiling and complexity artifacts |
| 3 | `recommend` | Generate pre-model fairness triage |
| 4 | `preprocess` | Split, scale, bin, and prepare fairness profiles |
| 5 | `hpo_study` | Run hyperparameter optimization |
| 6 | `feature_selection_study` | Run sensitive-attribute ablation study |
| 7 | `train` | Train baseline models |
| 8 | `assess` | Assess post-prediction fairness |
| 9 | `attribute_binning` | Analyze age-binning strategies |
| 10 | `mitigation` | Compare mitigation techniques |
| 11 | `combinatorial` | Run full experiment matrix |
| 12 | `compare` | Build comparison outputs, grouping evidence, and dissertation figures |

`src/fairxai/pipeline/stages.py` is the source of truth for names, aliases, and checkpoint markers.

## Repository Layout

| Path | Purpose |
|------|---------|
| `src/fairxai/` | Reusable Python package |
| `scripts/` | Bash, cardiac wrappers, common stage scripts, studies, experiments |
| `flows/` | Prefect orchestration |
| `configs/` | Pipeline, model, domain, experiment, profiling, and threshold configs |
| `docs/` | Architecture, guides, references, research notes, roadmap |
| `tests/` | Unit and integration tests |
| `data/` | External/raw/processed datasets; generated data is not the source of truth |
| `output/` | Generated run/study artifacts |
| `logs/` | Run logs and warning/error summaries |
| `notebooks/` | Exploratory notebooks plus exported tables/figures |

## Outputs

| Artifact | Path |
|----------|------|
| Run root | `output/cardiac/runs/<run_id>/` |
| Latest run pointer | `output/cardiac/latest_run` and `output/cardiac/latest_run.txt` |
| Logs | `logs/cardiac/runs/<run_id>/` |
| Processed splits | `data/processed/cardiac/<dataset>_<binning>/` |
| Study outputs | `output/cardiac/studies/<study_type>/` |
| Dissertation figures | `output/cardiac/studies/dissertation_figures/<run_id>/` |

## Documentation Map

- [docs/README.md](docs/README.md) - documentation index and reading order.
- [docs/guides/cheat-sheet.md](docs/guides/cheat-sheet.md) - commands, checks, stage table.
- [docs/architecture/pipeline-flow-control.md](docs/architecture/pipeline-flow-control.md) - resume/go-until/checkpoints.
- [docs/architecture/modules.md](docs/architecture/modules.md) - source module responsibilities.
- [docs/reference/results-schema.md](docs/reference/results-schema.md) - result JSON/table contracts.
- [docs/reference/plots.md](docs/reference/plots.md) - plotting APIs and figure outputs.
- [docs/planning/roadmap.md](docs/planning/roadmap.md) - implemented, partial, and deferred work.
- [docs/research/dissertation-evidence-check.md](docs/research/dissertation-evidence-check.md) - current dissertation evidence snapshot.

## Local Checks

```bash
python3 -m black --check src scripts flows tests
python3 -m isort --check-only src scripts flows tests
python3 -m ruff check src scripts flows tests
python3 -m pytest tests/unit/ -q
```

For all test commands and coverage notes, see [docs/guides/testing.md](docs/guides/testing.md).

## Current Limits

- Cardiac is the active end-to-end pipeline. Dermatology is scaffolded only.
- Counterfactual explanations are intentionally frozen behind `counterfactual_stub`.
- Clustering and similarity evidence should be framed as exploratory subgroup diagnostics, not standalone proof of fairness.
- GPU paths require a compatible CUDA/HPC environment.
