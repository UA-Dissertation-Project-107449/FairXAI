# FairXAI Cheat Sheet

Quick command and layout reference. For the longer flow-control guide, see
[../architecture/pipeline-flow-control.md](../architecture/pipeline-flow-control.md).

## Install Profiles

| Profile | Command | Use |
|---------|---------|-----|
| Core | `pip install -e .` | Imports and lightweight helpers |
| Experiment | `pip install -e ".[experiment]"` | Pipeline, fairness, HPO, mitigation |
| Dev | `pip install -e ".[dev]"` | Tests, notebooks, plots, lint tooling |
| Full | `pip install -e ".[full]"` | Local research stack except GPU |
| GPU | `pip install cuml-cu12==25.2.1` | HPC CUDA host after CUDA module load |

## Pipeline Stages

`fairxai.pipeline.stages.STAGES` is the source of truth.

| # | Name | Main entry point |
|---|------|------------------|
| 1 | `load` | `scripts/cardiac/load_data.py` |
| 2 | `profile` | `scripts/cardiac/profile_data.py` |
| 3 | `recommend` | `scripts/cardiac/generate_recommendations.py` |
| 4 | `preprocess` | `scripts/cardiac/preprocess.py` |
| 5 | `hpo_study` | `scripts/studies/run_hpo.py` |
| 6 | `feature_selection_study` | `scripts/studies/run_feature_selection_study.py` |
| 7 | `train` | `scripts/cardiac/train_baseline.py` |
| 8 | `assess` | `scripts/cardiac/assess_predictions.py` |
| 9 | `attribute_binning` | `scripts/experiments/run_attribute_binning_analysis.py` |
| 10 | `mitigation` | `scripts/cardiac/mitigation.py` |
| 11 | `combinatorial` | `scripts/cardiac/combinatorial.py` |
| 12 | `compare` | `scripts/cardiac/compare.py`, then grouping and dissertation-plot scripts |

Grouping (`scripts/studies/run_grouping_analysis.py`) currently runs during stage 12 after comparison. It is not a separate checkpointed stage.

## Common Runs

```bash
# Full bash pipeline
bash scripts/cardiac/cardiac_pipeline.sh

# Full Prefect flow
python3 flows/cardiac_pipeline.py

# Cleveland-only smoke run
python3 flows/cardiac_pipeline.py --datasets cleveland

# Stop after recommendations
bash scripts/cardiac/cardiac_pipeline.sh --go-until recommend

# Stop after baseline fairness assessment
bash scripts/cardiac/cardiac_pipeline.sh --go-until assess

# Resume latest run at training
bash scripts/cardiac/cardiac_pipeline.sh --resume-from train

# Selected dataset/model scope
bash scripts/cardiac/cardiac_pipeline.sh \
  --datasets cleveland \
  --model-types logistic_regression xgboost
```

## Source Layout

| Path | Role |
|------|------|
| `src/fairxai/` | Reusable Python package |
| `scripts/common/` | Pipeline-agnostic stage implementations |
| `scripts/cardiac/` | Cardiac wrappers and bash orchestrator |
| `scripts/experiments/` | Experiment-stage scripts |
| `scripts/studies/` | HPO, feature selection, grouping, dissertation plots |
| `flows/` | Prefect orchestration wrapper |
| `configs/` | Pipeline, model, domain, experiment, and threshold YAML/JSON |
| `tests/` | Unit and integration tests |

## Checks

```bash
python3 -m black --check src scripts flows tests
python3 -m isort --check-only src scripts flows tests
python3 -m ruff check src scripts flows tests
python3 -m pytest tests/unit/ -q
python3 -m pytest tests/ -m "not slow"
```

## Output Roots

| Output | Path |
|--------|------|
| Run artifacts | `output/cardiac/runs/<run_id>/` |
| Latest run pointer | `output/cardiac/latest_run` and `output/cardiac/latest_run.txt` |
| Logs | `logs/cardiac/runs/<run_id>/` |
| Processed splits | `data/processed/cardiac/<dataset>_<binning>/` |
| Study outputs | `output/cardiac/studies/<study_type>/` |
| Dissertation figures | `output/cardiac/studies/dissertation_figures/<run_id>/` |

## More Detail

- Architecture: [../architecture/modules.md](../architecture/modules.md)
- Testing: [testing.md](testing.md)
- Plots: [../reference/plots.md](../reference/plots.md)
- Results schema: [../reference/results-schema.md](../reference/results-schema.md)
