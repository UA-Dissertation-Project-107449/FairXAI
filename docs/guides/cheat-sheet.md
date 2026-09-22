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

`fairxai.pipeline.stages.STAGES` is the source of truth. The bash orchestrators
import it through `scripts/common/stage_registry.sh` rather than redeclaring it,
so this table is a convenience copy. When it disagrees with the catalog, the
catalog wins.

### Cardiac (1-12)

| # | Name | Main entry point |
|---|------|------------------|
| 1 | `load` | `scripts/cardiac/load_data.py` |
| 2 | `profile` | `scripts/cardiac/profile_data.py` |
| 3 | `recommend` | `scripts/cardiac/generate_recommendations.py` |
| 4 | `preprocess` | `scripts/cardiac/preprocess.py` |
| 5 | `tune` | `scripts/studies/run_hpo.py` |
| 6 | `select_features` | `scripts/studies/run_feature_selection_study.py` |
| 7 | `train` | `scripts/cardiac/train_baseline.py` |
| 8 | `assess` | `scripts/cardiac/assess_predictions.py` |
| 9 | `bin_attributes` | `scripts/experiments/run_attribute_binning_analysis.py` |
| 10 | `mitigate` | `scripts/cardiac/mitigation.py`, then `scripts/common/assess_mitigated_predictions.py` |
| 11 | `sweep` | `scripts/cardiac/combinatorial.py` |
| 12 | `compare` | `scripts/cardiac/compare.py`, then grouping and dissertation-plot scripts |

### Dermatology (1-4, 7-11; the gap at 5-6 is deliberate)

| # | Name |
|---|------|
| 1 | `load` |
| 2 | `profile` |
| 3 | `recommend` |
| 4 | `preprocess` |
| 7 | `train` |
| 8 | `assess` |
| 9 | `compare` |
| 10 | `explain` |
| 11 | `mitigate` |

Dermatology skips `tune` and `select_features` and keeps their numbers free, so
a stage number means the same thing in both domains.

### Pre-rename aliases

Renamed stages still resolve by their old names, and their old `.done` markers
are still accepted on resume. Only the canonical name is written.

| Canonical | Also accepts |
|-----------|--------------|
| `profile` | `profiling` |
| `recommend` | `recommendations`, `triage` |
| `preprocess` | `preprocessing` |
| `tune` | `hpo_study`, `hpo` |
| `select_features` | `feature_selection_study`, `feature_selection`, `fs_study` |
| `train` | `baseline`, `training` |
| `assess` | `fairness`, `assessment` |
| `bin_attributes` | `attribute_binning`, `age_binning` |
| `mitigate` | `mitigation` |
| `sweep` | `combinatorial`, `combo` |
| `compare` | `comparison` |

Grouping (`scripts/studies/run_grouping_analysis.py`) runs during stage 12 after
comparison. It is not a separate checkpointed stage.

## Common Runs

```bash
# Full bash pipeline
bash scripts/cardiac/cardiac_pipeline.sh

# Full Prefect flow
python3 flows/cardiac_pipeline.py

# Dermatology, either orchestrator
bash scripts/dermatology/dermatology_pipeline.sh
python3 flows/dermatology_pipeline.py --no-explain

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

# Drop an incomplete model family from the stage-12 ranking only
COMPARE_EXCLUDE_MODEL_TYPES="svm" bash scripts/cardiac/cardiac_pipeline.sh --resume-from compare

# Dermatology: skip saliency, reuse the frozen-feature cache
bash scripts/dermatology/dermatology_pipeline.sh --no-explain --cache-frozen-features
```

## Flags Worth Knowing

### Both domains

| Flag | Effect |
|------|--------|
| `--datasets`, `--model-types` | Narrow the run |
| `--resume-from`, `--go-until` | Stage range; canonical names or aliases |
| `--run-id` | Attach to an existing run root instead of creating one |

### Cardiac only

| Flag / variable | Effect |
|-----------------|--------|
| `--no-hpo-study`, `--no-feature-selection-study`, `--skip-studies` | Skip stages 5 and 6 |
| `--study-mode`, `--fs-jobs`, `--hpo-search-n-jobs`, `--hpo-model-n-jobs` | Study parallelism |
| `--parallel-studies` / `--no-parallel-studies` | Run stages 5 and 6 concurrently |
| `--parallel-experiments` / `--no-parallel-experiments` | Run stages 9-11 concurrently |
| `--max-cores`, `--cpu-fraction` | Core budget |
| `COMPARE_EXCLUDE_MODEL_TYPES` (env var, **not** a flag) | Space-separated families dropped from the stage-12 comparison. Not the same as `--model-types`: those families still ran and their results stay on disk. Use it when a family's grid is only partly complete, so its finished cells are a cost-biased subsample that cannot be ranked fairly against families that completed every cell. |

`sweep` (stage 11) is resumable and dispatches cheap cells first, so an
interrupted sweep restarts where it stopped rather than from the beginning.

### Dermatology only

| Flag | Effect |
|------|--------|
| `--explain` / `--no-explain` | Saliency stage. Usually 70-80% of wall-clock on a cached-feature run, so it is the first thing to drop for a fast iteration. |
| `--cache-frozen-features` / `--no-cache-frozen-features` | Reuse the frozen backbone features between runs instead of recomputing them |
| `--pretrained` / `--no-pretrained` | Backbone weights |
| `--augmentation` / `--no-augmentation` | Training augmentation |
| `--device`, `--epochs`, `--batch-size` | Training scope |
| `--figures` / `--no-figures`, `--group-views` / `--no-group-views` | Output artifacts |
| `--no-recommendations` | Skip stage 3 |

Explain precedence: `--explain` / `--no-explain` beats `RUN_EXPLAIN`, which beats
`xai.enabled` in the config. Both orchestrators take the same toggles.

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
