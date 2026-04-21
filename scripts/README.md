# Scripts

Entry points for running FairXAI pipelines and experiments.

## Structure

```
scripts/
├── common/                 # Pipeline-agnostic runners (train_baseline, assess_predictions, …)
├── cardiac/                # Cardiac orchestrator + thin wrappers (add --pipeline cardiac)
├── dermatology/            # Dermatology wrappers (TODO)
├── experiments/            # Pipeline experiment stages (called by flows/cardiac_pipeline.py)
│   ├── run_attribute_binning_analysis.py  # Stage 9: sweep 26+ age-binning strategies
│   ├── run_mitigation_comparison.py       # Stage 10: pre/in/post-processing mitigation
│   ├── run_combinatorial_experiments.py   # Stage 11: full experiment matrix
│   ├── run_experiment_comparison.py       # Stage 12: Pareto frontier + cross-experiment plots
│   └── _gates.py                          # Shared recall-gate utilities
└── studies/                # Studies-first stages + standalone study utilities
    ├── run_hpo.py                         # Stage 5: hyperparameter optimisation study
    ├── run_feature_selection_study.py     # Stage 6: sensitive-attribute ablation study
    ├── build_selector_contract.py         # Wiring helper: studies -> execution contract
    ├── run_grouping_analysis.py           # Clustering + similarity subgroup discovery
    └── generate_dissertation_plots.py     # Batch-generate dissertation figures from a run
```

## Run order (cardiac)

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `load_data.py` | Download / locate raw CSVs |
| 2 | `profile_data.py` | Generate profiling JSONs (complexity, imbalance, …) |
| 3 | `generate_recommendations.py` | Pre-model fairness triage (see `src/fairxai/recommendations/README.md`) |
| 4 | `preprocess.py` | Clean, encode, bin — all binning variants for combinatorial |
| 5 | `studies/run_hpo.py` | Hyperparameter optimisation study |
| 6 | `studies/run_feature_selection_study.py` | Sensitive-attribute ablation study |
| 7 | `train_baseline.py` | Train baseline models |
| 8 | `assess_predictions.py` | Compute fairness metrics on baselines |
| 9 | `experiments/run_attribute_binning_analysis.py` | Age-binning sensitivity analysis (26+ strategies) |
| 10 | `experiments/run_mitigation_comparison.py` | Pre-/in-/post-processing mitigation comparison |
| 11 | `experiments/run_combinatorial_experiments.py` | Full experiment matrix |
| 12 | `experiments/run_experiment_comparison.py` | Cross-experiment comparison + Pareto frontier |

Stage 3 is controlled by the `RUN_RECOMMENDATIONS` env var (default `true`).

## Explainability (XAI) configuration

XAI settings live in the pipeline config (`configs/pipelines/cardiac.yaml`) and
the combinatorial config (`configs/experiments/combinatorial.yaml`):

**Pipeline config** (`cardiac.yaml`):

```yaml
xai:
  enabled: true              # Master switch for all XAI (SHAP + LIME)
  cv_enabled: true           # Enable cross-validated XAI
  lime_instances: 3          # Holdout LIME: number of instances to explain
  cv_lime_instances: 3       # CV LIME: near-threshold instances to track
  global_max_samples: 1000   # SHAP background subsample cap
```

**Combinatorial config** (`combinatorial.yaml`):

```yaml
xai:
  enabled: true
  max_samples: 200
  lime_instances: 2
```

**XAI output layout** (baseline training):

```
xai/
└── {dataset}/
    ├── holdout/
    │   ├── shap/
    │   │   └── summary.csv       # mean, std, p25/p50/p75 per feature
    │   └── lime/
    │       └── examples.csv      # instance-level LIME weights
    └── cv/
        ├── shap/
        │   └── summary.csv       # aggregated across all CV folds
        └── lime/
            └── tracked.csv       # near-threshold instance explanations
```

**Combinatorial experiment output layout**:

All combinatorial outputs use `holdout/` and `cv/` sub-folders for
predictions, results, manifests, and XAI:

```
{experiment_root}/
├── predictions/{dataset}/
│   ├── holdout/predictions_{exp_id}.csv
│   └── cv/predictions_{exp_id}.csv
├── results/{dataset}/
│   ├── holdout/results_{exp_id}.json
│   └── cv/results_{exp_id}.json
├── manifests/{dataset}/
│   ├── holdout/experiment_{exp_id}.yaml
│   └── cv/experiment_{exp_id}.yaml
└── xai/{dataset}/holdout/
    ├── shap/{exp_id}_global.csv
    ├── shap/{exp_id}_local.csv
    ├── shap/global_summary.csv
    ├── shap/local_summary.csv
    └── lime/{exp_id}_examples.csv
```

CV LIME tracks **near-threshold** predictions (probability within ±0.1 of the
decision boundary) — these are the instances where explanations matter most
clinically.

## Verbosity

All scripts accept a `-v` / `-vv` flag (stacks with `action='count'`):

| Level | Flag | Console output |
|-------|------|----------------|
| 0 (default) | — | `[PHASE]`/`[SUCCESS]` markers + WARNING and above |
| 1 | `-v` | All INFO+ messages |
| 2 | `-vv` | All DEBUG+ messages |

File logs always capture **DEBUG+** regardless of verbosity.  
Dedicated `*_warnings.log` and `*_errors.log` files are always written alongside the main log.

### Log directory layout

When a `run_id` is active, logs are written to numbered phase directories that
mirror the pipeline stages:

```text
logs/cardiac/
├── latest_run → runs/<run_id>
└── runs/<run_id>/
    ├── 01_load/
    │   ├── load.log
    │   ├── load_warnings.log
    │   └── load_errors.log
    ├── 02_profile/
    │   ├── profile.log
    │   ├── profile_warnings.log
    │   └── profile_errors.log
    ├── …
    └── run_summary.json        # auto-generated at end of pipeline
```

Standalone (no `run_id`) runs still write to the flat `logs/cardiac/` directory.

**Bash pipeline** — set `VERBOSE=0`, `1`, or `2` (legacy `true`/`false` still accepted):

```bash
VERBOSE=2 bash scripts/cardiac/cardiac_pipeline.sh   # debug output
```

**Prefect flow:**

```bash
python3 flows/cardiac_pipeline.py -vv   # debug
python3 flows/cardiac_pipeline.py -v    # info
python3 flows/cardiac_pipeline.py       # quiet (default)
```

## Dataset and model scope overrides

Both orchestrators accept CLI scope overrides:

- `--datasets <d1> [d2 ...]`
- `--model-types <m1> [m2 ...]`

Precedence is: CLI flags > selector contract (when provided) > config > defaults/auto-discovery.

Examples:

```bash
bash scripts/cardiac/cardiac_pipeline.sh \
  --datasets cleveland \
  --model-types logistic_regression svm

python3 flows/cardiac_pipeline.py \
  --datasets cleveland \
  --model-types logistic_regression svm
```

## Outputs

Pipeline outputs go to `output/<pipeline>/runs/<run_id>/` — `RUN_ID` is always required.
Study outputs go to `output/<pipeline>/studies/<study_type>/` — standalone, not run-scoped.

```
output/cardiac/
├── runs/
│   └── run_<timestamp>_<pid>_<uuid>/
│       ├── profiling/
│       ├── recommendations/{selector_contract.json, ...}
│       ├── baseline/{models/, results/predictions/, prediction_fairness/}
│       └── experiments/{attribute_binning/, mitigation/, comparisons/data/, comparisons/plots/, …}
├── studies/
│   ├── hpo/                         # flat: best_params_*.json + latest.txt
│   ├── feature_selection/
│   │   └── <study_id>/{study_summary.json, runs/fs_<mode>__<model>/baseline/}
│   ├── grouping/
│   │   └── <study_id>/{cluster_assignments.csv, …}
│   └── dissertation_figures/
│       └── <run_id>/{fairness/, transformations/, cross_model/}
├── latest_run -> runs/<latest>
├── latest_run.txt
└── run_history.jsonl
```

## Utility scripts

Two helper scripts live at the **project root** (not inside `scripts/`):

- **`setup.sh`** — bootstraps the virtual environment, checks Python ≥ 3.10, installs `requirements.txt`.
- **`cleanup.sh`** — removes generated outputs (`output/`, `data/processed/`, `data/raw/`, `logs/`). Flags: `--output-only`, `--keep-latest`, `--nuke-env`, `--dry-run`, `-y`.

## Studies

Studies can run standalone, and two studies are also first-class pipeline stages in the cardiac flow.
The default cardiac orchestration executes HPO and feature-selection studies before baseline/experiment stages.

| Script | Purpose | Output |
|--------|---------|--------|
| `studies/run_hpo.py` | Stage 5 (or standalone): hyperparameter optimisation (GridSearchCV/RandomizedSearchCV). Auto-loaded by downstream runners. | `output/<pipeline>/studies/hpo/best_params_{dataset}_{model}.json` |
| `studies/run_feature_selection_study.py` | Stage 6 (or standalone): sensitive-attribute ablation study. | `output/<pipeline>/studies/feature_selection/<study_id>/` |
| `studies/build_selector_contract.py` | Internal wiring helper used by orchestrators before baseline/combinatorial. Emits recommendations derived from study outputs. | `output/<pipeline>/runs/<run_id>/recommendations/selector_contract.json` |
| `studies/run_grouping_analysis.py` | Clustering + similarity subgroup discovery. Writes `group_cluster` back to processed CSVs. | `output/<pipeline>/studies/grouping/<study_id>/` |
| `studies/generate_dissertation_plots.py` | Batch-generate dissertation figures from a completed pipeline run. | `output/<pipeline>/studies/dissertation_figures/<run_id>/` |

### HPO workflow

HPO can be run standalone, but the cardiac orchestrators now execute it by default before baseline and combinatorial stages:

```bash
python scripts/studies/run_hpo.py --pipeline cardiac --datasets cleveland
# → writes output/cardiac/studies/hpo/best_params_cleveland_<model>.json

# Main pipeline auto-loads HPO params when they exist:
python flows/cardiac_pipeline.py --datasets cleveland
```

HPO uses `n_jobs=-1` (it is the only process running). The combinatorial sweep then uses the
HPO-found params as model defaults, with hardware overrides (`device`, `n_jobs`) re-applied on
top so HPO cannot clobber GPU or threading settings.

## Experiments

Pipeline stages called automatically by `flows/cardiac_pipeline.py` and `scripts/cardiac/cardiac_pipeline.sh`.

| Script | Purpose |
|--------|---------|
| `experiments/run_attribute_binning_analysis.py` | Stage 9: age-binning sweep (26+ strategies across 5 families). |
| `experiments/run_mitigation_comparison.py` | Stage 10: focused mitigation strategy comparison. |
| `experiments/run_combinatorial_experiments.py` | Stage 11: full experiment matrix. Auto-loads HPO params and can consume `--selector-contract` for model-type recommendations. |
| `experiments/run_experiment_comparison.py` | Stage 12: cross-experiment comparison (Pareto frontier, trade-off scatter). |

### Feature selection mode

`train_baseline.py` accepts `--feature-selection-mode`, `--rfe-top-k`, and `--selector-contract`:

```bash
python scripts/common/train_baseline.py --pipeline cardiac --feature-selection-mode include_all_sensitive
python scripts/common/train_baseline.py --pipeline cardiac --feature-selection-mode rfe_top_k --rfe-top-k 10
python scripts/common/train_baseline.py --pipeline cardiac --selector-contract output/cardiac/runs/<run_id>/recommendations/selector_contract.json
```

See `src/fairxai/data/README.md` for all mode descriptions.

## Notes

- `RUN_ID` should be a single value for the whole run to keep outputs grouped.
- **Dermatology pipeline**: data acquisition scaffolded; pipeline not yet implemented. Cardiac only.
