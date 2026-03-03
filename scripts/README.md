# Scripts

Entry points for running FairXAI pipelines and experiments.

## Structure

```
scripts/
├── common/                 # Pipeline-agnostic runners
├── cardiac/                # Cardiac wrappers (add --pipeline cardiac)
├── dermatology/            # Dermatology wrappers (TODO)
└── experiments/            # Experiment runners (age binning, mitigation, combinatorial, comparison)
```

## Run order (cardiac)

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `load_data.py` | Download / locate raw CSVs |
| 2 | `profile_data.py` | Generate profiling JSONs (complexity, imbalance, …) |
| 3 | `generate_recommendations.py` | Pre-model fairness triage (see `src/fairxai/recommendations/README.md`) |
| 4 | `preprocess_data.py` | Clean, encode, bin — all binning variants for combinatorial |
| 5 | `train_baseline.py` | Train baseline models |
| 6 | `assess_baseline_fairness.py` | Compute fairness metrics on baselines |
| 7 | `age_binning_analysis.py` | Age-binning sensitivity analysis |
| 8 | `mitigation_comparison.py` | Pre-/in-/post-processing mitigation comparison |
| 9 | `run_combinatorial.py` | Combinatorial experiment matrix |
| 10 | `compare_experiments.py` | Cross-experiment comparison |

Phase 3 is controlled by the `RUN_RECOMMENDATIONS` env var (default `true`).

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
| 0 (default) | — | `[PHASE]`/`[SUCCESS]`/`[ERROR]` tags + WARNING and above |
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

## Outputs

All outputs are written under `output/<pipeline>/runs/<run_id>/` when `RUN_ID` is set. If not set, outputs go to the default pipeline folders under `output/<pipeline>/`.

## Utility scripts

Two helper scripts live at the **project root** (not inside `scripts/`):

- **`setup.sh`** — bootstraps the virtual environment, checks Python ≥ 3.10, installs `requirements.txt`.
- **`cleanup.sh`** — removes generated outputs (`output/`, `data/processed/`, `data/raw/`, `logs/`). Flags: `--output-only`, `--keep-latest`, `--nuke-env`, `--dry-run`, `-y`.

## Notes

- `RUN_ID` should be a single value for the whole run to keep outputs grouped.
- Dermatology pipeline runners are TODO.
