# Configs

Configuration files for pipelines, datasets, and experiments.

## Structure

```
configs/
├── datasets/               # Dataset registry (PLANNED -- not yet wired into code)
├── domain/                 # Domain metadata: feature maps, clinical constraints, sex mappings
├── experiments/            # Experiment configs: binning, mitigation, combinatorial, baseline
├── models/                 # One file per model type -- authoritative source for hyperparameters
├── pipelines/              # Pipeline runtime settings (paths, XAI, split)
├── profiling/              # Profiling tunables (complexity metrics, random seed, solver)
├── recommendations/        # Fairness triage thresholds
└── schema/                 # Unified JSON schema definitions (do not modify format)
```

## Usage

- **Pipeline runners** load `pipelines/<name>.yaml` for paths, model type list, and XAI settings.
- **Model hyperparameters** live in `models/<type>.yaml`. Runners load these at runtime;
  experiment configs only define variant overrides on top.
- **Experiment configs** (`experiments/`) are self-contained: each one can be run independently.
  They reference `domain/cardiac.yaml` as a base but do not inherit from each other.
- **Domain configs** (`domain/`) define dataset paths, feature mappings, clinical constraints,
  and sex/age normalisation constants.
- **Profiling tunables** (max samples, random seed, solver) live in `profiling/complexity.yaml`.
- **Fairness thresholds** (min_recall, max_fairness_violation) are the canonical source in
  `recommendations/thresholds.yaml`; experiment configs duplicate them intentionally so each
  config is independently runnable.

## Model configs (`models/`)

| File | Class | Notes |
|------|-------|-------|
| `logistic_regression.yaml` | `sklearn.linear_model.LogisticRegression` | Used as baseline for mitigation sweep |
| `random_forest.yaml` | `sklearn.ensemble.RandomForestClassifier` | Scale-invariant; uses SHAP TreeExplainer |
| `svm.yaml` | `sklearn.svm.SVC` | Requires standard scaling; SHAP skipped by default |
| `xgboost.yaml` | `xgboost.XGBClassifier` | `device` injected at runtime from accelerator setting |

## Experiment configs (`experiments/`)

| File | Status | Purpose |
|------|--------|---------|
| `baseline.yaml` | Active | Single LR baseline experiment |
| `age_binning.yaml` | Active | Attribute binning strategy sweep (22 strategies) |
| `mitigation.yaml` | Active | Fairness mitigation techniques comparison |
| `combinatorial.yaml` | Active | Full cross-product sweep (dataset x binning x mitigation x model) |
| `clustering.yaml` | **DEFERRED** | Subgroup discovery via clustering (design complete, not implemented) |

## Notes

- `schema/` is not modified by config refactors -- format is fixed for WebApp compatibility.
- Dermatology configs are TBD (registry.yaml placeholder exists under `datasets/`).
