# Configs

Configuration files for FairXAI pipelines, model families, experiments, domain metadata, profiling, and recommendation thresholds.

See [../docs/README.md](../docs/README.md) for the full docs index.

## Structure

```text
configs/
├── datasets/          # Dataset registry placeholder and future domain registry
├── domain/            # Domain metadata, feature maps, constraints, labels
├── experiments/       # Experiment and study configs
├── models/            # One YAML file per model type
├── pipelines/         # Pipeline runtime settings
├── profiling/         # Complexity/profiling tunables
├── recommendations/   # Fairness triage thresholds
└── schema/            # WebApp-compatible schema JSON
```

## Runtime Use

- `pipelines/cardiac.yaml` controls cardiac datasets, paths, sensitive attributes, XAI, scheduling, and default binning.
- `models/*.yaml` are the authoritative model hyperparameter defaults.
- `experiments/*.yaml` configure HPO, feature selection, attribute binning, mitigation, combinatorial, comparison, and clustering/grouping studies.
- `domain/cardiac.yaml` contains clinical constraints, sex/age mappings, and domain labels.
- `profiling/complexity.yaml` configures complexity metric runtime behavior.
- `recommendations/thresholds.yaml` is the central triage/fairness threshold file.
- `schema/cardiac.json` supports standardized dataset metadata and WebApp-compatible ingestion.

## Experiment Configs

| File | Status | Purpose |
|------|--------|---------|
| `baseline.yaml` | Active | Baseline experiment defaults |
| `hpo.yaml` | Active | Grid/random search settings per model |
| `feature_selection_study.yaml` | Active | Sensitive-attribute ablation settings |
| `age_binning.yaml` | Active | Attribute/age binning strategy sweep |
| `mitigation.yaml` | Active | Fairness mitigation comparison |
| `combinatorial.yaml` | Active | Dataset x binning x mitigation x model experiment matrix |
| `comparison.yaml` | Active | Canonical comparison outputs and dissertation figures |
| `clustering.yaml` | Active exploratory | Clustering/grouping study settings |

## Model Configs

| File | Model |
|------|-------|
| `logistic_regression.yaml` | `sklearn.linear_model.LogisticRegression` |
| `random_forest.yaml` | `sklearn.ensemble.RandomForestClassifier`, optional cuML path |
| `svm.yaml` | `sklearn.svm.SVC` |
| `xgboost.yaml` | `xgboost.XGBClassifier`, optional CUDA device |

## Notes

- Config files should stay declarative. Runtime behavior belongs in `src/` or `scripts/`.
- `schema/` format should remain stable for WebApp compatibility.
- Dermatology is scaffolded but not an active end-to-end pipeline.
- Architecture and flow-control details live in [../docs/architecture/pipeline-flow-control.md](../docs/architecture/pipeline-flow-control.md).
