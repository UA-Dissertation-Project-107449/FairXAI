# Configs

Configuration files for FairXAI pipelines, model families, experiments, domain metadata, profiling, and recommendation thresholds.

See [../docs/README.md](../docs/README.md) for the full docs index.

## Structure

```text
configs/
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
- `pipelines/dermatology.yaml` controls the dermatology dataset (`pad_ufes_20`), the patient-grouped split, image training, augmentation, XAI, and both stage-11 mitigation parts.
- `models/*.yaml` are the authoritative model hyperparameter defaults.
- `experiments/*.yaml` configure HPO, feature selection, attribute binning, mitigation, combinatorial, comparison, and clustering/grouping studies.
- `domain/cardiac.yaml` contains clinical constraints, sex/age mappings, and domain labels.
- `domain/<pipeline>_feature_map.yaml` maps source column names to canonical names; `scripts/common/load_data.py` passes it to the loader. Only the cardiac loader applies it; `DermatologyDataLoader` ignores `dermatology_feature_map.yaml`. For `cleveland_uci` and `four_site_uci` the numeric encoding comes from `scripts/utils/build_cardiac_uci_cohorts.py`, not from the cardiac map.
- `profiling/complexity.yaml` configures complexity metric runtime behavior.
- `recommendations/thresholds.yaml` is the central triage/fairness threshold file.
- `schema/cardiac.json` supports standardized dataset metadata and WebApp-compatible ingestion.
- `schema/dermatology.json` declares the image datasets (`pad_ufes_20`, `scin`) with their metadata files and standardizers. Only `pad_ufes_20` is in the default run.

## Experiment Configs

| File | Status | Purpose |
|------|--------|---------|
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
- Cardiac (`pipelines/cardiac.yaml`) and dermatology (`pipelines/dermatology.yaml`)
  both run end to end.
- Architecture and flow-control details live in [../docs/architecture/pipeline-flow-control.md](../docs/architecture/pipeline-flow-control.md).
