# FairXAI Module Architecture

This document maps `src/fairxai/` to responsibilities and dependency flow.
Each first-level package has a local `README.md` with file-level details.

## Module Overview

| Module | Purpose | Main Public Surface |
|--------|---------|---------------------|
| `cli` | Shared script bootstrapping, logging setup, run IDs, latest-run pointers | `resolve_run_id`, `get_run_root`, `setup_phase_logging` |
| `data` | Cardiac data loading, schema harmonization, preprocessing, profiling input generation | `loaders`, `preprocessors`, `profilers`, `schemas` |
| `experiments` | Attribute-binning analysis, experiment versioning, shared experiment data I/O | `create_binning_strategy`, `apply_binning`, `ExperimentVersioning` |
| `explainability` | Tabular SHAP/LIME wrappers and counterfactual placeholder | `shap_explain_tabular`, `lime_explain_instance` |
| `fairness` | Group/calibration/individual fairness metrics and mitigation engines | `FairnessMetrics`, `MitigationEngine` |
| `models` | Model registry, wrappers, baseline trainer, CV trainer | `MODEL_REGISTRY`, `get_model_class`, `CVTrainer` |
| `training` | Hyperparameter optimization helpers | `run_hpo` |
| `profiling` | Complexity metrics, profiling config, WebApp-compatible characterization | `compute_complexity_metrics`, `characterize_dataset` |
| `recommendations` | Pre-model fairness triage engine | `RecommendationEngine`, `TriageReport` |
| `pipeline` | Stage registry, aliases, checkpoint markers, resume validation | `STAGES`, `get_stage_range`, `mark_stage_complete` |
| `comparison` | Canonical experiment evidence tables and plotting frame helpers | `write_canonical_comparison_outputs`, `build_metric_plot_frame` |
| `clustering` | Latent subgroup discovery and per-cluster fairness diagnostics | `ClusteringEngine`, `ClusterProfiler`, `FairnessPerCluster` |
| `similarity` | k-NN individual fairness and violation-density mapping | `SimilarityEngine`, `ViolationDensityMapper` |
| `integration` | WebApp-oriented JSON adapters around characterization, binning, clustering | `characterize_dataset`, `run_binning`, `run_clustering` |
| `notebook_utils` | Notebook path, schema, loading, palette, and labeling helpers | `resolve_root_dir`, `load_raw_datasets`, label helpers |
| `utils` | Shared config, logging, warning/error capture, accelerator detection | `load_yaml_config`, `setup_logging`, `detect_accelerator` |
| `viz` | Plotting functions for EDA, fairness, clustering, dissertation figures | `fairness_comparison`, `distributions`, `clustering`, `transformations` exports |

## Dependency Direction

- `utils` sits at the bottom: config, logging, accelerator detection.
- `data` feeds `profiling`, `recommendations`, `models`, `fairness`, and experiments.
- `profiling` and `recommendations` run before model training.
- `models`, `training`, and `fairness` feed baseline and experiment scripts.
- `comparison`, `viz`, `clustering`, and `similarity` produce dissertation evidence and subgroup diagnostics.
- `pipeline` is orchestration metadata only; scripts and flows consume it.
- `integration` wraps stable internals for WebApp-facing JSON payloads.

## Status Snapshot

- Active cardiac pipeline: all 12 stages in `fairxai.pipeline.stages.STAGES`.
- Active model types: logistic regression, random forest, SVM, XGBoost.
- Active subgroup tooling: clustering and similarity modules exist and are tested.
- Active visualization modules: `distributions`, `comparisons`, `fairness`, `transformations`, `fairness_comparison`, and `clustering`.
- Scaffolded future domain: dermatology has scripts/config placeholders, but cardiac remains the only complete pipeline.

## Documentation Standard

See [../guides/style-guide.md](../guides/style-guide.md). Module READMEs should name purpose, files, public API, config/artifact dependencies, usage, and related tests.
