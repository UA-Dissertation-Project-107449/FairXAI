# Testing Guide

Testing focuses on fast unit coverage for reusable logic plus integration tests
that exercise script behavior on synthetic data.

## Layout

```text
tests/
├── conftest.py
├── fixtures/configs/   # Small pipeline configs used by script-level tests
├── unit/               # One file per module or stage behavior (grouped below)
└── integration/        # Script runs over synthetic data
```

Unit tests by area:

| Area | Files (`tests/unit/test_*.py`) |
|------|--------------------------------|
| Pipeline and run control | `pipeline_stages`, `pipeline_flag_recognition`, `run_manifest_guard`, `cardiac_run_axis_config`, `gpu_detection`, `memory_utils` |
| Data, cohorts and preprocessing | `build_cardiac_uci_cohorts`, `cardiac_preprocessor_split`, `fold_preprocessor`, `preprocess_age_bins`, `upload_profile`, `synthetic`, `profiling_sensitivity_runner`, `utils_archive_overlap` |
| Recommendations and tuning | `recommendations_data_quality`, `cli_triage`, `selector_contract_predeclare`, `hpo_loader_and_threshold`, `hyperparameter_parity`, `overfit_diagnostics` |
| Training and predictions | `model_wrapper_edge_cases`, `train_baseline_raw_meta`, `predictions_metadata`, `sweep_xai_model_resolution` |
| Fairness metrics and intervals | `fairness_metrics`, `fairness_uncertainty`, `paired_arm_effects`, `assess_predictions_decode`, `assess_predictions_uncertainty`, `assess_mitigated_predictions`, `attribute_binning`, `age_binning_sensitivity` |
| Mitigation and sweep | `mitigation_auc`, `mitigation_cv_protocol`, `mitigation_engine_model_types`, `mitigation_inprocessing_arms`, `mitigation_stage_multi_model`, `group_aware_resampling`, `constraint_attribute_axis`, `combinatorial_runner` |
| Explainability | `subgroup_shap`, `train_baseline_subgroup_shap`, `mitigation_subgroup_shap`, `shap_status` |
| Grouping and similarity | `clustering_engine`, `clustering_fairness`, `cluster_subgroups_output_scope`, `grouping_feature_alignment`, `grouping_pipeline`, `integration_clustering`, `similarity_fairness`, `similarity_pipeline` |
| Comparison and studies | `experiment_comparison`, `comparison_multi_model_evidence`, `dissertation_studies` |
| Dermatology and vision | `dermatology_pipeline_units`, `dermatology_readiness_figures`, `dermatology_fairness_figures`, `dermatology_comparison`, `image_assessment`, `image_explainability`, `image_mitigation`, `image_feature_mitigation`, `vision_transforms`, `vision_early_stopping`, `vision_frozen_batchnorm` |
| Visualization | `viz_smoke`, `viz_consistency` |

Integration tests: `test_clustering_pipeline.py`, `test_combinatorial_multi_model.py`,
`test_compare_multi_model.py`, `test_multi_model_baseline.py` and
`test_dermatology_feature_extraction.py`.

## Commands

```bash
cd Code/FairXAI

# Unit tests as PR CI runs them (laptop: -n 2; more workers can run out of memory)
python3 -m pytest tests/unit/ -q -n 2 -m "not local_data and not slow"

# Unit tests as main validation runs them (adds the slow bootstrap suites)
python3 -m pytest tests/unit/ -q -n 2 -m "not local_data"

# Everything, including integration and local-data tests
python3 -m pytest tests/ -n 2

# One file
python3 -m pytest tests/unit/test_pipeline_flag_recognition.py -q
```

`-n` needs `pytest-xdist`. `pyproject.toml` adds `-v --tb=short` to every run.

## Marks

| Mark | Meaning |
|------|---------|
| `slow` | Long tests: subprocess pipeline runs and the bootstrap suites (`test_fairness_uncertainty.py`, `test_paired_arm_effects.py`) |
| `local_data` | Needs gitignored datasets under `data/`; excluded from CI |
| `xgboost_model` | Includes XGBoost training paths |
| `integration` | Runs a full script through a subprocess |

Marks are registered in `pyproject.toml`. Mark a whole file with
`pytestmark = pytest.mark.slow` when every test in it is expensive.

## CI Checks

Both workflows in `.github/workflows/` lint, test and build:

```bash
black --check src scripts tests flows
isort --check-only src scripts tests flows
ruff check src scripts tests flows
python3 -m build
```

| Workflow | Tests |
|----------|-------|
| `pr-quick-ci.yml` (pull requests) | `pytest tests/unit/ -q -n auto -m "not local_data and not slow" --ignore=tests/unit/test_viz_smoke.py` |
| `main-validation.yml` (push to main) | `pytest tests/unit/ -q -n auto -m "not local_data"` |

Neither workflow runs `tests/integration/`. Main validation also performs a
characterization smoke run and validates the WebApp-facing JSON contract.

## Intentional Gaps

- Full cardiac combinatorial sweep: too expensive for routine tests.
- SHAP/LIME numerical correctness: generated during full research runs, not unit tests.
- Full HPC GPU validation: environment-specific.

## Related

- CI workflows: `.github/workflows/`
- Style guide: [style-guide.md](style-guide.md)
- Pipeline controls: [../architecture/pipeline-flow-control.md](../architecture/pipeline-flow-control.md)
