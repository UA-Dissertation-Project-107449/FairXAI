# Testing Guide

Testing focuses on fast unit coverage for reusable logic plus integration tests
that exercise script behavior on synthetic data.

## Structure

```text
tests/
├── conftest.py
├── fixtures/
├── unit/
│   ├── test_attribute_binning.py
│   ├── test_clustering_engine.py
│   ├── test_clustering_fairness.py
│   ├── test_combinatorial_runner.py
│   ├── test_dermatology_pipeline_units.py
│   ├── test_dissertation_studies.py
│   ├── test_experiment_comparison.py
│   ├── test_fairness_metrics.py
│   ├── test_gpu_detection.py
│   ├── test_hpo_loader_and_threshold.py
│   ├── test_memory_utils.py
│   ├── test_mitigation_auc.py
│   ├── test_model_wrapper_edge_cases.py
│   ├── test_pipeline_flag_recognition.py
│   ├── test_similarity_fairness.py
│   └── test_viz_smoke.py
└── integration/
    ├── test_clustering_pipeline.py
    ├── test_combinatorial_multi_model.py
    ├── test_compare_multi_model.py
    └── test_multi_model_baseline.py
```

## Commands

```bash
cd Code/FairXAI

# Fast unit tests
python3 -m pytest tests/unit/ -q

# Unit and integration tests without slow subprocess cases
python3 -m pytest tests/ -m "not slow"

# All tests
python3 -m pytest tests/

# Pipeline flag sanity check
python3 -m pytest tests/unit/test_pipeline_flag_recognition.py -q

# Viz smoke tests only
python3 -m pytest tests/unit/test_viz_smoke.py -q
```

## Marks

| Mark | Meaning |
|------|---------|
| `slow` | Real subprocess/pipeline-style tests |
| `xgboost_model` | Tests that include XGBoost paths |
| `integration` | Full-script or cross-module integration coverage |

Marks are registered in `pyproject.toml`.

## CI Checks

CI runs:

```bash
black --check src scripts tests flows
isort --check-only src scripts tests flows
ruff check src scripts tests flows
pytest tests/unit/ -q
python3 -m build
```

Main validation also performs a characterization smoke run and validates the
WebApp-facing JSON contract.

## What Is Covered

- Pipeline stage name/alias recognition.
- HPO loading and threshold behavior.
- Attribute binning and mitigation filtering logic.
- Fairness metric formulas and AUC handling.
- Clustering engine, clustering fairness, and clustering pipeline integration.
- Similarity fairness calculations.
- Model wrapper edge cases.
- Comparison table generation, baseline matching, per-group deltas, and cross-model summaries.
- Visualization smoke checks when plotting dependencies are installed.

## Intentional Gaps

- Full cardiac combinatorial sweep: too expensive for routine tests.
- SHAP/LIME numerical correctness: generated during full research runs, not unit tests.
- Full HPC GPU validation: environment-specific.

## Related

- CI workflows: `.github/workflows/`
- Style guide: [style-guide.md](style-guide.md)
- Pipeline controls: [../architecture/pipeline-flow-control.md](../architecture/pipeline-flow-control.md)
