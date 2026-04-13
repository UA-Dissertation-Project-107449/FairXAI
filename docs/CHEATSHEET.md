# FairXAI Cheat Sheet

Quick reference for the pipeline, tooling, and repo layout.
For detailed docs see the links at the end of each section.

---

## Install Profiles

| Profile | Command | Use Case |
|---------|---------|----------|
| Core only | `pip install -e .` | Minimal (CI import checks) |
| Experiment | `pip install -e ".[experiment]"` | HPC / full pipeline |
| Dev | `pip install -e ".[dev]"` | Local work (adds viz, notebooks, tests, ruff) |
| Full | `pip install -e ".[full]"` | Everything except GPU |
| GPU (HPC only) | `pip install cuml-cu12==25.2.1` | After `module load cuda/12.4.0` |

---

## Pipeline Stages

| # | Name | Sub-step | Script |
|---|------|----------|--------|
| 1 | `load` | — | `scripts/cardiac/load_data.py` |
| 2 | `profile` | — | `scripts/common/profile_data.py` |
| 3 | `recommend` | — | `scripts/common/generate_recommendations.py` |
| 4 | `preprocess` | — | `scripts/common/preprocess_data.py` |
| 4b | — | grouping | `scripts/experiments/run_grouping_analysis.py` |
| 5 | `train` | — | `scripts/cardiac/train_baseline.py` |
| 6 | `assess` | — | `scripts/cardiac/assess_predictions.py` |
| 7 | `attribute_binning` | — | `scripts/experiments/run_attribute_binning_analysis.py` |
| 8 | `mitigation` | — | `scripts/cardiac/mitigation.py` |
| 9 | `combinatorial` | — | `scripts/experiments/run_combinatorial_experiments.py` |
| 10 | `compare` | — | `scripts/experiments/run_experiment_comparison.py` |

Stage 4b (grouping/clustering) runs automatically after preprocess when `RUN_GROUPING=true`.  
Stages 7–10 each have an independent feature toggle (`RUN_ATTRIBUTE_BINNING`, `RUN_MITIGATION`, etc.).

### Common Runs

```bash
# Full pipeline
bash scripts/cardiac/cardiac_pipeline.sh

# Stages 1–3 only (profiling + triage, no training)
GO_UNTIL=recommend bash scripts/cardiac/cardiac_pipeline.sh

# Core pipeline without experiments (stages 1–6)
GO_UNTIL=assess bash scripts/cardiac/cardiac_pipeline.sh

# Single dataset, specific models
bash scripts/cardiac/cardiac_pipeline.sh \
  --datasets cleveland \
  --model-types logistic_regression xgboost

# Resume from training on latest run
RESUME_FROM=train bash scripts/cardiac/cardiac_pipeline.sh

# Resume, explicit run ID, stop before experiments
RESUME_FROM=train GO_UNTIL=assess \
  RUN_ID=run_20260224_143000_abc bash scripts/cardiac/cardiac_pipeline.sh

# Skip grouping and experiments, run only stages 1–6
RUN_GROUPING=false RUN_ATTRIBUTE_BINNING=false \
  RUN_MITIGATION=false RUN_COMBINATORIAL=false \
  GO_UNTIL=assess bash scripts/cardiac/cardiac_pipeline.sh
```

→ See `docs/pipeline-flow-control.md` for all flags, stage aliases, checkpoint details.

---

## Prefect Flow (alternative orchestrator)

```bash
# Same semantics as bash pipeline — identical flags
python flows/cardiac_pipeline.py
python flows/cardiac_pipeline.py --go-until recommend
python flows/cardiac_pipeline.py --resume-from train --datasets cleveland
python flows/cardiac_pipeline.py --help
```

`flows/cardiac_pipeline.py` wraps the same `scripts/` via subprocess.
Prefer the bash pipeline for HPC and ad-hoc runs; Prefect adds observability/retry.

---

## Source Layout (`src/fairxai/`)

| Module | Role |
|--------|------|
| `cli/` | Script runner helpers, logging setup, run pointer utilities |
| `data/` | Loaders, schema harmonization, preprocessing, profilers |
| `experiments/` | Versioning, attribute binning analysis, experiment I/O |
| `explainability/` | SHAP / LIME wrappers |
| `fairness/` | Fairness metrics + mitigation techniques |
| `clustering/` | K-Means / Hierarchical / DBSCAN / GMM + per-cluster fairness |
| `similarity/` | k-NN fairness consistency + violation density maps |
| `models/` | Baseline model wrapper + CV training |
| `notebook_utils/` | Notebook-facing convenience helpers |
| `pipeline/` | Stage definitions, checkpointing, flow-control helpers |
| `profiling/` | Dataset complexity metrics + EBM difficulty prediction |
| `recommendations/` | Fairness triage engine (rules → recommendations) |
| `utils/` | Config loading, logging setup |
| `viz/` | Plots: distributions, comparisons, experiment results, fairness |

→ See `docs/modules.md` for dependency graph and maturity notes.

---

## Scripts Layout

```
scripts/
├── common/              # Domain-agnostic stage scripts (used by cardiac + future domains)
│   ├── load_data.py
│   ├── profile_data.py
│   ├── generate_recommendations.py
│   ├── preprocess_data.py
│   ├── train_baseline.py
│   └── assess_predictions.py
├── cardiac/             # Cardiac-specific orchestration and wrappers
│   ├── cardiac_pipeline.sh   ← main entry point
│   ├── load_data.py          ← calls scripts/common/load_data.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── assess_predictions.py
│   ├── mitigation.py
│   ├── compare.py
│   ├── combinatorial.py
│   ├── profile_data.py
│   └── generate_recommendations.py
├── experiments/         # Optional experiment analysis scripts
│   ├── run_grouping_analysis.py       # Stage 4b: clustering + similarity
│   ├── run_attribute_binning_analysis.py
│   ├── run_combinatorial_experiments.py
│   ├── run_experiment_comparison.py
│   ├── run_mitigation_comparison.py
│   ├── run_feature_selection_study.py
│   └── run_hpo.py
└── generate_dissertation_plots.py     # Batch-generate all dissertation figures
```

---

## Tests

```
tests/
├── conftest.py          # Shared fixtures (synthetic data, tmp run dirs, fairness dicts)
├── unit/                # Fast, no pipeline, no real data (<5s total)
│   ├── test_clustering_engine.py
│   ├── test_clustering_fairness.py
│   ├── test_combinatorial_runner.py
│   ├── test_experiment_comparison.py
│   ├── test_fairness_metrics.py
│   ├── test_gpu_detection.py
│   ├── test_model_wrapper_edge_cases.py
│   ├── test_pipeline_flag_recognition.py
│   ├── test_similarity_fairness.py
│   └── test_viz_smoke.py
└── integration/         # Run real scripts on synthetic/temp data
    ├── test_clustering_pipeline.py
    ├── test_combinatorial_multi_model.py
    ├── test_compare_multi_model.py
    └── test_multi_model_baseline.py    # [slow] subprocess — ~5 min
```

```bash
# Fast unit tests only
pytest tests/unit/ -q

# All tests, skip slow subprocess test
pytest tests/ -m "not slow"

# Everything
pytest tests/

# Viz smoke tests require matplotlib — skip if not installed
pytest tests/unit/ --ignore=tests/unit/test_viz_smoke.py
```

→ See `docs/TESTING.md` for fixture reference and what is/isn't tested.

---

## CI Checks

Two workflows in `.github/workflows/`:

| Check | PR (`pr-quick-ci.yml`) | Main (`main-validation.yml`) |
|-------|-----------------------|------------------------------|
| Install | `.[experiment]` | `.[experiment]` |
| `black --check` | ✓ | ✓ |
| `isort --check-only` | ✓ | ✓ |
| `ruff check` | ✓ | ✓ |
| Build wheel + sdist | ✓ | ✓ |
| CLI smoke (`--help`) | ✓ | — |
| Unit tests (no viz) | ✓ (skip viz) | — |
| Unit tests (full) | — | ✓ (matplotlib installed) |
| Characterize smoke run | — | ✓ |
| JSON contract validation | — | ✓ |
| Artifact upload | — | ✓ |

### Running checks locally

```bash
# Format (apply)
black src scripts tests flows
isort src scripts tests flows

# Format (check only — same as CI)
black --check src scripts tests
isort --check-only src scripts tests

# Lint
ruff check src scripts

# All CI checks in one shot
black --check src scripts && isort --check-only src scripts && ruff check src scripts && pytest tests/unit/ -q
```

---

## Outputs

```
output/cardiac/runs/<run_id>/
├── .checkpoints/              # Stage completion markers (JSON)
├── raw/                       # Loaded + standardized datasets
├── profiling/                 # Complexity metrics + EBM scores
├── recommendations/           # Triage recommendations per dataset
├── processed/                 # Split + scaled CSVs (all binnings)
├── baseline/
│   ├── models/                # Trained model .pkl files (one per model type)
│   ├── predictions/           # Prediction CSVs (one per model type)
│   └── fairness/              # Stage-6 fairness assessment JSONs
├── grouping/                  # Cluster assignments + per-cluster fairness
└── experiments/full/
    ├── attribute_binning/
    ├── mitigation/
    ├── comparisons/           # full_comparison.csv, cross_model_summary.csv
    │   └── pareto_<dataset>_<model>.png
    └── dissertation_figures/  # Output of generate_dissertation_plots.py
```

Logs mirror the same structure under `logs/cardiac/runs/<run_id>/`.
