# Scripts

Entry points for FairXAI pipelines, studies, and experiment stages.

See [../docs/README.md](../docs/README.md) for the full docs index and
[../docs/guides/cheat-sheet.md](../docs/guides/cheat-sheet.md) for compact commands.

## Layout

```text
scripts/
├── common/       # Domain-agnostic stage implementations
├── cardiac/      # Cardiac bash orchestrator and thin wrappers
├── dermatology/  # Dermatology bash orchestrator and stage wrappers
├── experiments/  # Attribute binning, mitigation, combinatorial, comparison
├── studies/      # HPO, feature selection, selector contract, grouping, calibration, plots
├── thesis/       # Chapter 6 numbers ledger, figures, similarity, performance intervals
└── utils/        # Cohort building, provenance, overlap, run archiving (see utils/README.md)
```

`scripts/thesis/` reads finished runs only. Set the run IDs in `scripts/thesis/runs.yaml`,
then run `performance_intervals.py` (minutes; the ledger reads its CSVs), `build_ledger.py`,
`make_figures.py` and `similarity_heldout.py`. Each writes under `output/thesis/` by default;
pass `--out` to write elsewhere.

Both domains run end to end. Cardiac carries the quantitative fairness claims;
dermatology (images, PAD-UFES-20) adds explain and image mitigation stages.

## Cardiac Stage Order

| # | Stage | Main script |
|---|-------|-------------|
| 1 | `load` | `scripts/cardiac/load_data.py` |
| 2 | `profile` | `scripts/cardiac/profile_data.py` |
| 3 | `recommend` | `scripts/cardiac/generate_recommendations.py` |
| 4 | `preprocess` | `scripts/cardiac/preprocess.py` |
| 5 | `tune` | `scripts/studies/run_hpo.py` |
| 6 | `select_features` | `scripts/studies/run_feature_selection_study.py` |
| 7 | `train` | `scripts/cardiac/train_baseline.py` |
| 8 | `assess` | `scripts/cardiac/assess_predictions.py` |
| 9 | `bin_attributes` | `scripts/experiments/run_attribute_binning_analysis.py` |
| 10 | `mitigate` | `scripts/cardiac/mitigation.py` |
| 11 | `sweep` | `scripts/cardiac/combinatorial.py` |
| 12 | `compare` | `scripts/cardiac/compare.py`, `scripts/studies/run_grouping_analysis.py`, `scripts/studies/generate_dissertation_plots.py` |

Names come from `fairxai.pipeline.stages.STAGES`. Pre-rename names (`hpo_study`,
`feature_selection_study`, `attribute_binning`, `mitigation`, `combinatorial`)
still resolve as aliases; see the
[cheat sheet](../docs/guides/cheat-sheet.md#pre-rename-aliases).

Grouping currently runs during stage 12 and does not have its own checkpointed
stage marker.

Most `scripts/cardiac/` files are thin wrappers that call the shared
implementation in `scripts/common/` with the cardiac pipeline config.

### Optional Cardiac Steps

These steps are off by default. Each one is analysis-only, except clustering,
which adds a column to the splits.

| Step | Script | Runs | Enable | Output |
|------|--------|------|--------|--------|
| Subgroup clustering | `scripts/cardiac/cluster_subgroups.py` | before stage 7 | `RUN_GROUPING=1` or `grouping.enabled` | `group_cluster` in the splits; `runs/<run_id>/grouping_pretrain/<dataset>/` |
| SHAP status report | `scripts/common/report_shap_status.py` | after stage 7 | always | one report over every `shap_status.json`; `--strict` exits 1 on any fallback |
| Individual fairness | `scripts/cardiac/similarity_analysis.py` | after stage 8 | `RUN_SIMILARITY=1` or `similarity.enabled` | `runs/<run_id>/baseline/individual_fairness/` |
| Mitigation intervals | `scripts/common/assess_mitigated_predictions.py` | after stage 10 | `RUN_MITIGATION_INTERVALS=1` or `fairness.uncertainty.enabled` | `runs/<run_id>/experiments/mitigation/prediction_fairness/` |
| Age-binning sensitivity | `scripts/cardiac/age_binning_analysis.py` | after stage 10 | `RUN_AGE_BINNING=1` or `age_binning_sensitivity.enabled` | `runs/<run_id>/baseline/age_binning_sensitivity/` |

## Dermatology Stage Order

`scripts/dermatology/dermatology_pipeline.sh` runs these stages. There are no
stages 5 and 6; numbering follows `fairxai.pipeline.stages.DERMATOLOGY_STAGES`.

| # | Stage | Script |
|---|-------|--------|
| 1 | `load` | `scripts/dermatology/load_data.py` |
| 2 | `profile` | `scripts/dermatology/profile_data.py` |
| 3 | `recommend` | `scripts/dermatology/generate_recommendations.py` |
| 4 | `preprocess` | `scripts/dermatology/preprocess.py` |
| 7 | `train` | `scripts/dermatology/train_baseline.py` |
| 8 | `assess` | `scripts/dermatology/assess_predictions.py` |
| 9 | `compare` | `scripts/dermatology/compare.py` |
| 10 | `explain` | `scripts/dermatology/explain.py` |
| 11 | `mitigate` | `scripts/dermatology/mitigate.py` |

Outputs land in `output/dermatology/runs/<run_id>/`.

## Shared Implementations (`scripts/common/`)

| Script | Purpose |
|--------|---------|
| `load_data.py`, `profile_data.py`, `generate_recommendations.py`, `preprocess_data.py`, `train_baseline.py`, `assess_predictions.py` | Domain-agnostic stage 1–4, 7 and 8 bodies behind the cardiac wrappers |
| `assess_mitigated_predictions.py` | Bootstrap fairness intervals for the stage-10 mitigation arms |
| `report_shap_status.py` | One report over every `shap_status.json` a run wrote |
| `export_stage_registry.py` | Emits a domain stage catalog from `fairxai.pipeline.stages` as Bash declarations |
| `stage_registry.sh` | Sourced by both orchestrators; loads the catalog through `export_stage_registry.py` |

## Orchestrators

```bash
# Bash pipeline, preferred for HPC/ad-hoc shell runs
bash scripts/cardiac/cardiac_pipeline.sh

# Prefect flow, useful for local orchestration/observability
python3 flows/cardiac_pipeline.py
```

Common flags:

- `--datasets <name> [name ...]`
- `--model-types <type> [type ...]`
- `--resume-from <stage>`
- `--go-until <stage>`
- `--run-id <id>`
- `-v` / `-vv`

Dataset/model precedence is CLI flags, selector contract where applicable,
pipeline config, then defaults/auto-discovery.

## Studies

| Script | Purpose | Output |
|--------|---------|--------|
| `studies/run_hpo.py` | Hyperparameter optimization | `output/cardiac/studies/hpo/` |
| `studies/run_feature_selection_study.py` | Sensitive-attribute feature ablation | `output/cardiac/studies/feature_selection/` |
| `studies/build_selector_contract.py` | Converts study outputs into downstream selection hints | `output/cardiac/runs/<run_id>/recommendations/selector_contract.json` |
| `studies/run_grouping_analysis.py` | Clustering and similarity subgroup discovery | `output/cardiac/studies/grouping/` and run-linked grouping outputs |
| `studies/generate_dissertation_plots.py` | Batch dissertation figures | `output/cardiac/studies/dissertation_figures/<run_id>/` |
| `studies/run_bootstrap_calibration.py` | Null-calibration study for the fairness bootstrap | stdout; CSVs only with `--output <file>` |
| `studies/run_profiling_sensitivity_study.py` | Profiling metric sensitivity on synthetic datasets | `output/synthetic/studies/profiling_sensitivity/<study_id>/` |
| `studies/generate_profiling_sensitivity_plots.py` | Plots for the profiling sensitivity study | `<study_id>/figures/` |

## Thesis

| Script | Purpose | Output |
|--------|---------|--------|
| `thesis/runs.yaml` | Run IDs every thesis script reads | — |
| `thesis/thesis_runs.py` | Loads `runs.yaml`; shared run IDs and paths | — |
| `thesis/performance_intervals.py` | Bootstrap intervals for baseline performance and paired feature-selection ablation | `baseline_performance.csv`, `ablation_paired.csv` |
| `thesis/build_ledger.py` | Chapter 6 numbers ledger | `chapter6_ledger.md` |
| `thesis/make_figures.py` | Chapter 6 and 7 figures | `figures/` |
| `thesis/similarity_heldout.py` | Held-out k-NN consistency with pairwise bootstrap | `similarity_heldout.json` |

All thesis outputs land under `output/thesis/` unless `--out` says otherwise.

## Experiments

| Script | Purpose |
|--------|---------|
| `experiments/run_attribute_binning_analysis.py` | Age/attribute binning strategy sweep |
| `experiments/run_mitigation_comparison.py` | Mitigation comparison implementation used by wrappers |
| `experiments/run_combinatorial_experiments.py` | Full dataset x binning x mitigation x model matrix |
| `experiments/run_experiment_comparison.py` | Cross-experiment canonical comparison tables and plots |
| `experiments/_gates.py` | Shared recall/fairness gate helpers |

## XAI Outputs

Baseline and combinatorial scripts write SHAP/LIME summaries when XAI is enabled
in `configs/pipelines/cardiac.yaml` and `configs/experiments/combinatorial.yaml`.

Typical baseline layout:

```text
output/cardiac/runs/<run_id>/baseline/xai/<dataset>/
├── holdout/
│   ├── shap/summary.csv
│   └── lime/examples.csv
└── cv/
    ├── shap/summary.csv
    └── lime/tracked.csv
```

## Logs

Run logs mirror pipeline stages:

```text
logs/cardiac/runs/<run_id>/
├── 01_load/
├── 02_profile/
└── run_summary.json
```

`latest_run` and `latest_run.txt` pointers exist under both `output/cardiac/`
and `logs/cardiac/`.

## Generated Outputs

| Output | Path |
|--------|------|
| Run root | `output/cardiac/runs/<run_id>/` |
| Baseline | `output/cardiac/runs/<run_id>/baseline/` |
| Recommendations | `output/cardiac/runs/<run_id>/recommendations/` |
| Experiments | `output/cardiac/runs/<run_id>/experiments/` |
| Comparison tables | `output/cardiac/runs/<run_id>/experiments/comparisons/data/` |
| Dissertation figures | `output/cardiac/studies/dissertation_figures/<run_id>/` |

## Related Docs

- Pipeline controls: [../docs/architecture/pipeline-flow-control.md](../docs/architecture/pipeline-flow-control.md)
- Results schema: [../docs/reference/results-schema.md](../docs/reference/results-schema.md)
- Plots: [../docs/reference/plots.md](../docs/reference/plots.md)
- Testing: [../docs/guides/testing.md](../docs/guides/testing.md)
