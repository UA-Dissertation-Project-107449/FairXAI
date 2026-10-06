# Scripts

Entry points for FairXAI pipelines, studies, and experiment stages.

See [../docs/README.md](../docs/README.md) for the full docs index and
[../docs/guides/cheat-sheet.md](../docs/guides/cheat-sheet.md) for compact commands.

## Layout

```text
scripts/
├── common/       # Domain-agnostic stage implementations
├── cardiac/      # Cardiac bash orchestrator and thin wrappers
├── dermatology/  # Scaffolded future-domain wrapper area
├── experiments/  # Attribute binning, mitigation, combinatorial, comparison
├── studies/      # HPO, feature selection, selector contract, grouping, dissertation plots
└── thesis/       # Chapter 6 numbers ledger, figures, similarity, performance intervals
```

`scripts/thesis/` reads finished runs only. Set the run IDs in `scripts/thesis/runs.yaml`,
then run `performance_intervals.py` (minutes; the ledger reads its CSVs), `build_ledger.py`,
`make_figures.py` and `similarity_heldout.py`. Each writes under `output/thesis/` by default;
pass `--out` to write elsewhere.

Dermatology is scaffolded only. Cardiac is the active end-to-end pipeline.

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
