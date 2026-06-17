# Architecture And Design Decisions

This note captures decisions that affect how the repository is organized and
how generated artifacts should be interpreted.

## Repository Shape

| Path | Decision |
|------|----------|
| `src/fairxai/` | Reusable package code. Scripts should call into this layer instead of duplicating logic. |
| `scripts/` | Operational entry points for pipeline stages, studies, and experiments. |
| `flows/` | Prefect orchestration around the same scripts used by the bash pipeline. |
| `configs/` | Declarative runtime settings for pipeline, domain, model, profiling, recommendation, and experiment behavior. |
| `docs/` | Architecture, guides, references, research notes, and planning docs. |
| `data/` | External/raw/processed datasets. Generated data is not package source. |
| `output/` | Run-scoped and study-scoped generated artifacts. |
| `logs/` | Run logs, warning logs, error logs, and run summaries. |
| `notebooks/` | Exploration and presentation support. Durable logic belongs in `src/` or `scripts/`. |

## Workflow

1. Scripts are the primary execution unit.
2. Bash and Prefect orchestrators coordinate scripts and checkpoint stages.
3. Notebooks consume generated artifacts and should not be the only home for research logic.
4. Docs follow code/config truth when drift appears.

## Processed Data Layout

Processed train/test splits live in per-dataset subdirectories:

```text
data/processed/<pipeline>/<dataset>_<binning>/<dataset>_train.csv
data/processed/<pipeline>/<dataset>_<binning>/<dataset>_test.csv
data/processed/<pipeline>/<dataset>_<binning>/<dataset>_train_scaled.csv
data/processed/<pipeline>/<dataset>_<binning>/<dataset>_test_scaled.csv
```

Decision points:

- `runtime.default_binning` in `configs/pipelines/cardiac.yaml` is the canonical default.
- HPO, training, mitigation, grouping, and dissertation plots should use shared dataset resolvers in `fairxai.experiments.data_io`.
- Flat processed files directly under `data/processed/cardiac/` are legacy and should not be used for current pipeline behavior.
- Combinatorial experiments deliberately address explicit `<dataset>_<binning>/` directories because they sweep binning strategies.

## Run And Study Outputs

- Run-scoped artifacts live under `output/cardiac/runs/<run_id>/`.
- Study artifacts live under `output/cardiac/studies/<study_type>/`.
- Latest-run pointers live at `output/cardiac/latest_run` and `output/cardiac/latest_run.txt`.
- Logs mirror run IDs under `logs/cardiac/runs/<run_id>/`.

## Known Artifacts And Limits

### Logistic Regression Perfect Training Scores On Balanced Cleveland

Some mitigation configurations on the small Cleveland dataset can yield perfect
training metrics after resampling. This is treated as an overfit/stability
artifact, not as primary evidence. Dissertation framing should prefer selected
mitigations and test-set tradeoffs over these sensitivity rows.

### DBSCAN Sensitivity On Cleveland

Cleveland is small and moderately high-dimensional for density clustering.
DBSCAN requires a wider `eps` search than the initial grid. If DBSCAN remains
weak, the selected clustering solution should be described as exploratory
subgroup evidence rather than strong natural phenotype discovery.

### Dermatology Scope

Dermatology has scaffolding, but the end-to-end implemented research pipeline
is cardiac. Docs should avoid implying equivalent pipeline maturity.

## Dermatology Design Notes

### Image XAI Two-Layer Design

Image explainability (`explainability/image.py`) is split into two layers on
purpose:

- **Pure heatmap functions** (`gradcam_heatmap`, `lime_heatmap`, `shap_heatmap`)
  take a model + tensor and return a normalized `[0,1]` saliency array. No file
  I/O, no checkpoint loading, no sampling — trivial to unit-test and to reuse
  outside the pipeline.
- **Driver** (`select_images`, `explain_image_model`) owns the side effects:
  checkpoint loading, stratified group × outcome sampling, overlay rendering, and
  the `manifest.csv`.

Rationale: the heatmap math is the defensible methods-chapter contribution and
must be testable without GPU/checkpoints; the orchestration is pipeline glue.
Keeping them separate also lets the assessment/figures stages consume the
heatmap fns without inheriting the driver's I/O assumptions.

### Image Fairness Is Post-Prediction Only (No Retrain)

Dermatology fairness (`fairness/image_assessment.py`) scores from a saved
predictions CSV, never from model weights. Post-hoc **group views** (alternate
subgroup definitions, including one intersectional `sex_x_fitzpatrick`) are a CSV
`groupby`, not a retraining multiplier — "5 binnings" cost 5× a groupby, not 5×
training. Support gates (`min_group_samples=50`, intersectional `=30`) drop
undersized groups from metrics while reporting them as skipped, so small
subgroups never silently inflate a fairness delta. Mitigation for images, if
added, is post-processing only (`ThresholdOptimizer` on saved probabilities);
pre/in-processing for CNNs is explicitly out of scope.

## Related

- Module map: [modules.md](modules.md)
- Pipeline flow control: [pipeline-flow-control.md](pipeline-flow-control.md)
- Roadmap: [../planning/roadmap.md](../planning/roadmap.md)
