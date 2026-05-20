# Architecture & Design Decisions

## Project Structure

### Directory Organization

- **`src/fairxai/`**: Core Python package
  - `data/`: Data loading, preprocessing, validation
  - `models/`: Model architectures and base classes
  - `fairness/`: Fairness metrics, analysis, and mitigation algorithms
  - `explainability/`: Explainability and interpretability methods
  - `utils/`: Helper functions and utilities

- **`scripts/`**: Standalone Python scripts for running experiments
  - Run main code, can be called from bash or notebooks
  - Better for production-ready code and batch processing

- **`notebooks/`**: Jupyter notebooks for exploration and visualization
  - `runs/`: Output notebooks from script runs (for documentation)
  - Parallel to scripts, not replacing them

- **`experiments/`**: Experiment tracking and configuration
  - `configs/`: YAML/JSON configs for reproducibility
  - `baseline/`, `mitigation/`, `explainability/`: Experiment-specific outputs

- **`data/`**: Dataset management (gitignored for large files)
  - `raw/`: Original datasets
  - `processed/`: Cleaned/preprocessed data
  - `external/`: External reference data

- **`models/`**: Trained model checkpoints (gitignored)

- **`output/`**: All outputs and analysis results
  - `fairness/`, `explainability/`: Metric results by domain
  - `plots/`: Generated visualizations
  - `reports/`: Summary reports

- **`logs/`**: Application logs from script execution

## Workflow

1. **Scripts** (primary): Python scripts in `scripts/` handle core logic
2. **Bash**: Orchestrate scripts, run pipelines
3. **Notebooks** (secondary): Show results, explore interactively, document findings

## Key Principles

- Keep `src/` clean and modular for reusability
- Use `experiments/` for tracking different approaches
- All outputs (results, models, logs) are explicit and organized
- Large files are gitignored to keep repo lightweight

## Processed Data Layout

**Decision**: Processed train/test splits live in per-dataset subdirectories:
`data/processed/<pipeline>/{dataset}_{binning}/{dataset}_{train,test}{,_scaled}.csv`.

- The canonical binning is `runtime.default_binning` in the pipeline config
  (cardiac: `fixed_10yr`). It is the single source of truth — HPO, training,
  and mitigation all read it from there rather than hardcoding.
- Loaders fall back to `data/processed/<pipeline>/{dataset}/` (no binning suffix)
  for preprocess runs executed without `--all-binnings` / `--binning-strategy`.
- The legacy **flat** layout (`data/processed/<pipeline>/{dataset}_train.csv` etc.
  directly under the pipeline dir) is **retired**. It was never written by the
  current `preprocess_data.py` (which always uses subdirs); stale flat files left
  over from an older layout silently masked the bug — `cleveland` resolved via
  leftovers while `kaggle_heart` (no leftovers) was invisible to the train and
  mitigation stages.

**Rationale**: HPO (`run_hpo.py`) and combinatorial experiments
(`run_combinatorial_experiments.py`) already used the `{dataset}_{binning}/`
convention. `train_baseline.py` and `run_mitigation_comparison.py` were the
outliers reading flat paths. Unifying on the subdir layout makes every dataset
listed in `runtime.datasets` flow through all stages identically.

**Shared resolver**: `fairxai.experiments.data_io` exposes `resolve_dataset_dir`
(canonical `{dataset}_{binning}/` → `{dataset}/` fallback) and
`resolve_default_binning` (reads `runtime.default_binning`). All loaders —
training, mitigation, HPO, grouping, and dissertation plots — go through these,
so the layout and the default binning live in exactly one place. Combinatorial
experiments deliberately bypass the resolver: they iterate binning strategies
explicitly, so `{dataset}_{binning}/` is always addressed directly.

---

## Known Artefacts & Documented Limitations

### LR Perfect-Training-Score on Balanced Cleveland (Mitigation Stage)

**Symptom**: 4 `[OVERFIT-RISK]` log lines appear during the mitigation stage for
LogisticRegression when SMOTE / ROS / RUS / ADASYN are applied to Cleveland:
`train_auc_roc=1.0000 train_f1=1.0000 train_accuracy=1.0000`.

**Root cause**: Cleveland has ~188–224 samples after balancing (13 clean clinical features).
The resampled training set is small enough and clean enough that L2-regularised LR with
C=0.1 still achieves perfect linear separation.  The baseline LR (no resampling) is
unaffected (train_f1≈0.97, overfit_risk=low in `overfit_gap_table.csv`).

**Why it does not matter for the dissertation**:
- The primary selected mitigation is class reweighting (not SMOTE/ROS/RUS/ADASYN).
- Reweighting leaves the training distribution intact and does not trigger perfect separation.
- The four affected combinations are secondary sensitivity checks, not primary results.

**What was tried**: Tightening C from 1.0 to 0.1 (commit in `logistic_regression.yaml`).
The baseline improved; the resampling-stage warnings persisted.  Reducing C further
(e.g. 0.001) risks degrading test performance on the real distribution.

**Dissertation note**: Acknowledge in the "model stability" section that LR is sensitive
to small balanced samples and that resampling mitigations should be evaluated on larger
datasets or with non-linear models to avoid this artefact.

### DBSCAN Finds 0 Clusters on Cleveland

**Symptom**: All 12 DBSCAN configurations in `clustering.yaml` produced 0 clusters + 295
noise points from 303 samples (logged in `cluster_diagnostics.csv`).

**Root cause**: Cleveland has 13 features.  After StandardScaler the expected L2 distance
between two random samples is ≈ √(13 × 2) ≈ 5.1 (curse of dimensionality).  The original
eps grid [0.3, 0.5, 0.7, 1.0] was far below this threshold.

**Fix applied**: Extended eps grid to [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] in
`configs/experiments/clustering.yaml`.  Larger eps values should recover density structure.
If DBSCAN still underperforms after the grid expansion, the silhouette-based winner
(currently hierarchical k=3, silhouette=0.156) remains the selected solution.
