# FairXAI Roadmap

Single source of truth for what is implemented, what is deferred, and what is planned.
For architectural decisions behind specific choices, see `docs/DECISIONS.md`.

---

## Done (cardiac pipeline)

### Data & Preprocessing
- Data loading + schema harmonization (cleveland, kaggle_heart, cardio70k)
- 26 age-binning variants across 5 method families: fixed, quantile, equal_width, jenks, adaptive_quantile
- Clinical constraints validation — drops/flags physiologically impossible rows per `configs/domain/cardiac.yaml`
- Feature selection modes (6 modes) — sensitive-attribute visibility control for fairness research

### Profiling & Recommendations
- 12 complexity metrics (F2–F4, N2–N4, Raug, L1–L3, T1, BayesImbalance) + EBM difficulty score
- Pre-model fairness triage (6 recommendation categories: task framing, sensitive adequacy, representation, overlap/ambiguity, explainability, readiness)

### Training
- Four model types: Logistic Regression, Random Forest, SVM, XGBoost
- Baseline (holdout) + 5-fold cross-validated training
- Hyperparameter optimization via GridSearchCV / RandomizedSearchCV (`run_hpo.py`)
- GPU acceleration: XGBoost CUDA (`device='cuda'`), cuML Random Forest (`use_gpu=True`)

### Fairness
- Group fairness metrics: demographic parity, equalized odds, equal opportunity, predictive parity
- Calibration by group (Expected Calibration Error)
- Individual fairness (k-NN consistency)
- Pre-processing mitigation: reweighting, SMOTE, ADASYN
- In-processing mitigation: ExponentiatedGradient, GridSearchReduction (fairlearn)
- Post-processing mitigation: ThresholdOptimizer (fairlearn)
- Multi-stage mitigation combos (pre+in, pre+post, pre+in+post)

### Experiments
- Combinatorial experiment runner: dataset × binning × mitigation × model × training_method
- Age-binning sensitivity analysis (`run_attribute_binning_analysis.py`)
- Mitigation comparison (`run_mitigation_comparison.py`)
- Cross-experiment comparison with Pareto frontier (`run_experiment_comparison.py`)
- Feature selection study (6 modes, small datasets only)

### XAI
- SHAP: TreeExplainer (RF, XGBoost), LinearExplainer (LR, SVM linear) — holdout + CV
- LIME: holdout examples + near-threshold CV tracking
- Feature importance export for all 4 model types

### Integration
- WebApp CLI entry point: `fairxai-characterize` (`src/fairxai/cli/characterize.py`)
- JSON output schema matched to WebApp's expected format (jobId, metrics, pca2d)

### Visualization (implemented)
- Distribution plots: categorical/numeric distributions, target-by-group, missingness, outliers
- Comparison plots: correlation heatmaps, PCA scatter, drift heatmap
- Experiment plots: comparison heatmap, trade-off scatter, Pareto frontier

---

## Deferred / Not Yet Implemented

| Item | Status | Notes |
|------|--------|-------|
| Dermatology pipeline | Data acquisition scaffolded; cardiac-only for now | Start after cardiac results finalized |
| cuML SVM RBF | HPC-only path | 4 GB VRAM insufficient on PC1; add as HPC-specific config later |
| Viz: fairness plots (`fairness.py`) | Scaffolded — raise `NotImplementedError` | `plot_fairness_metric_heatmap`, `plot_group_performance_gaps`, `plot_bias_amplification_waterfall` |
| Viz: transformation plots (`transformations.py`) | Scaffolded — raise `NotImplementedError` | `plot_transformation_impact`, `plot_before_after_distributions`, `plot_scaling_effects` |
| Counterfactual explanations | Frozen per advisor direction | Stub exists in `explainability/tabular.py`; too manual to implement reliably |
| Clustering-based subgroup discovery | Designed, not wired | `configs/experiments/clustering.yaml` exists; implementation deferred |
| Similarity-based group discovery | Planned alongside clustering | No implementation yet |
| Dataset registry | Placeholder at `configs/datasets/registry.yaml` | Not wired into code |

---

## Integration Status

| Component | Status |
|-----------|--------|
| WebApp ↔ FairXAI CLI | Active (`fairxai_integration` branch on WebApp) |
| HPC (Pleiades) | Manual setup documented; run yet to be tested |
| WebApp Docker mount | FairXAI mounted at `/app/fairxai`; `pip install -e .[experiment]` at startup |
