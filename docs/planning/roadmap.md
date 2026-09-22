# FairXAI Roadmap

Current status reference for implemented, partial, and deferred work. For
design rationale, see [../architecture/decisions.md](../architecture/decisions.md).

Last reconciled against the code: 2026-09-21.

## Implemented For Cardiac

- Data loading and schema harmonization for Cleveland, Kaggle Heart, and Cardio70k.
- Profiling with complexity metrics, group/intersection diagnostics, and EBM difficulty.
- Pre-model recommendation engine with task framing, sensitive adequacy, representation, overlap, explainability, and readiness checks.
- Preprocessing with clinical constraints, age binning, feature selection modes, train/test split, and scaling.
- HPO and feature-selection studies before baseline/experiment stages.
- Four model families: logistic regression, random forest, SVM, XGBoost.
- Baseline holdout and cross-validated training with SHAP/LIME exports where enabled.
- Group fairness, calibration, individual fairness, and mitigation comparisons.
- Attribute-binning, mitigation, combinatorial, comparison, grouping, and dissertation plot scripts.
- Clustering subgroup discovery and per-cluster fairness diagnostics.
- Similarity-based individual fairness and violation-density support.
- WebApp-facing characterization/binning/clustering adapters.
- CI checks for formatting, linting, unit tests, packaging, and characterization smoke validation.
- Statistical evidence layer: bootstrap confidence intervals for every group
  fairness scalar and parity gap (`fairness/uncertainty.py`), subgroup SHAP
  disparity (`explainability/subgroup.py`), and a null-calibration harness
  (`scripts/studies/run_bootstrap_calibration.py`). Every mitigation arm reports
  through it.
- Resumable combinatorial sweep that dispatches cheap cells first, and a
  stage-12 comparison that can exclude model families whose grid is incomplete.

## Implemented For Dermatology

- Full nine-stage pipeline (`load`, `profile`, `recommend`, `preprocess`,
  `train`, `assess`, `compare`, `explain`, `mitigate`), numbered to leave the
  cardiac-only stages 5-6 as a gap.
- Both orchestrators: `scripts/dermatology/dermatology_pipeline.sh` and
  `flows/dermatology_pipeline.py`, taking the same flags.
- Image baseline training with pretraining and augmentation toggles, a
  frozen-feature cache shared between runs, and a saliency stage that can be
  switched off (`--no-explain`), which is 70-80% of wall-clock on a cached run.
- Fairness assessment, model comparison and mitigation experiments over the
  image models, including group views.

## Partially Implemented / Constrained

| Item | Status | Notes |
|------|--------|-------|
| Dermatology evidence | Pipeline complete, evidence thinner | The stages all run end to end, but cardiac remains the pipeline the dissertation's quantitative claims rest on. Dermatology results are reported as a second domain, not as an independent replication. |
| Counterfactual explanations | Frozen placeholder | `counterfactual_stub` remains explicit because reliable counterfactual generation was deferred. |
| GPU paths | Environment-specific | XGBoost CUDA and cuML hooks require compatible HPC/CUDA setup. Local CPU fallback remains normal. |
| Historical recommendation defaults | Needs accumulated evidence | Recommendation history can fall back to literature defaults when prior runs are sparse. |
| Clustering evidence | Implemented, exploratory | Current clusters support subgroup interpretation and diagnostics, not strong fairness claims by themselves. |

## Deferred

- HPC-specific cuML SVM RBF configuration.
- Interactive recommendation-confirmation UI/TUI.
- Broader domain-specific recommendation rules such as temporal drift detection.
- Formal docs site tooling such as MkDocs; current docs are Markdown-only by design.

## Integration Status

| Component | Status |
|-----------|--------|
| Cardiac bash pipeline | Active |
| Cardiac Prefect flow | Active local/orchestration alternative |
| Dermatology bash pipeline | Active |
| Dermatology Prefect flow | Active local/orchestration alternative |
| WebApp characterization CLI | Active via `fairxai characterize` / `fairxai triage`; the `fairxai-characterize` shim was removed once the WebApp and HPC scripts migrated |
| WebApp JSON adapters | Active in `fairxai.integration` |
| HPC deployment | Configured for manual environment setup; run validation remains environment-dependent |
