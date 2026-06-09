# FairXAI Roadmap

Current status reference for implemented, partial, and deferred work. For
design rationale, see [../architecture/decisions.md](../architecture/decisions.md).

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

## Partially Implemented / Constrained

| Item | Status | Notes |
|------|--------|-------|
| Dermatology pipeline | Scaffolded only | Cardiac is the only complete research pipeline. Dermatology scripts/config placeholders should not be presented as production-ready. |
| Counterfactual explanations | Frozen placeholder | `counterfactual_stub` remains explicit because reliable counterfactual generation was deferred. |
| GPU paths | Environment-specific | XGBoost CUDA and cuML hooks require compatible HPC/CUDA setup. Local CPU fallback remains normal. |
| Historical recommendation defaults | Needs accumulated evidence | Recommendation history can fall back to literature defaults when prior runs are sparse. |
| Clustering evidence | Implemented, exploratory | Current clusters support subgroup interpretation and diagnostics, not strong fairness claims by themselves. |

## Deferred

- Full dermatology end-to-end pipeline.
- HPC-specific cuML SVM RBF configuration.
- Interactive recommendation-confirmation UI/TUI.
- Broader domain-specific recommendation rules such as temporal drift detection.
- Formal docs site tooling such as MkDocs; current docs are Markdown-only by design.

## Integration Status

| Component | Status |
|-----------|--------|
| Cardiac bash pipeline | Active |
| Cardiac Prefect flow | Active local/orchestration alternative |
| WebApp characterization CLI | Active via `fairxai-characterize` |
| WebApp JSON adapters | Active in `fairxai.integration` |
| HPC deployment | Configured for manual environment setup; run validation remains environment-dependent |
