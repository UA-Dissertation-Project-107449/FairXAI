# Metric Notes (Profiling)

This document records the profiling metric behavior implemented in FairXAI.

Current objective: keep one implementation path aligned with
Domain_characterization formulas while preserving FairXAI packaging and WebApp
JSON contract.

## Scope

- `src/fairxai/profiling/complexity.py`
- `src/fairxai/profiling/domain_characterization.py`
- `configs/profiling/complexity.yaml`

## Contract Stability

The WebApp-facing output shape is unchanged:

- count fields: `nSamples`, `nFeatures`, `nClasses`
- complexity aliases: `F2Imbalance..BayesImbalance`
- `ebmDifficulty` from fixed `EBM_FEATURE_ORDER`
- clipping to `[0, 1]` remains in `domain_characterization.py`

## Metric Behavior Mapping

All metrics are implemented with Domain_characterization-equivalent logic:

- `F2`, `F3`: low-cardinality feature filtering (`unique > 3`)
- `F4`: binary-feature exclusion style (`unique > 2`) before overlap pruning
- `L1`, `L2`, `L3`: `LinearSVC`-based definitions and synthetic sampling pattern
- `N2`, `N3`, `N4`: nearest-neighbor implementations matching legacy behavior
- `T1`: loop-based hypersphere pruning with early-stop heuristic
- `Raug`: thresholded opposite-neighbor counting (`delta=2`) and imbalance weighting
- `BayesImbalance`: neighborhood BI-style calculation

## Runtime Parameters

`complexity.yaml` now keeps only runtime parameters, not implementation modes:

- `raug_k`
- `raug_delta`
- `bayes_k`
- `bayes_search_depth`
- `linear_svc_max_iter`
- `random_seed`

## Index Column Handling

When `characterize_dataset(..., index_column=...)` is provided:

- index column is excluded from feature matrix used for metrics and PCA
- target handling is unchanged
- if index equals target, target takes precedence

## Reproducibility

Use fixed seed and fixed environment versions when comparing with historical
results.

Recommended procedure:

1. Run characterization on canonical datasets.
2. Compare per-metric values against Domain_characterization reports.
3. Investigate outliers larger than practical tolerance.

## Numerical Caveats

Small differences may still appear due to:

- scikit-learn version differences
- linear solver convergence details
- floating-point ordering

These should be treated as numerical noise unless they materially affect
ranking/decision thresholds in downstream analysis.
