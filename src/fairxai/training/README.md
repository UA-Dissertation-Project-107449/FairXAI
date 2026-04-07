# Training Module

Hyperparameter optimization (HPO) utilities for FairXAI model training.

This module provides grid and randomized search wrappers that are orthogonal
to fairlearn's mitigation-level `GridSearchReduction`. HPO tunes model
hyperparameters before the combinatorial mitigation sweep runs.

## Files

| File | Purpose |
|------|---------|
| `grid_search.py` | `run_hpo()` and `load_hpo_params()` — sklearn GridSearchCV / RandomizedSearchCV wrappers |
| `__init__.py` | Public re-exports |

## Key Functions

### `run_hpo(model_type, X_train, y_train, param_grid, ...)`

Runs grid or randomized search and returns the best hyperparameter dict.

```python
from fairxai.training.grid_search import run_hpo

best_params = run_hpo(
    model_type="random_forest",
    X_train=X_train,
    y_train=y_train,
    param_grid={"n_estimators": [100, 200, 300], "max_depth": [8, 14, None]},
    cv=5,
    scoring="f1",
    search="random",   # "grid" or "random"
    n_iter=20,
    n_jobs=-1,
)
# → {"n_estimators": 200, "max_depth": 14}
```

**Scoring note**: `scoring='f1'` optimizes the HPO search, but a hard recall floor
is enforced post-search by the calling script (via `min_recall` gate in
`configs/recommendations/thresholds.yaml`).

### `load_hpo_params(hpo_dir, dataset, model_type)`

Loads pre-computed best params from disk. Returns `{}` if the file does not exist
(so callers can safely call it unconditionally — missing file = no HPO override).

```python
from fairxai.training.grid_search import load_hpo_params

params = load_hpo_params(
    hpo_dir=Path("output/cardiac/hpo"),
    dataset="cleveland",
    model_type="logistic_regression",
)
# → {"C": 0.1} if file exists, {} otherwise
```

## Output Format

HPO writes one JSON file per (dataset, model_type) combination:

```
output/<pipeline>/hpo/
├── best_params_cleveland_logistic_regression.json
├── best_params_cleveland_random_forest.json
├── best_params_cleveland_svm.json
├── best_params_cleveland_xgboost.json
└── best_params_kaggle_heart_logistic_regression.json
```

File contents: flat dict of hyperparameter name → best value, e.g.:
```json
{"C": 0.1, "penalty": "l2"}
```

## Integration with Combinatorial Runner

`scripts/experiments/run_combinatorial_experiments.py` auto-loads HPO params if
`output/<pipeline>/hpo/` exists at runtime — no CLI flag needed:

1. At startup, checks if `hpo_output_dir` exists.
2. If yes, logs a notice and passes `hpo_dir` to `_resolve_model_variants()`.
3. `_resolve_model_variants()` merges HPO params into `base_params`, then
   re-applies hardware overrides (`device`, `n_jobs`) so HPO cannot clobber them.

## HPO Config

Search grids and settings live in `configs/experiments/hpo.yaml`:
- `scoring`: objective metric (default `f1`)
- `cv_folds`: cross-validation folds during search (default 5)
- `n_iter_random`: iterations for RandomizedSearchCV (default 20)
- `max_rows_for_rbf_svm`: skip RBF SVM grid above this row count (default 5000)
- `grids.<model_type>.search`: `"grid"` or `"random"`
- `grids.<model_type>.params`: parameter space

## Usage

```bash
# Run HPO first (uses n_jobs=-1, runs alone)
python scripts/experiments/run_hpo.py --pipeline cardiac

# Then run combinatorial — picks up HPO params automatically
python scripts/experiments/run_combinatorial_experiments.py --pipeline cardiac
```
