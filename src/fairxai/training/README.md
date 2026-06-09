# Training Module

Hyperparameter optimization helpers used before baseline and combinatorial
experiment stages.

## Files

| File | Purpose |
|------|---------|
| `grid_search.py` | GridSearchCV/RandomizedSearchCV wrappers, save/load HPO params |
| `__init__.py` | Public exports |

## Public API

- `run_hpo`

`grid_search.py` also contains `save_hpo_results` and `load_hpo_params` for
script-level use.

## Config And Artifacts

- HPO config: `configs/experiments/hpo.yaml`
- Model defaults: `configs/models/*.yaml`
- HPO outputs: `output/cardiac/studies/hpo/best_params_<dataset>_<model>.json`

HPO runs before baseline/combinatorial stages in the current cardiac pipeline.
Downstream scripts reload best params and then re-apply runtime hardware/job
settings.

## Usage

```python
from fairxai.training import run_hpo

results = run_hpo(
    model_type="logistic_regression",
    X_train=X_train,
    y_train=y_train,
    param_grid={"C": [0.1, 1.0]},
)
```

## Related

- Models: [../models/README.md](../models/README.md)
- Scripts: [../../../scripts/README.md](../../../scripts/README.md)
