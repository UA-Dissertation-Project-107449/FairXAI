"""Hyperparameter optimisation (HPO) via sklearn GridSearchCV / RandomizedSearchCV.

This module is intentionally separate from fairness mitigation (fairlearn's
GridSearchReduction) and from the main combinatorial sweep.  The intended flow
is:

    1. Run :func:`run_hpo` once per model × dataset → best params stored as JSON.
    2. The combinatorial runner (or ``train_baseline.py``) loads those params
       as model defaults so every experiment benefits from tuned hyperparameters.

All searches use ``f1`` as the primary score.  The recall hard-floor is
checked post-search and logged as a warning when the best config is below it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Models that benefit from RandomizedSearchCV (large search space).
_RANDOM_SEARCH_MODELS = {"random_forest", "xgboost"}

# Maximum n_rows for which RBF SVM is tractable on consumer hardware.
_RBF_SVM_MAX_ROWS = 5_000


def _build_estimator(model_type: str, base_params: Dict[str, Any]):
    """Instantiate a fresh estimator for the given model type."""
    from fairxai.models import get_model_class

    cls = get_model_class(model_type)
    # Strip keys not accepted by this model class (e.g. device, use_gpu when cuml absent)
    import inspect

    sig = inspect.signature(cls.__init__).parameters
    safe_params = {k: v for k, v in base_params.items() if k in sig}
    return cls(**safe_params)


def _sklearn_estimator(model_type: str, base_params: Dict[str, Any]):
    """Return the underlying sklearn estimator wrapped inside the model class."""
    model = _build_estimator(model_type, base_params)
    # sklearn GridSearchCV needs the raw estimator, not our wrapper
    return model.model


def run_hpo(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List[Any]],
    base_params: Optional[Dict[str, Any]] = None,
    search: str = "grid",
    cv: int = 5,
    scoring: str = "f1",
    n_iter: int = 20,
    n_jobs: int = -1,
    recall_hard_floor: float = 0.60,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run hyperparameter optimisation for one model × dataset pair.

    Args:
        model_type: Registry key (e.g. ``'logistic_regression'``).
        X_train: Training features.
        y_train: Training labels.
        param_grid: Search space; keys must match the **sklearn** estimator's
            constructor parameters (not the FairXAI wrapper).
        base_params: Fixed params merged into the estimator before the search
            (e.g. ``{'random_state': 42, 'n_jobs': 1}``).
        search: ``'grid'`` for exhaustive :class:`~sklearn.model_selection.GridSearchCV`,
            ``'random'`` for :class:`~sklearn.model_selection.RandomizedSearchCV`.
        cv: Number of CV folds.
        scoring: sklearn scorer string (default ``'f1'``).
        n_iter: Number of parameter settings sampled when ``search='random'``.
        n_jobs: Parallelism for the search (``-1`` = all cores).
        recall_hard_floor: After search, warn if best CV recall < this value.
        random_state: Seed for ``RandomizedSearchCV``.

    Returns:
        ``{'best_params': dict, 'best_score': float, 'cv_results': dict,
        'model_type': str, 'n_train': int}``
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    base = dict(base_params or {})
    # Always set a fixed random_state for reproducibility
    base.setdefault("random_state", random_state)

    estimator = _sklearn_estimator(model_type, base)

    # Guard: skip RBF SVM on large datasets
    if model_type == "svm" and len(X_train) > _RBF_SVM_MAX_ROWS:
        filtered = {
            k: [v for v in vals if not (k == "kernel" and v == "rbf")]
            for k, vals in param_grid.items()
        }
        if filtered != param_grid:
            logger.warning(
                f"HPO: RBF kernel removed from SVM grid (n_train={len(X_train)} > "
                f"{_RBF_SVM_MAX_ROWS}). Only linear kernel will be searched."
            )
        param_grid = filtered

    if search == "random" or model_type in _RANDOM_SEARCH_MODELS:
        searcher = RandomizedSearchCV(
            estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=True,
        )
    else:
        searcher = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
        )

    logger.info(
        f"HPO [{model_type}]: {search} search, cv={cv}, scoring={scoring}, "
        f"n_train={len(X_train)}"
    )
    searcher.fit(X_train, y_train)

    best_params = dict(searcher.best_params_)
    best_score = float(searcher.best_score_)
    logger.info(f"HPO [{model_type}]: best_score={best_score:.4f}, best_params={best_params}")

    # Warn when best CV recall is below clinical floor (checked via refit scorer only)
    best_idx = searcher.best_index_
    cv_results = searcher.cv_results_
    recall_key = "mean_test_score" if scoring == "recall" else None
    if recall_key:
        best_recall = float(cv_results[recall_key][best_idx])
        if best_recall < recall_hard_floor:
            logger.warning(
                f"HPO [{model_type}]: best CV recall ({best_recall:.3f}) is below "
                f"recall_hard_floor ({recall_hard_floor}). Consider widening the grid."
            )

    return {
        "model_type": model_type,
        "best_params": best_params,
        "best_score": best_score,
        "n_train": int(len(X_train)),
        "scoring": scoring,
        "search": search,
    }


def save_hpo_results(results: Dict[str, Any], output_path: Path) -> None:
    """Persist HPO results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"HPO results saved: {output_path}")


def load_hpo_params(hpo_dir: Path, dataset: str, model_type: str) -> Optional[Dict[str, Any]]:
    """Load best params from a previous HPO run.

    Returns ``None`` if no HPO file exists for this dataset × model_type pair.
    """
    path = hpo_dir / f"best_params_{dataset}_{model_type}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("best_params")
