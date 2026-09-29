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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from fairxai.utils.config import load_yaml_config

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
    max_rows_for_rbf_svm: Optional[int] = None,
    random_state: int = 42,
    preprocess_in_cv: bool = False,
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
        max_rows_for_rbf_svm: Maximum train rows to keep ``kernel='rbf'`` for
            SVM grid search. When exceeded, RBF is removed from the grid.
        random_state: Seed for ``RandomizedSearchCV``.
        preprocess_in_cv: When True, *X_train* holds RAW (unimputed, unscaled)
            features and the estimator is wrapped in a ``Pipeline`` of
            ``SimpleImputer(median) -> StandardScaler -> estimator`` so
            imputation/scaling are refit inside every search fold on that fold's
            training rows only (leak-free). ``param_grid`` keys stay unprefixed;
            they are remapped onto the estimator step internally, and the
            returned ``best_params`` are stripped back to plain estimator keys.

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
    rbf_threshold = int(max_rows_for_rbf_svm or _RBF_SVM_MAX_ROWS)
    if rbf_threshold <= 0:
        rbf_threshold = _RBF_SVM_MAX_ROWS
    if model_type == "svm" and len(X_train) > rbf_threshold:
        filtered = {
            k: [v for v in vals if not (k == "kernel" and v == "rbf")]
            for k, vals in param_grid.items()
        }
        if filtered != param_grid:
            logger.warning(
                f"HPO: RBF kernel removed from SVM grid (n_train={len(X_train)} > "
                f"{rbf_threshold}). Only linear kernel will be searched."
            )
        param_grid = filtered

    # Refit imputation + scaling inside every search fold (leak-free) when the
    # caller passes raw features. Remap the (unprefixed) grid onto the estimator
    # step so hpo.yaml stays estimator-native.
    if preprocess_in_cv:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
        param_grid = {f"model__{key}": vals for key, vals in param_grid.items()}

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

    best_params = {
        (key.split("__", 1)[1] if key.startswith("model__") else key): value
        for key, value in searcher.best_params_.items()
    }
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
        "preprocess_in_cv": bool(preprocess_in_cv),
        "preprocessing_mode": "fold_safe_pipeline" if preprocess_in_cv else "prescaled_panel",
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
    filename = f"best_params_{dataset}_{model_type}.json"

    def _read_params(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return data.get("best_params")

    # Legacy flat layout: studies/hpo/best_params_*.json
    direct = _read_params(hpo_dir / filename)
    if direct is not None:
        return direct

    # Run-scoped layout: studies/hpo/<study_id>/best_params_*.json
    latest_txt = hpo_dir / "latest.txt"
    if latest_txt.exists():
        study_id = latest_txt.read_text(encoding="utf-8").strip()
        if study_id:
            latest = _read_params(hpo_dir / study_id / filename)
            if latest is not None:
                return latest

    if hpo_dir.exists() and hpo_dir.is_dir():
        candidates = sorted(
            [p for p in hpo_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            params = _read_params(candidate / filename)
            if params is not None:
                return params

    return None


# --------------------------------------------------------------------------- #
# Resolving a family's hyperparameters for a stage
#
# Stages 7, 10 and 11 each built their own parameter dictionary and disagreed:
# 7 merged HPO over the model YAML, 10 ignored HPO, 11 merged HPO and then let
# the sweep's model_variants overwrite it. Every stage now calls
# resolve_model_params, in one order: model YAML, random_state, HPO best,
# explicit overrides, hardware (last, so nothing moves a fit to another device).
# --------------------------------------------------------------------------- #

_LOGGED_HPO: set = set()


def _log_once(key: Tuple[Any, ...], message: str, *args: Any) -> None:
    """Log *message* at info level the first time *key* is seen, debug after."""
    if key in _LOGGED_HPO:
        logger.debug(message, *args)
        return
    _LOGGED_HPO.add(key)
    logger.info(message, *args)


@dataclass(frozen=True)
class ResolvedParams:
    """Resolved hyperparameters plus where the tuned values came from."""

    params: Dict[str, Any] = field(default_factory=dict)
    hpo_used: bool = False
    hpo_keys: Tuple[str, ...] = ()

    @property
    def variant_name(self) -> str:
        """``tuned`` when HPO supplied the values, else ``default``, so a results
        table never calls an untuned model tuned."""
        return "tuned" if self.hpo_used else "default"


def hpo_params_dir(project_root: Path, pipeline: str, enabled: bool = True) -> Optional[Path]:
    """Directory holding ``best_params_<cohort>_<family>.json``, or ``None`` when
    HPO is off or was never run (callers treat both as "no tuning available")."""
    if not enabled:
        return None
    hpo_dir = Path(project_root) / "output" / pipeline / "studies" / "hpo"
    return hpo_dir if hpo_dir.exists() else None


def load_base_params(project_root: Path, model_type: str) -> Dict[str, Any]:
    """Base hyperparameters from ``configs/models/<family>.yaml``. A missing file
    is not fatal: the wrapper class defaults apply instead."""
    cfg_path = Path(project_root) / "configs" / "models" / f"{model_type}.yaml"
    if not cfg_path.exists():
        logger.warning(
            "No model config at %s - falling back to wrapper class defaults for %s",
            cfg_path,
            model_type,
        )
        return {}
    return dict(load_yaml_config(str(cfg_path)).get("hyperparameters", {}) or {})


def resolve_model_params(
    project_root: Path,
    model_type: str,
    dataset: Optional[str] = None,
    hpo_dir: Optional[Path] = None,
    random_state: Optional[int] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    hardware: Optional[Mapping[str, Any]] = None,
) -> ResolvedParams:
    """Resolve the hyperparameters for one (cohort, family) pair.

    Args:
        project_root: Repository root, used to find ``configs/models/``.
        model_type: Family key, e.g. ``"xgboost"``.
        dataset: Cohort key the HPO file is named after. HPO is skipped without it.
        hpo_dir: Directory from :func:`hpo_params_dir`. ``None`` skips HPO.
        random_state: Applied only when the model config sets no ``random_state``.
        overrides: Explicit per-experiment params, applied over the tuned ones.
        hardware: Device/thread params, applied last.
    """
    params = load_base_params(project_root, model_type)
    if random_state is not None:
        params.setdefault("random_state", random_state)

    hpo_best: Dict[str, Any] = {}
    if hpo_dir is not None and dataset:
        hpo_best = load_hpo_params(Path(hpo_dir), dataset, model_type) or {}

    if hpo_best:
        params.update(hpo_best)
        # Info, not debug: the smoke gate greps this line. Logged once per pair,
        # since the sweep resolves the same pair for every cell.
        _log_once(
            (model_type, dataset, "loaded"),
            "[HPO] Loaded best params for %s/%s: %s",
            model_type,
            dataset,
            hpo_best,
        )
    elif hpo_dir is not None and dataset:
        _log_once(
            (model_type, dataset, "missing"),
            "[HPO] No saved params for %s/%s; using config defaults.",
            model_type,
            dataset,
        )

    if overrides:
        params.update(dict(overrides))
    if hardware:
        params.update(dict(hardware))

    return ResolvedParams(params=params, hpo_used=bool(hpo_best), hpo_keys=tuple(sorted(hpo_best)))
