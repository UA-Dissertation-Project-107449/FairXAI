"""Feature selection with sensitive-attribute awareness.

Supports the dissertation research goal: *demonstrate that sensitive attributes are essential*
for fair model performance.  The primary comparison is:

- ``exclude_sensitive``      — model-blind baseline (privacy-preserving, typical deployment)
- ``include_all_sensitive``  — model-aware (what we argue should be used)
- ``include_<attr>_only``    — ablation: which single attribute drives the fairness gain?
- ``rfe_top_k``              — importance-based selection (model-agnostic, keeps top-k features)

Usage
-----
>>> X_selected, kept_cols = build_feature_set(
...     X, sensitive_attrs=["age_group", "sex"],
...     mode="include_all_sensitive",
... )
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Modes that require no trained model (fast, always available)
_STATIC_MODES = {
    "exclude_sensitive",
    "include_all_sensitive",
}


def build_feature_set(
    X: pd.DataFrame,
    sensitive_attrs: List[str],
    mode: str = "exclude_sensitive",
    top_k: int = 10,
    trained_model: Optional[Any] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return a filtered feature DataFrame and the list of kept column names.

    Args:
        X: Full feature DataFrame (may or may not include sensitive columns).
        sensitive_attrs: Column names considered sensitive (demographic proxies).
        mode: Selection strategy — one of:
            ``exclude_sensitive``     : drop all columns in *sensitive_attrs*
            ``include_all_sensitive`` : keep all columns (no filtering)
            ``include_<attr>_only``   : keep only the named attribute from *sensitive_attrs*;
                                        e.g. ``include_sex_only`` keeps ``sex`` and all
                                        non-sensitive columns.
            ``rfe_top_k``             : rank by model importance, keep ``top_k`` features
                                        (requires ``trained_model``).
        top_k: Number of features to keep in ``rfe_top_k`` mode.
        trained_model: A FairXAI model wrapper with a ``get_feature_importance()`` method.
            Required for ``rfe_top_k`` mode; ignored otherwise.

    Returns:
        Tuple of ``(X_filtered, kept_columns)``.

    Raises:
        ValueError: When an unrecognised mode is passed or required args are missing.
    """
    all_cols = list(X.columns)
    present_sensitive = [c for c in sensitive_attrs if c in all_cols]

    if mode == "exclude_sensitive":
        kept = [c for c in all_cols if c not in present_sensitive]
        logger.debug(f"[feature_selection] exclude_sensitive: dropped {present_sensitive}")

    elif mode == "include_all_sensitive":
        kept = all_cols
        logger.debug("[feature_selection] include_all_sensitive: all columns retained")

    elif mode.startswith("include_") and mode.endswith("_only"):
        # e.g. "include_sex_only" → keep only "sex" from sensitive_attrs
        attr_name = mode[len("include_"):-len("_only")]
        if attr_name not in present_sensitive:
            available = present_sensitive or sensitive_attrs
            raise ValueError(
                f"[feature_selection] mode='{mode}' requests attribute '{attr_name}' "
                f"but it is not present in X.  Available sensitive cols: {available}"
            )
        drop = [c for c in present_sensitive if c != attr_name]
        kept = [c for c in all_cols if c not in drop]
        logger.debug(
            f"[feature_selection] {mode}: kept '{attr_name}', dropped {drop}"
        )

    elif mode == "rfe_top_k":
        kept = _rfe_top_k(X, all_cols, present_sensitive, top_k, trained_model)

    else:
        raise ValueError(
            f"[feature_selection] Unknown mode '{mode}'. "
            f"Valid modes: exclude_sensitive, include_all_sensitive, "
            f"include_<attr>_only, rfe_top_k"
        )

    if not kept:
        logger.warning(
            f"[feature_selection] mode='{mode}' produced an empty feature set. "
            "Falling back to exclude_sensitive."
        )
        kept = [c for c in all_cols if c not in present_sensitive]

    return X[kept].copy(), kept


def _rfe_top_k(
    X: pd.DataFrame,
    all_cols: List[str],
    present_sensitive: List[str],
    top_k: int,
    trained_model: Optional[Any],
) -> List[str]:
    """Return top-k columns by model importance (or permutation importance fallback)."""
    if trained_model is None:
        raise ValueError(
            "[feature_selection] rfe_top_k requires a trained model "
            "(pass trained_model=<fitted FairXAI model wrapper>)."
        )

    # Try native get_feature_importance (LR coef_, RF/XGB feature_importances_)
    importances: Optional[pd.Series] = None
    try:
        imp = trained_model.get_feature_importance()
        if isinstance(imp, dict):
            importances = pd.Series(imp)
        elif isinstance(imp, pd.Series):
            importances = imp
        elif isinstance(imp, (list, np.ndarray)):
            importances = pd.Series(imp, index=all_cols[: len(imp)])
    except Exception as exc:
        logger.warning(f"[feature_selection] get_feature_importance failed: {exc}")

    # Fallback: permutation importance (slow — skip for large datasets)
    if importances is None:
        n_rows, n_cols = X.shape
        if n_rows > 5_000 or n_cols > 50:
            logger.warning(
                f"[feature_selection] rfe_top_k: no native importances and dataset is "
                f"large ({n_rows}×{n_cols}); returning top {top_k} columns by order "
                f"(permutation importance skipped for performance)."
            )
            return all_cols[:top_k]

        logger.info(
            "[feature_selection] rfe_top_k: falling back to permutation importance "
            f"(n={n_rows}, p={n_cols})"
        )
        from sklearn.inspection import permutation_importance

        try:
            y_pred = trained_model.predict(X)
            perm = permutation_importance(
                trained_model.model, X, y_pred, n_repeats=5, random_state=42
            )
            importances = pd.Series(perm.importances_mean, index=all_cols)
        except Exception as exc:
            logger.warning(
                f"[feature_selection] permutation importance failed: {exc}. "
                f"Returning first {top_k} columns."
            )
            return all_cols[:top_k]

    # Align importances index to actual columns present
    importances = importances.reindex(all_cols).fillna(0.0)
    top_cols = importances.abs().nlargest(top_k).index.tolist()
    logger.debug(f"[feature_selection] rfe_top_k={top_k}: {top_cols}")
    return top_cols
