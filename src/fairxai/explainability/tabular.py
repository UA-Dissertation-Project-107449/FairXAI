"""Explainability helpers for tabular models (SHAP, LIME, counterfactual hooks).

These wrappers are thin and defer heavy lifting to third-party libs. They keep
inputs/output formats consistent so pipeline callers can swap explainers.

Runtime behavior (sample caps, enable/disable toggles, CV usage) is configured
by caller scripts via YAML `xai` sections.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd


def adaptive_shap_sample_cap(n_rows: int, base_cap: int = 1000, large_cap: int = 200) -> int:
    """Return a SHAP sample cap appropriate for the dataset size.

    Keeps SHAP tractable on large datasets (e.g. cardio70k with 70 k rows)
    without reducing it unnecessarily on small ones.

    Args:
        n_rows: Number of rows in the dataset being explained.
        base_cap: Cap used for small datasets (n_rows ≤ 10 000).
        large_cap: Cap used for large datasets (n_rows > 10 000).

    Returns:
        The smaller of ``base_cap`` / ``large_cap`` and ``n_rows``.
    """
    threshold = 10_000
    cap = base_cap if n_rows <= threshold else large_cap
    return min(cap, n_rows)


def _is_svm_like_model(model: Any) -> bool:
    """Best-effort check for SVM estimators where generic SHAP is unstable/slow."""
    model_name = type(model).__name__.lower()
    return "svc" in model_name or model_name.startswith("svm")


def _normalize_shap_values(raw_values: Any) -> np.ndarray:
    """Normalize SHAP outputs to a 2D array [n_samples, n_features]."""
    if isinstance(raw_values, list):
        arrays = [np.asarray(v) for v in raw_values if v is not None]
        if not arrays:
            raise ValueError("Received empty SHAP values list")
        # Binary classification convention: prefer positive class attribution.
        if len(arrays) == 2:
            arr = arrays[1]
        else:
            arr = np.mean(np.stack(arrays, axis=0), axis=0)
    else:
        arr = np.asarray(raw_values)

    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        if arr.shape[2] == 2:
            return arr[:, :, 1]
        return np.mean(arr, axis=2)

    return arr.reshape(arr.shape[0], -1)


@dataclass
class ShapExplanation:
    shap_values: Any
    base_values: Any
    expected_value: Any
    feature_names: List[str]
    data: pd.DataFrame


def shap_explain_tabular(
    model: Any,
    data: pd.DataFrame,
    feature_names: Optional[Iterable[str]] = None,
    max_samples: int = 1000,
    allow_svm: bool = False,
) -> ShapExplanation:
    """Compute SHAP values for a tabular model.

    Args:
        model: Trained model with predict/proba compatible with shap.Explainer.
        data: Feature dataframe (unscaled or scaled as desired for explainer).
        feature_names: Optional feature name list; defaults to dataframe columns.
        max_samples: Subsample cap for speed.
        allow_svm: Allow SHAP on SVM-like estimators.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover
        raise ImportError("shap is required for SHAP explanations. pip install shap") from exc

    if feature_names is None:
        feature_names = list(data.columns)
    df = data.copy()
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    if _is_svm_like_model(model) and not allow_svm:
        raise ValueError(
            "SHAP is skipped for SVM models by default due high runtime and fragile"
            " additivity behavior; use LIME outputs for SVM interpretability."
        )

    # Suppress sklearn stratified-fold warnings emitted during PermutationExplainer
    # runs (model called on tiny subsets → imbalanced folds). These are expected
    # when using fine-grained bins with small group sizes and are not actionable.
    # Note: must be a plain filterwarnings call (not catch_warnings) so the filter
    # persists across threads spawned internally by SHAP/joblib workers.
    warnings.filterwarnings(
        "ignore",
        message=".*least populated class.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*n_splits.*",
        category=UserWarning,
    )

    try:
        explainer = shap.Explainer(model, df)
    except TypeError:
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(lambda X: model.predict_proba(X), df)
        elif hasattr(model, "predict"):
            explainer = shap.Explainer(lambda X: model.predict(X), df)
        else:
            raise
    try:
        shap_values = explainer(df, check_additivity=False)
    except TypeError:
        shap_values = explainer(df)

    normalized_values = _normalize_shap_values(getattr(shap_values, "values", shap_values))
    return ShapExplanation(
        shap_values=normalized_values,
        base_values=shap_values.base_values,
        expected_value=getattr(shap_values, "expected_value", None),
        feature_names=list(feature_names),
        data=df,
    )


@dataclass
class LimeExplanation:
    weights: List[tuple]
    intercept: float
    score: float
    local_pred: float


def build_lime_explainer(
    training_data: pd.DataFrame,
    feature_names: Optional[Iterable[str]] = None,
    class_names: Optional[List[str]] = None,
) -> Any:
    """Build a :class:`~lime.lime_tabular.LimeTabularExplainer` for reuse.

    Pre-building the explainer and passing it to :func:`lime_explain_instance`
    avoids the cost of reconstructing it on every single-instance call — which
    is especially noticeable in CV loops where the same training set is reused
    across multiple tracked instances in the same fold.

    Args:
        training_data: Background training set (as used for the current fold).
        feature_names: Column names; defaults to ``training_data.columns``.
        class_names: Class labels for display (e.g. ``['no_disease', 'disease']``).

    Returns:
        A ``LimeTabularExplainer`` instance ready to call ``explain_instance``.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("lime is required for LIME explanations. pip install lime") from exc

    names = list(feature_names) if feature_names is not None else list(training_data.columns)
    return LimeTabularExplainer(
        training_data.values,
        feature_names=names,
        class_names=class_names,
        discretize_continuous=True,
        mode="classification",
    )


def lime_explain_instance(
    model: Any,
    data_row: pd.Series,
    training_data: pd.DataFrame,
    feature_names: Optional[Iterable[str]] = None,
    class_names: Optional[List[str]] = None,
    num_features: int = 10,
    explainer: Optional[Any] = None,
) -> LimeExplanation:
    """Compute a single-instance LIME explanation for tabular data.

    Args:
        model: Trained model with ``predict_proba`` or ``decision_function``.
        data_row: Single row to explain (as a ``pd.Series``).
        training_data: Background dataset for the explainer.
        feature_names: Column names (defaults to ``training_data.columns``).
        class_names: Class labels for display.
        num_features: Number of top features to return.
        explainer: Pre-built ``LimeTabularExplainer`` to reuse across calls.
            When provided, ``training_data``, ``feature_names``, and
            ``class_names`` are ignored for explainer construction — pass
            them only for the first call or use :func:`build_lime_explainer`.
            Reusing avoids the overhead of rebuilding discretisation bins on
            every invocation (important in CV fold loops).
    """
    if feature_names is None:
        feature_names = list(training_data.columns)

    def _to_proba_matrix(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores)
        if scores.ndim == 1:
            return np.vstack([1 - scores, scores]).T
        if scores.ndim == 2 and scores.shape[1] == 2:
            return scores
        # Multi-class or unexpected shape: apply softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _to_frame(X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        try:
            return pd.DataFrame(X, columns=list(feature_names))
        except Exception:
            return pd.DataFrame(X)

    def _predict_proba(X: np.ndarray) -> np.ndarray:
        X_df = _to_frame(X)
        if hasattr(model, "predict_proba"):
            return _to_proba_matrix(model.predict_proba(X_df))
        if hasattr(model, "decision_function"):
            raw = model.decision_function(X_df)
            raw = np.asarray(raw)
            if raw.ndim == 1:
                prob_pos = 1.0 / (1.0 + np.exp(-raw))
                return np.vstack([1 - prob_pos, prob_pos]).T
            exp_scores = np.exp(raw - np.max(raw, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        raise ValueError("Model must provide predict_proba or decision_function for LIME")

    if explainer is None:
        explainer = build_lime_explainer(training_data, feature_names, class_names)

    exp = explainer.explain_instance(
        data_row.values,
        _predict_proba,
        num_features=num_features,
    )

    def _extract_first(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, dict):
            if 1 in value:
                return float(value[1])
            if 0 in value:
                return float(value[0])
            return float(next(iter(value.values())))
        if isinstance(value, (list, tuple, np.ndarray)):
            return float(value[0]) if len(value) else 0.0
        return float(value)

    return LimeExplanation(
        weights=exp.as_list(),
        intercept=_extract_first(exp.intercept),
        score=float(exp.score) if exp.score is not None else np.nan,
        local_pred=_extract_first(getattr(exp, "local_pred", None)),
    )


def counterfactual_stub(*_: Any, **__: Any) -> None:
    """Placeholder for future counterfactual implementations (tabular).

    Roadmap:
        Planned implementation target: Q2 2026.
    """
    raise NotImplementedError("Counterfactual explanations not implemented yet for tabular data.")
