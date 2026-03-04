"""Explainability helpers for tabular models (SHAP, LIME, counterfactual hooks).

These wrappers are thin and defer heavy lifting to third-party libs. They keep
inputs/output formats consistent so pipeline callers can swap explainers.

Runtime behavior (sample caps, enable/disable toggles, CV usage) is configured
by caller scripts via YAML `xai` sections.
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Union
import warnings
import numpy as np
import pandas as pd


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
) -> ShapExplanation:
    """Compute SHAP values for a tabular model.

    Args:
        model: Trained model with predict/proba compatible with shap.Explainer.
        data: Feature dataframe (unscaled or scaled as desired for explainer).
        feature_names: Optional feature name list; defaults to dataframe columns.
        max_samples: Subsample cap for speed.
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

    explainer = shap.Explainer(model, df)
    shap_values = explainer(df)
    return ShapExplanation(
        shap_values=shap_values.values,
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


def lime_explain_instance(
    model: Any,
    data_row: pd.Series,
    training_data: pd.DataFrame,
    feature_names: Optional[Iterable[str]] = None,
    class_names: Optional[List[str]] = None,
    num_features: int = 10,
) -> LimeExplanation:
    """Compute a single-instance LIME explanation for tabular data."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("lime is required for LIME explanations. pip install lime") from exc

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
        if hasattr(model, 'predict_proba'):
            return _to_proba_matrix(model.predict_proba(X_df))
        if hasattr(model, 'decision_function'):
            raw = model.decision_function(X_df)
            raw = np.asarray(raw)
            if raw.ndim == 1:
                prob_pos = 1.0 / (1.0 + np.exp(-raw))
                return np.vstack([1 - prob_pos, prob_pos]).T
            exp_scores = np.exp(raw - np.max(raw, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        raise ValueError("Model must provide predict_proba or decision_function for LIME")

    explainer = LimeTabularExplainer(
        training_data.values,
        feature_names=list(feature_names),
        class_names=class_names,
        discretize_continuous=True,
        mode="classification",
    )
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
