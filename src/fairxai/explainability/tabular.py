"""Explainability helpers for tabular models (SHAP, LIME, counterfactual hooks).

These wrappers are thin and defer heavy lifting to third-party libs. They keep
inputs/output formats consistent so pipeline callers can swap explainers.
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional
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

    explainer = LimeTabularExplainer(
        training_data.values,
        feature_names=list(feature_names),
        class_names=class_names,
        discretize_continuous=True,
        mode="classification",
    )
    exp = explainer.explain_instance(
        data_row.values,
        model.predict_proba,
        num_features=num_features,
    )
    return LimeExplanation(
        weights=exp.as_list(),
        intercept=exp.intercept[0] if exp.intercept is not None else 0.0,
        score=exp.score,
        local_pred=exp.local_pred[0] if hasattr(exp, "local_pred") else np.nan,
    )


def counterfactual_stub(*_: Any, **__: Any) -> None:
    """Placeholder for future counterfactual implementations (tabular)."""
    raise NotImplementedError("Counterfactual explanations not implemented yet for tabular data.")
