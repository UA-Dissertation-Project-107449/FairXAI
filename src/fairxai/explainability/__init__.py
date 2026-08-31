"""Explainability public API for tabular SHAP/LIME utilities.

This module re-exports the stable explainability surface used by scripts.
Configuration is provided by caller-level YAML (`xai` blocks), not by module
environment variables.
"""

from .subgroup import SubgroupShapSummary, summarise_subgroup_shap
from .tabular import (
    LimeExplanation,
    ShapExplanation,
    counterfactual_stub,
    lime_explain_instance,
    shap_explain_tabular,
)

__all__ = [
    "ShapExplanation",
    "SubgroupShapSummary",
    "summarise_subgroup_shap",
    "LimeExplanation",
    "shap_explain_tabular",
    "lime_explain_instance",
    "counterfactual_stub",
]
