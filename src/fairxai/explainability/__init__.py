"""Explainability public API for tabular SHAP/LIME utilities.

This module re-exports the stable explainability surface used by scripts.
Configuration is provided by caller-level YAML (`xai` blocks), not by module
environment variables.
"""

from .tabular import (
	ShapExplanation,
	LimeExplanation,
	shap_explain_tabular,
	lime_explain_instance,
	counterfactual_stub,
)

__all__ = [
	"ShapExplanation",
	"LimeExplanation",
	"shap_explain_tabular",
	"lime_explain_instance",
	"counterfactual_stub",
]
