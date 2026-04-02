"""Experiments package public API.

Exports attribute-binning analysis utilities and experiment artifact
versioning helpers consumed by experiment scripts.
"""

from .attribute_binning import (
    analyze_strategy_comprehensive,
    apply_binning,
    compare_strategies,
    compute_fairness_metrics,
    compute_strategy_score,
    create_binning_strategy,
    generate_summary_report,
    sensitive_attribute_distribution,
    validate_and_repair,
)
from .versioning import ExperimentVersioning

__all__ = [
    "create_binning_strategy",
    "apply_binning",
    "sensitive_attribute_distribution",
    "compute_fairness_metrics",
    "analyze_strategy_comprehensive",
    "compare_strategies",
    "compute_strategy_score",
    "generate_summary_report",
    "validate_and_repair",
    "ExperimentVersioning",
]
