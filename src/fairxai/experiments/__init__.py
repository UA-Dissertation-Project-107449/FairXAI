"""Experiments package public API.

Exports attribute-binning analysis utilities and experiment artifact
versioning helpers consumed by experiment scripts.
"""

from .age_binning_sensitivity import (
    before_after_deltas,
    compute_age_binning_sensitivity,
    load_mitigation_predictions,
    run_age_binning,
    run_age_binning_sensitivity,
)
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
    "compute_age_binning_sensitivity",
    "before_after_deltas",
    "run_age_binning",
    "run_age_binning_sensitivity",
    "load_mitigation_predictions",
    "ExperimentVersioning",
]
