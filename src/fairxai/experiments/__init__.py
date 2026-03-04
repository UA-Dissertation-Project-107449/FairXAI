"""Experiments package public API.

Exports age-binning analysis utilities and experiment artifact versioning
helpers consumed by experiment scripts.
"""

from .age_binning import (
    create_binning_strategy,
    apply_binning,
    sensitive_attribute_distribution,
    compute_fairness_metrics,
    analyze_strategy_comprehensive,
    compare_strategies,
    compute_strategy_score,
    generate_summary_report
)
from .versioning import ExperimentVersioning

__all__ = [
    'create_binning_strategy',
    'apply_binning',
    'sensitive_attribute_distribution',
    'compute_fairness_metrics',
    'analyze_strategy_comprehensive',
    'compare_strategies',
    'compute_strategy_score',
    'generate_summary_report',
    'ExperimentVersioning'
]
