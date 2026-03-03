"""Profiling package public API.

Exports dataset complexity metric utilities used by profiling and
recommendation workflows.
"""

from .complexity import (
    compute_complexity_metrics,
    get_supported_complexity_metrics,
    is_complexity_metric_key,
    is_primary_complexity_metric,
)

__all__ = [
    "compute_complexity_metrics",
    "get_supported_complexity_metrics",
    "is_complexity_metric_key",
    "is_primary_complexity_metric",
]
