"""Profiling package public API.

Exports dataset complexity metric utilities used by profiling and
recommendation workflows, plus the configuration loader.
"""

from .complexity import (
    compute_complexity_metrics,
    get_supported_complexity_metrics,
    is_complexity_metric_key,
    is_primary_complexity_metric,
)
from .config import ComplexityConfig, load_complexity_config
from .domain_characterization import characterize_dataset

__all__ = [
    "compute_complexity_metrics",
    "get_supported_complexity_metrics",
    "is_complexity_metric_key",
    "is_primary_complexity_metric",
    "ComplexityConfig",
    "load_complexity_config",
    "characterize_dataset",
]
