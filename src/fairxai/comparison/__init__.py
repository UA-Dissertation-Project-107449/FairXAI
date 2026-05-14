"""Comparison-stage helpers for canonical FairXAI experiment evidence tables."""

from .baseline_matching import (
    baseline_key_from_row,
    build_baseline_lookups,
    find_matching_baseline,
    normalize_sensitive_attr,
    safe_float,
    safe_int,
)
from .config import load_comparison_config
from .metric_tables import write_canonical_comparison_outputs

__all__ = [
    "baseline_key_from_row",
    "build_baseline_lookups",
    "find_matching_baseline",
    "load_comparison_config",
    "normalize_sensitive_attr",
    "safe_float",
    "safe_int",
    "write_canonical_comparison_outputs",
]

