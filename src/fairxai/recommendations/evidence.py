"""Helpers to extract specific metric values from profiling output dicts.

Every function in this module takes a raw profile dict (as produced by
``DataProfiler.profile_dataset``) and returns a simple Python scalar (or
*None* when the requested value is absent).  These utilities centralise
all dict-key look-ups so that the rule functions in ``rules.py`` never
need to hard-code nested key paths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Basic stats
# ---------------------------------------------------------------------------


def get_n_samples(profile: Dict) -> int:
    return profile.get("basic_stats", {}).get("n_samples", 0)


def get_n_features(profile: Dict) -> int:
    return profile.get("basic_stats", {}).get("n_features", 0)


def get_target_prevalence(profile: Dict) -> Optional[float]:
    return profile.get("basic_stats", {}).get("target_prevalence")


def get_n_classes(profile: Dict) -> int:
    """Number of distinct target values."""
    counts = profile.get("target_distribution", {}).get("counts", {})
    return len(counts)


def get_imbalance_ratio(profile: Dict) -> float:
    return profile.get("target_distribution", {}).get("imbalance_ratio", 1.0)


# ---------------------------------------------------------------------------
# Sensitive attribute helpers
# ---------------------------------------------------------------------------


def get_sensitive_attrs(profile: Dict) -> List[str]:
    """Return the list of sensitive attributes present in the profile."""
    return list(profile.get("sensitive_attr_distribution", {}).keys())


def get_group_counts(profile: Dict, attr: str) -> Dict[str, int]:
    dist = profile.get("sensitive_attr_distribution", {}).get(attr, {})
    return {str(k): int(v) for k, v in dist.get("counts", {}).items()}


def get_group_proportions(profile: Dict, attr: str) -> Dict[str, float]:
    dist = profile.get("sensitive_attr_distribution", {}).get(attr, {})
    return {str(k): float(v) for k, v in dist.get("proportions", {}).items()}


# ---------------------------------------------------------------------------
# Representation balance
# ---------------------------------------------------------------------------


def get_size_ratio(profile: Dict, attr: str) -> Optional[float]:
    return profile.get("representation_balance", {}).get(attr, {}).get("size_ratio")


def get_min_group_size(profile: Dict, attr: str) -> Optional[int]:
    val = profile.get("representation_balance", {}).get(attr, {}).get("min_group_size")
    return int(val) if val is not None else None


def get_cv(profile: Dict, attr: str) -> Optional[float]:
    return profile.get("representation_balance", {}).get(attr, {}).get("coefficient_of_variation")


# ---------------------------------------------------------------------------
# Label imbalance
# ---------------------------------------------------------------------------


def get_statistical_parity_diff(profile: Dict, attr: str) -> Optional[float]:
    li = profile.get("label_imbalance_by_group", {}).get(attr, {})
    return li.get("statistical_parity_difference", {}).get("max_difference")


def get_positive_rates(profile: Dict, attr: str) -> Dict[str, float]:
    li = profile.get("label_imbalance_by_group", {}).get(attr, {})
    return {str(k): float(v) for k, v in li.get("positive_rates", {}).items()}


# ---------------------------------------------------------------------------
# Complexity metrics (global)
# ---------------------------------------------------------------------------


def get_complexity_metric(profile: Dict, metric: str) -> Optional[float]:
    return profile.get("complexity_metrics", {}).get(metric)


def get_all_complexity(profile: Dict) -> Dict[str, Optional[float]]:
    return dict(profile.get("complexity_metrics", {}))


# ---------------------------------------------------------------------------
# Group / intersection complexity
# ---------------------------------------------------------------------------


def get_group_complexity(profile: Dict, attr: str, group: str, metric: str) -> Optional[float]:
    gc = profile.get("group_complexity_metrics", {}).get(attr, {}).get(str(group), {})
    return gc.get("complexity_metrics", {}).get(metric)


def get_group_complexity_status(profile: Dict, attr: str, group: str) -> str:
    gc = profile.get("group_complexity_metrics", {}).get(attr, {}).get(str(group), {})
    return gc.get("status", "unavailable")


def get_intersection_slices(profile: Dict) -> Dict[str, Dict]:
    """Return intersection complexity dict  ``pair_key → {slice_key → info}``."""
    return dict(profile.get("intersection_complexity_metrics", {}))


def get_low_support_intersections(
    profile: Dict, min_samples: int = 50
) -> List[Tuple[str, str, int]]:
    """Return ``(pair_key, slice_key, n_samples)`` for every intersectional
    slice that was skipped or has fewer than *min_samples*."""
    results: List[Tuple[str, str, int]] = []
    for pair_key, slices in get_intersection_slices(profile).items():
        for slice_key, info in slices.items():
            n = info.get("n_samples", 0)
            if n < min_samples or "skipped" in info.get("status", ""):
                results.append((pair_key, slice_key, n))
    return results


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def get_total_missing(profile: Dict) -> int:
    return profile.get("missing_value_analysis", {}).get("total_missing", 0)


def get_missing_fraction(profile: Dict, column: str) -> float:
    """Fraction of missing values for *column* relative to total rows."""
    n = get_n_samples(profile) or 1
    cols_missing = profile.get("missing_value_analysis", {}).get("columns_with_missing", {})
    return cols_missing.get(column, 0) / n


# ---------------------------------------------------------------------------
# Group statistics (target prevalence per group)
# ---------------------------------------------------------------------------


def get_group_target_prevalence(profile: Dict, attr: str) -> Dict[str, float]:
    gs = profile.get("group_statistics", {}).get(attr, {})
    return {k: v.get("target_prevalence", 0.0) for k, v in gs.items()}


def get_group_class_support(profile: Dict, attr: str) -> Dict[str, Dict[str, int]]:
    """Return ``{group: {class_label: count}}`` from group_statistics."""
    gs = profile.get("group_statistics", {}).get(attr, {})
    return {k: v.get("target_counts", {}) for k, v in gs.items()}


# ---------------------------------------------------------------------------
# Reference comparison helper
# ---------------------------------------------------------------------------


def compare_to_reference(
    value: Optional[float],
    reference: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """Compare a metric value against a reference distribution.

    Parameters
    ----------
    value : float or None
    reference : dict
        Must contain at least ``median``.  May also have ``p25``, ``p75``,
        ``min``, ``max``.

    Returns
    -------
    dict or None
        ``{"value", "median", "percentile_approx", "above_p75"}`` or None
        if value is unavailable.
    """
    if value is None:
        return None
    median = reference.get("median")
    if median is None:
        return None
    p25 = reference.get("p25", median)
    p75 = reference.get("p75", median)
    # Very rough percentile approximation using IQR
    if value <= p25:
        pct = (
            25.0 * (value - reference.get("min", p25)) / max(p25 - reference.get("min", p25), 1e-9)
        )
        pct = max(0.0, min(25.0, pct))
    elif value <= median:
        pct = 25.0 + 25.0 * (value - p25) / max(median - p25, 1e-9)
    elif value <= p75:
        pct = 50.0 + 25.0 * (value - median) / max(p75 - median, 1e-9)
    else:
        pct = 75.0 + 25.0 * (value - p75) / max(reference.get("max", p75) - p75, 1e-9)
        pct = min(100.0, pct)

    return {
        "value": value,
        "median": median,
        "percentile_approx": round(pct, 1),
        "above_p75": value > p75,
    }
