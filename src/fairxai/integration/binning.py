"""WebApp adapter for attribute binning subgroup analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from fairxai.experiments.attribute_binning import (
    apply_binning,
    compute_strategy_score,
    create_binning_strategy,
)

logger = logging.getLogger(__name__)

_DISPARITY_P0_THRESHOLD = 4.0
_DISPARITY_P1_THRESHOLD = 2.0

# Equal-thirds weighting: sample size, balance, and fairness contribute equally
# to the overall strategy score (no component privileged).
_SCORE_WEIGHTS = {"sample_size": 1 / 3, "balance": 1 / 3, "fairness": 1 / 3}

SUPPORTED_STRATEGIES = [
    "equal_width_3",
    "equal_width_5",
    "equal_width_7",
    "quantile_3",
    "quantile_5",
    "quantile_7",
]


def run_binning(
    csv_path: str | Path,
    target_column: str,
    attribute: str,
    strategy: str,
    min_group_size: int = 10,
) -> dict[str, Any]:
    """Bin a continuous attribute and compute per-bin target rate statistics.

    Args:
        csv_path: Absolute path to the dataset CSV file.
        target_column: Name of the binary target column.
        attribute: Continuous numerical column to bin.
        strategy: Binning strategy name (e.g. ``"quantile_5"``).
        min_group_size: Minimum samples per bin; smaller bins are merged.

    Returns:
        JSON-serializable dict with bins, summary, and recommendations.

    Raises:
        ValueError: If attribute or target_column not found, or strategy unknown.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if attribute not in df.columns:
        raise ValueError(f"Attribute '{attribute}' not in dataset columns")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset columns")

    df = df.dropna(subset=[attribute, target_column]).copy()
    df[attribute] = pd.to_numeric(df[attribute], errors="coerce")
    df = df.dropna(subset=[attribute])

    bins, labels = create_binning_strategy(
        df, strategy, col=attribute, min_group_size=min_group_size
    )
    df_binned = apply_binning(df, bins, labels, col=attribute, output_col="_bin_group")

    bin_stats = _compute_bin_stats(df_binned, "_bin_group", target_column, attribute)
    summary = _compute_summary(bin_stats)
    summary["strategy_score"] = _compute_strategy_score(bin_stats, strategy)
    summary["score_weights"] = {k: round(v, 4) for k, v in _SCORE_WEIGHTS.items()}
    recommendations = _generate_recommendations(attribute, strategy, summary)

    return {
        "attribute": attribute,
        "strategy": strategy,
        "bins": bin_stats,
        "summary": summary,
        "recommendations": recommendations,
    }


def _compute_bin_stats(
    df: pd.DataFrame, bin_col: str, target_col: str, attribute_col: str | None = None
) -> list[dict[str, Any]]:
    total = len(df)
    groups = df.groupby(bin_col, observed=True)

    stats = []
    for label, grp in groups:
        count = len(grp)
        pct = round(count / total * 100, 1) if total > 0 else 0.0
        target_vals = pd.to_numeric(grp[target_col], errors="coerce").dropna()
        target_rate = round(float(target_vals.mean()), 4) if len(target_vals) > 0 else None
        observed_min = _observed_min(grp, attribute_col)
        lower_bound, upper_bound, closed = _get_bin_bounds(label, observed_min)
        display_label = _format_bin_label(label, lower_bound, upper_bound, closed)
        stats.append(
            {
                "label": display_label,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "closed": closed,
                "count": count,
                "pct": pct,
                "target_rate": target_rate,
            }
        )
    return stats


def _observed_min(grp: pd.DataFrame, attribute_col: str | None) -> float | None:
    if attribute_col is None or attribute_col not in grp.columns:
        return None
    values = pd.to_numeric(grp[attribute_col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min())


def _get_bin_bounds(
    label: Any, observed_min: float | None = None
) -> tuple[float | None, float | None, str | None]:
    if isinstance(label, pd.Interval):
        left = float(label.left)
        right = float(label.right)
        closed = label.closed
        if observed_min is not None and label.closed == "right" and left < observed_min <= right:
            left = observed_min
            closed = "both"
        return left, right, closed
    return None, None, None


def _format_bin_label(
    raw_label: Any,
    lower_bound: float | None,
    upper_bound: float | None,
    closed: str | None,
) -> str:
    if lower_bound is None or upper_bound is None or closed is None:
        return str(raw_label)

    left_bracket = "[" if closed in {"left", "both"} else "("
    right_bracket = "]" if closed in {"right", "both"} else ")"
    return (
        f"{left_bracket}{_format_bound(lower_bound)}, {_format_bound(upper_bound)}{right_bracket}"
    )


def _format_bound(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:g}"


def _compute_summary(bin_stats: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [b["target_rate"] for b in bin_stats if b["target_rate"] is not None]
    if len(rates) < 2:
        return {
            "disparity_ratio": None,
            "max_target_rate": rates[0] if rates else None,
            "min_target_rate": rates[0] if rates else None,
            "max_bin": None,
            "min_bin": None,
        }

    max_rate = max(rates)
    min_rate = min(rates)
    disparity = round(max_rate / min_rate, 2) if min_rate > 0 else None

    bins_with_rates = [b for b in bin_stats if b["target_rate"] is not None]
    max_bin = max(bins_with_rates, key=lambda b: b["target_rate"])["label"]
    min_bin = min(bins_with_rates, key=lambda b: b["target_rate"])["label"]

    return {
        "disparity_ratio": disparity,
        "max_target_rate": round(max_rate, 4),
        "min_target_rate": round(min_rate, 4),
        "max_bin": max_bin,
        "min_bin": min_bin,
    }


def _compute_strategy_score(bin_stats: list[dict[str, Any]], strategy: str) -> float | None:
    """Score a binning strategy with equal (1/3) weighting of its components.

    Reuses :func:`compute_strategy_score` so the WebApp surfaces the same
    equal-thirds score used by the offline experiment report.  The component
    metrics are derived from the per-bin counts and target rates:

    - ``min_group_size`` / ``mean_group_size`` — statistical power.
    - ``group_balance_cv`` — coefficient of variation of bin counts (evenness).
    - ``max_sp_difference`` — max absolute positive-rate gap across bins.

    Returns ``None`` when fewer than two populated bins exist (score undefined).
    """
    counts = [b["count"] for b in bin_stats if b["count"] > 0]
    rates = [b["target_rate"] for b in bin_stats if b["target_rate"] is not None]
    if len(counts) < 2 or len(rates) < 2:
        return None

    mean_count = sum(counts) / len(counts)
    if mean_count <= 0:
        return None
    variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
    cv = (variance**0.5) / mean_count

    metrics = {
        "min_group_size": min(counts),
        "mean_group_size": mean_count,
        "group_balance_cv": cv,
        "max_sp_difference": max(rates) - min(rates),
    }
    score = compute_strategy_score(
        {"strategy": strategy, "fairness_metrics": metrics},
        _SCORE_WEIGHTS["sample_size"],
        _SCORE_WEIGHTS["balance"],
        _SCORE_WEIGHTS["fairness"],
    )
    return round(float(score), 4)


def _generate_recommendations(
    attribute: str, strategy: str, summary: dict[str, Any]
) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    ratio = summary.get("disparity_ratio")
    if ratio is None:
        return recs

    if ratio >= _DISPARITY_P0_THRESHOLD:
        priority = "P0"
        title = f"Critical disparity across {attribute} bins"
        action = (
            f"The '{strategy}' binning reveals a {ratio:.1f}× positive-rate gap between "
            f"'{summary['max_bin']}' ({summary['max_target_rate']:.0%}) and "
            f"'{summary['min_bin']}' ({summary['min_target_rate']:.0%}). "
            f"Apply a fairness constraint (e.g. demographic parity) or stratified sampling "
            f"before model training."
        )
        outcome = "Reduced outcome disparity across subgroups defined by this attribute."
    elif ratio >= _DISPARITY_P1_THRESHOLD:
        priority = "P1"
        title = f"Notable disparity across {attribute} bins"
        action = (
            f"The '{strategy}' binning shows a {ratio:.1f}× positive-rate gap "
            f"('{summary['max_bin']}' vs '{summary['min_bin']}'). "
            f"Consider whether this reflects a real domain signal or a data artefact. "
            f"Monitor fairness metrics during model evaluation."
        )
        outcome = "Improved awareness of potential disparate impact linked to this attribute."
    else:
        return recs

    recs.append(
        {"priority": priority, "title": title, "action": action, "expected_outcome": outcome}
    )
    return recs
