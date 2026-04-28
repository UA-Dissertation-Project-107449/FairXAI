"""WebApp adapter for attribute binning subgroup analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from fairxai.experiments.attribute_binning import apply_binning, create_binning_strategy

logger = logging.getLogger(__name__)

_DISPARITY_P0_THRESHOLD = 4.0
_DISPARITY_P1_THRESHOLD = 2.0

SUPPORTED_STRATEGIES = [
    "equal_width_3",
    "equal_width_5",
    "quantile_3",
    "quantile_5",
    "jenks_3",
    "jenks_5",
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

    bin_stats = _compute_bin_stats(df_binned, "_bin_group", target_column)
    summary = _compute_summary(bin_stats)
    recommendations = _generate_recommendations(attribute, strategy, summary)

    return {
        "attribute": attribute,
        "strategy": strategy,
        "bins": bin_stats,
        "summary": summary,
        "recommendations": recommendations,
    }


def _compute_bin_stats(
    df: pd.DataFrame, bin_col: str, target_col: str
) -> list[dict[str, Any]]:
    total = len(df)
    groups = df.groupby(bin_col, observed=True)

    stats = []
    for label, grp in groups:
        count = len(grp)
        pct = round(count / total * 100, 1) if total > 0 else 0.0
        target_vals = pd.to_numeric(grp[target_col], errors="coerce").dropna()
        target_rate = round(float(target_vals.mean()), 4) if len(target_vals) > 0 else None
        stats.append(
            {"label": str(label), "count": count, "pct": pct, "target_rate": target_rate}
        )
    return stats


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
