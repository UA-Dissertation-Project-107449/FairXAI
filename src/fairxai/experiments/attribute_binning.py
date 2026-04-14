"""
Attribute binning strategies — reusable functions for experiments.

This module provides utilities for:
- Creating various binning strategies for continuous attributes (config-driven
  or built-in)
- Analyzing sensitive attribute distribution within bins
- Computing fairness metrics per binning strategy
- Computing cross-attribute fairness impact (how binning one attribute
  affects statistical parity of other sensitive attributes)
- Generating comparison reports

Strategies are defined declaratively in YAML (e.g.
``configs/experiments/age_binning.yaml``) under the ``binning_strategies``
key.  The code reads method / bins / labels / n_bins from the config dict so
users can add or modify strategies without touching Python.

Supported methods:
  - ``fixed``             — user supplies ``bins`` (edges) and ``labels``
  - ``quantile``          — user supplies ``n_bins``; edges from ``pd.qcut``
  - ``equal_width``       — user supplies ``n_bins``; edges from ``pd.cut``
  - ``jenks``             — natural breaks via ``jenkspy`` (optional dep)
  - ``adaptive_quantile`` — quantile + merge under-populated bins

All strategies pass through a safeguard layer (``validate_and_repair``)
that clips edges to the actual data range and merges any bin whose count
falls below ``min_group_size``.

Designed to be imported by notebooks and experiment scripts.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants for scoring and interpretation
MAX_EXPECTED_SP_DIFF = 0.5  # Typical max statistical parity difference for age groups
MAX_EXPECTED_CV = 1.0  # Typical max coefficient of variation for reasonable binnings
MIN_SAMPLE_SIZE = 30  # Minimum recommended group size for statistical validity

# ---------------------------------------------------------------------------
# Built-in strategy defaults (mirrors the canonical YAML structure).
# Used as fallback when no strategy_config dict is passed to
# ``create_binning_strategy``.
# ---------------------------------------------------------------------------
BUILTIN_STRATEGIES: Dict[str, Dict[str, Any]] = {
    # --- Fixed strategies ---------------------------------------------------
    "fixed_2": {
        "description": "Binary split at cardiovascular risk boundary",
        "method": "fixed",
        "bins": [0, 55, 100],
        "labels": ["<55", "55+"],
    },
    "fixed_3": {
        "description": "Coarse 3-bin split",
        "method": "fixed",
        "bins": [0, 45, 65, 100],
        "labels": ["<45", "45-64", "65+"],
    },
    "fixed_5yr": {
        "description": "Finer granularity - 5-year fixed intervals",
        "method": "fixed",
        "bins": [0, 35, 40, 45, 50, 55, 60, 65, 70, 75, 100],
        "labels": [
            "<35",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75+",
        ],
    },
    "fixed_10yr": {
        "description": "Current baseline - 10-year fixed intervals",
        "method": "fixed",
        "bins": [0, 40, 50, 60, 70, 100],
        "labels": ["<40", "40-49", "50-59", "60-69", "70+"],
    },
    "clinical": {
        "description": "Clinical cardiovascular risk guidelines (ACC/AHA)",
        "method": "fixed",
        "bins": [0, 45, 55, 65, 100],
        "labels": ["<45", "45-54", "55-64", "65+"],
    },
    # --- Quantile strategies ------------------------------------------------
    "quantile_3": {
        "description": "Data-driven terciles (equal sample sizes)",
        "method": "quantile",
        "n_bins": 3,
    },
    "quantile_4": {
        "description": "Data-driven quartiles",
        "method": "quantile",
        "n_bins": 4,
    },
    "quantile_5": {
        "description": "Data-driven quintiles",
        "method": "quantile",
        "n_bins": 5,
    },
    "quantile_6": {
        "description": "Data-driven sextiles",
        "method": "quantile",
        "n_bins": 6,
    },
    "quantile_7": {
        "description": "Data-driven septiles",
        "method": "quantile",
        "n_bins": 7,
    },
    "quantile_8": {
        "description": "Data-driven octiles",
        "method": "quantile",
        "n_bins": 8,
    },
    "quantile_9": {
        "description": "Data-driven noniles",
        "method": "quantile",
        "n_bins": 9,
    },
    "quantile_10": {
        "description": "Data-driven deciles",
        "method": "quantile",
        "n_bins": 10,
    },
    # --- Equal-width strategies ---------------------------------------------
    "equal_width_3": {
        "description": "3 equal-width bins spanning the data range",
        "method": "equal_width",
        "n_bins": 3,
    },
    "equal_width_4": {
        "description": "4 equal-width bins spanning the data range",
        "method": "equal_width",
        "n_bins": 4,
    },
    "equal_width_5": {
        "description": "5 equal-width bins spanning the data range",
        "method": "equal_width",
        "n_bins": 5,
    },
    "equal_width_6": {
        "description": "6 equal-width bins spanning the data range",
        "method": "equal_width",
        "n_bins": 6,
    },
    "equal_width_8": {
        "description": "8 equal-width bins spanning the data range",
        "method": "equal_width",
        "n_bins": 8,
    },
    "equal_width_10": {
        "description": "10 equal-width bins spanning the data range",
        "method": "equal_width",
        "n_bins": 10,
    },
    # --- Jenks natural breaks -----------------------------------------------
    "jenks_3": {
        "description": "3 bins via Jenks natural breaks",
        "method": "jenks",
        "n_bins": 3,
    },
    "jenks_4": {
        "description": "4 bins via Jenks natural breaks",
        "method": "jenks",
        "n_bins": 4,
    },
    "jenks_5": {
        "description": "5 bins via Jenks natural breaks",
        "method": "jenks",
        "n_bins": 5,
    },
    # --- Adaptive quantile --------------------------------------------------
    "adaptive_quantile_5": {
        "description": "Quantile quintiles with forced merge of small bins",
        "method": "adaptive_quantile",
        "n_bins": 5,
    },
}


def create_binning_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    col: str = "age_raw",
    strategy_config: Optional[Dict[str, Any]] = None,
    min_group_size: int = MIN_SAMPLE_SIZE,
    **kwargs,
) -> Tuple[List[float], Optional[List[str]]]:
    """Create bins and labels for a given binning strategy.

    The strategy is resolved in order:

    1. If *strategy_config* is supplied (a dict with at least ``method``),
       it is used directly.
    2. Otherwise, if *strategy_name* matches a key in
       :data:`BUILTIN_STRATEGIES`, those defaults are used.
    3. For names like ``quantile_N`` or ``equal_width_N``, the method and
       *n_bins* are inferred from the name.
    4. If none of the above match, a ``ValueError`` is raised.

    After resolving, all strategies pass through
    :func:`validate_and_repair` which clips edges to the data range and
    merges under-populated bins.

    Parameters
    ----------
    df : DataFrame
        DataFrame that contains the column to bin.
    strategy_name : str
        Human-readable strategy identifier.
    col : str
        Column with continuous values to bin (default ``age_raw``).
    strategy_config : dict, optional
        Config dict (typically one entry from the YAML
        ``binning_strategies`` section).  When provided, *strategy_name*
        is used only for logging/labelling.
    min_group_size : int
        Minimum acceptable bin count.  Bins below this are merged by
        :func:`validate_and_repair`.  Set to 0 to disable merging.
    **kwargs
        Legacy pass-through (``n_bins`` etc.).

    Returns
    -------
    (bins, labels)
        *bins* is a list of edge values.  *labels* may be ``None`` for
        auto-generated interval labels (quantile / equal_width).
    """
    logger.info(f"Creating binning strategy: {strategy_name}")

    # -- Resolve config dict -------------------------------------------------
    cfg = strategy_config or BUILTIN_STRATEGIES.get(strategy_name)

    if cfg is None:
        # Attempt to infer from the name (e.g. "quantile_7", "equal_width_4")
        cfg = _infer_config_from_name(strategy_name, **kwargs)

    if cfg is None:
        known = ", ".join(sorted(BUILTIN_STRATEGIES))
        raise ValueError(
            f"Unknown strategy '{strategy_name}' and no strategy_config "
            f"provided.  Built-in strategies: {known}.  "
            f"Patterns: quantile_N, equal_width_N."
        )

    method = cfg.get("method", "").lower()
    # Per-strategy min_group_size override
    effective_min = int(cfg.get("min_group_size", min_group_size))
    # Adaptive flag — forces stricter repair for fixed strategies
    adaptive = cfg.get("adaptive", method in ("adaptive_quantile",))

    # -- Fixed edges ---------------------------------------------------------
    if method == "fixed":
        bins = list(cfg["bins"])
        labels = list(cfg["labels"]) if cfg.get("labels") else None

    # -- Quantile (data-driven, equal-frequency) -----------------------------
    elif method == "quantile":
        n_bins = int(cfg.get("n_bins", kwargs.get("n_bins", 5)))
        bins, labels = _quantile_bins(df, col, n_bins, strategy_name)

    # -- Equal-width (data-driven, equal-range) ------------------------------
    elif method == "equal_width":
        n_bins = int(cfg.get("n_bins", kwargs.get("n_bins", 5)))
        bins, labels = _equal_width_bins(df, col, n_bins, strategy_name)

    # -- Jenks natural breaks ------------------------------------------------
    elif method == "jenks":
        n_bins = int(cfg.get("n_bins", kwargs.get("n_bins", 5)))
        bins, labels = _jenks_bins(df, col, n_bins, strategy_name)

    # -- Adaptive quantile (quantile + forced repair) ------------------------
    elif method == "adaptive_quantile":
        n_bins = int(cfg.get("n_bins", kwargs.get("n_bins", 5)))
        bins, labels = _quantile_bins(df, col, n_bins, strategy_name)
        adaptive = True  # ensure repair runs

    else:
        raise ValueError(
            f"Unknown method '{method}' in strategy '{strategy_name}'.  "
            f"Supported methods: fixed, quantile, equal_width, jenks, "
            f"adaptive_quantile."
        )

    # -- Safeguard layer -----------------------------------------------------
    bins, labels = validate_and_repair(
        df[col],
        bins,
        labels,
        min_group_size=effective_min if adaptive else 0,
        strategy_name=strategy_name,
    )

    logger.info(f"  Created {len(bins) - 1} bins")
    return bins, labels


# ---------------------------------------------------------------------------
# Internal helpers for dynamic bin methods
# ---------------------------------------------------------------------------


def _infer_config_from_name(name: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Try to build a config dict from a conventionally-named strategy."""
    for prefix in ("quantile", "equal_width", "jenks", "adaptive_quantile"):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix.startswith("_") and suffix[1:].isdigit():
                n_bins = int(suffix[1:])
            else:
                n_bins = kwargs.get("n_bins")
            if n_bins is not None:
                return {"method": prefix, "n_bins": n_bins}
    return None


def _quantile_bins(
    df: pd.DataFrame, col: str, n_bins: int, strategy_name: str
) -> Tuple[List[float], None]:
    """Compute quantile-based bin edges."""
    logger.info(f"  Computing {n_bins} quantile bins from data")
    try:
        _, edges = pd.qcut(df[col], q=n_bins, retbins=True, duplicates="drop")
        actual = len(edges) - 1
        if actual < n_bins:
            logger.warning(f"  Only created {actual} bins due to duplicate edges")
    except ValueError as e:
        logger.error(f"  Failed to create {n_bins} quantile bins: {e}")
        logger.info(f"  Falling back to {n_bins - 1} bins")
        try:
            _, edges = pd.qcut(df[col], q=n_bins - 1, retbins=True, duplicates="drop")
        except ValueError:
            raise ValueError(
                f"Cannot create quantile bins for '{strategy_name}'.  "
                "Dataset may be too small or have insufficient unique values."
            )
    return list(edges), None


def _equal_width_bins(
    df: pd.DataFrame, col: str, n_bins: int, strategy_name: str
) -> Tuple[List[float], None]:
    """Compute equal-width bin edges spanning the data range."""
    logger.info(f"  Computing {n_bins} equal-width bins from data")
    lo, hi = float(df[col].min()), float(df[col].max())
    if lo == hi:
        raise ValueError(
            f"Cannot create equal-width bins for '{strategy_name}': "
            f"column '{col}' has zero range ({lo})."
        )
    edges = np.linspace(lo, hi, n_bins + 1)
    # Widen endpoints slightly so pd.cut(include_lowest=True) catches all values
    edges[0] -= 0.001
    edges[-1] += 0.001
    return list(edges), None


def _jenks_bins(
    df: pd.DataFrame, col: str, n_bins: int, strategy_name: str
) -> Tuple[List[float], None]:
    """Compute bin edges via Jenks natural breaks (minimises within-group variance)."""
    try:
        from jenkspy import jenks_breaks
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "jenkspy is required for Jenks natural breaks. pip install jenkspy"
        ) from exc

    logger.info(f"  Computing {n_bins} Jenks natural-break bins from data")
    values = df[col].dropna().values
    breaks = jenks_breaks(values.astype(float), n_classes=n_bins)
    # jenkspy returns n_bins+1 values including min and max
    edges = list(breaks)
    # Widen endpoints slightly to ensure all values are captured
    edges[0] -= 0.001
    edges[-1] += 0.001
    return edges, None


# ---------------------------------------------------------------------------
# Safeguard layer — clips edges, merges under-populated bins
# ---------------------------------------------------------------------------


def validate_and_repair(
    series: pd.Series,
    bins: List[float],
    labels: Optional[List[str]],
    min_group_size: int = 0,
    strategy_name: str = "unknown",
) -> Tuple[List[float], Optional[List[str]]]:
    """Clip bin edges to the data range and merge under-populated bins.

    Parameters
    ----------
    series : Series
        The raw continuous column (before binning).
    bins : list[float]
        Bin edges.
    labels : list[str] or None
        Bin labels (must have ``len(bins) - 1`` entries, or ``None``).
    min_group_size : int
        Minimum acceptable count per bin.  Set to 0 to disable merging
        (edge clipping still runs).
    strategy_name : str
        Used for log messages only.

    Returns
    -------
    (bins, labels)
        Possibly shortened after merging.
    """

    data_min, data_max = float(series.min()), float(series.max())
    bins = list(bins)

    # -- 1. Clip edges to actual data range ----------------------------------
    # Keep first edge ≤ data_min and last edge ≥ data_max; remove any
    # intermediate edges that fall outside the data range.
    if bins[0] > data_min:
        bins[0] = data_min - 0.001
    if bins[-1] < data_max:
        bins[-1] = data_max + 0.001

    # Remove intermediate edges that create empty tail bins
    while len(bins) > 2 and bins[1] <= data_min:
        bins.pop(1)
        if labels is not None and len(labels) > 1:
            logger.warning(
                f"  strategy '{strategy_name}': dropped empty leading bin "
                f"(edge {bins[0]:.1f} below data minimum {data_min:.1f})"
            )
            labels.pop(0)
    while len(bins) > 2 and bins[-2] >= data_max:
        bins.pop(-2)
        if labels is not None and len(labels) > 1:
            logger.warning(
                f"  strategy '{strategy_name}': dropped empty trailing bin "
                f"(edge {bins[-1]:.1f} above data maximum {data_max:.1f})"
            )
            labels.pop(-1)

    # Re-sync labels if they were None (auto-generated)
    if labels is not None and len(labels) != len(bins) - 1:
        labels = None  # fall back to auto labels after clipping

    if min_group_size <= 0 or len(bins) <= 2:
        return bins, labels

    # -- 2. Merge under-populated bins ---------------------------------------
    # Do a trial cut, then iteratively merge the smallest bin with its
    # smaller neighbour until all bins meet the threshold.
    max_merges = len(bins) - 2  # can't merge below 1 bin
    for _ in range(max_merges):
        trial = pd.cut(series, bins=bins, include_lowest=True)
        counts = trial.value_counts(sort=False)
        # Find first bin below threshold
        below = [(i, c) for i, c in enumerate(counts) if c < min_group_size]
        if not below:
            break
        idx, cnt = min(below, key=lambda x: x[1])

        n_bins_now = len(bins) - 1
        if n_bins_now <= 1:
            break

        # Decide merge direction: left or right neighbour (pick smaller)
        if idx == 0:
            merge_right = True
        elif idx == n_bins_now - 1:
            merge_right = False
        else:
            left_count = int(counts.iloc[idx - 1])
            right_count = int(counts.iloc[idx + 1])
            merge_right = right_count <= left_count

        if merge_right:
            removed_edge = bins.pop(idx + 1)
        else:
            removed_edge = bins.pop(idx)

        old_label = None
        if labels is not None and len(labels) == n_bins_now:
            merge_idx = idx if merge_right else idx - 1
            old_label = labels.pop(max(merge_idx, 0))

        logger.warning(
            f"  strategy '{strategy_name}': merged bin with {cnt} samples "
            f"(edge {removed_edge:.1f} removed, label '{old_label}' dropped) "
            f"— below min_group_size={min_group_size}"
        )

    # If labels got out of sync, fall back to None (auto)
    if labels is not None and len(labels) != len(bins) - 1:
        labels = None

    return bins, labels


def apply_binning(
    df: pd.DataFrame,
    bins: List[float],
    labels: Optional[List[str]],
    col: str = "age_raw",
    output_col: str = "age_group_exp",
) -> pd.DataFrame:
    """
    Apply binning to a DataFrame column.

    Args:
        df: Input DataFrame
        bins: Bin edges (from create_binning_strategy)
        labels: Bin labels (or None for auto-generation)
        col: Column containing numeric values to bin
        output_col: Name for the new binned column

    Returns:
        DataFrame with new binned column added

    Example:
        >>> bins, labels = create_binning_strategy(df, 'clinical')
        >>> df_binned = apply_binning(df, bins, labels)
        >>> print(df_binned['age_group_exp'].value_counts())
    """
    df = df.copy()
    df[output_col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    logger.info(f"Applied binning: {df[output_col].nunique()} unique groups created")
    return df


def sensitive_attribute_distribution(
    df: pd.DataFrame, bin_col: str, sensitive_col: str = "sex"
) -> pd.DataFrame:
    """
    Calculate percentage distribution of sensitive attribute within each bin.

    Useful for understanding if certain sensitive groups are concentrated
    in specific age bins (representation bias).

    Args:
        df: DataFrame with binned age data
        bin_col: Column with age bins
        sensitive_col: Sensitive attribute column (e.g., 'sex')

    Returns:
        DataFrame with percentage distribution per bin
        Columns: [bin_col, <sensitive_value_1>_pct, <sensitive_value_2>_pct, ...]

    Example:
        >>> dist = sensitive_attribute_distribution(df, 'age_group_exp', 'sex')
        >>> print(dist)
        age_group_exp  female_pct  male_pct
        <40                  45.2      54.8
        40-49                38.1      61.9
    """
    # Validation
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    if df[bin_col].isna().any():
        n_missing = df[bin_col].isna().sum()
        logger.warning(f"Found {n_missing} NaN values in {bin_col}, they will be excluded")
        df = df[df[bin_col].notna()]

    grouped = (
        df.groupby(bin_col, observed=True)[sensitive_col]
        .value_counts(normalize=True)
        .rename("pct")
        .reset_index()
    )

    pivot = grouped.pivot(index=bin_col, columns=sensitive_col, values="pct").fillna(0)
    pivot = (pivot * 100).round(2)

    # Rename columns to be more descriptive
    new_cols = {col: f"{str(col).lower()}_pct" for col in pivot.columns}
    pivot = pivot.rename(columns=new_cols)

    logger.info(f"Calculated sensitive attribute distribution for {len(pivot)} bins")
    return pivot.reset_index()


def compute_fairness_metrics(
    df: pd.DataFrame, bin_col: str, target_col: str = "heart_disease"
) -> Dict:
    """
    Compute fairness-relevant metrics for an attribute binning strategy.

    Metrics include:
    - Group size statistics (min, max, mean, balance coefficient of variation)
    - Statistical parity: positive rate per group and max difference
    - Overall positive rate for context

    Args:
        df: DataFrame with binned age data and target
        bin_col: Column with age bins
        target_col: Binary target variable column

    Returns:
        Dictionary with fairness metrics
        Keys: n_groups, min_group_size, max_group_size, mean_group_size,
              group_balance_cv, max_sp_difference, overall_positive_rate,
              group_sizes (dict), positive_rates_by_group (dict)

    Example:
        >>> metrics = compute_fairness_metrics(df, 'age_group_exp')
        >>> print(f"Max statistical parity diff: {metrics['max_sp_difference']:.3f}")
    """
    # Group size statistics
    group_counts = df[bin_col].value_counts()

    # Positive rate per group (statistical parity indicator)
    positive_rates = df.groupby(bin_col, observed=True)[target_col].mean()

    # Coefficient of variation for group balance (lower = more balanced)
    cv = float(group_counts.std() / group_counts.mean())

    # Statistical parity difference (max - min positive rate)
    sp_diff = float(positive_rates.max() - positive_rates.min())

    # Demographic parity ratio (min / max positive rate)
    max_rate = float(positive_rates.max())
    min_rate = float(positive_rates.min())
    dp_ratio = (min_rate / max_rate) if max_rate > 0 else 0.0

    # Max within-bin label imbalance: largest deviation of any bin's
    # positive rate from the dataset-wide positive rate.
    overall_rate = float(df[target_col].mean())
    label_imbalance = float((positive_rates - overall_rate).abs().max())

    metrics = {
        "n_groups": len(group_counts),
        "min_group_size": int(group_counts.min()),
        "max_group_size": int(group_counts.max()),
        "mean_group_size": float(group_counts.mean()),
        "group_balance_cv": cv,
        "max_sp_difference": sp_diff,
        "demographic_parity_ratio": round(dp_ratio, 4),
        "max_within_bin_label_imbalance": round(label_imbalance, 4),
        "overall_positive_rate": overall_rate,
        "group_sizes": {str(k): int(v) for k, v in group_counts.to_dict().items()},
        "positive_rates_by_group": {str(k): float(v) for k, v in positive_rates.to_dict().items()},
    }

    logger.info(
        f"Computed fairness metrics: {metrics['n_groups']} groups, "
        f"SP diff={sp_diff:.3f}, balance CV={cv:.3f}"
    )

    return metrics


# ---------------------------------------------------------------------------
# Cross-attribute fairness impact
# ---------------------------------------------------------------------------


def compute_cross_attribute_impact(
    df: pd.DataFrame,
    bin_col: str,
    sensitive_cols: List[str],
    target_col: str = "heart_disease",
) -> Dict[str, Dict[str, Any]]:
    """Measure how a binning strategy affects SP of *other* sensitive attrs.

    For every sensitive attribute in *sensitive_cols* that is **not** the
    binned column itself:

    1. Compute the overall (global) SP difference for that attribute.
    2. Compute the SP difference for that attribute **within each bin
       group** and take the max.
    3. Report the delta (within-bin max − global).

    A large positive delta means the binning amplifies existing disparities
    for that attribute inside certain groups.

    Parameters
    ----------
    df : DataFrame
        DataFrame that already contains *bin_col* (e.g. after
        ``apply_binning``).
    bin_col : str
        The binned column (e.g. ``age_group_exp``).
    sensitive_cols : list[str]
        All sensitive attribute columns to evaluate.
    target_col : str
        Binary target variable.

    Returns
    -------
    dict
        Mapping ``{attr: {global_sp, max_within_bin_sp, delta, per_bin}}``
        where *per_bin* maps bin labels to per-group SP values.
    """
    impact: Dict[str, Dict[str, Any]] = {}

    for attr in sensitive_cols:
        # Skip if the attribute *is* the binned column or not present
        if attr == bin_col or attr not in df.columns:
            continue
        if df[attr].nunique(dropna=True) < 2:
            continue

        # Global SP for this attr
        global_rates = df.groupby(attr, observed=True)[target_col].mean()
        global_sp = float(global_rates.max() - global_rates.min())

        # Within-bin SP: for each bin, compute SP of this attr
        per_bin: Dict[str, float] = {}
        for bin_label, group_df in df.groupby(bin_col, observed=True):
            if group_df[attr].nunique(dropna=True) < 2:
                per_bin[str(bin_label)] = 0.0
                continue
            rates = group_df.groupby(attr, observed=True)[target_col].mean()
            per_bin[str(bin_label)] = float(rates.max() - rates.min())

        max_within = max(per_bin.values()) if per_bin else 0.0
        delta = round(max_within - global_sp, 4)

        impact[attr] = {
            "global_sp": round(global_sp, 4),
            "max_within_bin_sp": round(max_within, 4),
            "delta": delta,
            "per_bin": {k: round(v, 4) for k, v in per_bin.items()},
        }

        logger.info(
            f"  Cross-attr impact [{attr}]: global_sp={global_sp:.4f}, "
            f"max_within_bin_sp={max_within:.4f}, delta={delta:+.4f}"
        )

    return impact


def analyze_strategy_comprehensive(
    df: pd.DataFrame,
    strategy_name: str,
    bins: List[float],
    labels: Optional[List[str]],
    dataset_name: str,
    col: str = "age_raw",
    sensitive_col: Union[str, List[str]] = "sex",
    target_col: str = "heart_disease",
) -> Dict:
    """
    Comprehensive analysis of a single binning strategy.

    Combines:
    - Fairness metrics (group balance, statistical parity)
    - Sensitive attribute distribution within bins
    - Cross-attribute fairness impact (ΔSP for every other sensitive attr)
    - Overall population statistics

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    strategy_name : str
        Name of binning strategy.
    bins : list
        Bin edges.
    labels : list or None
        Bin labels.
    dataset_name : str
        Dataset identifier for tracking.
    col : str
        Column with continuous values.
    sensitive_col : str or list[str]
        One or more sensitive attribute columns.  The first element
        is the "primary" (used for within-bin distribution); all others
        contribute to the cross-attribute impact matrix.
        Legacy: a plain ``str`` is accepted and wrapped automatically.
    target_col : str
        Binary target variable column.

    Returns
    -------
    dict
        Complete analysis results with keys: dataset, strategy,
        fairness_metrics, sensitive_distribution,
        overall_sensitive_distribution, cross_attribute_impact,
        bins, labels.
    """
    # Normalise sensitive_col → list
    if isinstance(sensitive_col, str):
        sensitive_cols = [sensitive_col]
    else:
        sensitive_cols = list(sensitive_col)

    primary_attr = sensitive_cols[0]

    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing strategy: {strategy_name} on {dataset_name}")
    logger.info(f"{'='*60}")

    # Apply binning
    df_binned = apply_binning(df, bins, labels, col, "age_group_exp")

    # Fairness metrics
    fairness = compute_fairness_metrics(df_binned, "age_group_exp", target_col)

    # Sensitive attribute distribution within bins (primary attr)
    sensitive_dist = sensitive_attribute_distribution(df_binned, "age_group_exp", primary_attr)

    # Overall sensitive attribute distribution (for context)
    overall_sensitive = df[primary_attr].value_counts(normalize=True)
    overall_sensitive_pct = {str(k): round(v * 100, 2) for k, v in overall_sensitive.items()}

    # Cross-attribute fairness impact
    cross_impact = compute_cross_attribute_impact(
        df_binned,
        bin_col="age_group_exp",
        sensitive_cols=sensitive_cols,
        target_col=target_col,
    )

    result = {
        "dataset": dataset_name,
        "strategy": strategy_name,
        "fairness_metrics": fairness,
        "sensitive_distribution": sensitive_dist.to_dict(orient="records"),
        "overall_sensitive_distribution": overall_sensitive_pct,
        "cross_attribute_impact": cross_impact,
        "bins": [float(b) for b in bins],
        "labels": labels if labels else "auto-generated",
    }

    logger.info(f"Analysis complete for {strategy_name}")

    return result


def compare_strategies(results: List[Dict], by_dataset: bool = True) -> pd.DataFrame:
    """
    Create comparison table across multiple binning strategies.

    Extracts key metrics from analysis results for easy comparison.

    Args:
        results: List of analysis results from analyze_strategy_comprehensive
        by_dataset: Whether to group by dataset first

    Returns:
        Comparison DataFrame with columns:
        [dataset, strategy, n_groups, min_group_size, max_group_size,
         group_balance_cv, max_sp_difference]

    Example:
        >>> results = []
        >>> for strategy in ['fixed_10yr', 'clinical', 'quantile_5']:
        ...     bins, labels = create_binning_strategy(df, strategy)
        ...     result = analyze_strategy_comprehensive(df, strategy, bins, labels, 'cleveland')
        ...     results.append(result)
        >>> comparison = compare_strategies(results)
        >>> print(comparison.to_string(index=False))
    """
    logger.info(f"Creating comparison table for {len(results)} strategy configurations")

    comparison_data = []

    for result in results:
        metrics = result["fairness_metrics"]
        comparison_data.append(
            {
                "dataset": result["dataset"],
                "strategy": result["strategy"],
                "n_groups": metrics["n_groups"],
                "min_group_size": metrics["min_group_size"],
                "max_group_size": metrics["max_group_size"],
                "group_balance_cv": round(metrics["group_balance_cv"], 3),
                "max_sp_difference": round(metrics["max_sp_difference"], 3),
                "demographic_parity_ratio": metrics.get("demographic_parity_ratio", float("nan")),
                "max_label_imbalance": metrics.get("max_within_bin_label_imbalance", float("nan")),
            }
        )

    df = pd.DataFrame(comparison_data)

    if by_dataset:
        df = df.sort_values(["dataset", "strategy"])
    else:
        df = df.sort_values(["strategy", "dataset"])

    logger.info(f"Comparison table created with {len(df)} rows")

    return df


def compute_strategy_score(
    result: Dict,
    sample_size_weight: float = 0.40,
    balance_weight: float = 0.30,
    fairness_weight: float = 0.30,
) -> float:
    """
    Compute overall score for a binning strategy.

    Score components:
    1. Sample size: Penalize small min group size (need statistical power)
    2. Balance: Reward low CV (even group sizes)
    3. Fairness: Reward low statistical parity difference

    Args:
        result: Analysis result from analyze_strategy_comprehensive
        sample_size_weight: Weight for sample size component [0-1]
        balance_weight: Weight for balance component [0-1]
        fairness_weight: Weight for fairness component [0-1]

    Returns:
        Overall score (higher is better)

    Note:
        Weights should sum to 1.0 for interpretable scores

    Example:
        >>> score = compute_strategy_score(result, 0.4, 0.3, 0.3)
        >>> print(f"Strategy score: {score:.3f}")
    """
    metrics = result["fairness_metrics"]

    # Sample size score: min_group_size / mean_group_size (normalize to [0,1])
    sample_score = min(metrics["min_group_size"] / metrics["mean_group_size"], 1.0)

    # Balance score: 1 - (CV / MAX_EXPECTED_CV)
    balance_score = max(1.0 - (metrics["group_balance_cv"] / MAX_EXPECTED_CV), 0.0)

    # Fairness score: 1 - (sp_diff / MAX_EXPECTED_SP_DIFF)
    fairness_score = max(1.0 - (metrics["max_sp_difference"] / MAX_EXPECTED_SP_DIFF), 0.0)

    # Weighted combination
    overall_score = (
        sample_size_weight * sample_score
        + balance_weight * balance_score
        + fairness_weight * fairness_score
    )

    logger.info(
        f"Computed score for {result['strategy']}: {overall_score:.3f} "
        f"(sample={sample_score:.3f}, balance={balance_score:.3f}, "
        f"fairness={fairness_score:.3f})"
    )

    return overall_score


def generate_summary_report(
    results: List[Dict], output_file: Path, scoring_weights: Dict[str, float] = None
):
    """
    Generate markdown summary report for attribute binning analysis.

    Includes:
    - Overview statistics
    - Comparison table
    - Strategy scores
    - Interpretation guidelines

    Args:
        results: List of analysis results
        output_file: Path to save markdown report
        scoring_weights: Optional dict with 'sample_size', 'balance', 'fairness' weights

    Example:
        >>> generate_summary_report(results, Path('output/attribute_binning_report.md'))
    """
    logger.info(f"Generating summary report: {output_file}")

    if scoring_weights is None:
        scoring_weights = {"sample_size": 0.40, "balance": 0.30, "fairness": 0.30}

    # Compute scores for ranking
    for result in results:
        result["score"] = compute_strategy_score(
            result,
            scoring_weights["sample_size"],
            scoring_weights["balance"],
            scoring_weights["fairness"],
        )

    comparison = compare_strategies(results)

    # Add scores to comparison
    scores = {(r["dataset"], r["strategy"]): r["score"] for r in results}
    comparison["score"] = comparison.apply(
        lambda row: scores.get((row["dataset"], row["strategy"]), 0.0), axis=1
    )
    comparison["score"] = comparison["score"].round(3)

    # Generate report content
    report = ["# Attribute Binning Strategy Analysis Report\n"]

    report.append("## Overview\n")
    report.append(f"- **Datasets analyzed**: {len(set(r['dataset'] for r in results))}")
    report.append(f"- **Strategies tested**: {len(set(r['strategy'] for r in results))}")
    report.append(f"- **Total configurations**: {len(results)}\n")

    report.append("## Scoring Weights\n")
    report.append(f"- Sample Size: {scoring_weights['sample_size']:.0%}")
    report.append(f"- Group Balance: {scoring_weights['balance']:.0%}")
    report.append(f"- Fairness Sensitivity: {scoring_weights['fairness']:.0%}\n")

    report.append("## Comparison Table\n")
    report.append("```")
    report.append(comparison.to_string(index=False))
    report.append("```\n")

    report.append("## Top Strategies by Score\n")
    top_strategies = comparison.nlargest(5, "score")
    report.append("```")
    report.append(top_strategies[["dataset", "strategy", "score"]].to_string(index=False))
    report.append("```\n")

    report.append("## Interpretation Guidelines\n")
    report.append("### Statistical Parity Difference:")
    report.append("- **< 0.10**: Low bias - groups have similar positive rates")
    report.append("- **0.10-0.20**: Moderate bias - may require mitigation")
    report.append("- **> 0.20**: High bias - strong fairness concerns\n")

    report.append("### Group Balance CV (Coefficient of Variation):")
    report.append("- **< 0.30**: Well-balanced groups")
    report.append("- **0.30-0.50**: Moderate imbalance")
    report.append("- **> 0.50**: Highly imbalanced - consider different strategy\n")

    report.append("### Sample Size:")
    report.append("- **min_group_size ≥ 30**: Adequate for statistical tests")
    report.append("- **min_group_size < 30**: Risk of unstable estimates\n")

    report.append("## Recommendations\n")
    report.append("1. **Prioritize strategies with score ≥ 0.70**")
    report.append("2. **Ensure min_group_size ≥ 30** for statistical validity")
    report.append("3. **Test top strategies** across both datasets for consistency")
    report.append("4. **Consider clinical interpretability** in final selection\n")

    # -----------------------------------------------------------------
    # Cross-Attribute Fairness Impact
    # -----------------------------------------------------------------
    _append_cross_attribute_section(report, results)

    # Write report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(report))

    logger.info(f"Summary report saved: {output_file}")


def _append_cross_attribute_section(report: List[str], results: List[Dict]) -> None:
    """Append a Cross-Attribute Fairness Impact section to *report*.

    Builds a table:  strategy × other-sensitive-attr → ΔSP
    (delta = max within-bin SP − global SP).
    """
    # Collect all cross-attr data across results
    all_attrs: set[str] = set()
    for r in results:
        all_attrs.update(r.get("cross_attribute_impact", {}).keys())

    if not all_attrs:
        return

    sorted_attrs = sorted(all_attrs)

    report.append("## Cross-Attribute Fairness Impact\n")
    report.append(
        "How each binning strategy affects statistical parity of **other** "
        "sensitive attributes.  ΔSP = max(within-bin SP) − global SP.  "
        "Positive values mean the binning *amplifies* disparities inside "
        "at least one bin.\n"
    )

    # Group by dataset
    datasets = sorted({r["dataset"] for r in results})
    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        if not ds_results:
            continue

        report.append(f"### {ds}\n")

        # Header row
        header = "| Strategy | " + " | ".join(f"ΔSP ({a})" for a in sorted_attrs) + " |"
        sep = "|" + "|".join(["---"] * (len(sorted_attrs) + 1)) + "|"
        report.append(header)
        report.append(sep)

        for r in ds_results:
            impact = r.get("cross_attribute_impact", {})
            cells = []
            for attr in sorted_attrs:
                info = impact.get(attr)
                if info is None:
                    cells.append("—")
                else:
                    d = info["delta"]
                    cells.append(f"{d:+.4f}")
            row = f"| {r['strategy']} | " + " | ".join(cells) + " |"
            report.append(row)

        report.append("")  # blank line after table
