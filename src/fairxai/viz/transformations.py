"""Transformation-impact visualizations."""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_BEFORE_COLOR = "#AAAAAA"
_AFTER_COLOR = "#0072B2"


def plot_transformation_impact(before_dict, after_dict, output_file):
    """Side-by-side bar chart comparing metrics before and after a mitigation.

    Parameters
    ----------
    before_dict : dict[str, float]
        Metric values before the transformation.
        Expected keys (any subset): ``f1``, ``recall``, ``precision``,
        ``auc_roc``, ``fairness_gap``.
    after_dict : dict[str, float]
        Same structure as ``before_dict``, after the transformation.
    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """
    if not before_dict or not after_dict:
        return None

    metrics = [k for k in before_dict if k in after_dict]
    if not metrics:
        return None

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.6), 5))

    ax.bar(
        x - width / 2,
        [before_dict[m] for m in metrics],
        width,
        label="Before",
        color=_BEFORE_COLOR,
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        [after_dict[m] for m in metrics],
        width,
        label="After",
        color=_AFTER_COLOR,
        alpha=0.85,
    )

    for i, m in enumerate(metrics):
        delta = after_dict[m] - before_dict[m]
        sign = "+" if delta >= 0 else ""
        y_top = max(before_dict[m], after_dict[m])
        ax.text(x[i], y_top + 0.02, f"{sign}{delta:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], rotation=20, ha="right")
    ax.set_ylabel("Value")
    ax.set_ylim(
        0, min(1.15, max(list(before_dict.values()) + list(after_dict.values())) * 1.25 + 0.05)
    )
    ax.set_title("Transformation Impact: Before vs After")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def plot_before_after_distributions(X_before, X_after, feature_cols, output_file):
    """Overlaid KDE plots per feature, sorted by KS statistic (largest shift first).

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature data before transformation (e.g., pre-SMOTE training split).
    X_after : pd.DataFrame
        Feature data after transformation (e.g., post-SMOTE training split).
    feature_cols : list[str]
        Numeric feature columns to plot.
    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """
    if not feature_cols:
        return None

    try:
        from scipy import stats as sp_stats
    except ImportError:
        logger.warning("scipy not installed; install fairxai[experiment] for KS statistics")
        sp_stats = None

    ks_stats = {}
    for col in feature_cols:
        a = X_before[col].dropna().values
        b = X_after[col].dropna().values
        if sp_stats is not None and len(a) > 1 and len(b) > 1:
            ks_stat, _ = sp_stats.ks_2samp(a, b)
            ks_stats[col] = ks_stat
        else:
            ks_stats[col] = 0.0

    sorted_cols = sorted(feature_cols, key=lambda c: ks_stats.get(c, 0), reverse=True)

    n_cols = 3
    n_rows = math.ceil(len(sorted_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(sorted_cols):
        ax = axes_flat[i]
        a = X_before[col].dropna()
        b = X_after[col].dropna()
        sns.kdeplot(a, ax=ax, color=_BEFORE_COLOR, label="Before", fill=True, alpha=0.3)
        sns.kdeplot(b, ax=ax, color=_AFTER_COLOR, label="After", fill=True, alpha=0.3)
        ax.set_title(f"{col}\nKS={ks_stats[col]:.3f}", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if i == 0:
            ax.legend(fontsize=7)
        elif ax.get_legend():
            ax.get_legend().remove()

    for j in range(len(sorted_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions: Before vs After Transformation", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def plot_scaling_effects(X_raw, X_scaled, output_file):
    """Box plots comparing raw vs scaled numeric features side by side.

    Parameters
    ----------
    X_raw : pd.DataFrame
        Raw (unscaled) feature values.
    X_scaled : pd.DataFrame
        Scaled feature values (same columns as ``X_raw``).
    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """
    cols = [c for c in X_raw.columns if c in X_scaled.columns]
    if not cols:
        return None

    raw_long = X_raw[cols].melt(var_name="feature", value_name="value")
    raw_long["condition"] = "raw"
    scaled_long = X_scaled[cols].melt(var_name="feature", value_name="value")
    scaled_long["condition"] = "scaled"
    combined = pd.concat([raw_long, scaled_long], ignore_index=True)

    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.8), 5))
    sns.boxplot(
        data=combined,
        x="feature",
        y="value",
        hue="condition",
        palette={"raw": _BEFORE_COLOR, "scaled": _AFTER_COLOR},
        ax=ax,
        linewidth=0.8,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Scaling Effects: Raw vs Scaled Feature Distributions")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    ax.legend(title="Condition")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file
