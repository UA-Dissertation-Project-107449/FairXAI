"""Cluster evidence visualisation for dissertation figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fairxai.viz.save_utils import save_figure

logger = logging.getLogger(__name__)

# Consistent palette for cluster IDs (up to 10 clusters)
_CLUSTER_PALETTE = [
    "#0072B2",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
    "#999999",
    "#CC6677",
]


def save_cluster_profile_bars(
    fairness_df: pd.DataFrame,
    out_path: Path,
) -> Path | None:
    """Bar chart of cluster sizes and mean fairness disparity.

    Uses ``fairness_by_cluster.csv`` (columns: cluster_id, sensitive_attr,
    dp_max_diff, eo_tpr_diff, eo_fpr_diff, n_samples, is_fair).  Plots two
    sub-panels:
    - top: sample count per cluster (one bar per cluster, annotated with N);
    - bottom: mean dp_max_diff per cluster across all sensitive attributes.

    Clusters with ``n_samples < 10`` are annotated with a warning marker.
    """
    out_path = Path(out_path)

    required = {"cluster_id", "n_samples", "dp_max_diff"}
    if fairness_df is None or fairness_df.empty:
        logger.warning("[SKIP] save_cluster_profile_bars: empty dataframe")
        return None
    if not required.issubset(fairness_df.columns):
        missing = required - set(fairness_df.columns)
        logger.warning("[SKIP] save_cluster_profile_bars: missing columns %s", missing)
        return None

    df = fairness_df.copy()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df["n_samples"] = pd.to_numeric(df["n_samples"], errors="coerce")
    df["dp_max_diff"] = pd.to_numeric(df["dp_max_diff"], errors="coerce")

    # Per-cluster aggregates
    # n_samples should be the same across sensitive_attr rows for the same cluster
    cluster_sizes = (
        df.groupby("cluster_id")["n_samples"].first().sort_index()
    )
    cluster_dp = (
        df.groupby("cluster_id")["dp_max_diff"].mean().sort_index()
    )
    cluster_ids = cluster_sizes.index.tolist()
    colors = [_CLUSTER_PALETTE[int(c) % len(_CLUSTER_PALETTE)] for c in cluster_ids]
    small_bin = cluster_sizes < 10

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(5, len(cluster_ids) * 1.2 + 1.5), 6),
                                          gridspec_kw={"height_ratios": [1.6, 1]})

    # Top panel: sample counts
    bars = ax_top.bar(cluster_ids, cluster_sizes.values, color=colors, edgecolor="white")
    for bar, cid, small in zip(bars, cluster_ids, small_bin):
        h = bar.get_height()
        label = f"N={int(h)}"
        if small:
            label += " ⚠"
        ax_top.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#CC0000" if small else "#333333",
        )
    ax_top.set_ylabel("Sample count")
    ax_top.set_title("Cluster Size and Fairness Disparity")
    ax_top.set_xticks(cluster_ids)
    ax_top.set_xticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=8)

    # Bottom panel: mean DP gap
    ax_bot.bar(cluster_ids, cluster_dp.values, color=colors, edgecolor="white")
    ax_bot.axhline(0.1, color="#CC0000", linewidth=0.8, linestyle="--", label="0.10 threshold")
    ax_bot.set_ylabel("Mean DP max-diff")
    ax_bot.set_xticks(cluster_ids)
    ax_bot.set_xticklabels([f"Cluster {c}" for c in cluster_ids], fontsize=8)
    ax_bot.set_ylim(0, max(cluster_dp.max() * 1.2, 0.15))
    ax_bot.legend(fontsize=7)

    if small_bin.any():
        fig.text(
            0.5,
            0.01,
            "⚠ clusters with N<10 are statistically fragile — interpret with caution",
            ha="center",
            fontsize=7,
            style="italic",
            color="#CC0000",
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)
    return out_path


def save_cluster_fairness_heatmap(
    fairness_df: pd.DataFrame,
    out_path: Path,
) -> Path | None:
    """Heatmap of fairness metrics per cluster and sensitive attribute.

    Uses ``fairness_by_cluster.csv`` (columns: cluster_id, sensitive_attr,
    dp_max_diff, eo_tpr_diff, eo_fpr_diff, n_samples, is_fair).  Produces
    one heatmap per sensitive_attr (side by side if multiple).  Rows = cluster,
    columns = [dp_max_diff, eo_tpr_diff, eo_fpr_diff], cells annotated with
    rounded values.  Red tones indicate higher disparity.
    """
    import seaborn as sns

    out_path = Path(out_path)

    required = {"cluster_id", "sensitive_attr", "dp_max_diff", "eo_tpr_diff", "eo_fpr_diff"}
    if fairness_df is None or fairness_df.empty:
        logger.warning("[SKIP] save_cluster_fairness_heatmap: empty dataframe")
        return None
    if not required.issubset(fairness_df.columns):
        missing = required - set(fairness_df.columns)
        logger.warning("[SKIP] save_cluster_fairness_heatmap: missing columns %s", missing)
        return None

    df = fairness_df.copy()
    for col in ["cluster_id", "dp_max_diff", "eo_tpr_diff", "eo_fpr_diff"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    metrics = ["dp_max_diff", "eo_tpr_diff", "eo_fpr_diff"]
    metric_labels = {"dp_max_diff": "DP gap", "eo_tpr_diff": "EO TPR gap", "eo_fpr_diff": "EO FPR gap"}
    attrs = sorted(df["sensitive_attr"].dropna().unique())
    n_attrs = max(len(attrs), 1)

    fig, axes = plt.subplots(1, n_attrs, figsize=(5 * n_attrs + 0.5, max(3, df["cluster_id"].nunique() * 0.6 + 1.5)))
    axes_flat = np.atleast_1d(axes)

    for ax_idx, attr in enumerate(attrs):
        sub = df[df["sensitive_attr"] == attr].set_index("cluster_id")[metrics].sort_index()
        sub.columns = [metric_labels.get(c, c) for c in sub.columns]
        ax = axes_flat[ax_idx]
        sns.heatmap(
            sub,
            ax=ax,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"Fairness by Cluster — {attr.replace('_', ' ').title()}")
        ax.set_ylabel("Cluster ID" if ax_idx == 0 else "")
        ax.set_xlabel("")

    fig.suptitle("Cluster-Level Fairness Disparities", fontsize=11, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=300)
    plt.close(fig)
    return out_path
