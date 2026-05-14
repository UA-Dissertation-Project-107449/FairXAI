"""Fairness-specific visualizations."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fairxai.viz.labels import display_mitigation
from fairxai.viz.save_utils import heatmap_size, save_figure

logger = logging.getLogger(__name__)

_THRESHOLD = 0.10  # standard fairness gap threshold line


def plot_fairness_metric_heatmap(df, sensitive_attr, output_file):
    """Heatmap of fairness metric gaps per mitigation technique.

    Rows = mitigation technique (non-baseline rows only), columns = three
    fairness metrics for ``sensitive_attr``.  Cell values are means over all
    binning strategies present in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        ``full_comparison.csv`` content.  Expected columns include
        ``mitigation_technique``, ``binning_strategy``, and the per-attribute
        fairness columns ``dem_parity_{attr}_max_diff``,
        ``eq_odds_{attr}_tpr_diff``, ``eq_odds_{attr}_fpr_diff``.
    sensitive_attr : str
        Attribute key as used in column names, e.g. ``"age_group_cat"``
        or ``"sex_cat"``.
    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """
    # Column names in full_comparison.csv use short attr (without _cat suffix)
    attr_short = sensitive_attr.replace("_cat", "")
    metric_cols = {
        f"dem_parity_{attr_short}_max_diff": "Dem. Parity ΔMax",
        f"eq_odds_{attr_short}_tpr_diff": "Eq. Odds ΔTPR",
        f"eq_odds_{attr_short}_fpr_diff": "Eq. Odds ΔFPR",
    }

    available = {k: v for k, v in metric_cols.items() if k in df.columns}
    if not available:
        logger.warning(
            "plot_fairness_metric_heatmap: no metric cols found for attr '%s'", sensitive_attr
        )
        return None

    plot_df = df[df["mitigation_technique"] != "baseline"].copy()
    if plot_df.empty:
        return None

    agg = (
        plot_df.groupby("mitigation_technique")[list(available.keys())]
        .mean()
        .rename(columns=available)
    )
    if agg.empty:
        return None
    agg.index = [display_mitigation(idx) for idx in agg.index]

    attr_label = attr_short.replace("_", " ").title()
    width, height = heatmap_size(agg.index, len(available), min_width=10, min_height=5)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        agg,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        center=_THRESHOLD,
        vmin=0,
        vmax=0.5,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Fairness Metric Gaps by Mitigation — {attr_label}", fontsize=12)
    ax.set_xlabel("Fairness Metric")
    ax.set_ylabel("Mitigation Technique")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    save_figure(fig, output_file, dpi=300)
    plt.close(fig)
    return output_file


def plot_group_performance_gaps(before_json, after_json, sensitive_attr, output_file):
    """Grouped bar chart of per-group TPR / FPR / precision before and after mitigation.

    Parameters
    ----------
    before_json : path-like or dict
        Stage-6 baseline fairness assessment JSON (path or already-loaded dict).
    after_json : path-like or dict
        Experiment fairness assessment JSON with the same structure.
    sensitive_attr : str
        Attribute key, e.g. ``"age_group_cat"`` or ``"sex_cat"``.
    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """

    def _load(src):
        if isinstance(src, dict):
            return src
        return json.loads(Path(src).read_text())

    def _extract(data, attr):
        records = []
        # Support both top-level and nested under "test_metrics"
        root = data.get("test_metrics", data)
        gf = root.get("group_fairness", {}).get(attr, {})
        eq_odds = gf.get("equalized_odds", {}).get("group_metrics", {})
        pp = gf.get("predictive_parity", {}).get("group_precision", {})
        groups = sorted(set(list(eq_odds) + list(pp)))
        for g in groups:
            if g in eq_odds:
                records.append(
                    {"group": g, "metric": "TPR", "value": eq_odds[g].get("tpr", float("nan"))}
                )
                records.append(
                    {"group": g, "metric": "FPR", "value": eq_odds[g].get("fpr", float("nan"))}
                )
            if g in pp:
                records.append(
                    {
                        "group": g,
                        "metric": "Precision",
                        "value": pp[g].get("precision", float("nan")),
                    }
                )
        return records

    try:
        before_data = _load(before_json)
        after_data = _load(after_json)
    except Exception as exc:
        logger.warning("plot_group_performance_gaps: could not load JSONs: %s", exc)
        return None

    before_records = _extract(before_data, sensitive_attr)
    after_records = _extract(after_data, sensitive_attr)
    if not before_records and not after_records:
        return None

    before_df = pd.DataFrame(before_records)
    before_df["condition"] = "Baseline"
    after_df = pd.DataFrame(after_records)
    after_df["condition"] = "After Mitigation"
    combined = pd.concat([before_df, after_df], ignore_index=True)

    metrics = combined["metric"].unique().tolist()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 4.5, 4.5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    attr_label = sensitive_attr.replace("_cat", "").replace("_", " ").title()
    palette = {"Baseline": "#AAAAAA", "After Mitigation": "#0072B2"}

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        sub = combined[combined["metric"] == metric]
        sns.barplot(data=sub, x="group", y="value", hue="condition", palette=palette, ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("Group")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
        if i > 0 and ax.get_legend():
            ax.get_legend().remove()

    fig.suptitle(f"Per-Group Performance Gaps — {attr_label}", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def plot_bias_amplification_waterfall(stages_dict, output_file):
    """Waterfall chart showing how fairness gap changes across pipeline stages.

    Each bar represents the fairness gap at one pipeline stage; colour encodes
    whether it improved (green) or worsened (red) relative to the previous stage.

    Parameters
    ----------
    stages_dict : dict[str, float]
        Ordered mapping of stage name → fairness gap value.  Example::

            {
                "raw_data": 0.31,
                "preprocessed": 0.28,
                "trained_baseline": 0.22,
                "mitigated": 0.14,
            }

    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """
    if not stages_dict or len(stages_dict) < 2:
        return None

    stages = list(stages_dict.keys())
    values = [stages_dict[s] for s in stages]
    deltas = [0.0] + [values[i] - values[i - 1] for i in range(1, len(values))]

    colors = []
    for i, d in enumerate(deltas):
        if i == 0:
            colors.append("#888888")
        elif d <= 0:
            colors.append("#2E8B57")  # green: gap decreased → improved
        else:
            colors.append("#B22222")  # red: gap increased → worsened

    fig, ax = plt.subplots(figsize=(max(8, len(stages) * 1.8), 5))
    bars = ax.bar(stages, values, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)

    for bar, val, delta in zip(bars, values, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        if delta != 0.0:
            sign = "+" if delta > 0 else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{sign}{delta:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    ax.axhline(
        y=_THRESHOLD,
        color="orange",
        linestyle="--",
        linewidth=1.2,
        label=f"Fairness threshold ({_THRESHOLD})",
    )
    ax.set_xlabel("Pipeline Stage")
    ax.set_ylabel("Fairness Gap")
    ax.set_title("Bias Amplification Across Pipeline Stages")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s.replace("_", " ").title() for s in stages], rotation=25, ha="right")
    ax.set_ylim(0, max(values) * 1.25)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file
