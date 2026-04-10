"""
Experiment comparison plots — heatmaps, scatter trade-offs, Pareto frontiers,
cross-model radar, intersectional fairness, and mitigation effectiveness.

Migrated from the legacy ``fairxai.visualization.plots`` module.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Colour palette per model type (colour-blind friendly)
PALETTE_MODEL = {
    "logistic_regression": "#0072B2",
    "random_forest": "#009E73",
    "svm": "#D55E00",
    "xgboost": "#CC79A7",
}


def save_comparison_heatmap(
    pivot: pd.DataFrame,
    title: str,
    output_file,
    fmt: str = ".3f",
    cmap: str = "viridis",
):
    """Save a heatmap for a comparison pivot table."""
    if pivot is None or pivot.empty:
        return None

    fig = plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_tradeoff_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    style_col: str,
    title: str,
    output_file,
):
    """Save a scatter plot showing performance vs fairness trade-offs."""
    if df is None or df.empty:
        return None

    plot_df = df[[x_col, y_col, hue_col, style_col]].dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return None

    use_hue = hue_col in plot_df.columns and plot_df[hue_col].notna().any()
    use_style = style_col in plot_df.columns and plot_df[style_col].notna().any()

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue=hue_col if use_hue else None,
        style=style_col if use_style else None,
        alpha=0.8,
    )
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if use_hue or use_style:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_file,
):
    """Save a Pareto frontier plot (maximize x, minimize y)."""
    if df is None or df.empty:
        return None

    data = df[[x_col, y_col]].dropna().copy()
    if data.empty:
        return None

    # Compute Pareto frontier: maximize x, minimize y
    data = data.sort_values([x_col, y_col], ascending=[False, True])
    pareto = []
    best_y = None
    for _, row in data.iterrows():
        y_val = row[y_col]
        if best_y is None or y_val <= best_y:
            pareto.append(row)
            best_y = y_val

    pareto_df = pd.DataFrame(pareto)

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6)
    sns.lineplot(data=pareto_df, x=x_col, y=y_col, color="red", marker="o")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_intersectional_heatmap(
    per_group_df: pd.DataFrame,
    metric: str,
    output_file,
):
    """Heatmap of per-subgroup fairness delta — mitigation × demographic group.

    Rows = mitigation technique, columns = subgroup label (sensitive_attr + group),
    cell value = mean ``delta`` (experiment − baseline).  Green = improvement,
    red = worsening.

    Parameters
    ----------
    per_group_df : pd.DataFrame
        ``per_group_comparison.csv`` content.  Required columns:
        ``mitigation_technique``, ``sensitive_attr``, ``group``, ``metric``, ``delta``.
    metric : str
        Metric name to filter on, e.g. ``"demographic_parity_rate"`` or ``"tpr"``.
    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """
    sub = per_group_df[per_group_df["metric"] == metric].copy()
    if sub.empty:
        logger.warning("save_intersectional_heatmap: no rows for metric '%s'", metric)
        return None

    value_col = "delta"

    # Guard against non-numeric/empty deltas in partially populated outputs.
    if "delta" in sub.columns:
        sub["delta"] = pd.to_numeric(sub["delta"], errors="coerce")
    else:
        sub["delta"] = np.nan

    # Fallback 1: derive delta from experiment_value and baseline_value.
    if sub["delta"].dropna().empty and {"experiment_value", "baseline_value"}.issubset(sub.columns):
        exp_val = pd.to_numeric(sub["experiment_value"], errors="coerce")
        base_val = pd.to_numeric(sub["baseline_value"], errors="coerce")
        sub["delta"] = exp_val - base_val

    # Fallback 2: if no baseline pairing is available, visualize experiment values directly.
    if sub["delta"].dropna().empty and "experiment_value" in sub.columns:
        sub["experiment_value"] = pd.to_numeric(sub["experiment_value"], errors="coerce")
        value_col = "experiment_value"

    sub = sub.dropna(subset=[value_col])
    if sub.empty:
        logger.warning("save_intersectional_heatmap: no numeric values for metric '%s'", metric)
        return None

    sub["group_label"] = (
        sub["sensitive_attr"].str.replace("_cat", "").str.title()
        + "\n"
        + sub["group"].astype(str)
    )

    agg = (
        sub.groupby(["mitigation_technique", "group_label"])[value_col]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot(index="mitigation_technique", columns="group_label", values=value_col)

    if pivot.empty:
        return None

    values = pivot.to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        logger.warning("save_intersectional_heatmap: all delta values are NaN for metric '%s'", metric)
        return None
    abs_max = max(float(np.abs(values[valid]).max()), 0.01)

    fig, ax = plt.subplots(
        figsize=(max(10, pivot.shape[1] * 1.5), max(5, pivot.shape[0] * 0.6 + 2))
    )
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-abs_max,
        vmax=abs_max,
        linewidths=0.5,
        ax=ax,
    )
    metric_label = metric.replace("_", " ").title()
    value_label = (
        "Delta (Experiment − Baseline)"
        if value_col == "delta"
        else "Experiment Value (baseline unavailable)"
    )
    ax.set_title(
        f"Intersectional Fairness — {metric_label}\n{value_label}",
        fontsize=12,
    )
    ax.set_xlabel("Subgroup")
    ax.set_ylabel("Mitigation Technique")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_cross_model_radar(summary_df: pd.DataFrame, output_file):
    """Spider / radar chart comparing model types across 5 performance + fairness axes.

    Axes: F1, Recall, Precision, AUC-ROC, Fairness (= 1 − fairness_gap).
    One filled polygon per model type.

    Parameters
    ----------
    summary_df : pd.DataFrame
        ``cross_model_summary.csv`` content.  Required columns:
        ``model_type``, ``f1_score``, ``recall``, ``precision``,
        ``auc_roc``, ``fairness_gap``.
    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """
    if summary_df is None or summary_df.empty:
        logger.warning("save_cross_model_radar: empty DataFrame")
        return None

    if "model_type" not in summary_df.columns:
        logger.warning("save_cross_model_radar: missing required column model_type")
        return None

    df = summary_df.copy()

    aliases = {
        "f1_score": ["f1_score", "f1", "f1_value"],
        "recall": ["recall", "recall_value"],
        "precision": ["precision", "precision_value"],
        "auc_roc": ["auc_roc", "auc", "auc_value"],
    }

    def _resolve_alias(candidates: list[str]) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    metric_axes = []
    for canonical, candidates in aliases.items():
        src = _resolve_alias(candidates)
        if src is None:
            continue
        df[canonical] = pd.to_numeric(df[src], errors="coerce")
        metric_axes.append((canonical, canonical.replace("_", " ").upper()))

    if "fairness_gap" not in df.columns:
        if {"dp_max_diff", "eq_odds_max_diff"}.issubset(df.columns):
            df["fairness_gap"] = df[["dp_max_diff", "eq_odds_max_diff"]].max(axis=1, skipna=True)
        else:
            logger.warning("save_cross_model_radar: fairness_gap missing and cannot be derived")
            return None

    df["fairness_gap"] = pd.to_numeric(df["fairness_gap"], errors="coerce")
    df["fairness_score"] = 1 - df["fairness_gap"].clip(0, 1)

    # Keep a meaningful radar even when some metrics are unavailable.
    axes_specs = metric_axes + [("fairness_score", "Fairness\n(1−gap)")]
    if len(axes_specs) < 3:
        logger.warning("save_cross_model_radar: insufficient metric axes after normalization")
        return None

    df = df.groupby("model_type", as_index=False)[[col for col, _ in axes_specs]].mean()
    df = df.dropna(subset=[col for col, _ in axes_specs])
    if df.empty:
        logger.warning("save_cross_model_radar: no complete rows after normalization")
        return None

    pretty_labels = {
        "f1_score": "F1",
        "recall": "Recall",
        "precision": "Precision",
        "auc_roc": "AUC-ROC",
        "fairness_score": "Fairness\n(1−gap)",
    }

    categories = [pretty_labels.get(col, label) for col, label in axes_specs]
    n = len(categories)
    angles = [i * 2 * np.pi / n for i in range(n)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for _, row in df.iterrows():
        model = row["model_type"]
        values = [row[col] for col, _ in axes_specs]
        values_closed = values + [values[0]]
        color = PALETTE_MODEL.get(model, "#333333")
        label = model.replace("_", " ").title()
        ax.plot(angles, values_closed, color=color, linewidth=2, label=label)
        ax.fill(angles, values_closed, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Cross-Model Performance + Fairness Radar", fontsize=12, pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.1), fontsize=9)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_mitigation_effectiveness_matrix(full_df: pd.DataFrame, output_file):
    """Side-by-side heatmaps: fairness gain % and performance cost % per mitigation.

    Rows = mitigation technique (means over all binnings / training methods).
    Left panel = fairness_gain_pct (green = high), right = performance cost (red = high).

    Parameters
    ----------
    full_df : pd.DataFrame
        ``full_comparison.csv`` content.  Required columns:
        ``mitigation_technique``, ``fairness_gain_pct``.
        ``score_drop_pct`` is used if present; otherwise computed from
        ``baseline_score`` and ``score_value``.
    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """
    if "fairness_gain_pct" not in full_df.columns:
        logger.warning("save_mitigation_effectiveness_matrix: fairness_gain_pct not in DataFrame")
        return None

    plot_df = full_df[full_df["mitigation_technique"] != "baseline"].copy()
    if plot_df.empty:
        return None

    if "score_drop_pct" not in plot_df.columns:
        if "score_value" in plot_df.columns and "baseline_score" in plot_df.columns:
            plot_df["score_drop_pct"] = (
                (plot_df["baseline_score"] - plot_df["score_value"])
                / plot_df["baseline_score"].clip(lower=1e-9)
                * 100
            ).clip(lower=0)
        else:
            plot_df["score_drop_pct"] = float("nan")

    agg = (
        plot_df.groupby("mitigation_technique")[["fairness_gain_pct", "score_drop_pct"]]
        .mean()
    )
    if agg.empty:
        return None

    has_cost = not agg["score_drop_pct"].isna().all()
    n_panels = 2 if has_cost else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels + 1, max(5, len(agg) * 0.6 + 2)))
    if n_panels == 1:
        axes = [axes]

    sns.heatmap(
        agg[["fairness_gain_pct"]],
        annot=True, fmt=".1f", cmap="Greens",
        linewidths=0.5, vmin=0, ax=axes[0],
    )
    axes[0].set_title("Fairness Gain %", fontsize=11)
    axes[0].set_ylabel("Mitigation Technique")
    axes[0].set_xlabel("")

    if has_cost:
        sns.heatmap(
            agg[["score_drop_pct"]],
            annot=True, fmt=".1f", cmap="Reds",
            linewidths=0.5, vmin=0, ax=axes[1],
        )
        axes[1].set_title("Performance Cost %", fontsize=11)
        axes[1].set_ylabel("")
        axes[1].set_xlabel("")

    fig.suptitle(
        "Mitigation Effectiveness: Fairness Gain vs Performance Cost", fontsize=12
    )
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_pareto_all_models(
    full_df: pd.DataFrame,
    output_file,
    x_col: str = "f1_value",
    y_col: str = "fairness_gap",
):
    """Pareto frontier scatter with one coloured point cloud per model type.

    Parameters
    ----------
    full_df : pd.DataFrame
        ``full_comparison.csv`` content.  Required columns:
        ``model_type``, plus whichever columns are passed as ``x_col`` / ``y_col``.
    output_file : path-like
        Destination path for the saved PNG.
    x_col : str
        Column to maximise (x-axis).  Default ``"f1_value"``.
    y_col : str
        Column to minimise (y-axis).  Default ``"fairness_gap"``.

    Returns
    -------
    path-like or None
    """
    needed = {"model_type", x_col, y_col}
    if not needed.issubset(full_df.columns):
        logger.warning("save_pareto_all_models: missing columns %s", needed - set(full_df.columns))
        return None

    data = full_df[["model_type", x_col, y_col]].dropna(subset=[x_col, y_col]).copy()
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_type, group in data.groupby("model_type"):
        color = PALETTE_MODEL.get(model_type, "#333333")
        label = model_type.replace("_", " ").title()

        ax.scatter(group[x_col], group[y_col], alpha=0.45, color=color, s=30, label=label)

        sorted_g = group.sort_values([x_col, y_col], ascending=[False, True])
        pareto = []
        best_y = None
        for _, row in sorted_g.iterrows():
            if best_y is None or row[y_col] <= best_y:
                pareto.append(row)
                best_y = row[y_col]

        if len(pareto) >= 2:
            pf = pd.DataFrame(pareto)
            ax.plot(pf[x_col], pf[y_col], color=color, linewidth=2, marker="o", markersize=5)

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title("Pareto Frontier — All Model Types")
    ax.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file
