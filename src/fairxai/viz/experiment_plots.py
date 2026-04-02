"""
Experiment comparison plots — heatmaps, scatter trade-offs, Pareto frontiers.

Migrated from the legacy ``fairxai.visualization.plots`` module.
Only the three save-oriented functions that were actually used are kept here.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


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
