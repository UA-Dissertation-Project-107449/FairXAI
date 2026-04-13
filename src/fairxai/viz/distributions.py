from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_non_empty_datasets(datasets: dict[str, pd.DataFrame]) -> None:
    if not datasets:
        raise ValueError("`datasets` must contain at least one dataframe.")


def _ensure_column_exists(datasets: dict[str, pd.DataFrame], column: str) -> None:
    missing = [name for name, df in datasets.items() if column not in df.columns]
    if missing:
        raise ValueError(f"Column '{column}' not found in datasets: {missing}")


def plot_categorical_distribution_grid(
    datasets: dict[str, pd.DataFrame],
    column: str,
    title: str = None,
    subtitle: str = None,
    palette: dict = None,
    show_percentages: bool = True,
    annotate_imbalance: bool = False,
    fairness_context: str = None,
    figsize: tuple = None,
    save_path: Path = None,
    category_order: list[str] | None = None,
    as_proportion: bool = False,
    show_counts_in_labels: bool = False,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create a grid of categorical distribution plots for multiple datasets.

    Educational Features:
    - Auto-generates interpretive subtitle if distributions differ significantly
    - Highlights imbalanced categories (potential fairness issue)
    - Optionally adds fairness context annotation

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        {dataset_name: dataframe} pairs
    column : str
        Column name to plot (must exist in all datasets)
    title : str, optional
        Main plot title (auto-generated if None)
    subtitle : str, optional
        Interpretive subtitle (auto-generated based on data if None)
    palette : dict, optional
        {category: color} mapping
    show_percentages : bool, default True
        Annotate bars with percentages
    annotate_imbalance : bool, default False
        Add warning box if any category >65% (fairness red flag)
    fairness_context : str, optional
        Educational text box explaining fairness implications
    figsize : tuple, optional
        Figure size (auto-calculated based on dataset count if None)
    save_path : Path, optional
        Save figure to this path

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Array of subplot axes

    Examples
    --------
    >>> # Simple usage
    >>> plot_categorical_distribution_grid(
    ...     datasets={'Train': train_df, 'Test': test_df},
    ...     column='sex_cat',
    ...     title="Sex Distribution: Train vs Test"
    ... )

    >>> # With fairness annotations
    >>> plot_categorical_distribution_grid(
    ...     datasets={'Cleveland': clev, 'Kaggle': kag},
    ...     column='sex_cat',
    ...     annotate_imbalance=True,
    ...     fairness_context=(
    ...         "Imbalanced sensitive attributes can cause models to "
    ...         "perform worse on underrepresented groups."
    ...     )
    ... )
    """
    _ensure_non_empty_datasets(datasets)
    _ensure_column_exists(datasets, column)

    n_datasets = len(datasets)

    # Auto-calculate layout
    ncols = min(3, n_datasets)  # Max 3 columns
    nrows = (n_datasets + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows + 1)  # +1 for title space

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    if category_order is None:
        category_values = set()
        for _, df in datasets.items():
            category_values.update(df[column].dropna().astype(str).unique())
        category_order = sorted(category_values)

    # Compute distribution stats for subtitle generation
    dist_stats = {}
    for name, df in datasets.items():
        counts = df[column].astype(str).value_counts(normalize=True)
        dist_stats[name] = counts

    # Auto-generate subtitle if not provided
    if subtitle is None and n_datasets > 1:
        subtitle = _generate_distribution_subtitle(dist_stats, column)

    # Plot each dataset
    for idx, (name, df) in enumerate(datasets.items()):
        ax = axes[idx]
        series = df[column].astype(str)
        total = len(series)

        if as_proportion:
            proportions = series.value_counts(normalize=True).reindex(category_order, fill_value=0)
            if palette:
                if name in palette:
                    bar_colors = [palette.get(name, "#7f7f7f")] * len(category_order)
                else:
                    bar_colors = [palette.get(cat, "#7f7f7f") for cat in category_order]
            else:
                bar_colors = None
            sns.barplot(
                x=proportions.index.astype(str), y=proportions.values, palette=bar_colors, ax=ax
            )
            ax.set_ylabel("proportion")
            ax.set_ylim(0, 1.08)
        else:
            if palette:
                if name in palette:
                    sns.countplot(
                        data=df.astype({column: str}),
                        x=column,
                        order=category_order,
                        color=palette.get(name, "#7f7f7f"),
                        ax=ax,
                    )
                else:
                    bar_colors = [palette.get(cat, "#7f7f7f") for cat in category_order]
                    sns.countplot(
                        data=df.astype({column: str}),
                        x=column,
                        order=category_order,
                        palette=bar_colors,
                        ax=ax,
                    )
            else:
                sns.countplot(data=df.astype({column: str}), x=column, order=category_order, ax=ax)
            ax.set_ylabel("Count")

        ax.set_title(f"{name}\n(n={len(df):,})", fontsize=12, fontweight="bold")
        ax.set_xlabel("")

        # Add percentage annotations
        if show_percentages:
            counts = series.value_counts().reindex(category_order, fill_value=0)
            for container in ax.containers:
                if as_proportion:
                    if show_counts_in_labels:
                        labels = [
                            f"{v:.1%}\n({int(counts.loc[cat])})"
                            for cat, v in zip(category_order, container.datavalues)
                        ]
                    else:
                        labels = [f"{v:.1%}" for v in container.datavalues]
                else:
                    labels = [
                        f"{(v/total*100 if total else 0):.1f}%\n({int(v)})"
                        for v in container.datavalues
                    ]
                ax.bar_label(container, labels=labels, fontsize=9)

            if as_proportion:
                ymax = max((container.datavalues.max() for container in ax.containers), default=1.0)
                ax.set_ylim(0, max(1.08, float(ymax) * 1.12))
            else:
                ymax = max((container.datavalues.max() for container in ax.containers), default=0)
                ax.set_ylim(0, float(ymax) * 1.18 if ymax else 1.0)

        # Imbalance warning box
        if annotate_imbalance:
            max_pct = df[column].value_counts(normalize=True).max()
            if max_pct > 0.65:  # Fairness threshold
                ax.text(
                    0.98,
                    0.98,
                    "⚠️ Imbalanced",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
                    fontsize=9,
                    ha="right",
                    va="top",
                )

    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    # Main title and subtitle
    layout_top = 0.95
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
        layout_top = 0.88
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", fontsize=11, style="italic", color="#555555")
        layout_top = 0.80 if title else 0.88

    # Fairness context box
    if fairness_context:
        fig.text(
            0.5,
            0.02,
            f"📊 Fairness Note: {fairness_context}",
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
            wrap=True,
        )

    layout_bottom = 0.08 if fairness_context else 0.03
    plt.tight_layout(rect=[0, layout_bottom, 1, layout_top])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_numeric_distribution_comparison(
    datasets: dict[str, pd.DataFrame],
    column: str,
    title: str | None = None,
    bins: int = 20,
    kde: bool = False,
    colors: dict[str, str] | None = None,
    figsize: tuple | None = None,
    value_getter: Callable[[str, pd.DataFrame], pd.Series] | None = None,
    xlabel: str | None = None,
    ylabel: str = "count",
    save_path: Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    _ensure_non_empty_datasets(datasets)

    n_datasets = len(datasets)
    ncols = min(3, n_datasets)
    nrows = (n_datasets + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows + 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for idx, (name, df) in enumerate(datasets.items()):
        ax = axes[idx]
        if value_getter is not None:
            series = value_getter(name, df)
        else:
            if column not in df.columns:
                ax.set_title(f"{name} (missing {column})")
                ax.axis("off")
                continue
            series = pd.to_numeric(df[column], errors="coerce")

        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            ax.set_title(f"{name} ({column} unavailable)")
            ax.axis("off")
            continue

        color = colors.get(name) if colors else None
        sns.histplot(series, bins=bins, kde=kde, ax=ax, color=color)
        ax.set_title(f"{name} {column}")
        ax.set_xlabel(xlabel if xlabel else column)
        ax.set_ylabel(ylabel)

    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_target_distribution_by_group(
    datasets: dict[str, pd.DataFrame],
    target_col: str,
    group_col: str,
    title: str | None = None,
    palette: dict | None = None,
    kind: str = "bar",
    figsize: tuple | None = None,
    y_lim: tuple[float, float] = (0.0, 1.05),
    save_path: Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    _ensure_non_empty_datasets(datasets)
    if kind not in {"bar", "line"}:
        raise ValueError("`kind` must be either 'bar' or 'line'.")

    n_datasets = len(datasets)
    ncols = min(3, n_datasets)
    nrows = (n_datasets + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows + 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for idx, (name, df) in enumerate(datasets.items()):
        ax = axes[idx]
        if target_col not in df.columns or group_col not in df.columns:
            ax.set_title(f"{name} (missing {target_col} or {group_col})")
            ax.axis("off")
            continue

        grouped = df.groupby(group_col, dropna=False)[target_col].mean().reset_index()
        grouped[group_col] = grouped[group_col].astype(str)

        if kind == "bar":
            if palette:
                colors = [palette.get(value, "#7f7f7f") for value in grouped[group_col]]
                sns.barplot(data=grouped, x=group_col, y=target_col, ax=ax, palette=colors)
            else:
                sns.barplot(data=grouped, x=group_col, y=target_col, ax=ax)

            for container in ax.containers:
                ax.bar_label(
                    container, labels=[f"{v:.0%}" for v in container.datavalues], fontsize=9
                )
        else:
            sns.lineplot(data=grouped, x=group_col, y=target_col, ax=ax, marker="o")
            for x_idx, value in enumerate(grouped[target_col]):
                ax.annotate(
                    f"{value:.0%}",
                    (x_idx, value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 3),
                    textcoords="offset points",
                )

        ax.set_title(f"{name} {target_col} by {group_col}")
        ax.set_xlabel(group_col)
        ax.set_ylabel("prevalence")
        ax.set_ylim(*y_lim)

    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_stacked_group_distribution_grid(
    datasets: dict[str, pd.DataFrame],
    group_col: str,
    stack_col: str,
    title: str | None = None,
    subtitle: str | None = None,
    group_order_by_dataset: dict[str, list[str]] | None = None,
    stack_order: list[str] | None = None,
    stack_palette: dict[str, str] | None = None,
    figsize: tuple | None = None,
    annotate_totals: bool = True,
    save_path: Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    _ensure_non_empty_datasets(datasets)

    n_datasets = len(datasets)
    ncols = min(3, n_datasets)
    nrows = (n_datasets + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows + 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for idx, (name, df) in enumerate(datasets.items()):
        ax = axes[idx]

        if group_col not in df.columns or stack_col not in df.columns:
            ax.set_title(f"{name} (missing {group_col} or {stack_col})")
            ax.axis("off")
            continue

        temp = pd.DataFrame(
            {
                group_col: df[group_col].astype(str),
                stack_col: df[stack_col].astype(str),
            }
        )

        grouped = (
            temp.groupby([group_col, stack_col], observed=True).size().reset_index(name="count")
        )
        pivoted = grouped.pivot(index=group_col, columns=stack_col, values="count").fillna(0)

        if group_order_by_dataset and name in group_order_by_dataset:
            ordered_groups = [str(value) for value in group_order_by_dataset[name]]
            pivoted = pivoted.reindex(ordered_groups, fill_value=0)

        if stack_order is not None:
            ordered_stack = [str(value) for value in stack_order]
            pivoted = pivoted.reindex(columns=ordered_stack, fill_value=0)
        else:
            pivoted = pivoted.sort_index(axis=1)

        colors = None
        if stack_palette:
            colors = [stack_palette.get(str(column), "#7f7f7f") for column in pivoted.columns]

        pivoted.plot(kind="bar", stacked=True, ax=ax, color=colors, legend=False)

        if annotate_totals:
            totals = pivoted.sum(axis=1)
            for x_pos, total in enumerate(totals):
                ax.annotate(
                    f"{int(total)}",
                    (x_pos, total),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 3),
                    textcoords="offset points",
                )
            ymax = totals.max() if len(totals) else 0
            ax.set_ylim(0, float(ymax) * 1.18 if ymax else 1.0)

        ax.set_title(f"{name} {group_col} x {stack_col}")
        ax.set_xlabel(group_col)
        ax.set_ylabel("count")

        if idx == 0 and len(pivoted.columns) > 0:
            ax.legend(title=stack_col, loc="best")

    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    layout_top = 0.95
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
        layout_top = 0.88
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", fontsize=11, style="italic", color="#555555")
        layout_top = 0.80 if title else 0.88

    plt.tight_layout(rect=[0, 0.03, 1, layout_top])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_missing_data_patterns(
    datasets: dict[str, pd.DataFrame],
    title: str = "Missing Data Patterns",
    figsize: tuple[float, float] | None = None,
    save_path: Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    _ensure_non_empty_datasets(datasets)

    n_datasets = len(datasets)
    if figsize is None:
        figsize = (12, 4 * n_datasets)

    fig, axes = plt.subplots(n_datasets, 2, figsize=figsize)
    if n_datasets == 1:
        axes = np.array([axes])

    for idx, (name, df) in enumerate(datasets.items()):
        missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]

        if missing_pct.empty:
            axes[idx, 0].text(
                0.5,
                0.5,
                f"{name}: No missing values",
                ha="center",
                va="center",
                fontsize=13,
            )
            axes[idx, 0].axis("off")
            axes[idx, 1].axis("off")
            continue

        axes[idx, 0].barh(missing_pct.index.astype(str), missing_pct.values, color="coral")
        axes[idx, 0].set_xlabel("Missing (%)")
        axes[idx, 0].set_title(f"{name}: Missing Data by Feature")
        axes[idx, 0].grid(axis="x", alpha=0.3)

        for row_i, (_, pct) in enumerate(missing_pct.items()):
            axes[idx, 0].text(
                pct + 0.5,
                row_i,
                f"{pct:.1f}%",
                va="center",
                fontsize=9,
            )

        missing_mask = df[missing_pct.index].isna().astype(int)
        if len(missing_pct) > 1:
            cooccur = missing_mask.T @ missing_mask
            cooccur_pct = cooccur / len(df) * 100
            sns.heatmap(
                cooccur_pct,
                annot=True,
                fmt=".1f",
                cmap="Reds",
                cbar_kws={"label": "Co-occurrence (%)"},
                ax=axes[idx, 1],
            )
            axes[idx, 1].set_title(f"{name}: Missing Co-occurrence")
        else:
            axes[idx, 1].text(
                0.5,
                0.5,
                "Only 1 feature\nwith missing data",
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[idx, 1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def plot_outlier_analysis(
    datasets: dict[str, pd.DataFrame],
    features: list[str],
    bounds: dict[str, tuple[float, float]] | None = None,
    method: str = "iqr",
    figsize: tuple[float, float] | None = None,
    save_path: Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    _ensure_non_empty_datasets(datasets)
    if method not in {"iqr", "zscore", "clinical"}:
        raise ValueError("`method` must be one of: 'iqr', 'zscore', 'clinical'.")

    default_bounds = {
        "age": (18, 100),
        "age_raw": (18, 100),
        "trestbps": (60, 250),
        "resting_bp": (60, 250),
        "chol": (100, 500),
        "cholesterol": (100, 500),
        "thalach": (60, 220),
        "max_heart_rate": (60, 220),
        "ap_hi": (60, 250),
        "ap_lo": (40, 150),
        "height": (130, 220),
        "weight": (40, 200),
    }

    if bounds:
        default_bounds.update(bounds)
    bounds = default_bounds

    n_features = len(features)
    n_datasets = len(datasets)

    if n_features == 0:
        raise ValueError("`features` must contain at least one feature.")

    if figsize is None:
        figsize = (max(10, n_datasets * 3), n_features * 2.5)

    fig, axes = plt.subplots(n_features, 1, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    dataset_names = list(datasets.keys())

    for feat_idx, feature in enumerate(features):
        ax = axes[feat_idx]

        plot_data = []
        for name in dataset_names:
            df = datasets[name]
            if feature not in df.columns:
                continue
            vals = pd.to_numeric(df[feature], errors="coerce").dropna()
            if vals.empty:
                continue
            plot_data.extend([(name, val) for val in vals])

        if not plot_data:
            ax.text(0.5, 0.5, f"{feature}: No data", ha="center", va="center")
            ax.axis("off")
            continue

        plot_df = pd.DataFrame(plot_data, columns=["dataset", feature])
        sns.boxplot(data=plot_df, x="dataset", y=feature, ax=ax)

        if feature in bounds:
            low_bound, high_bound = bounds[feature]
            ax.axhline(low_bound, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.axhline(high_bound, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(
                0.98,
                0.98,
                f"Clinical bounds:\n[{low_bound}, {high_bound}]",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
                fontsize=8,
                va="top",
                ha="right",
            )

        outlier_counts = []
        for name in dataset_names:
            df = datasets[name]
            if feature not in df.columns:
                continue
            vals = pd.to_numeric(df[feature], errors="coerce").dropna()
            if vals.empty:
                continue

            if method == "clinical" and feature in bounds:
                low, high = bounds[feature]
                n_outliers = int(((vals < low) | (vals > high)).sum())
            elif method == "iqr":
                q1, q3 = vals.quantile([0.25, 0.75])
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_outliers = int(((vals < low) | (vals > high)).sum())
            elif method == "zscore":
                std = vals.std()
                if std == 0 or np.isnan(std):
                    n_outliers = 0
                else:
                    z = np.abs((vals - vals.mean()) / std)
                    n_outliers = int((z > 3).sum())
            else:
                n_outliers = 0

            pct = (n_outliers / len(vals) * 100) if len(vals) > 0 else 0.0
            outlier_counts.append(f"{name}: {n_outliers} ({pct:.1f}%)")

        ax.set_title(
            f"{feature} distribution - Outliers ({method}): {', '.join(outlier_counts)}",
            fontsize=10,
        )
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Outlier Analysis Across Datasets", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


def _generate_distribution_subtitle(dist_stats: dict[str, pd.Series], column: str) -> str:
    """
    Auto-generate interpretive subtitle based on distribution differences.

    Examples:
    - "Cleveland is 68% male, Kaggle 79% male, Cardio70k is balanced"
    - "Age distributions are similar across all datasets"
    """
    if not dist_stats:
        return None

    dataset_names = list(dist_stats.keys())

    # Collect all possible categories across datasets
    all_categories = set()
    for series in dist_stats.values():
        all_categories.update(series.index)

    # Align distributions (fill missing categories with 0)
    aligned = {}
    for name, series in dist_stats.items():
        aligned[name] = series.reindex(all_categories, fill_value=0)

    # Compute max absolute difference per category across datasets
    max_diff = 0
    for category in all_categories:
        values = [aligned[name][category] for name in dataset_names]
        diff = max(values) - min(values)
        max_diff = max(max_diff, diff)

    # Thresholds for interpretation
    SIMILAR_THRESHOLD = 0.05  # <5% difference
    MODERATE_THRESHOLD = 0.15  # 5–15%

    # If very similar
    if max_diff < SIMILAR_THRESHOLD:
        return f"{column} distributions are similar across all datasets."

    # Identify dominant category per dataset
    dominance_statements = []
    for name in dataset_names:
        series = aligned[name]
        top_category = series.idxmax()
        top_pct = series.max() * 100

        if top_pct >= 50:
            dominance_statements.append(f"{name} is {top_pct:.0f}% {top_category}")
        else:
            dominance_statements.append(f"{name} is relatively balanced")

    if max_diff < MODERATE_THRESHOLD:
        return "Minor distribution differences observed. " + ", ".join(dominance_statements) + "."
    else:
        return "Notable distribution differences detected. " + ", ".join(dominance_statements) + "."


def plot_mixed_feature_batches(
    df: pd.DataFrame,
    features: list[str],
    dataset_name: str,
    color: str,
    units: dict[str, str] | None = None,
    batch_size: int = 4,
    categorical_unique_threshold: int = 5,
    save_path: Path | None = None,
    show: bool = False,
) -> list[tuple[plt.Figure, np.ndarray]]:
    figures: list[tuple[plt.Figure, np.ndarray]] = []
    if not features:
        return figures

    units = units or {}
    batches = [
        features[index : index + batch_size] for index in range(0, len(features), batch_size)
    ]

    for batch in batches:
        fig, axes = plt.subplots(1, len(batch), figsize=(4 * len(batch), 3))
        axes = np.atleast_1d(axes).ravel()

        for idx, feature in enumerate(batch):
            ax = axes[idx]
            series = df[feature]

            if series.nunique(dropna=False) <= categorical_unique_threshold:
                counts = series.value_counts(dropna=False).sort_index()
                sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color=color)
                ax.set_ylabel("count")
                ymax = counts.max() if len(counts) else 0
                ax.set_ylim(0, float(ymax) * 1.15 if ymax else 1.0)
                for container in ax.containers:
                    ax.bar_label(
                        container,
                        labels=[str(int(value)) for value in container.datavalues],
                        fontsize=9,
                    )
            else:
                sns.histplot(series, bins=20, ax=ax, color=color)
                unit = units.get(feature)
                if unit:
                    ax.set_xlabel(f"{feature} ({unit})")

            ax.set_title(f"{dataset_name} {feature}")

        plt.tight_layout()
        if save_path:
            base = save_path.stem
            ext = save_path.suffix or ".png"
            out_path = save_path.parent / f"{base}_batch_{len(figures)+1}{ext}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        figures.append((fig, axes))

    return figures


def plot_bmi_and_bp_relationship(
    df: pd.DataFrame,
    color: str,
    height_col: str = "height",
    weight_col: str = "weight",
    systolic_col: str = "ap_hi",
    diastolic_col: str = "ap_lo",
    save_path: Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray] | None:
    if not {height_col, weight_col}.issubset(df.columns):
        return None

    bmi = df[weight_col] / (df[height_col] / 100) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(bmi, bins=30, ax=axes[0], color=color)
    axes[0].set_title("cardio70k BMI distribution")
    axes[0].set_xlabel("BMI (kg/m^2)")

    if {systolic_col, diastolic_col}.issubset(df.columns):
        sns.scatterplot(x=df[systolic_col], y=df[diastolic_col], ax=axes[1], s=8, color=color)
        axes[1].set_title("cardio70k systolic vs diastolic BP")
        axes[1].set_xlabel("ap_hi (mm Hg)")
        axes[1].set_ylabel("ap_lo (mm Hg)")
    else:
        axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes
