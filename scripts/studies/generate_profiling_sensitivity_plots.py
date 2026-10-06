"""Figures for the profiling-sensitivity study.

Reads the CSVs written by ``run_profiling_sensitivity_study.py`` and emits
matplotlib/seaborn figures under the study's ``figures/`` directory. The study is
replicated with paired seeds, so per-dataset plots aggregate the replicates
(mean and 95% CI), and the knob-response figures read the paired deltas from
``knob_response_summary.csv``. Each plot is skipped (with a warning) when its
input is missing, so a partial study still produces as many figures as possible.

Usage
-----
    python scripts/studies/generate_profiling_sensitivity_plots.py --study-id latest
    python scripts/studies/generate_profiling_sensitivity_plots.py --study-id run_2026... --pipeline synthetic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logger = logging.getLogger(__name__)

STUDY_TYPE = "profiling_sensitivity"

# Measures in the knob-response forest plot; the heatmap shows all of them.
FOREST_METRICS = ["ebmDifficulty", "N3Imbalance", "F3Imbalance"]
HEATMAP_METRICS = [
    "ebmDifficulty",
    "F2Imbalance",
    "F3Imbalance",
    "F4Imbalance",
    "L1Imbalance",
    "L2Imbalance",
    "L3Imbalance",
    "N2Imbalance",
    "N3Imbalance",
    "N4Imbalance",
    "T1Imbalance",
    "RaugImbalance",
    "BayesImbalance",
]
TIER_COLORS = {"abstract": "#1f77b4", "healthcare": "#d62728"}


def _resolve_study_root(pipeline: str, study_id: str) -> Path:
    base = _ROOT / "output" / pipeline / "studies" / STUDY_TYPE
    if study_id in ("latest", "", None):
        pointer = base / "latest.txt"
        if not pointer.exists():
            raise SystemExit(f"No latest study pointer at {pointer}")
        study_id = pointer.read_text().strip()
    root = base / study_id
    if not root.exists():
        raise SystemExit(f"Study root not found: {root}")
    return root


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("[WARNING] missing input: %s", path.name)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[WARNING] could not read %s: %s", path.name, exc)
        return None


def _save(fig: plt.Figure, out_dir: Path, name: str, pdf: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("[SUCCESS] wrote %s", path)


def _condition_label(condition: str) -> str:
    """Readable label for a runner condition key (``missingness_mcar_0.2`` etc.)."""
    parts = condition.split("_")
    knob, value = parts[0], parts[-1]
    if knob == "missingness":
        return f"{parts[1].upper()} {float(value):.0%}"
    labels = {
        "imbalance": f"minority {float(value):.0%}",
        "separability": f"class_sep {value}",
        "size": f"n = {value}",
        "cardinality": f"{value} levels, 3 high-card",
        "duplicates": f"duplicates {float(value):.0%}",
    }
    return labels.get(knob, condition)


def _knob_conditions(summary_df: pd.DataFrame) -> list[str]:
    """Non-baseline conditions in grid order (the runner writes them in that order)."""
    conditions = summary_df.loc[summary_df["knob"] != "base", "condition"]
    return list(dict.fromkeys(conditions))


def plot_missingness(dataset_df: pd.DataFrame, fig_dir: Path) -> None:
    sub = dataset_df[dataset_df["label"] == "missingness"]
    if sub.empty or "top_missing_pct" not in sub:
        logger.warning("[WARNING] no missingness rows; skipping missingness plot")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=sub,
        x="missing_pct",
        y="top_missing_pct",
        hue="missing_mechanism",
        style="tier",
        markers=True,
        errorbar=("ci", 95),
        err_style="bars",
        ax=ax,
    )
    lims = [0, max(sub["missing_pct"].max() * 100, sub["top_missing_pct"].max()) + 5]
    ax.plot([0, lims[1] / 100], [0, lims[1]], ls="--", c="grey", lw=1, label="design (x100)")
    ax.set_xlabel("Designed missing fraction")
    ax.set_ylabel("Observed top-column missing %")
    ax.set_title("Observed vs designed missingness (MCAR vs MAR)")
    _save(fig, fig_dir / "missingness", "observed_vs_design_missing.png")


def plot_class_balance(dataset_df: pd.DataFrame, fig_dir: Path) -> None:
    sub = dataset_df[dataset_df["label"] == "imbalance"]
    if sub.empty or "class_balance_delta" not in sub:
        logger.warning("[WARNING] no imbalance rows; skipping class-balance plot")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=sub.sort_values("minority_ratio"),
        x="minority_ratio",
        y="class_balance_delta",
        hue="tier",
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("Designed minority-class ratio")
    ax.set_ylabel("class_balance_delta (max/min count)")
    ax.set_title("Class-balance response to minority ratio")
    _save(fig, fig_dir / "class_balance", "balance_delta_vs_minority_ratio.png")


def plot_paired_forest(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    """Paired delta vs baseline (mean, 95% CI) per condition, one panel per measure."""
    metrics = [m for m in FOREST_METRICS if m in set(summary_df["metric"])]
    conditions = _knob_conditions(summary_df)
    if not metrics or not conditions:
        logger.warning("[WARNING] no paired deltas; skipping forest plot")
        return
    y = np.arange(len(conditions))
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(3.2 * len(metrics), 0.32 * len(conditions) + 1.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        for offset, (tier, color) in zip((-0.17, 0.17), TIER_COLORS.items()):
            rows = (
                summary_df[(summary_df["metric"] == metric) & (summary_df["tier"] == tier)]
                .set_index("condition")
                .reindex(conditions)
            )
            mean = rows["delta_mean"].to_numpy()
            err = np.vstack(
                [
                    mean - rows["delta_ci95_low"].to_numpy(),
                    rows["delta_ci95_high"].to_numpy() - mean,
                ]
            )
            ax.errorbar(
                mean, y + offset, xerr=err, fmt="o", ms=4, capsize=2, color=color, label=tier
            )
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("Paired delta vs baseline")
    axes[0].set_yticks(y, [_condition_label(c) for c in conditions])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    _save(fig, fig_dir / "complexity", "paired_delta_forest.png", pdf=True)


def plot_paired_heatmap(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    """Mean paired delta for every measure; cells whose 95% CI covers zero are faded."""
    conditions = _knob_conditions(summary_df)
    sub = summary_df[summary_df["metric"].isin(HEATMAP_METRICS)]
    if sub.empty or not conditions:
        logger.warning("[WARNING] no paired deltas; skipping delta heatmap")
        return
    tiers = [t for t in TIER_COLORS if t in set(sub["tier"])]
    metrics = [m for m in HEATMAP_METRICS if m in set(sub["metric"])]
    limit = float(sub.loc[sub["knob"] != "base", "delta_mean"].abs().max()) or 1.0
    fig, axes = plt.subplots(
        1, len(tiers), figsize=(7 * len(tiers), 0.4 * len(conditions) + 2), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, tier in zip(axes, tiers):
        rows = sub[sub["tier"] == tier]
        pivot = rows.pivot(index="condition", columns="metric", values="delta_mean")
        low = rows.pivot(index="condition", columns="metric", values="delta_ci95_low")
        high = rows.pivot(index="condition", columns="metric", values="delta_ci95_high")
        pivot, low, high = (
            f.reindex(index=conditions, columns=metrics) for f in (pivot, low, high)
        )
        covers_zero = (low <= 0) & (high >= 0)
        sns.heatmap(
            pivot,
            mask=covers_zero,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 7},
            cbar=ax is axes[-1],
            ax=ax,
        )
        sns.heatmap(
            pivot.where(covers_zero),
            cmap=["#f0f0f0"],
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 7, "color": "#9a9a9a"},
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"{tier} tier", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticks(np.arange(len(conditions)) + 0.5, [_condition_label(c) for c in conditions])
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    _save(fig, fig_dir / "complexity", "paired_delta_heatmap.png", pdf=True)


def plot_duplicates(dataset_df: pd.DataFrame, fig_dir: Path) -> None:
    sub = dataset_df[dataset_df["label"] == "duplicates"]
    if sub.empty or "duplicate_pct_observed" not in sub:
        logger.warning("[WARNING] no duplicates rows; skipping duplicates plot")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=sub,
        x="duplicate_pct",
        y="duplicate_pct_observed",
        hue="tier",
        style="tier",
        markers=True,
        errorbar=("ci", 95),
        err_style="bars",
        ax=ax,
    )
    lims = [0, max(sub["duplicate_pct"].max(), sub["duplicate_pct_observed"].max()) + 0.05]
    ax.plot(lims, lims, ls="--", c="grey", lw=1, label="design = observed")
    ax.set_xlabel("Designed duplicate fraction")
    ax.set_ylabel("Observed duplicate-row fraction")
    ax.set_title("Observed vs designed duplicate rows")
    ax.legend()
    _save(fig, fig_dir / "duplicates", "observed_vs_design_duplicates.png")


def plot_type_accuracy(dataset_df: pd.DataFrame, fig_dir: Path) -> None:
    if "semantic_type_accuracy" not in dataset_df:
        logger.warning("[WARNING] no accuracy column; skipping accuracy plot")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=dataset_df, x="label", y="semantic_type_accuracy", hue="tier", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Knob family")
    ax.set_ylabel("Semantic-type accuracy")
    ax.set_title("Semantic-type accuracy by knob (after type-inference fix)")
    plt.xticks(rotation=20)
    _save(fig, fig_dir / "type_inference", "semantic_type_accuracy_by_knob.png")


def plot_type_confusion(confusion_df: pd.DataFrame, fig_dir: Path) -> None:
    if confusion_df is None or confusion_df.empty:
        logger.warning("[WARNING] no confusion data; skipping heatmap")
        return
    pivot = confusion_df.pivot_table(
        index="expected", columns="observed", values="count", fill_value=0, aggfunc="sum"
    ).astype(int)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Semantic-type confusion (expected vs observed)")
    _save(fig, fig_dir / "type_inference", "type_confusion_heatmap.png")


def plot_lowcard_boundary(column_df: pd.DataFrame, fig_dir: Path) -> None:
    sub = column_df[column_df["name"].astype(str).str.startswith("lowcard_")]
    if sub.empty:
        logger.warning("[WARNING] no low-card columns; skipping boundary plot")
        return
    counts = sub.groupby(["n_unique", "observed_semantic_type"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="n_unique", y="count", hue="observed_semantic_type", ax=ax)
    ax.set_xlabel("Low-cardinality column distinct values")
    ax.set_ylabel("Column count")
    ax.set_title("Low-cardinality numeric: observed semantic type")
    _save(fig, fig_dir / "type_inference", "lowcard_numeric_boundary.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", default="latest")
    parser.add_argument("--pipeline", default="synthetic")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    study_root = _resolve_study_root(args.pipeline, args.study_id)
    fig_dir = study_root / "figures"
    logger.info("[PHASE] plotting study %s", study_root.name)

    dataset_df = _safe_read_csv(study_root / "dataset_results.csv")
    column_df = _safe_read_csv(study_root / "column_results.csv")
    confusion_df = _safe_read_csv(study_root / "type_confusion.csv")
    summary_df = _safe_read_csv(study_root / "knob_response_summary.csv")

    sns.set_theme(style="whitegrid")

    if dataset_df is not None:
        plot_missingness(dataset_df, fig_dir)
        plot_class_balance(dataset_df, fig_dir)
        plot_duplicates(dataset_df, fig_dir)
        plot_type_accuracy(dataset_df, fig_dir)
    if summary_df is not None:
        plot_paired_forest(summary_df, fig_dir)
        plot_paired_heatmap(summary_df, fig_dir)
    if confusion_df is not None:
        plot_type_confusion(confusion_df, fig_dir)
    if column_df is not None:
        plot_lowcard_boundary(column_df, fig_dir)

    logger.info("[SUCCESS] figures under %s", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
