"""Figures for the profiling-sensitivity study.

Reads the aggregated CSVs written by ``run_profiling_sensitivity_study.py`` and
emits matplotlib/seaborn figures under the study's ``figures/`` directory. Each
plot is skipped (with a warning) when its input columns are missing, so a partial
study still produces as many figures as possible.

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
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logger = logging.getLogger(__name__)

STUDY_TYPE = "profiling_sensitivity"


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


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("[SUCCESS] wrote %s", path.relative_to(_ROOT))


def plot_missingness(dataset_df: pd.DataFrame, fig_dir: Path) -> None:
    sub = dataset_df[dataset_df["label"] == "missingness"]
    if sub.empty or "top_missing_pct" not in sub:
        logger.warning("[WARNING] no missingness rows; skipping missingness plot")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=sub,
        x="missing_pct",
        y="top_missing_pct",
        hue="missing_mechanism",
        style="tier",
        s=120,
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


def plot_complexity_vs_knob(dataset_df: pd.DataFrame, fig_dir: Path) -> None:
    for label, xcol, xlabel in [
        ("separability", "class_sep", "class_sep"),
        ("size", "n_samples", "n_samples"),
    ]:
        sub = dataset_df[dataset_df["label"] == label]
        if sub.empty or "ebmDifficulty" not in sub:
            logger.warning("[WARNING] no %s rows; skipping complexity plot", label)
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.lineplot(
            data=sub.sort_values(xcol),
            x=xcol,
            y="ebmDifficulty",
            hue="tier",
            marker="o",
            ax=ax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ebmDifficulty")
        ax.set_title(f"EBM difficulty vs {xlabel}")
        _save(fig, fig_dir / "complexity", f"ebm_difficulty_vs_{label}.png")


def plot_duplicates(dataset_df: pd.DataFrame, fig_dir: Path) -> None:
    sub = dataset_df[dataset_df["label"] == "duplicates"]
    if sub.empty or "duplicate_pct_observed" not in sub:
        logger.warning("[WARNING] no duplicates rows; skipping duplicates plot")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=sub,
        x="duplicate_pct",
        y="duplicate_pct_observed",
        style="tier",
        s=120,
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

    sns.set_theme(style="whitegrid")

    if dataset_df is not None:
        plot_missingness(dataset_df, fig_dir)
        plot_class_balance(dataset_df, fig_dir)
        plot_complexity_vs_knob(dataset_df, fig_dir)
        plot_duplicates(dataset_df, fig_dir)
        plot_type_accuracy(dataset_df, fig_dir)
    if confusion_df is not None:
        plot_type_confusion(confusion_df, fig_dir)
    if column_df is not None:
        plot_lowcard_boundary(column_df, fig_dir)

    logger.info("[SUCCESS] figures under %s", fig_dir.relative_to(_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
