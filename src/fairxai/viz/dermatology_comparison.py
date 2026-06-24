"""Dermatology baseline comparison figures.

Renders the optional PNGs for stage 9 from the same per-model rows the comparison
collator produces (see :mod:`fairxai.comparison.dermatology`). Kept in ``viz`` so
it reuses the shared model palette (:data:`fairxai.viz.fairness_comparison.PALETTE_MODEL`)
and save helpers, and so :mod:`fairxai.comparison.dermatology` stays matplotlib-free
on the plain CSV/Markdown path (matplotlib is imported only when figures are asked
for, via this module).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: stage 9 runs in the pipeline, never a GUI
import matplotlib.pyplot as plt  # noqa: E402

from fairxai.viz.fairness_comparison import PALETTE_MODEL  # noqa: E402
from fairxai.viz.save_utils import save_figure  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_COLOR = "#333333"


def _num(value: Any) -> Optional[float]:
    """Coerce to float, treating None/blank/non-numeric as missing."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    return "_".join(part for part in slug.split("_") if part)


def _plot_grouped_bars(
    *,
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[tuple[str, str]],
    output_file: Path,
    title: str,
    ylabel: str,
) -> Optional[Path]:
    """Grouped bar chart: models on x-axis, one bar per metric. Returns path or None."""
    if not rows or not metrics:
        return None

    models = [str(r.get("model", "model")) for r in rows]
    x_positions = list(range(len(models)))
    bar_width = 0.8 / max(len(metrics), 1)
    fig_width = max(8.0, len(models) * max(1.5, len(metrics) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    any_value = False
    for idx, (column, label) in enumerate(metrics):
        values = [_num(r.get(column)) for r in rows]
        if all(v is None for v in values):
            continue
        any_value = True
        offset = (idx - (len(metrics) - 1) / 2.0) * bar_width
        numeric_values = [0.0 if v is None else v for v in values]
        ax.bar(
            [x + offset for x in x_positions],
            numeric_values,
            width=bar_width * 0.9,
            label=label,
        )

    if not any_value:
        plt.close(fig)
        return None

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    if ylabel.lower().startswith("score"):
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, output_file)
    plt.close(fig)
    return output_file


def _plot_performance(
    rows: Sequence[Mapping[str, Any]], perf_columns: Sequence[str], figures_dir: Path
) -> Optional[Path]:
    metrics = [
        (column, column.upper() if column != "f1" else "F1")
        for column in perf_columns
        if any(_num(r.get(column)) is not None for r in rows)
    ]
    return _plot_grouped_bars(
        rows=rows,
        metrics=metrics,
        output_file=figures_dir / "performance_metrics.png",
        title="Dermatology Baseline Performance",
        ylabel="Score",
    )


def _plot_runtime_vs_auc(rows: Sequence[Mapping[str, Any]], figures_dir: Path) -> Optional[Path]:
    plot_rows = [
        r
        for r in rows
        if _num(r.get("train_time_seconds")) is not None and _num(r.get("auc")) is not None
    ]
    if not plot_rows:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for row in plot_rows:
        model = str(row.get("model", "model"))
        color = PALETTE_MODEL.get(model, _DEFAULT_COLOR)
        x_val = _num(row.get("train_time_seconds")) or 0.0
        y_val = _num(row.get("auc")) or 0.0
        ax.scatter([x_val], [y_val], s=80, color=color, edgecolors="black", linewidths=0.5)
        ax.annotate(
            model,
            (x_val, y_val),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    ax.set_title("AUC vs Training Time")
    ax.set_xlabel("Training time (seconds)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, figures_dir / "runtime_vs_auc.png")
    plt.close(fig)
    return figures_dir / "runtime_vs_auc.png"


def _plot_fairness_attr(
    rows: Sequence[Mapping[str, Any]],
    attr: str,
    fairness_labels: Mapping[str, str],
    figures_dir: Path,
) -> Optional[Path]:
    metrics = []
    for suffix, label in fairness_labels.items():
        column = f"{attr}_{suffix}"
        if any(_num(r.get(column)) is not None for r in rows):
            metrics.append((column, label))
    if not metrics:
        return None
    return _plot_grouped_bars(
        rows=rows,
        metrics=metrics,
        output_file=figures_dir / f"fairness_deltas_{_slug(attr)}.png",
        title=f"Fairness Deltas by Model - {attr}",
        ylabel="Delta",
    )


def _plot_learning_curve(
    run_key: str, history: Sequence[Mapping[str, Any]], best_epoch: Any, figures_dir: Path
) -> Optional[Path]:
    """Train/val loss (left axis) + val AUC (right axis) vs epoch for one model."""
    epochs = [h.get("epoch") for h in history]
    if not epochs:
        return None
    train_loss = [_num(h.get("train_loss")) for h in history]
    val_loss = [_num(h.get("val_loss")) for h in history]
    val_auc = [_num(h.get("val_auc")) for h in history]

    fig, ax1 = plt.subplots(figsize=(7.5, 5.0))
    ax1.plot(epochs, train_loss, marker="o", color="#0072B2", label="train loss")
    if any(v is not None for v in val_loss):
        ax1.plot(
            epochs,
            [float("nan") if v is None else v for v in val_loss],
            marker="s",
            linestyle="--",
            color="#D55E00",
            label="val loss",
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.25)

    handles, labels = ax1.get_legend_handles_labels()
    if any(v is not None for v in val_auc):
        ax2 = ax1.twinx()
        ax2.plot(
            epochs,
            [float("nan") if v is None else v for v in val_auc],
            marker="^",
            color="#009E73",
            label="val AUC",
        )
        ax2.set_ylabel("Validation AUC")
        ax2.set_ylim(0, 1.05)
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    best = _num(best_epoch)
    if best is not None:
        ax1.axvline(best, color="#555555", linestyle=":", linewidth=1.0)
        ax1.annotate(
            f"best epoch {int(best)}",
            xy=(best, ax1.get_ylim()[1]),
            fontsize=8,
            ha="center",
            va="top",
        )
    ax1.legend(handles, labels, loc="best")
    ax1.set_title(f"Learning Curve - {run_key}")
    fig.tight_layout()
    out_path = figures_dir / f"learning_curve_{_slug(run_key)}.png"
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def render_learning_curves(metrics: Mapping[str, Mapping[str, Any]], out_dir: Path) -> list[Path]:
    """One learning-curve PNG per model from each metrics dict's ``history``."""
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for run_key, data in sorted(metrics.items()):
        history = data.get("history") or []
        path = _plot_learning_curve(run_key, history, data.get("best_epoch"), figures_dir)
        if path:
            written.append(path)
    logger.info("[SUCCESS] Learning curves saved to %s (%d figure(s))", figures_dir, len(written))
    return written


def render_comparison_figures(
    rows: Sequence[Mapping[str, Any]],
    attrs: Sequence[str],
    out_dir: Path,
    *,
    perf_columns: Sequence[str],
    fairness_labels: Mapping[str, str],
) -> list[Path]:
    """Render comparison figures under ``out_dir/figures``; return the written paths.

    ``perf_columns`` and ``fairness_labels`` are supplied by the comparison collator
    so the canonical column names live in one place. No manifest file is written —
    the run logs plus the clickable output folder are the record.
    """
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    performance_path = _plot_performance(rows, perf_columns, figures_dir)
    if performance_path:
        written.append(performance_path)

    runtime_path = _plot_runtime_vs_auc(rows, figures_dir)
    if runtime_path:
        written.append(runtime_path)

    for attr in attrs:
        fairness_path = _plot_fairness_attr(rows, attr, fairness_labels, figures_dir)
        if fairness_path:
            written.append(fairness_path)

    logger.info(
        "[SUCCESS] Comparison figures saved to %s (%d figure(s))", figures_dir, len(written)
    )
    return written
