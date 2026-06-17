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
