"""Dermatology stage-8 fairness figures: subgroup heatmaps + intersectional views.

Renders from the flattened per-group rows the assessment stage already produces
(see :func:`fairxai.fairness.image_assessment._flatten_for_csv` and
:func:`_flatten_group_views_for_csv`). Kept in ``viz`` so the JSON/Markdown/CSV
assessment path stays matplotlib-free — matplotlib is imported only when figures
are requested.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: the pipeline never has a GUI
import matplotlib.pyplot as plt  # noqa: E402

from fairxai.viz.save_utils import heatmap_size, save_figure  # noqa: E402

logger = logging.getLogger(__name__)

# Performance columns shown in every subgroup heatmap; all live in [0, 1].
_METRIC_COLUMNS = ["accuracy", "recall", "auc"]


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    return "_".join(part for part in slug.split("_") if part)


def _num(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_heatmap(
    groups: Sequence[str], rows: Sequence[Sequence[Any]], title: str, out_path: Path
) -> Optional[Path]:
    """Group (rows) x metric (cols) heatmap, values in [0,1]. Returns path or None."""
    if not groups:
        return None
    data = np.array(
        [[np.nan if _num(v) is None else _num(v) for v in row] for row in rows], dtype=float
    )
    if np.isnan(data).all():
        return None
    width, height = heatmap_size(groups, len(_METRIC_COLUMNS))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(_METRIC_COLUMNS)))
    ax.set_xticklabels([c.upper() for c in _METRIC_COLUMNS])
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(list(groups))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.55 else "black",
                    fontsize=8,
                )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def render_subgroup_heatmaps(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    attrs: Optional[Sequence[str]] = None,
) -> list[Path]:
    """One group x metric heatmap per (run_key, sensitive_attribute).

    ``rows`` are the flattened fairness-group records (one per run_key x attr x group).
    Output: ``out_dir/figures/subgroup_heatmap_<run_key>_<attr>.png``.
    """
    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for r in rows:
        key = (str(r.get("run_key")), str(r.get("sensitive_attribute")))
        grouped.setdefault(key, []).append(r)

    written: list[Path] = []
    for (run_key, attr), recs in sorted(grouped.items()):
        if attrs and attr not in attrs:
            continue
        recs = sorted(recs, key=lambda x: str(x.get("group")))
        groups = [str(r.get("group")) for r in recs]
        matrix = [[r.get(col) for col in _METRIC_COLUMNS] for r in recs]
        out_path = figures_dir / f"subgroup_heatmap_{_slug(run_key)}_{_slug(attr)}.png"
        path = _metric_heatmap(
            groups, matrix, f"Subgroup performance - {run_key} / {attr}", out_path
        )
        if path:
            written.append(path)
    logger.info("[SUCCESS] Subgroup heatmaps saved to %s (%d figure(s))", figures_dir, len(written))
    return written


def render_group_view_figures(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    exploratory_only: bool = True,
) -> list[Path]:
    """One intersection-cell x metric heatmap per (run_key, group_view).

    ``rows`` are the flattened group-view records. By default only exploratory
    (intersectional) views are plotted. Output:
    ``out_dir/figures/<view>__<run_key>.png``.
    """
    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for r in rows:
        if exploratory_only and not bool(r.get("exploratory")):
            continue
        key = (str(r.get("run_key")), str(r.get("group_view")))
        grouped.setdefault(key, []).append(r)

    written: list[Path] = []
    for (run_key, view), recs in sorted(grouped.items()):
        recs = sorted(recs, key=lambda x: str(x.get("group")))
        groups = [str(r.get("group")) for r in recs]
        matrix = [[r.get(col) for col in _METRIC_COLUMNS] for r in recs]
        out_path = figures_dir / f"{_slug(view)}__{_slug(run_key)}.png"
        path = _metric_heatmap(groups, matrix, f"{view} - {run_key}", out_path)
        if path:
            written.append(path)
    logger.info(
        "[SUCCESS] Group-view figures saved to %s (%d figure(s))", figures_dir, len(written)
    )
    return written
