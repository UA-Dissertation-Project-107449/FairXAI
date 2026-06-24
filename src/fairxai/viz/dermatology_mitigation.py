"""Dermatology mitigation before/after figures (stage 11).

Renders baseline-vs-mitigated fairness gaps per sensitive attribute x constraint
from the flattened mitigation rows (see
:func:`fairxai.fairness.image_mitigation._flatten_for_csv`). Kept in ``viz`` so the
JSON/Markdown/CSV path in :mod:`fairxai.fairness.image_mitigation` stays
matplotlib-free — matplotlib is imported only when figures are requested.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: the pipeline never has a GUI
import matplotlib.pyplot as plt  # noqa: E402

from fairxai.viz.save_utils import save_figure  # noqa: E402

logger = logging.getLogger(__name__)

# (before_col, after_col, panel label) for the headline fairness gaps the
# mitigation CSV carries. Lower is fairer, so a downward after-bar is an improvement.
_METRIC_PANELS = [
    ("dp_before", "dp_after", "DP gap"),
    ("tpr_before", "tpr_after", "TPR gap"),
    ("fpr_before", "fpr_after", "FPR gap"),
]


def _num(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    return "_".join(part for part in slug.split("_") if part)


def _panel(
    ax: Any, constraints: Sequence[str], before: Sequence[Any], after: Sequence[Any]
) -> bool:
    """Grouped before/after bars for one metric on ``ax``. Returns True if anything drawn."""
    b = [_num(v) for v in before]
    a = [_num(v) for v in after]
    if all(v is None for v in b) and all(v is None for v in a):
        return False
    x = list(range(len(constraints)))
    width = 0.38
    ax.bar(
        [i - width / 2 for i in x],
        [0.0 if v is None else v for v in b],
        width=width,
        label="before",
        color="#999999",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [0.0 if v is None else v for v in a],
        width=width,
        label="after",
        color="#0072B2",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in constraints], fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return True


def _plot_attr(attr: str, rows: Sequence[Mapping[str, Any]], figures_dir: Path) -> Optional[Path]:
    """One figure for *attr*: a before/after panel per fairness metric over constraints."""
    if not rows:
        return None
    constraints: list[str] = []
    for r in rows:
        c = str(r.get("constraint"))
        if c not in constraints:
            constraints.append(c)
    by_constraint = {str(r.get("constraint")): r for r in rows}

    panels: list[tuple[str, list[Any], list[Any]]] = []
    for before_col, after_col, label in _METRIC_PANELS:
        before = [by_constraint.get(c, {}).get(before_col) for c in constraints]
        after = [by_constraint.get(c, {}).get(after_col) for c in constraints]
        if any(_num(v) is not None for v in before) or any(_num(v) is not None for v in after):
            panels.append((label, before, after))
    if not panels:
        return None

    fig, axes = plt.subplots(
        1, len(panels), figsize=(max(4.0, len(constraints) * 1.3) * len(panels), 4.5), squeeze=False
    )
    drew = False
    for ax, (label, before, after) in zip(axes[0], panels):
        if _panel(ax, constraints, before, after):
            drew = True
        ax.set_title(label)
        ax.set_ylabel("Max group difference")
    if not drew:
        plt.close(fig)
        return None
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Mitigation before/after - {attr}")
    fig.tight_layout()
    out_path = figures_dir / f"mitigation_before_after_{_slug(attr)}.png"
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def render_mitigation_figures(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    attrs: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Render one before/after figure per attribute under ``out_dir/figures``.

    ``rows`` are the flattened mitigation records (one per run_key x attr x
    constraint). Rows carrying an ``error`` (a fairlearn-rejected combo) are skipped.
    """
    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if not r.get("error")]
    by_attr: dict[str, list[Mapping[str, Any]]] = {}
    for r in valid:
        by_attr.setdefault(str(r.get("attr")), []).append(r)

    written: list[Path] = []
    for attr in attrs or sorted(by_attr):
        path = _plot_attr(attr, by_attr.get(attr, []), figures_dir)
        if path:
            written.append(path)
    logger.info(
        "[SUCCESS] Mitigation figures saved to %s (%d figure(s))", figures_dir, len(written)
    )
    return written
