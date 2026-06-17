"""Readiness/data-validity figures for dermatology runs.

These plots are intentionally pre-model: they visualize split-aware data
fairness profiles so dissertation claims can separate data limitations from
model behavior. Rendered at stage 4 (preprocess) from the split profile JSON.

``matplotlib``'s config dir is set once in :mod:`fairxai.viz` (package import),
so this module does not touch ``MPLCONFIGDIR`` itself.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fairxai.viz.save_utils import save_figure  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUTPUTS = ("subgroup_support", "target_prevalence")
_DEFAULT_COLOR = "#3b4cc0"
_LOW_SUPPORT_COLOR = "#b40426"


def _load_json(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read readiness figure input %s: %s", path, exc)
        return None


def _profile_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_stats = profile.get("group_statistics", {})
    distributions = profile.get("sensitive_attr_distribution", {})
    for attr, groups in group_stats.items():
        counts = distributions.get(attr, {}).get("counts", {})
        for group, stats in groups.items():
            rows.append(
                {
                    "attribute": str(attr),
                    "group": str(group),
                    "label": f"{attr}: {group}",
                    "n": int(stats.get("n_samples", counts.get(group, 0)) or 0),
                    "prevalence": stats.get("target_prevalence"),
                }
            )
    return rows


def _plot_subgroup_support(
    profile: dict[str, Any],
    output_file: Path,
    *,
    min_group_samples: int,
) -> Optional[Path]:
    rows = sorted(_profile_rows(profile), key=lambda r: (r["attribute"], r["n"]))
    if not rows:
        return None

    labels = [r["label"] for r in rows]
    values = [r["n"] for r in rows]
    colors = [_LOW_SUPPORT_COLOR if n < min_group_samples else _DEFAULT_COLOR for n in values]
    height = max(5.0, len(rows) * 0.42 + 1.8)
    fig, ax = plt.subplots(figsize=(10.0, height))
    ax.barh(range(len(rows)), values, color=colors)
    ax.axvline(min_group_samples, color="#222222", linestyle="--", linewidth=1.0)
    ax.text(
        min_group_samples,
        len(rows) - 0.25,
        f" min n={min_group_samples}",
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.set_title("Sensitive Subgroup Support")
    ax.set_xlabel("Samples")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_file)
    plt.close(fig)
    return output_file


def _plot_target_prevalence(profile: dict[str, Any], output_file: Path) -> Optional[Path]:
    rows = [r for r in _profile_rows(profile) if isinstance(r.get("prevalence"), (int, float))]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: (r["attribute"], r["prevalence"]))
    labels = [r["label"] for r in rows]
    values = [float(r["prevalence"]) for r in rows]
    overall = profile.get("basic_stats", {}).get("target_prevalence")

    height = max(5.0, len(rows) * 0.42 + 1.8)
    fig, ax = plt.subplots(figsize=(10.0, height))
    ax.barh(range(len(rows)), values, color=_DEFAULT_COLOR)
    if isinstance(overall, (int, float)):
        ax.axvline(float(overall), color="#222222", linestyle="--", linewidth=1.0)
        ax.text(
            float(overall),
            len(rows) - 0.25,
            " overall",
            va="top",
            ha="left",
            fontsize=9,
        )
    ax.set_title("Target Prevalence By Sensitive Group")
    ax.set_xlabel("Positive-label prevalence")
    ax.set_xlim(0, 1.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_file)
    plt.close(fig)
    return output_file


def render_readiness_figures(
    *,
    profile_path: Optional[Path],
    out_dir: Path,
    outputs: Optional[Iterable[str]] = None,
    min_group_samples: int = 50,
) -> list[Path]:
    """Render configured dermatology readiness figures; return the written paths.

    Each figure logs a written/skip line (no manifest file — the run logs plus the
    clickable output folder are the record).
    """
    requested = list(outputs or DEFAULT_OUTPUTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = _load_json(profile_path)
    written: list[Path] = []

    for output in requested:
        path: Optional[Path] = None
        reason: Optional[str] = None
        if output == "subgroup_support":
            if profile is None:
                reason = "missing_profile"
            else:
                path = _plot_subgroup_support(
                    profile,
                    out_dir / "subgroup_support.png",
                    min_group_samples=min_group_samples,
                )
        elif output == "target_prevalence":
            if profile is None:
                reason = "missing_profile"
            else:
                path = _plot_target_prevalence(profile, out_dir / "target_prevalence.png")
        else:
            reason = "unknown_output"

        if path:
            written.append(path)
            logger.info("[SUCCESS] readiness figure %s -> %s", output, path)
        else:
            logger.warning("Skipped readiness figure %s: %s", output, reason or "no_plot_data")

    return written
