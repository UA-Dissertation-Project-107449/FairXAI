"""Unit tests for stage-8/11 dermatology figure modules (synthetic rows only)."""

from __future__ import annotations

from pathlib import Path

from fairxai.viz.dermatology_fairness import (
    render_group_view_figures,
    render_subgroup_heatmaps,
)
from fairxai.viz.dermatology_mitigation import render_mitigation_figures


def test_subgroup_heatmap_per_run_and_attr(tmp_path: Path) -> None:
    rows = [
        {
            "run_key": "pad_ufes_20_resnet18",
            "sensitive_attribute": "sex",
            "group": "Female",
            "accuracy": 0.80,
            "recall": 0.70,
            "auc": 0.82,
        },
        {
            "run_key": "pad_ufes_20_resnet18",
            "sensitive_attribute": "sex",
            "group": "Male",
            "accuracy": 0.75,
            "recall": 0.60,
            "auc": 0.78,
        },
    ]
    written = render_subgroup_heatmaps(rows, tmp_path)
    assert (tmp_path / "figures" / "subgroup_heatmap_pad_ufes_20_resnet18_sex.png").exists()
    assert written


def test_group_view_figures_render_exploratory_only(tmp_path: Path) -> None:
    rows = [
        {
            "run_key": "k",
            "group_view": "sex_x_fitzpatrick",
            "group": "Female x I-II",
            "accuracy": 0.80,
            "recall": 0.70,
            "auc": 0.80,
            "exploratory": True,
        },
        {
            "run_key": "k",
            "group_view": "sex_x_fitzpatrick",
            "group": "Male x III-IV",
            "accuracy": 0.70,
            "recall": 0.60,
            "auc": 0.75,
            "exploratory": True,
        },
        {
            "run_key": "k",
            "group_view": "sex",
            "group": "Female",
            "accuracy": 0.80,
            "recall": 0.70,
            "auc": 0.80,
            "exploratory": False,
        },
    ]
    written = render_group_view_figures(rows, tmp_path)
    names = {p.name for p in written}
    assert any("sex_x_fitzpatrick" in n for n in names)
    # the non-exploratory plain "sex" view is skipped by default
    assert not any(n.startswith("sex__") for n in names)


def test_mitigation_before_after_figure(tmp_path: Path) -> None:
    rows = [
        {
            "run_key": "pad_ufes_20_resnet18",
            "attr": "sex",
            "constraint": "demographic_parity",
            "dp_before": 0.20,
            "dp_after": 0.10,
            "tpr_before": 0.15,
            "tpr_after": 0.08,
            "fpr_before": 0.10,
            "fpr_after": 0.12,
        },
        {
            "run_key": "pad_ufes_20_resnet18",
            "attr": "sex",
            "constraint": "equalized_odds",
            "dp_before": 0.20,
            "dp_after": 0.13,
            "tpr_before": 0.15,
            "tpr_after": 0.05,
            "fpr_before": 0.10,
            "fpr_after": 0.09,
        },
    ]
    written = render_mitigation_figures(rows, tmp_path, attrs=["sex"])
    assert (tmp_path / "figures" / "mitigation_before_after_sex.png").exists()
    assert written


def test_mitigation_figure_skips_error_rows(tmp_path: Path) -> None:
    rows = [
        {"run_key": "k", "attr": "sex", "constraint": "demographic_parity", "error": "boom"},
    ]
    written = render_mitigation_figures(rows, tmp_path)
    assert written == []
