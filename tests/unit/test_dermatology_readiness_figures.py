"""Unit tests for dermatology readiness/data-validity figures."""

from __future__ import annotations

import json
from pathlib import Path

from fairxai.viz.dermatology_readiness import render_readiness_figures


def _write_profile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_name": "pad_ufes_20_test",
                "basic_stats": {"n_samples": 120, "target_prevalence": 0.45},
                "sensitive_attr_distribution": {
                    "sex": {"counts": {"0": 60, "1": 55, "-1": 5}},
                    "fitzpatrick_group": {"counts": {"I-II": 70, "III-IV": 40, "V-VI": 10}},
                },
                "group_statistics": {
                    "sex": {
                        "0": {"n_samples": 60, "target_prevalence": 0.50},
                        "1": {"n_samples": 55, "target_prevalence": 0.40},
                        "-1": {"n_samples": 5, "target_prevalence": 0.0},
                    },
                    "fitzpatrick_group": {
                        "I-II": {"n_samples": 70, "target_prevalence": 0.55},
                        "III-IV": {"n_samples": 40, "target_prevalence": 0.35},
                        "V-VI": {"n_samples": 10, "target_prevalence": 0.20},
                    },
                },
            }
        )
    )


def test_readiness_figures_render_from_profile(tmp_path: Path) -> None:
    profile = tmp_path / "pad_ufes_20_test.json"
    out_dir = tmp_path / "figures"
    _write_profile(profile)

    written = render_readiness_figures(
        profile_path=profile,
        out_dir=out_dir,
        outputs=["subgroup_support", "target_prevalence"],
        min_group_samples=50,
    )

    assert {p.name for p in written} == {"subgroup_support.png", "target_prevalence.png"}
    assert (out_dir / "subgroup_support.png").exists()
    assert (out_dir / "target_prevalence.png").exists()
    assert not (out_dir / "figures_manifest.json").exists()


def test_readiness_figures_skip_missing_profile_gracefully(tmp_path: Path) -> None:
    out_dir = tmp_path / "figures"

    written = render_readiness_figures(
        profile_path=tmp_path / "missing_profile.json",
        out_dir=out_dir,
        outputs=["subgroup_support", "target_prevalence"],
    )

    assert written == []
    assert out_dir.exists()
    assert not (out_dir / "figures_manifest.json").exists()
