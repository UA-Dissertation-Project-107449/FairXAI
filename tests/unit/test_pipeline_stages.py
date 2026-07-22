from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from fairxai.pipeline.stages import (
    CARDIAC_STAGES,
    DERMATOLOGY_STAGES,
    STAGES,
    get_completed_stages,
    get_stage_range,
    get_stages,
    mark_stage_complete,
    resolve_stage,
    validate_prior_stages,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = ROOT_DIR / "scripts" / "common" / "export_stage_registry.py"

CARDIAC_ORDER = [
    "load",
    "profile",
    "recommend",
    "preprocess",
    "tune",
    "select_features",
    "train",
    "assess",
    "bin_attributes",
    "mitigate",
    "sweep",
    "compare",
]
DERMATOLOGY_ORDER = [
    "load",
    "profile",
    "recommend",
    "preprocess",
    "train",
    "assess",
    "compare",
    "explain",
    "mitigate",
]


# --- registry shape ---------------------------------------------------------


def test_cardiac_subset_is_ordered_and_contiguous():
    assert [s.name for s in CARDIAC_STAGES] == CARDIAC_ORDER
    assert [s.number for s in CARDIAC_STAGES] == list(range(1, 13))


def test_dermatology_subset_preserves_numbering_gaps():
    assert [s.name for s in DERMATOLOGY_STAGES] == DERMATOLOGY_ORDER
    # Stages 5 (tune) and 6 (select_features) do not apply to image pipelines,
    # and their numbers stay unused rather than being reassigned.
    assert [s.number for s in DERMATOLOGY_STAGES] == [1, 2, 3, 4, 7, 8, 9, 10, 11]


def test_stage_numbers_are_unique_within_each_domain():
    for domain in ("cardiac", "dermatology"):
        numbers = [s.number for s in get_stages(domain)]
        assert len(numbers) == len(set(numbers))


def test_shared_stage_names_keep_their_own_numbers_per_domain():
    # "compare" and "mitigate" exist in both subsets under different numbers.
    assert resolve_stage("compare", "cardiac").number == 12
    assert resolve_stage("compare", "dermatology").number == 9
    assert resolve_stage("mitigate", "cardiac").number == 10
    assert resolve_stage("mitigate", "dermatology").number == 11


def test_stages_alias_still_points_at_the_cardiac_subset():
    assert STAGES is CARDIAC_STAGES


def test_str_has_no_misleading_total():
    # Dermatology has 9 stages numbered up to 11, so "n/total" cannot be right.
    assert str(DERMATOLOGY_STAGES[-1]) == "[stage 11] mitigate"


# --- resolution -------------------------------------------------------------


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("tune", 5),
        ("hpo_study", 5),
        ("hpo", 5),
        ("select_features", 6),
        ("feature_selection_study", 6),
        ("fs_study", 6),
        ("bin_attributes", 9),
        ("attribute_binning", 9),
        ("age_binning", 9),
        ("mitigate", 10),
        ("mitigation", 10),
        ("sweep", 11),
        ("combinatorial", 11),
        ("combo", 11),
        ("triage", 3),
        ("9", 9),
        ("stage9", 9),
        ("phase11", 11),
        ("SWEEP", 11),
    ],
)
def test_cardiac_aliases_resolve(identifier, expected):
    assert resolve_stage(identifier).number == expected


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        resolve_stage("does_not_exist")


def test_unknown_domain_raises():
    with pytest.raises(ValueError, match="Unknown pipeline domain"):
        get_stages("cardiology")


def test_stage_range_is_inclusive_and_skips_gaps():
    names = [s.name for s in get_stage_range("preprocess", "assess", domain="dermatology")]
    assert names == ["preprocess", "train", "assess"]


def test_stage_range_defaults_to_the_whole_domain():
    assert get_stage_range(domain="dermatology") == list(DERMATOLOGY_STAGES)


def test_inverted_stage_range_raises():
    with pytest.raises(ValueError, match="is after"):
        get_stage_range("compare", "load")


# --- checkpoint markers -----------------------------------------------------


def test_marker_filenames_list_canonical_first_then_legacy():
    stage = resolve_stage("bin_attributes")
    assert stage.marker_filenames == ("9_bin_attributes.done", "9_attribute_binning.done")


def test_mark_stage_complete_writes_the_canonical_name(tmp_path):
    marker = mark_stage_complete(tmp_path, resolve_stage("attribute_binning"))
    assert marker.name == "9_bin_attributes.done"


def test_legacy_marker_counts_as_completed(tmp_path):
    ckpt = tmp_path / ".checkpoints"
    ckpt.mkdir()
    (ckpt / "9_attribute_binning.done").write_text("{}", encoding="utf-8")
    assert [s.name for s in get_completed_stages(tmp_path)] == ["bin_attributes"]


def test_resume_validation_accepts_legacy_markers(tmp_path):
    ckpt = tmp_path / ".checkpoints"
    ckpt.mkdir()
    legacy = {
        1: "load",
        2: "profile",
        3: "recommend",
        4: "preprocess",
        5: "hpo_study",
        6: "feature_selection_study",
        7: "train",
        8: "assess",
        9: "attribute_binning",
    }
    for number, name in legacy.items():
        (ckpt / f"{number}_{name}.done").write_text("{}", encoding="utf-8")

    # Should not raise: every prior stage has a marker under its old name.
    validate_prior_stages(tmp_path, resolve_stage("mitigate"), tmp_path)


def test_resume_validation_reports_missing_markers(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        validate_prior_stages(tmp_path, resolve_stage("assess"), tmp_path)
    assert "9_bin_attributes.done" not in str(excinfo.value)
    assert "5_tune.done" in str(excinfo.value)


# --- shell export -----------------------------------------------------------


def _export(domain: str) -> str:
    return subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), domain],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT_DIR,
    ).stdout


def test_shell_export_declares_the_expected_bash_variables():
    out = _export("cardiac")
    for declaration in (
        "declare -gA STAGE_NUM=(",
        "declare -gA STAGE_NAME=(",
        "declare -gA STAGE_MARKERS=(",
        "declare -ga STAGE_ORDER=(",
        "STAGE_FIRST=1",
        "STAGE_LAST=12",
    ):
        assert declaration in out


def test_shell_export_carries_aliases_and_legacy_markers():
    out = _export("cardiac")
    assert "[attribute_binning]=9" in out
    assert "[bin_attributes]=9" in out
    assert "'9_bin_attributes.done 9_attribute_binning.done'" in out


def test_shell_export_preserves_dermatology_gaps():
    out = _export("dermatology")
    assert "declare -ga STAGE_ORDER=(1 2 3 4 7 8 9 10 11)" in out
    assert "STAGE_LAST=11" in out


def test_shell_export_rejects_unknown_domain():
    result = subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "cardiology"],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )
    assert result.returncode != 0
