"""The resume guard: what a run ran on, not just how far it got.

Checkpoint markers prove a stage finished. They say nothing about the datasets,
families or config that stage finished on, so a resume after a config edit
silently mixes two experiments under one run ID. These tests pin the manifest
contract and the shell entry point the pipelines call.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fairxai.pipeline import (  # noqa: E402
    RUN_MANIFEST_FILENAME,
    build_run_manifest,
    compare_run_manifests,
    read_run_manifest,
    write_run_manifest,
)
from fairxai.utils.config import dataset_excluded_model_types  # noqa: E402


def _manifest(tmp_path: Path, config_text: str = "a: 1", **overrides) -> dict:
    config = tmp_path / "some.yaml"
    config.write_text(config_text)
    kwargs = {
        "pipeline": "cardiac",
        "datasets": ["cleveland_uci", "four_site_uci"],
        "model_types": ["logistic_regression"],
        "config_paths": [config],
        "flags": {"hpo_study": "true"},
        "project_root": tmp_path,
    }
    kwargs.update(overrides)
    return build_run_manifest(**kwargs)


class TestManifestContract:
    def test_round_trip(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path)
        write_run_manifest(tmp_path / "run", manifest)

        recorded = read_run_manifest(tmp_path / "run")
        assert compare_run_manifests(recorded, manifest) == []

    def test_absent_manifest_reads_as_none(self, tmp_path: Path) -> None:
        assert read_run_manifest(tmp_path) is None

    def test_unreadable_manifest_reads_as_none(self, tmp_path: Path) -> None:
        (tmp_path / RUN_MANIFEST_FILENAME).write_text("{not json")
        assert read_run_manifest(tmp_path) is None

    def test_field_order_does_not_count_as_a_change(self, tmp_path: Path) -> None:
        first = _manifest(tmp_path, datasets=["a", "b"])
        second = _manifest(tmp_path, datasets=["b", "a"])
        assert compare_run_manifests(first, second) == []

    def test_a_config_edit_is_reported_by_name(self, tmp_path: Path) -> None:
        before = _manifest(tmp_path, config_text="a: 1")
        after = _manifest(tmp_path, config_text="a: 2")

        differences = compare_run_manifests(before, after)
        assert len(differences) == 1
        assert differences[0].startswith("config_hashes[some.yaml]")

    def test_narrowed_scope_and_flipped_flag_are_both_reported(self, tmp_path: Path) -> None:
        before = _manifest(tmp_path)
        after = _manifest(tmp_path, datasets=["cleveland_uci"], flags={"hpo_study": "false"})

        differences = compare_run_manifests(before, after)
        assert any(d.startswith("datasets:") for d in differences)
        assert any(d.startswith("flags[hpo_study]") for d in differences)

    def test_a_missing_config_file_is_recorded_rather_than_raising(self, tmp_path: Path) -> None:
        manifest = build_run_manifest(
            "cardiac", [], [], [tmp_path / "gone.yaml"], project_root=tmp_path
        )
        assert manifest["config_hashes"]["gone.yaml"] == "missing"


def _guard(tmp_path: Path, run_root: Path, config: Path, *, resuming: int, **env) -> tuple:
    """Call run_manifest_guard the way the pipelines do."""
    script = (
        f'source "{ROOT}/scripts/common/stage_registry.sh"\n'
        f'run_manifest_guard "{ROOT}" "{run_root}" cardiac {resuming} '
        f'"cleveland_uci" "logistic_regression" "hpo_study={env.pop("hpo", "true")}" "{config}"'
    )
    process = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    return process.returncode, process.stderr


class TestShellGuard:
    def test_a_fresh_run_records_the_manifest(self, tmp_path: Path) -> None:
        config = tmp_path / "c.yaml"
        config.write_text("a: 1")
        run_root = tmp_path / "run"

        code, _ = _guard(tmp_path, run_root, config, resuming=0)
        assert code == 0
        assert json.loads((run_root / RUN_MANIFEST_FILENAME).read_text())["pipeline"] == "cardiac"

    def test_an_unchanged_resume_passes(self, tmp_path: Path) -> None:
        config = tmp_path / "c.yaml"
        config.write_text("a: 1")
        run_root = tmp_path / "run"
        _guard(tmp_path, run_root, config, resuming=0)

        code, _ = _guard(tmp_path, run_root, config, resuming=1)
        assert code == 0

    def test_a_changed_resume_stops_the_run(self, tmp_path: Path) -> None:
        config = tmp_path / "c.yaml"
        config.write_text("a: 1")
        run_root = tmp_path / "run"
        _guard(tmp_path, run_root, config, resuming=0)
        config.write_text("a: 2")

        code, stderr = _guard(tmp_path, run_root, config, resuming=1, hpo="false")
        assert code == 1
        assert "config_hashes[c.yaml]" in stderr
        assert "flags[hpo_study]" in stderr

    def test_the_override_proceeds_and_rewrites_the_manifest(self, tmp_path: Path) -> None:
        config = tmp_path / "c.yaml"
        config.write_text("a: 1")
        run_root = tmp_path / "run"
        _guard(tmp_path, run_root, config, resuming=0)
        config.write_text("a: 2")

        code, _ = _guard(tmp_path, run_root, config, resuming=1, ALLOW_MANIFEST_CHANGE="1")
        assert code == 0
        # Rewritten, so a second resume with the same inputs is clean again.
        assert _guard(tmp_path, run_root, config, resuming=1)[0] == 0

    def test_a_run_that_predates_the_manifest_records_one(self, tmp_path: Path) -> None:
        config = tmp_path / "c.yaml"
        config.write_text("a: 1")
        run_root = tmp_path / "run"
        run_root.mkdir()

        code, _ = _guard(tmp_path, run_root, config, resuming=1)
        assert code == 0
        assert (run_root / RUN_MANIFEST_FILENAME).exists()


class TestPipelinesCallTheGuard:
    @pytest.mark.parametrize(
        "script",
        ["scripts/cardiac/cardiac_pipeline.sh", "scripts/dermatology/dermatology_pipeline.sh"],
    )
    def test_the_guard_runs_before_any_stage(self, script: str) -> None:
        text = (ROOT / script).read_text()
        assert "run_manifest_guard" in text
        assert text.index("run_manifest_guard") < text.index("should_run 1")


class TestPerCohortFamilyExclusions:
    def test_the_configured_cohort_drops_its_family(self) -> None:
        config = {"dataset_overrides": {"cardio70k": {"exclude_model_types": ["SVM"]}}}
        assert dataset_excluded_model_types(config, "cardio70k") == {"svm"}
        assert dataset_excluded_model_types(config, "cleveland_uci") == set()

    def test_no_overrides_excludes_nothing(self) -> None:
        assert dataset_excluded_model_types({}, "cardio70k") == set()

    @pytest.mark.parametrize(
        "config_name", ["experiments/combinatorial.yaml", "experiments/mitigation.yaml"]
    )
    def test_the_svm_scale_decision_is_recorded_not_remembered(self, config_name: str) -> None:
        """The comments telling an operator to drop SVM on cardio70k are now config."""
        with open(ROOT / "configs" / config_name) as handle:
            config = yaml.safe_load(handle)
        assert dataset_excluded_model_types(config, "cardio70k") == {"svm"}
