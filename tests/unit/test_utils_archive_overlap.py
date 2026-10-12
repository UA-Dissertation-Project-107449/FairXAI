"""Unit tests for scripts/utils archival and record-overlap utilities."""

# ruff: noqa: E402

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils import cardiac_overlap_matrix as overlap_matrix  # noqa: E402
from scripts.utils import cardiac_record_overlap as record_overlap_util  # noqa: E402
from scripts.utils.archive_run import main as archive_main  # noqa: E402
from scripts.utils.cardiac_overlap_matrix import (  # noqa: E402
    COMMON_KEYS,
    audit_common_frames,
)
from scripts.utils.cardiac_record_overlap import overlap  # noqa: E402
from scripts.utils.cleveland_provenance import (
    DEFAULT_UCI,
    DEFAULT_WORKING,
    UCI_COLUMNS,
)
from scripts.utils.cleveland_provenance import compare as compare_cleveland  # noqa: E402
from scripts.utils.cleveland_provenance import (
    load_uci,
    load_working,
)
from scripts.utils.cleveland_provenance import main as provenance_main

KEYS = ["age_raw", "sex_bin", "trestbps", "chol", "thalach", "oldpeak"]


def _make_run(output_root: Path, domain: str, run_id: str) -> Path:
    run = output_root / domain / "runs" / run_id
    (run / "profiling").mkdir(parents=True)
    (run / "profiling" / "p.json").write_text('{"n": 1}')
    return run


def _manifest(output_root: Path, domain: str) -> list[dict]:
    path = output_root / domain / "archived_runs" / "archive_manifest.json"
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# archive_run
# --------------------------------------------------------------------------- #
class TestArchiveRun:
    def test_copy_preserves_source_and_writes_manifest(self, tmp_path):
        _make_run(tmp_path, "cardiac", "run_a")
        rc = archive_main(["run_a", "--domain", "cardiac", "--output-root", str(tmp_path)])
        assert rc == 0
        dest = tmp_path / "cardiac" / "archived_runs" / "run_a"
        assert (dest / "profiling" / "p.json").exists()
        # Source still present after a copy.
        assert (tmp_path / "cardiac" / "runs" / "run_a").exists()
        entry = _manifest(tmp_path, "cardiac")[0]
        assert entry["original_run_id"] == "run_a"
        assert entry["archived_name"] == "run_a"
        assert entry["operation"] == "copy"
        assert entry["archived_at"]

    def test_rename_via_name(self, tmp_path):
        _make_run(tmp_path, "cardiac", "run_b")
        archive_main(
            [
                "run_b",
                "--domain",
                "cardiac",
                "--name",
                "nice_name",
                "--output-root",
                str(tmp_path),
                "--note",
                "hello",
            ]
        )
        dest = tmp_path / "cardiac" / "archived_runs" / "nice_name"
        assert dest.exists()
        entry = _manifest(tmp_path, "cardiac")[0]
        assert entry["archived_name"] == "nice_name"
        assert entry["original_run_id"] == "run_b"
        assert entry["note"] == "hello"

    def test_move_removes_source(self, tmp_path):
        _make_run(tmp_path, "cardiac", "run_c")
        archive_main(["run_c", "--domain", "cardiac", "--move", "--output-root", str(tmp_path)])
        assert not (tmp_path / "cardiac" / "runs" / "run_c").exists()
        assert (tmp_path / "cardiac" / "archived_runs" / "run_c").exists()
        assert _manifest(tmp_path, "cardiac")[0]["operation"] == "move"

    def test_copy_archives_paired_logs(self, tmp_path):
        _make_run(tmp_path, "cardiac", "run_logs")
        log_dir = tmp_path / "logs" / "cardiac" / "runs" / "run_logs"
        log_dir.mkdir(parents=True)
        (log_dir / "preprocessing.log").write_text("ok")
        rc = archive_main(
            [
                "run_logs",
                "--domain",
                "cardiac",
                "--output-root",
                str(tmp_path),
                "--logs-root",
                str(tmp_path / "logs"),
            ]
        )
        assert rc == 0
        dest = tmp_path / "cardiac" / "archived_runs" / "run_logs"
        assert (dest / "_logs" / "preprocessing.log").read_text() == "ok"
        assert log_dir.exists()  # copy leaves the live log dir in place
        entry = _manifest(tmp_path, "cardiac")[0]
        assert entry["logs_archived_path"] == str(dest / "_logs")

    def test_no_logs_flag_skips_log_archive(self, tmp_path):
        _make_run(tmp_path, "cardiac", "run_nolog")
        log_dir = tmp_path / "logs" / "cardiac" / "runs" / "run_nolog"
        log_dir.mkdir(parents=True)
        (log_dir / "preprocessing.log").write_text("ok")
        archive_main(
            [
                "run_nolog",
                "--domain",
                "cardiac",
                "--output-root",
                str(tmp_path),
                "--logs-root",
                str(tmp_path / "logs"),
                "--no-logs",
            ]
        )
        dest = tmp_path / "cardiac" / "archived_runs" / "run_nolog"
        assert not (dest / "_logs").exists()
        assert _manifest(tmp_path, "cardiac")[0]["logs_archived_path"] is None

    def test_missing_logs_dir_is_tolerated(self, tmp_path):
        _make_run(tmp_path, "cardiac", "run_noldir")
        rc = archive_main(
            [
                "run_noldir",
                "--domain",
                "cardiac",
                "--output-root",
                str(tmp_path),
                "--logs-root",
                str(tmp_path / "logs"),
            ]
        )
        assert rc == 0
        dest = tmp_path / "cardiac" / "archived_runs" / "run_noldir"
        assert not (dest / "_logs").exists()
        assert _manifest(tmp_path, "cardiac")[0]["logs_archived_path"] is None

    def test_manifest_appends(self, tmp_path):
        _make_run(tmp_path, "cardiac", "run_d1")
        _make_run(tmp_path, "cardiac", "run_d2")
        archive_main(["run_d1", "--domain", "cardiac", "--output-root", str(tmp_path)])
        archive_main(["run_d2", "--domain", "cardiac", "--output-root", str(tmp_path)])
        assert len(_manifest(tmp_path, "cardiac")) == 2

    def test_refuses_overwrite_without_force(self, tmp_path):
        _make_run(tmp_path, "cardiac", "run_e")
        archive_main(["run_e", "--domain", "cardiac", "--output-root", str(tmp_path)])
        _make_run(tmp_path, "cardiac", "run_e2")
        rc = archive_main(
            [
                "run_e2",
                "--domain",
                "cardiac",
                "--name",
                "run_e",
                "--output-root",
                str(tmp_path),
            ]
        )
        assert rc == 2

    def test_force_overwrites(self, tmp_path):
        _make_run(tmp_path, "cardiac", "run_f")
        archive_main(["run_f", "--domain", "cardiac", "--output-root", str(tmp_path)])
        _make_run(tmp_path, "cardiac", "run_f2")
        rc = archive_main(
            [
                "run_f2",
                "--domain",
                "cardiac",
                "--name",
                "run_f",
                "--force",
                "--output-root",
                str(tmp_path),
            ]
        )
        assert rc == 0

    def test_missing_source_errors(self, tmp_path):
        rc = archive_main(["nope", "--domain", "cardiac", "--output-root", str(tmp_path)])
        assert rc == 2

    def test_requires_run_id_or_source_path(self, tmp_path):
        rc = archive_main(["--domain", "cardiac", "--output-root", str(tmp_path)])
        assert rc == 2

    def test_register_only(self, tmp_path):
        # Pre-existing archived dir with no live source.
        arch = tmp_path / "dermatology" / "archived_runs" / "manual_x"
        arch.mkdir(parents=True)
        (arch / "f.txt").write_text("x")
        rc = archive_main(
            [
                "--source-path",
                str(arch),
                "--domain",
                "dermatology",
                "--name",
                "manual_x",
                "--register-only",
                "--output-root",
                str(tmp_path),
            ]
        )
        assert rc == 0
        assert _manifest(tmp_path, "dermatology")[0]["operation"] == "register-only"

    def test_register_only_records_supplied_legacy_path(self, tmp_path):
        legacy = tmp_path / "_ARCHIVED" / "dermatology" / "manual_x"
        legacy.mkdir(parents=True)
        rc = archive_main(
            [
                "--source-path",
                str(legacy),
                "--domain",
                "dermatology",
                "--name",
                "manual_x",
                "--register-only",
                "--output-root",
                str(tmp_path / "output"),
            ]
        )

        assert rc == 0
        entry = _manifest(tmp_path / "output", "dermatology")[0]
        assert entry["source_path"] == str(legacy)
        assert entry["archived_path"] == str(legacy)

    @pytest.mark.parametrize(
        "unsafe_args",
        [
            ["../outside", "--domain", "cardiac"],
            ["run_a", "--domain", "../outside"],
            ["run_a", "--domain", "cardiac", "--name", "../../outside"],
        ],
    )
    def test_rejects_path_traversal_components(self, tmp_path, unsafe_args):
        rc = archive_main([*unsafe_args, "--output-root", str(tmp_path)])
        assert rc == 2

    def test_force_rejects_source_destination_overlap(self, tmp_path):
        archived = tmp_path / "cardiac" / "archived_runs" / "same"
        archived.mkdir(parents=True)
        sentinel = archived / "keep.txt"
        sentinel.write_text("do not delete")

        rc = archive_main(
            [
                "--source-path",
                str(archived),
                "--domain",
                "cardiac",
                "--name",
                "same",
                "--force",
                "--output-root",
                str(tmp_path),
            ]
        )

        assert rc == 2
        assert sentinel.read_text() == "do not delete"

    def test_corrupt_manifest_fails_closed_without_overwrite(self, tmp_path):
        source = _make_run(tmp_path, "cardiac", "run_corrupt")
        manifest = tmp_path / "cardiac" / "archived_runs" / "archive_manifest.json"
        manifest.parent.mkdir(parents=True)
        corrupt_content = "{ definitely not json"
        manifest.write_text(corrupt_content)

        rc = archive_main(["run_corrupt", "--domain", "cardiac", "--output-root", str(tmp_path)])

        assert rc == 2
        assert manifest.read_text() == corrupt_content
        assert source.exists()
        assert not (manifest.parent / "run_corrupt").exists()


# --------------------------------------------------------------------------- #
# cardiac_record_overlap
# --------------------------------------------------------------------------- #
def _df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=KEYS)


class TestRecordOverlap:
    def test_identical_frames_full_overlap(self):
        df = _df([(60, 1, 140, 240, 150, 1.0), (50, 0, 120, 200, 170, 0.5)])
        r = overlap(df, df.copy(), KEYS)
        assert r["intersection_unique_fingerprints"] == 2
        assert r["matched_row_pairs"] == 2
        assert r["a_pct_matched"] == 100.0
        assert r["b_pct_matched"] == 100.0

    def test_disjoint_frames_zero_overlap(self):
        a = _df([(60, 1, 140, 240, 150, 1.0)])
        b = _df([(41, 0, 121, 201, 171, 0.6)])
        r = overlap(a, b, KEYS)
        assert r["intersection_unique_fingerprints"] == 0
        assert r["matched_row_pairs"] == 0

    def test_subset_containment(self):
        # a is fully contained in b (b has an extra record).
        a = _df([(60, 1, 140, 240, 150, 1.0), (50, 0, 120, 200, 170, 0.5)])
        b = _df(
            [
                (60, 1, 140, 240, 150, 1.0),
                (50, 0, 120, 200, 170, 0.5),
                (33, 1, 110, 180, 190, 0.0),
            ]
        )
        r = overlap(a, b, KEYS)
        assert r["a_pct_matched"] == 100.0
        assert r["matched_row_pairs"] == 2
        assert round(r["b_pct_matched"]) == 67

    def test_oldpeak_rounding_matches(self):
        # oldpeak differing below the 0.1 rounding grain still matches.
        a = _df([(60, 1, 140, 240, 150, 1.04)])
        b = _df([(60, 1, 140, 240, 150, 1.03)])
        r = overlap(a, b, KEYS)
        assert r["intersection_unique_fingerprints"] == 1

    def test_float_int_coercion(self):
        # Same record stored as floats vs ints matches after coercion.
        a = _df([(60.0, 1.0, 140.0, 240.0, 150.0, 1.0)])
        b = _df([(60, 1, 140, 240, 150, 1.0)])
        r = overlap(a, b, KEYS)
        assert r["intersection_unique_fingerprints"] == 1

    def test_duplicate_fingerprints_are_matched_only_once(self):
        duplicate = (60, 1, 140, 240, 150, 1.0)
        a = _df([duplicate, duplicate])
        b = _df([duplicate])

        r = overlap(a, b, KEYS)

        assert r["intersection_unique_fingerprints"] == 1
        assert r["matched_row_pairs"] == 1
        assert r["a_pct_matched"] == 50.0
        assert r["b_pct_matched"] == 100.0


# --------------------------------------------------------------------------- #
# cleveland_provenance
# --------------------------------------------------------------------------- #
def _write_provenance_fixture(root: Path) -> tuple[Path, Path]:
    uci_path = root / DEFAULT_UCI
    working_path = root / DEFAULT_WORKING
    uci_path.parent.mkdir(parents=True)
    working_path.parent.mkdir(parents=True)

    uci_rows = [
        (63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0),
        (67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 7, 2),
        (55, 0, 3, 130, 250, 0, 0, 170, 0, 0.0, 1, None, 3, 1),
    ]
    pd.DataFrame(uci_rows, columns=UCI_COLUMNS).to_csv(
        uci_path,
        header=False,
        index=False,
        na_rep="?",
    )

    working_rows = [
        (63, 1, 0, 145, 233, 1, 2, 150, 0, 2.3, 2, 0, 1, 0),
        (67, 1, 3, 160, 286, 0, 2, 108, 1, 1.5, 1, 3, 2, 1),
    ]
    working_columns = [
        "age_raw",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "heart_disease",
    ]
    pd.DataFrame(working_rows, columns=working_columns).to_csv(working_path, index=False)
    return uci_path, working_path


class TestClevelandProvenance:
    def test_exact_complete_case_and_rowwise_mappings(self, tmp_path):
        uci_path, working_path = _write_provenance_fixture(tmp_path)

        report = compare_cleveland(load_uci(uci_path), load_working(working_path))

        assert report["dropped_rows_exactly_missing_ca_thal"] is True
        assert report["complete_case_stable_keys_equal_both_directions"] is True
        assert report["complete_case_exact_row_multisets_equal"] is True
        assert report["dropped_row_indices_uci"] == [2]
        assert all(check["verified"] for check in report["mapping_checks"].values())

    def test_detects_reverse_extra_row_and_mapping_error(self, tmp_path):
        uci_path, working_path = _write_provenance_fixture(tmp_path)
        uci = load_uci(uci_path)
        working = load_working(working_path)
        working.loc[working.index[0], "cp"] = 2
        working = pd.concat([working, working.iloc[[1]]], ignore_index=True)

        report = compare_cleveland(uci, working)

        assert report["complete_case_stable_keys_equal_both_directions"] is False
        assert report["complete_case_exact_row_multisets_equal"] is False
        assert report["working_rows_absent_from_uci"] == 1
        assert report["mapping_checks"]["cp_minus_one"]["mismatches"] == 1

    def test_cli_defaults_use_repository_paths(self, tmp_path, monkeypatch):
        _write_provenance_fixture(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert provenance_main([]) == 0


# --------------------------------------------------------------------------- #
# cardiac_overlap_matrix source-union audit
# --------------------------------------------------------------------------- #
def _common_df(record_ids: list[int]) -> pd.DataFrame:
    rows = []
    for i in record_ids:
        rows.append(
            (
                40 + i,
                i % 2,
                (i % 4) + 1,
                110 + i,
                180 + i,
                i % 2,
                i % 3,
                180 - i,
                i % 2,
                round(i / 10, 1),
                (i % 3) + 1,
                i % 2,
            )
        )
    return pd.DataFrame(rows, columns=COMMON_KEYS)


class TestSourceUnionAudit:
    def test_statlog_subset_adds_no_unique_records(self):
        uci = {
            "uci_cleveland_303": _common_df([0, 1, 2]),
            "uci_hungarian_294": _common_df([3]),
            "uci_switzerland_123": _common_df([4]),
            "uci_va_200": _common_df([5]),
        }
        statlog = _common_df([0, 1])
        kaggle = _common_df([0, 1, 2, 3, 4, 5])

        r = audit_common_frames(uci, statlog, kaggle)

        assert r["statlog_matches_cleveland"] == 2
        assert r["statlog_unique_beyond_cleveland"] == 0
        assert r["statlog_unique_beyond_four_uci"] == 0
        assert r["combined_source_rows_before_dedup"] == 8
        assert r["combined_unique_records_after_dedup"] == 6
        assert r["duplicate_rows_removed_by_dedup"] == 2
        assert r["deduplicated_source_count_matches_kaggle_rows"] is True
        assert r["exact_common_keysets_equal"] is True

    def test_statlog_new_record_is_reported(self):
        uci = {
            "uci_cleveland_303": _common_df([0]),
            "uci_hungarian_294": _common_df([1]),
            "uci_switzerland_123": _common_df([2]),
            "uci_va_200": _common_df([3]),
        }
        statlog = _common_df([0, 4])
        kaggle = _common_df([0, 1, 2, 3, 4])

        r = audit_common_frames(uci, statlog, kaggle)

        assert r["statlog_matches_cleveland"] == 1
        assert r["statlog_unique_beyond_cleveland"] == 1
        assert r["statlog_unique_beyond_four_uci"] == 1

    def test_default_source_union_builder_uses_three_input_groups(self, tmp_path, monkeypatch):
        paths = {
            "uci_cleveland_303": tmp_path / "cleveland.data",
            "uci_hungarian_294": tmp_path / "hungarian.data",
            "uci_switzerland_123": tmp_path / "switzerland.data",
            "uci_va_200": tmp_path / "va.data",
        }
        statlog_path = tmp_path / "statlog.data"
        kaggle_path = tmp_path / "heart.csv"
        for path in [*paths.values(), statlog_path, kaggle_path]:
            path.touch()

        frames = {
            paths["uci_cleveland_303"]: _common_df([0, 1]),
            paths["uci_hungarian_294"]: _common_df([2]),
            paths["uci_switzerland_123"]: _common_df([3]),
            paths["uci_va_200"]: _common_df([4]),
        }
        monkeypatch.setattr(overlap_matrix, "UCI_SOURCE_PATHS", paths)
        monkeypatch.setattr(overlap_matrix, "STATLOG_PATH", statlog_path)
        monkeypatch.setattr(overlap_matrix, "KAGGLE_HEART_PATH", kaggle_path)
        monkeypatch.setattr(overlap_matrix, "_load_uci_common", lambda path: frames[path])
        monkeypatch.setattr(
            overlap_matrix,
            "_load_statlog_common",
            lambda _path: _common_df([0]),
        )
        monkeypatch.setattr(
            overlap_matrix,
            "_load_heart918_common",
            lambda _path: _common_df([0, 1, 2, 3, 4]),
        )

        report = overlap_matrix.build_source_union_audit()

        assert report["available"] is True
        assert report["statlog_unique_beyond_four_uci"] == 0
        assert report["combined_unique_records_after_dedup"] == 5


@pytest.mark.local_data
def test_local_default_provenance_and_overlap_smoke(monkeypatch):
    """Exercise gitignored cardiac sources locally; CI deselects this marker."""
    required = [
        ROOT / DEFAULT_UCI,
        ROOT / DEFAULT_WORKING,
        ROOT / record_overlap_util.DEFAULT_A,
        ROOT / record_overlap_util.DEFAULT_B,
        *(ROOT / path for path in overlap_matrix.UCI_SOURCE_PATHS.values()),
        ROOT / overlap_matrix.STATLOG_PATH,
        ROOT / overlap_matrix.KAGGLE_HEART_PATH,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"Local cardiac sources unavailable: {missing}")

    monkeypatch.chdir(ROOT)
    assert provenance_main([]) == 0
    assert record_overlap_util.main([]) == 0
    assert overlap_matrix.main([]) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
