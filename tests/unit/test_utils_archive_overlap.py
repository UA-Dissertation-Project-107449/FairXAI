"""Unit tests for scripts/utils archival and record-overlap utilities."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.archive_run import main as archive_main  # noqa: E402
from scripts.utils.cardiac_overlap_matrix import (  # noqa: E402
    COMMON_KEYS,
    audit_common_frames,
)
from scripts.utils.cardiac_record_overlap import overlap  # noqa: E402

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


# --------------------------------------------------------------------------- #
# cardiac_record_overlap
# --------------------------------------------------------------------------- #
def _df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=KEYS)


class TestRecordOverlap:
    def test_identical_frames_full_overlap(self):
        df = _df([(60, 1, 140, 240, 150, 1.0), (50, 0, 120, 200, 170, 0.5)])
        r = overlap(df, df.copy(), KEYS)
        assert r["intersection_unique_keys"] == 2
        assert r["a_pct_in_intersection"] == 100.0
        assert r["b_pct_in_intersection"] == 100.0

    def test_disjoint_frames_zero_overlap(self):
        a = _df([(60, 1, 140, 240, 150, 1.0)])
        b = _df([(41, 0, 121, 201, 171, 0.6)])
        r = overlap(a, b, KEYS)
        assert r["intersection_unique_keys"] == 0
        assert r["a_rows_in_intersection"] == 0

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
        assert r["a_pct_in_intersection"] == 100.0
        assert r["b_rows_in_intersection"] == 2
        assert round(r["b_pct_in_intersection"]) == 67

    def test_oldpeak_rounding_matches(self):
        # oldpeak differing below the 0.1 rounding grain still matches.
        a = _df([(60, 1, 140, 240, 150, 1.04)])
        b = _df([(60, 1, 140, 240, 150, 1.03)])
        r = overlap(a, b, KEYS)
        assert r["intersection_unique_keys"] == 1

    def test_float_int_coercion(self):
        # Same record stored as floats vs ints matches after coercion.
        a = _df([(60.0, 1.0, 140.0, 240.0, 150.0, 1.0)])
        b = _df([(60, 1, 140, 240, 150, 1.0)])
        r = overlap(a, b, KEYS)
        assert r["intersection_unique_keys"] == 1


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
