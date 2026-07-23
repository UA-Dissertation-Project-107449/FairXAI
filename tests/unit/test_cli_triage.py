from __future__ import annotations

import json

import pandas as pd
import pytest

from fairxai.cli.main import main


@pytest.fixture
def dataset(tmp_path):
    """A small dataset with one free-text column and one string identifier."""
    path = tmp_path / "cohort.csv"
    rows = 40
    pd.DataFrame(
        {
            "patient_ref": [f"P{i:04d}" for i in range(rows)],
            "age": [20 + (i % 45) for i in range(rows)],
            "sex": [i % 2 for i in range(rows)],
            "clinical_note": [f"free text note number {i}" for i in range(rows)],
            "target": [i % 2 for i in range(rows)],
        }
    ).to_csv(path, index=False)
    return path


def _run(argv: list[str]) -> int:
    return main(argv)


# --- characterize no longer runs triage -------------------------------------


def test_characterize_emits_no_triage_keys(dataset, tmp_path, capsys):
    out_dir = tmp_path / "out"
    rc = _run(
        [
            "characterize",
            "--filename",
            str(dataset),
            "--output-dir",
            str(out_dir),
            "--target-column",
            "target",
            "--print-json",
        ]
    )
    assert rc == 0

    result = json.loads(capsys.readouterr().out)
    assert "triage_report" not in result
    assert "triage_status" not in result
    assert "triage_error" not in result

    written = json.loads((out_dir / "cohort.json").read_text(encoding="utf-8"))
    assert "triage_report" not in written


def test_characterize_rejects_removed_triage_flags(dataset, tmp_path):
    with pytest.raises(SystemExit):
        _run(
            [
                "characterize",
                "--filename",
                str(dataset),
                "--output-dir",
                str(tmp_path),
                "--include-triage",
            ]
        )


# --- fairxai triage ---------------------------------------------------------


def test_triage_prints_only_the_report(dataset, tmp_path, capsys):
    rc = _run(
        [
            "triage",
            "--filename",
            str(dataset),
            "--target-column",
            "target",
            "--index-column",
            "patient_ref",
            "--sensitive-columns",
            "sex",
        ]
    )
    assert rc == 0

    report = json.loads(capsys.readouterr().out)
    assert isinstance(report, dict)
    # The report is the whole payload — not nested under a characterization result.
    assert "metrics" not in report
    assert "pca2d" not in report

    # Nothing is written to disk.
    assert list(tmp_path.glob("*.json")) == []


def test_triage_accepts_an_all_unique_string_index(dataset, capsys):
    assert (
        _run(
            [
                "triage",
                "--filename",
                str(dataset),
                "--target-column",
                "target",
                "--index-column",
                "patient_ref",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)


def test_triage_rejects_a_text_target(dataset, capsys):
    rc = _run(
        [
            "triage",
            "--filename",
            str(dataset),
            "--target-column",
            "clinical_note",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "free text" in captured.err
    assert captured.out == ""


def test_triage_rejects_a_text_sensitive_column(dataset, capsys):
    rc = _run(
        [
            "triage",
            "--filename",
            str(dataset),
            "--target-column",
            "target",
            "--sensitive-columns",
            "clinical_note",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "sensitive attributes" in captured.err
    assert captured.out == ""


def test_triage_rejects_unknown_columns(dataset, capsys):
    rc = _run(
        [
            "triage",
            "--filename",
            str(dataset),
            "--target-column",
            "nope",
        ]
    )
    assert rc == 1
    assert "not found" in capsys.readouterr().err
