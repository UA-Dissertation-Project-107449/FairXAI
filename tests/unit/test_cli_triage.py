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


# --- deprecated shim --------------------------------------------------------


def test_deprecated_shim_still_merges_triage_into_the_output_json(dataset, tmp_path, capsys):
    from fairxai.cli.characterize import main as legacy_main

    out_dir = tmp_path / "legacy"
    rc = legacy_main(
        [
            "--filename",
            str(dataset),
            "--output-dir",
            str(out_dir),
            "--target-column",
            "target",
            "--include-triage",
            "--sensitive-columns",
            "sex",
        ]
    )
    assert rc == 0
    assert "DEPRECATION" in capsys.readouterr().err

    written = json.loads((out_dir / "cohort.json").read_text(encoding="utf-8"))
    assert written["triage_status"] == "success"
    assert written["triage_report"]


def test_deprecated_shim_keeps_characterization_when_triage_fails(dataset, tmp_path):
    from fairxai.cli.characterize import main as legacy_main

    out_dir = tmp_path / "legacy_fail"
    rc = legacy_main(
        [
            "--filename",
            str(dataset),
            "--output-dir",
            str(out_dir),
            "--target-column",
            "target",
            "--include-triage",
            "--sensitive-columns",
            "clinical_note",
        ]
    )
    assert rc == 0

    written = json.loads((out_dir / "cohort.json").read_text(encoding="utf-8"))
    assert written["triage_status"] == "failed"
    assert "sensitive attributes" in written["triage_error"]
    assert written["metrics"]


@pytest.mark.parametrize("target_args", [[], ["--target-column", "not_a_column"]])
def test_deprecated_shim_triages_the_resolved_target(dataset, tmp_path, target_args):
    """The old contract triaged whatever column characterization settled on.

    ``fairxai triage`` rejects an omitted or unknown target, so the shim has to
    pass the resolved column rather than the raw argument.
    """
    from fairxai.cli.characterize import main as legacy_main

    out_dir = tmp_path / "legacy_resolved"
    rc = legacy_main(
        [
            "--filename",
            str(dataset),
            "--output-dir",
            str(out_dir),
            *target_args,
            "--include-triage",
        ]
    )
    assert rc == 0

    written = json.loads((out_dir / "cohort.json").read_text(encoding="utf-8"))
    assert written["target_column"] == "target"
    assert written["triage_status"] == "success"
    assert written["triage_report"]


def test_deprecated_shim_marks_triage_not_requested(dataset, tmp_path):
    from fairxai.cli.characterize import main as legacy_main

    out_dir = tmp_path / "legacy_none"
    assert (
        legacy_main(
            [
                "--filename",
                str(dataset),
                "--output-dir",
                str(out_dir),
                "--target-column",
                "target",
            ]
        )
        == 0
    )
    written = json.loads((out_dir / "cohort.json").read_text(encoding="utf-8"))
    assert written["triage_status"] == "not_requested"
    assert "triage_report" not in written
