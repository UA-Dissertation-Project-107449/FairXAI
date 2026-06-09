from __future__ import annotations

import json

import pandas as pd

from fairxai.cli.main import main
from fairxai.profiling import domain_characterization as dc


def _write_csv(tmp_path):
    path = tmp_path / "adult.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "age": [22, 38, 45, 61],
            "sex": [0, 1, 0, 1],
            "income": [0, 1, 0, 1],
        }
    ).to_csv(path, index=False)
    return path


def test_profile_dataset_returns_flat_upload_metadata(tmp_path):
    csv_path = _write_csv(tmp_path)

    result = dc.profile_dataset(str(csv_path))

    assert "metadata" not in result
    assert result["columns"] == ["id", "age", "sex", "income"]
    assert result["target_guess"] == "income"
    assert result["index_guess"] == "id"
    assert result["row_count"] == 4
    assert result["dataset_size_bytes"] == csv_path.stat().st_size
    id_profile = next(profile for profile in result["column_profiles"] if profile["name"] == "id")
    assert id_profile["n_unique"] == 4
    assert id_profile["is_all_unique"] is True


def test_profile_cli_prints_flat_json(tmp_path, capsys):
    csv_path = _write_csv(tmp_path)

    assert main(["profile", "--filename", str(csv_path)]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["columns"] == ["id", "age", "sex", "income"]
    assert result["target_guess"] == "income"
    assert result["row_count"] == 4


def test_characterize_dataset_adds_profile_fields_without_wrapping(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(dc, "_predict_ebm_difficulty", lambda **_kwargs: 0.42)

    result = dc.characterize_dataset(
        filename=str(csv_path),
        output_dir=out_dir,
        target_column="income",
        index_column="id",
    )

    assert "metadata" not in result
    assert result["metrics"]["ebmDifficulty"] == 0.42
    assert result["columns"] == ["id", "age", "sex", "income"]
    assert result["feature_columns"] == ["age", "sex"]
    assert result["target_column"] == "income"
    assert result["index_column"] == "id"
    assert result["row_count"] == 4
    assert any(profile["name"] == "income" for profile in result["column_profiles"])
