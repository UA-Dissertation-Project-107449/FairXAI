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
    assert id_profile["n_distinct"] == 4
    assert id_profile["distinct_ratio"] == 1.0
    assert id_profile["is_all_unique"] is True
    assert id_profile["semantic_type"] == "identifier"
    assert id_profile["binning_guidance"] == "avoid"
    sex_profile = next(profile for profile in result["column_profiles"] if profile["name"] == "sex")
    assert sex_profile["semantic_type"] == "binary"
    assert sex_profile["binning_guidance"] == "use_directly"
    assert sex_profile["recommended_bin_counts"] == [2]


def test_profile_dataset_adds_guidance_for_cardinality_regimes(tmp_path):
    csv_path = tmp_path / "profile_regimes.csv"
    pd.DataFrame(
        {
            "row_id": range(100),
            "age": list(range(50)) * 2,
            "risk_band": list(range(5)) * 20,
            "sex": [0, 1] * 50,
            "mostly_missing": [None] * 60 + list(range(40)),
            "target": [0, 1] * 50,
        }
    ).to_csv(csv_path, index=False)

    result = dc.profile_dataset(str(csv_path))
    profiles = {profile["name"]: profile for profile in result["column_profiles"]}

    assert profiles["row_id"]["semantic_type"] == "identifier"
    assert profiles["age"]["semantic_type"] == "continuous"
    assert profiles["age"]["binning_guidance"] == "great_for_binning"
    assert profiles["age"]["recommended_bin_counts"] == [2, 5, 10]
    assert profiles["risk_band"]["semantic_type"] == "categorical"
    assert profiles["risk_band"]["binning_guidance"] == "limited_bins"
    assert profiles["risk_band"]["recommended_bin_counts"] == [5]
    assert profiles["mostly_missing"]["binning_guidance"] == "caution"


def test_profile_dataset_marks_ten_distinct_numeric_ok_for_binning(tmp_path):
    csv_path = tmp_path / "ten_values.csv"
    pd.DataFrame(
        {
            "score_0_to_9": list(range(10)) * 10,
            "target": [0, 1] * 50,
        }
    ).to_csv(csv_path, index=False)

    result = dc.profile_dataset(str(csv_path))
    profile = next(
        profile for profile in result["column_profiles"] if profile["name"] == "score_0_to_9"
    )

    assert profile["semantic_type"] == "continuous"
    assert profile["binning_guidance"] == "ok_for_binning"
    assert profile["recommended_bin_counts"] == [2, 5, 10]


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
