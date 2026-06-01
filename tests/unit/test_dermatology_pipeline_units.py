"""Focused tests for dermatology PAD baseline plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fairxai.data.loaders import DermatologyDataLoader
from fairxai.data.preprocessors import DermatologyPreprocessor
from fairxai.data.profilers import DataProfiler


def _write_pad_fixture(root: Path) -> Path:
    dataset_dir = root / "pad_ufes_20"
    image_dir = dataset_dir / "images" / "imgs_part_1"
    image_dir.mkdir(parents=True)
    rows = [
        {
            "patient_id": "p1",
            "lesion_id": "l1",
            "age": 55,
            "gender": "FEMALE",
            "fitspatrick": 2,
            "diagnostic": "BCC",
            "img_id": "img1.png",
        },
        {
            "patient_id": "p2",
            "lesion_id": "l2",
            "age": 35,
            "gender": "MALE",
            "fitspatrick": 5,
            "diagnostic": "NEV",
            "img_id": "img2.png",
        },
    ]
    pd.DataFrame(rows).to_csv(dataset_dir / "metadata.csv", index=False)
    for name in ["img1.png", "img2.png"]:
        (image_dir / name).write_bytes(b"not-used-by-loader")
    return dataset_dir


def _write_schema(path: Path) -> Path:
    payload = {
        "datasets": {
            "pad_ufes_20": {
                "metadata_filename": "metadata.csv",
                "relative_dir": "pad_ufes_20",
                "target": "diagnostic",
                "image_id_column": "img_id",
                "image_globs": ["images/*.png", "images/imgs_part_*/*.png"],
                "positive_labels": ["BCC", "SCC", "MEL"],
                "negative_labels": ["ACK", "NEV", "SEK"],
            }
        },
        "dermatology_relevant_datasets": ["pad_ufes_20"],
    }
    path.write_text(json.dumps(payload))
    return path


def test_pad_loader_maps_target_and_resolves_split_images(tmp_path: Path) -> None:
    _write_pad_fixture(tmp_path)
    schema_path = _write_schema(tmp_path / "dermatology.json")

    loader = DermatologyDataLoader(str(schema_path))
    df = loader.load_dataset("pad_ufes_20", str(tmp_path))

    assert df["skin_cancer"].tolist() == [1, 0]
    assert df["diagnostic_label"].tolist() == ["BCC", "NEV"]
    assert df["image_path"].map(Path).map(Path.exists).all()
    assert loader.last_image_reports["pad_ufes_20"]["missing_images"] == 0


def test_dermatology_patient_split_has_no_patient_leakage() -> None:
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(12)],
            "image_path": [f"/tmp/{i}.png" for i in range(12)],
            "skin_cancer": [0, 1] * 6,
            "age_group": ["20-39", "60-79"] * 6,
            "sex": [0, 1] * 6,
            "fitzpatrick_group": ["I-II", "III-IV"] * 6,
        }
    )
    preprocessor = DermatologyPreprocessor(
        sensitive_attrs=["age_group", "sex", "fitzpatrick_group"]
    )
    train_df, test_df = preprocessor.patient_stratified_split(
        df, target="skin_cancer", patient_col="patient_id", test_size=0.25, random_state=7
    )

    assert set(train_df["patient_id"]).isdisjoint(set(test_df["patient_id"]))
    assert len(train_df) + len(test_df) == len(df)


def test_dermatology_raw_profile_allows_metadata_nans() -> None:
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(80)],
            "image_path": [f"/tmp/{i}.png" for i in range(80)],
            "skin_cancer": [0, 1] * 40,
            "age": [float(i) if i % 5 else None for i in range(80)],
            "age_group": ["20-39", "60-79"] * 40,
            "sex": [0, 1] * 40,
            "fitzpatrick_group": ["I-II", "III-IV"] * 40,
            "diameter_1": [float(i) if i % 7 else None for i in range(80)],
            "diagnostic": ["NEV", "BCC"] * 40,
        }
    )
    profiler = DataProfiler(sensitive_attrs=["age_group", "sex", "fitzpatrick_group"])

    profile = profiler.profile_dataset(df, target="skin_cancer", dataset_name="pad_ufes_20")

    assert profile["missing_value_analysis"]["total_missing"] > 0
    assert profile["complexity_metrics"]
