"""Focused tests for dermatology PAD baseline plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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


def _resolve_head_linear(model, model_name: str):
    """Return the final classification Linear for a built image model."""
    if model_name == "resnet18":
        return model.fc
    if model_name == "densenet121":
        return model.classifier
    # mobilenet_v3_large / efficientnet_b0 keep the head as the last Sequential entry.
    return model.classifier[-1]


@pytest.mark.parametrize(
    "model_name",
    ["resnet18", "mobilenet_v3_large", "efficientnet_b0", "densenet121"],
)
def test_image_model_registry_builds_and_swaps_head(model_name: str) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torch.nn as nn
    from torchvision import models as tv_models

    from fairxai.training.vision import _build_image_model

    # pretrained=False: no weight download, no forward pass, CPU-only architecture build.
    model, weights_enum, weights_name = _build_image_model(
        tv_models,
        nn,
        model_name,
        pretrained=False,
        freeze_backbone=True,
        num_classes=2,
    )

    head = _resolve_head_linear(model, model_name)
    assert isinstance(head, nn.Linear)
    assert head.out_features == 2
    assert weights_name is None  # not pretrained
    assert weights_enum.endswith("_Weights")

    # Frozen backbone, trainable head.
    head_param_ids = {id(p) for p in head.parameters()}
    assert all(p.requires_grad for p in head.parameters())
    assert all(not p.requires_grad for p in model.parameters() if id(p) not in head_param_ids)


def test_image_model_registry_rejects_unknown_model() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torch.nn as nn
    from torchvision import models as tv_models

    from fairxai.training.vision import _build_image_model

    with pytest.raises(ValueError, match="Unsupported model_name"):
        _build_image_model(tv_models, nn, "not_a_model", pretrained=False, freeze_backbone=True)


def test_csv_image_dataset_is_picklable(tmp_path: Path) -> None:
    # forkserver/spawn DataLoader workers pickle the dataset; a local class would
    # break, so the dataset must stay importable and picklable at module level.
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import pickle

    from torchvision import transforms

    from fairxai.training.vision import _CsvImageDataset

    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame({"image_path": ["/tmp/x.png"], "skin_cancer": [1]}).to_csv(csv_path, index=False)
    transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])

    dataset = _CsvImageDataset(csv_path, transform, "image_path", "skin_cancer")
    restored = pickle.loads(pickle.dumps(dataset))

    assert len(restored) == 1
    assert restored.image_col == "image_path"
    assert restored.target_col == "skin_cancer"
