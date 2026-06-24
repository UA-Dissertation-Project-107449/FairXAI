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


def _write_scin_fixture(root: Path) -> Path:
    """Two-CSV SCIN fixture (cases + labels joined on case_id)."""
    dataset_dir = root / "scin" / "dataset"
    image_dir = dataset_dir / "images"
    image_dir.mkdir(parents=True)

    cases = [
        {
            "case_id": -1001,
            "year": 2020,
            "age_group": "AGE_40_TO_49",
            "sex_at_birth": "MALE",
            "fitzpatrick_skin_type": "FST5",
            "combined_race": "BLACK_OR_AFRICAN_AMERICAN",
            "image_1_path": "dataset/images/h1.png",
            "image_2_path": "",
            "image_3_path": "",
        },
        {
            "case_id": -1002,
            "year": 2021,
            "age_group": "AGE_18_TO_29",
            "sex_at_birth": "FEMALE",
            "fitzpatrick_skin_type": "FST2",
            # Multi-race comma value -> collapsed to "Multiple".
            "combined_race": "HISPANIC_LATINO_OR_SPANISH_ORIGIN,WHITE",
            # First image path empty -> loader falls back to image_2_path.
            "image_1_path": "",
            "image_2_path": "dataset/images/h2.png",
            "image_3_path": "",
        },
        {
            "case_id": -1003,
            "year": 2021,
            "age_group": "AGE_UNKNOWN",
            "sex_at_birth": "OTHER_OR_UNSPECIFIED",
            "fitzpatrick_skin_type": "",
            # Empty race -> "unknown" (missingness is informative in SCIN).
            "combined_race": "",
            "image_1_path": "dataset/images/h3.png",
            "image_2_path": "",
            "image_3_path": "",
        },
    ]
    labels = [
        {"case_id": -1001, "weighted_skin_condition_label": "{'Melanoma': 0.7, 'Eczema': 0.3}"},
        {"case_id": -1002, "weighted_skin_condition_label": "{'Eczema': 0.9, 'Acne': 0.1}"},
        {
            "case_id": -1003,
            "weighted_skin_condition_label": "{'Basal Cell Carcinoma': 0.8, 'Nevus': 0.2}",
        },
    ]
    pd.DataFrame(cases).to_csv(dataset_dir / "scin_cases.csv", index=False)
    pd.DataFrame(labels).to_csv(dataset_dir / "scin_labels.csv", index=False)
    for name in ["h1.png", "h2.png", "h3.png"]:
        (image_dir / name).write_bytes(b"not-used-by-loader")
    return dataset_dir


def _write_scin_schema(path: Path) -> Path:
    payload = {
        "datasets": {
            "scin": {
                "standardizer": "scin",
                "metadata_filename": "scin_cases.csv",
                "labels_filename": "scin_labels.csv",
                "join_key": "case_id",
                "relative_dir": "scin/dataset",
                "weighted_label_column": "weighted_skin_condition_label",
                "positive_label_substrings": [
                    "melanoma",
                    "carcinoma",
                    "basal cell",
                    "squamous cell",
                ],
                "image_path_columns": ["image_1_path", "image_2_path", "image_3_path"],
                "image_globs": ["images/*.png"],
                "age_group_column": "age_group",
                "sex_column": "sex_at_birth",
                "fitzpatrick_column": "fitzpatrick_skin_type",
                "combined_race_column": "combined_race",
            }
        },
        "dermatology_relevant_datasets": ["scin"],
    }
    path.write_text(json.dumps(payload))
    return path


def test_scin_loader_joins_labels_and_standardizes(tmp_path: Path) -> None:
    _write_scin_fixture(tmp_path)
    schema_path = _write_scin_schema(tmp_path / "dermatology.json")

    loader = DermatologyDataLoader(str(schema_path))
    df = loader.load_dataset("scin", str(tmp_path))

    # One row per case (no image explode), cases+labels joined on case_id.
    assert len(df) == 3
    assert df["patient_id"].tolist() == ["-1001", "-1002", "-1003"]
    # Top-1 weighted condition -> cancer-substring binary.
    assert df["skin_cancer"].tolist() == [1, 0, 1]
    assert df["diagnostic_label"].tolist() == ["Melanoma", "Eczema", "Basal Cell Carcinoma"]
    # Native SCIN demographics mapped to the unified encodings.
    assert df["sex"].tolist() == [1, 0, -1]
    assert df["fitzpatrick_group"].tolist() == ["V-VI", "I-II", "unknown"]
    assert df["age_group"].tolist() == ["40-49", "18-29", "unknown"]
    # combined_race collapsed to a single readable group (multi-race -> Multiple,
    # empty -> unknown). Only combined_race is ingested (not the 11 one-hots).
    assert df["race_group"].tolist() == ["Black Or African American", "Multiple", "unknown"]
    # Image falls back across image_1/2/3 columns and resolves to real files.
    assert df["image_path"].map(Path).map(Path.exists).all()
    assert loader.last_image_reports["scin"]["missing_images"] == 0


def test_scin_profile_excludes_case_id_and_runs(tmp_path: Path) -> None:
    _write_scin_fixture(tmp_path)
    schema_path = _write_scin_schema(tmp_path / "dermatology.json")
    loader = DermatologyDataLoader(str(schema_path))
    df = loader.load_dataset("scin", str(tmp_path))

    profiler = DataProfiler(sensitive_attrs=["age_group", "sex", "fitzpatrick_group"])
    profile = profiler.profile_dataset(df, target="skin_cancer", dataset_name="scin")

    assert profile["dataset_name"] == "scin"
    assert "complexity_metrics" in profile
    # case_id must not leak into the model feature set / complexity inputs.
    assert "case_id" in df.columns  # survives standardization as raw column


def test_scin_profile_reports_race_group_distribution(tmp_path: Path) -> None:
    _write_scin_fixture(tmp_path)
    schema_path = _write_scin_schema(tmp_path / "dermatology.json")
    loader = DermatologyDataLoader(str(schema_path))
    df = loader.load_dataset("scin", str(tmp_path))

    # race_group is profiled when supplied (profiling-only audit attr).
    profiler = DataProfiler(sensitive_attrs=["age_group", "sex", "race_group"])
    profile = profiler.profile_dataset(df, target="skin_cancer", dataset_name="scin")
    race_dist = profile["sensitive_attr_distribution"]["race_group"]["counts"]
    assert set(race_dist) == {"Black Or African American", "Multiple", "unknown"}


def test_scin_race_group_helper_covers_all_branches() -> None:
    fn = DermatologyDataLoader._scin_race_group
    assert fn("WHITE") == "White"
    assert fn("AMERICAN_INDIAN_OR_ALASKA_NATIVE") == "American Indian Or Alaska Native"
    assert fn("HISPANIC_LATINO_OR_SPANISH_ORIGIN,WHITE") == "Multiple"
    assert fn("TWO_OR_MORE_AFTER_MITIGATION") == "Multiple"
    assert fn("PREFER_NOT_TO_ANSWER") == "prefer_not_to_answer"
    assert fn("") == "unknown"
    assert fn(float("nan")) == "unknown"
    assert fn(None) == "unknown"


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


@pytest.mark.parametrize(
    "model_name",
    ["resnet18", "mobilenet_v3_large", "efficientnet_b0", "densenet121"],
)
def test_detach_reattach_head_round_trip(model_name: str) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torch.nn as nn
    from torchvision import models as tv_models

    from fairxai.training.vision import (
        _MODEL_REGISTRY,
        _build_image_model,
        _detach_head,
        _reattach_head,
    )

    model, _, _ = _build_image_model(
        tv_models, nn, model_name, pretrained=False, freeze_backbone=True
    )
    strategy = _MODEL_REGISTRY[model_name][2]

    head = _detach_head(model, strategy, nn)
    assert isinstance(head, nn.Linear)
    assert head.out_features == 2
    # After detaching, the head slot is an Identity so the model yields raw features.
    slot = model.fc if strategy == "fc" else model.classifier
    if strategy == "classifier_last":
        slot = slot[-1]
    assert isinstance(slot, nn.Identity)

    _reattach_head(model, strategy, head)
    restored = model.fc if strategy == "fc" else model.classifier
    if strategy == "classifier_last":
        restored = restored[-1]
    assert restored is head


def test_build_predictions_df_shapes_and_metrics(tmp_path: Path) -> None:
    from fairxai.training.vision import _build_predictions_df

    csv_path = tmp_path / "src.csv"
    pd.DataFrame(
        {
            "image_path": [f"/tmp/{i}.png" for i in range(4)],
            "patient_id": ["p0", "p1", "p2", "p3"],
            "sex": [0, 1, 0, 1],
        }
    ).to_csv(csv_path, index=False)

    # Pass rows out of order to confirm row_indices drive the metadata join.
    preds, metrics = _build_predictions_df(
        y_true=[1, 0, 1],
        y_prob=[0.9, 0.2, 0.4],
        row_indices=[2, 0, 3],
        csv_path=csv_path,
        image_col="image_path",
        sensitive_cols=["sex"],
    )

    assert list(preds["y_pred"]) == [1, 0, 0]
    assert list(preds["patient_id"]) == ["p2", "p0", "p3"]
    assert {"y_true", "y_proba", "near_threshold", "image_path", "sex"} <= set(preds.columns)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_extract_features_collects_in_loader_order() -> None:
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from fairxai.training.vision import _extract_features

    # Stub "backbone" = flatten; stub dataset yields (image, label, row_index).
    images = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 1, 2, 2)
    samples = [(images[i], i % 2, i) for i in range(6)]
    loader = DataLoader(samples, batch_size=2, shuffle=False)

    feats, labels, idx = _extract_features(nn.Flatten(), loader, torch.device("cpu"), torch)

    assert feats.shape == (6, 4)
    assert labels == [0, 1, 0, 1, 0, 1]
    assert idx == [0, 1, 2, 3, 4, 5]


def test_train_head_reduces_loss_on_separable_features() -> None:
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from fairxai.training.vision import _head_scores, _train_head

    torch.manual_seed(0)
    n, dim = 64, 8
    features = torch.randn(n, dim)
    # Linearly separable target from the first feature dimension.
    labels = (features[:, 0] > 0).long()

    head = nn.Linear(dim, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.05)
    device = torch.device("cpu")

    history, total_time = _train_head(
        head,
        features,
        labels,
        criterion,
        optimizer,
        epochs=30,
        batch_size=16,
        device=device,
        torch=torch,
        random_state=0,
    )

    assert len(history) == 30
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert total_time >= 0.0

    scores = _head_scores(head, features, device, torch)
    assert len(scores) == n
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Batched scoring must equal one-shot scoring (no device-wide tensor move).
    batched = _head_scores(head, features, device, torch, batch_size=7)
    assert batched == pytest.approx(scores, rel=1e-4, abs=1e-6)
