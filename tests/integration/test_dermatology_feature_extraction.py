"""Integration test: rebuild frozen-backbone features from a saved checkpoint.

The rest of the feature-space mitigation path is unit-tested with an injected
extractor. This exercises the one part that cannot be faked — architecture
rebuild, head detach, eval-mode forward pass, row alignment — against a real
torchvision model and real image files, so a registry or head-strategy mistake
surfaces here rather than on the mini-PC mid-run.

No pretrained download: weights are random (``pretrained=False``). Only the
*shape* and *alignment* of the output are asserted, never its values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from fairxai.fairness.image_feature_mitigation import (  # noqa: E402
    extract_frozen_features,
    mitigate_features_frame,
)

# resnet18 pools to 512; mobilenet_v3_large's classifier-last head sees 1280.
_EXPECTED_DIM = {"resnet18": 512, "mobilenet_v3_large": 1280}


def _write_split(tmp_path: Path, n: int) -> Path:
    from PIL import Image

    rng = np.random.default_rng(7)
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    rows = []
    for i in range(n):
        path = images_dir / f"img_{i}.png"
        Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(path)
        rows.append(
            {
                "image_path": str(path),
                "skin_cancer": i % 2,
                "sex": i % 2,
                "patient_id": f"p{i}",
            }
        )
    csv_path = tmp_path / "split_train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _write_checkpoint(tmp_path: Path, model_name: str) -> Path:
    import torch.nn as nn
    import torchvision.models as tv_models

    from fairxai.training.vision import _build_image_model

    model, _, _ = _build_image_model(
        tv_models, nn, model_name, pretrained=False, freeze_backbone=True
    )
    path = tmp_path / f"{model_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "image_size": 64,
            "transform": {
                "resize": 73,
                "center_crop": 64,
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
        },
        path,
    )
    return path


@pytest.mark.parametrize("model_name", sorted(_EXPECTED_DIM))
def test_features_match_the_backbone_width_and_split_rows(tmp_path: Path, model_name: str) -> None:
    csv_path = _write_split(tmp_path, 6)
    checkpoint = _write_checkpoint(tmp_path, model_name)

    features, frame = extract_frozen_features(
        checkpoint, csv_path, image_col="image_path", target_col="skin_cancer", batch_size=4
    )

    assert features.shape == (6, _EXPECTED_DIM[model_name])
    # Row alignment is what makes the sensitive columns usable: row i of the
    # matrix must be row i of the split CSV, in file order.
    assert frame["image_path"].tolist() == pd.read_csv(csv_path)["image_path"].tolist()
    assert np.isfinite(features).all()


def test_extracted_features_feed_the_mitigation_matrix(tmp_path: Path) -> None:
    """The extractor's output is directly consumable by the mitigation frame."""
    csv_path = _write_split(tmp_path, 40)
    checkpoint = _write_checkpoint(tmp_path, "resnet18")
    features, frame = extract_frozen_features(
        checkpoint, csv_path, image_col="image_path", target_col="skin_cancer", batch_size=8
    )
    meta = frame.assign(y_true=frame["skin_cancer"])

    report = mitigate_features_frame(
        features,
        meta,
        features,
        meta,
        ["sex"],
        techniques=["reweighting"],
        min_group_samples=5,
    )
    assert report["n_features"] == 512
    assert "reweighting" in report["sensitive_attributes"]["sex"]["techniques"]
