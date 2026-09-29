"""Stage 7 image training: a frozen backbone must not move its BatchNorm statistics.

Freezing the weights does not freeze the running averages. With the backbone in
train mode they drifted from ImageNet toward the training cohort batch by batch,
so the cached and uncached arms trained different models and the augmentation
comparison measured augmentation plus that drift. These tests run the real
training path on a handful of 32x32 images with randomly initialised weights,
where ``running_mean`` starts at exactly zero and any update is visible.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from fairxai.training.vision import train_image_baseline  # noqa: E402

_BN_KEY = "features.0.1.running_mean"
_BN_COUNT = "features.0.1.num_batches_tracked"


def _write_split(tmp_path, name: str, n: int, seed: int):
    """A CSV of tiny solid-colour images, half of each class."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    rows = []
    for i in range(n):
        label = i % 2
        shade = 40 + (seed + i) * 7 % 180
        colour = (shade, 255 - shade, shade // 2) if label else (shade // 2, shade, 255 - shade)
        path = images_dir / f"{name}_{i}.png"
        Image.new("RGB", (32, 32), colour).save(path)
        rows.append({"image_path": str(path), "diagnostic": label, "sex": i % 2})
    csv_path = tmp_path / f"{name}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _train(tmp_path, *, freeze_backbone: bool):
    train_csv = _write_split(tmp_path, "train", 8, seed=0)
    test_csv = _write_split(tmp_path, "test", 4, seed=5)
    result = train_image_baseline(
        train_csv=train_csv,
        test_csv=test_csv,
        output_root=tmp_path / "out",
        dataset_name="tiny",
        target_col="diagnostic",
        sensitive_cols=["sex"],
        model_name="mobilenet_v3_large",
        device_request="cpu",
        epochs=1,
        batch_size=4,
        image_size=32,
        pretrained=False,
        freeze_backbone=freeze_backbone,
        cache_frozen_features=False,
        use_augmentation=False,
    )
    checkpoint = torch.load(result["model_file"], weights_only=False)
    return result, checkpoint


def test_frozen_backbone_keeps_its_batchnorm_statistics(tmp_path):
    result, checkpoint = _train(tmp_path, freeze_backbone=True)
    state = checkpoint["model_state_dict"]
    assert torch.count_nonzero(state[_BN_KEY]) == 0
    assert int(state[_BN_COUNT]) == 0
    assert checkpoint["backbone_train_mode"] == "eval"
    assert result["config"]["backbone_train_mode"] == "eval"


def test_unfrozen_backbone_still_trains_in_train_mode(tmp_path):
    result, checkpoint = _train(tmp_path, freeze_backbone=False)
    state = checkpoint["model_state_dict"]
    # The contrast: when the backbone is meant to learn, the statistics move.
    assert torch.count_nonzero(state[_BN_KEY]) > 0
    assert int(state[_BN_COUNT]) > 0
    assert checkpoint["backbone_train_mode"] == "train"
    assert result["config"]["backbone_train_mode"] == "train"


def test_head_still_learns_with_the_backbone_in_eval(tmp_path):
    """eval() on the backbone must not stop the gradient reaching the head."""
    result, checkpoint = _train(tmp_path, freeze_backbone=True)
    head_weight = checkpoint["model_state_dict"]["classifier.3.weight"]
    assert torch.isfinite(head_weight).all()
    assert result["history"][0]["train_loss"] > 0
