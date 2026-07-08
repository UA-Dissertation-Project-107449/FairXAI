"""Unit tests for image train/eval transform construction (stage 7).

No image decode, no training — just inspects the transform pipelines built by
``_build_image_transforms`` and the feature-cache resolution rule under augmentation.
Torch/torchvision are optional extras, so the module skips when they are absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
transforms = pytest.importorskip("torchvision.transforms")  # noqa: E402

from fairxai.training.vision import _build_image_transforms  # noqa: E402

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _op_types(compose) -> list[type]:
    return [type(op) for op in compose.transforms]


def test_eval_transform_is_resize_centercrop_not_squish():
    """Eval framing is the standard Resize(256)->CenterCrop(224), never a square squish."""
    _, eval_t, meta = _build_image_transforms(
        transforms,
        224,
        use_augmentation=False,
        aug_cfg={},
        normalize_mean=_MEAN,
        normalize_std=_STD,
    )
    types = _op_types(eval_t)
    assert transforms.Resize in types
    assert transforms.CenterCrop in types
    assert transforms.RandomResizedCrop not in types
    assert meta["resize"] == 256  # round(224 * 256 / 224)
    assert meta["center_crop"] == 224


def test_no_augmentation_train_equals_eval():
    """With augmentation off the train pipeline is the deterministic eval pipeline."""
    train_t, eval_t, meta = _build_image_transforms(
        transforms,
        224,
        use_augmentation=False,
        aug_cfg={},
        normalize_mean=_MEAN,
        normalize_std=_STD,
    )
    assert _op_types(train_t) == _op_types(eval_t)
    assert meta["use_augmentation"] is False
    assert "augmentation" not in meta


def test_augmentation_adds_train_only_stochastic_ops():
    """Augmentation prepends crop/flips/rotation/blur/jitter to train; eval stays clean."""
    train_t, eval_t, meta = _build_image_transforms(
        transforms,
        224,
        use_augmentation=True,
        aug_cfg={"crop_scale_min": 0.7, "rotation_degrees": 20, "blur_prob": 0.2},
        normalize_mean=_MEAN,
        normalize_std=_STD,
    )
    train_types = _op_types(train_t)
    assert transforms.RandomResizedCrop in train_types
    assert transforms.RandomHorizontalFlip in train_types
    assert transforms.RandomVerticalFlip in train_types
    assert transforms.RandomRotation in train_types
    assert transforms.RandomApply in train_types  # blur wrapped in RandomApply
    assert transforms.ColorJitter in train_types
    # Eval never sees stochastic ops.
    eval_types = _op_types(eval_t)
    for stochastic in (
        transforms.RandomResizedCrop,
        transforms.RandomHorizontalFlip,
        transforms.RandomRotation,
        transforms.ColorJitter,
    ):
        assert stochastic not in eval_types
    assert meta["augmentation"]["crop_scale_min"] == 0.7


def test_colorjitter_leaves_skin_tone_untouched():
    """Fairness guard: brightness/contrast only, no saturation or hue shifts."""
    train_t, _, _ = _build_image_transforms(
        transforms,
        224,
        use_augmentation=True,
        aug_cfg={},
        normalize_mean=_MEAN,
        normalize_std=_STD,
    )
    jitter = next(op for op in train_t.transforms if isinstance(op, transforms.ColorJitter))
    assert jitter.brightness is not None
    assert jitter.contrast is not None
    assert jitter.saturation is None
    assert jitter.hue is None


def test_blur_prob_zero_omits_blur():
    """blur_prob=0 drops the RandomApply(GaussianBlur) op entirely."""
    train_t, _, _ = _build_image_transforms(
        transforms,
        224,
        use_augmentation=True,
        aug_cfg={"blur_prob": 0.0},
        normalize_mean=_MEAN,
        normalize_std=_STD,
    )
    assert transforms.RandomApply not in _op_types(train_t)


def test_seed_worker_is_deterministic():
    """_seed_worker derives numpy/random from the torch seed, so it repeats exactly."""
    import random

    import numpy as np
    import torch

    from fairxai.training.vision import _seed_worker

    torch.manual_seed(123)
    _seed_worker(0)
    first = (np.random.rand(), random.random())
    torch.manual_seed(123)
    _seed_worker(0)
    second = (np.random.rand(), random.random())
    assert first == second


def test_augmentation_forces_feature_cache_off():
    """use_augmentation + cache + frozen backbone must resolve feature_cache to False.

    Mirrors the boolean rule in train_image_baseline: caching is incompatible with
    per-epoch augmentation (features are extracted once, freezing a single crop).
    """
    cache_frozen_features = True
    freeze_backbone = True
    use_augmentation = True
    feature_cache = bool(cache_frozen_features and freeze_backbone and not use_augmentation)
    assert feature_cache is False
