"""Unit tests for post-hoc image explainability (stage 10).

No real model weights, no training: pure heatmap functions run on a tiny hand-
built CNN, and the driver is exercised with a monkeypatched checkpoint loader
over a handful of generated images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
nn = torch.nn

from fairxai.explainability import image as xai  # noqa: E402


def _tiny_cnn() -> "nn.Module":
    """Minimal conv classifier: has a Conv2d (for Grad-CAM) and 2 logits."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )


def _write_image(path: Path, seed: int, size: int = 16) -> None:
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


# --------------------------------------------------------------------------- #
# Pure heatmap functions
# --------------------------------------------------------------------------- #
def test_gradcam_returns_normalized_map() -> None:
    model = _tiny_cnn()
    x = torch.randn(1, 3, 16, 16)
    heat = xai.gradcam_heatmap(model, x, target_class=1)
    assert heat.shape == (16, 16)
    assert heat.min() >= 0.0 and heat.max() <= 1.0


def test_find_last_conv_raises_without_conv() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, 2))
    with pytest.raises(ValueError):
        xai._find_last_conv(model)


def test_lime_heatmap_shape() -> None:
    pytest.importorskip("skimage")

    def predict_fn(images: np.ndarray) -> np.ndarray:
        return np.tile([0.3, 0.7], (len(images), 1))

    rng = np.random.default_rng(1)
    img = rng.random((16, 16, 3))
    heat = xai.lime_heatmap(predict_fn, img, target_class=1, num_samples=40)
    assert heat.shape == (16, 16)
    assert heat.min() >= 0.0 and heat.max() <= 1.0


def test_shap_heatmap_shape() -> None:
    pytest.importorskip("shap")
    model = _tiny_cnn()
    background = torch.randn(2, 3, 16, 16)
    x = torch.randn(1, 3, 16, 16)
    heat = xai.shap_heatmap(model, x, background, target_class=1)
    assert heat.shape == (16, 16)
    assert heat.min() >= 0.0 and heat.max() <= 1.0


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_outcome_mapping() -> None:
    assert xai._outcome(1, 1) == "TP"
    assert xai._outcome(0, 1) == "FP"
    assert xai._outcome(0, 0) == "TN"
    assert xai._outcome(1, 0) == "FN"


def test_select_images_stratifies_and_caps() -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, 80),
            "y_pred": rng.integers(0, 2, 80),
            "sex": rng.choice(["Female", "Male"], 80),
            "image_path": [f"img_{i}.png" for i in range(80)],
        }
    )
    sel = xai.select_images(df, ["sex"], n_samples=6, per_cell=1)
    assert len(sel) <= 6
    assert "outcome" in sel.columns
    # both sexes represented when budget allows
    assert sel["sex"].nunique() == 2


def _first_cell_index(df: pd.DataFrame, attr: str) -> int:
    """Index of the row ``select_images`` draws for ``attr``'s first cell."""
    _key, cell = next(iter(df.groupby([attr, "y_true"], dropna=False)))
    return cell.sample(1, random_state=42).index[0]


def test_select_images_draws_from_every_attribute_under_a_tight_budget() -> None:
    """A budget smaller than the first attribute's stratum count still reaches the rest.

    Cells used to be drained attribute by attribute, so with age_group listed
    first (40 strata here) a 9-image budget never reached fitzpatrick_group or
    sex, and the dermatology stage explained age strata only.
    """
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, n),
            "y_pred": rng.integers(0, 2, n),
            "age_group": rng.choice([f"g{i}" for i in range(10)], n),
            "fitzpatrick_group": rng.choice(["I-II", "III-IV", "V-VI"], n),
            "sex": rng.choice(["Female", "Male"], n),
            "image_path": [f"img_{i}.png" for i in range(n)],
        }
    )
    attrs = ["age_group", "fitzpatrick_group", "sex"]

    sel = xai.select_images(df, attrs, n_samples=9, per_cell=1)

    assert len(sel) == 9
    for attr in attrs:
        assert _first_cell_index(df, attr) in sel.index, attr


def test_select_images_ignores_predictions() -> None:
    """Two models scored on the same rows explain the same images.

    Strata used to be (group, outcome), so each model's own predictions decided
    which images it explained and the dermatology arms barely overlapped.
    """
    rng = np.random.default_rng(4)
    n = 200
    df = pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, n),
            "fitzpatrick_group": rng.choice(["I-II", "III-IV", "V-VI"], n),
            "sex": rng.choice(["Female", "Male"], n),
            "image_path": [f"img_{i}.png" for i in range(n)],
        }
    )
    arm_a = df.assign(y_pred=rng.integers(0, 2, n))
    arm_b = df.assign(y_pred=rng.integers(0, 2, n))
    attrs = ["fitzpatrick_group", "sex"]

    sel_a = xai.select_images(arm_a, attrs, n_samples=20, per_cell=2)
    sel_b = xai.select_images(arm_b, attrs, n_samples=20, per_cell=2)

    assert 0 < len(sel_a) <= 20  # a row drawn for both attributes counts once
    assert list(sel_a["image_path"]) == list(sel_b["image_path"])
    # the outcome column still reflects each model's own predictions
    assert list(sel_a["outcome"]) != list(sel_b["outcome"])


def test_select_images_no_sensitive_attr_falls_back() -> None:
    df = pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1],
            "y_pred": [0, 1, 1, 0],
            "image_path": ["a.png", "b.png", "c.png", "d.png"],
        }
    )
    sel = xai.select_images(df, ["sex"], n_samples=3)
    assert len(sel) == 3


# --------------------------------------------------------------------------- #
# Driver (monkeypatched checkpoint loader; Grad-CAM only for speed/determinism)
# --------------------------------------------------------------------------- #
def test_explain_image_model_writes_manifest_and_png(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "runs" / "run_x"
    img_dir = tmp_path / "imgs"
    rows = []
    for i in range(6):
        p = img_dir / f"lesion_{i}.png"
        _write_image(p, seed=i)
        rows.append(
            {
                "image_path": str(p),
                "y_true": i % 2,
                "y_pred": (i + 1) % 2,
                "y_proba": 0.6,
                "sex": "Female" if i % 2 else "Male",
            }
        )
    preds = pd.DataFrame(rows)

    ckpt = {
        "model_name": "tiny",
        "image_size": 16,
        "transform": {
            "resize": [16, 16],
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5],
        },
    }
    monkeypatch.setattr(xai, "_load_model", lambda path, device: (_tiny_cnn(), ckpt))

    manifest = xai.explain_image_model(
        run_root,
        "pad_ufes_20_tiny",
        tmp_path / "fake.pt",
        preds,
        image_col="image_path",
        sensitive_attrs=["sex"],
        methods=["gradcam"],
        n_samples=4,
    )

    assert manifest, "expected at least one explanation"
    out = run_root / "baseline" / "explanations" / "pad_ufes_20_tiny"
    assert (out / "manifest.csv").exists()
    assert all((Path(row["png_path"]).exists()) for row in manifest)
    assert {row["method"] for row in manifest} == {"gradcam"}
    assert {"sex", "outcome"} <= set(manifest[0])


def test_explain_image_model_picks_do_not_depend_on_row_order(tmp_path: Path, monkeypatch) -> None:
    """Arms that wrote their test predictions in different orders explain the same images."""
    img_dir = tmp_path / "imgs"
    rows = []
    for i in range(12):
        p = img_dir / f"lesion_{i:02d}.png"
        _write_image(p, seed=i)
        rows.append(
            {
                "image_path": str(p),
                "y_true": i % 2,
                "y_pred": (i // 2) % 2,
                "sex": "Female" if (i // 3) % 2 else "Male",
            }
        )
    preds = pd.DataFrame(rows)
    shuffled = preds.sample(frac=1, random_state=7).assign(y_pred=lambda d: 1 - d["y_pred"])

    ckpt = {"model_name": "tiny", "image_size": 16, "transform": {"resize": [16, 16]}}
    monkeypatch.setattr(xai, "_load_model", lambda path, device: (_tiny_cnn(), ckpt))

    def _paths(run_key: str, df: pd.DataFrame) -> list[str]:
        manifest = xai.explain_image_model(
            tmp_path / "runs" / "run_x",
            run_key,
            tmp_path / "fake.pt",
            df,
            image_col="image_path",
            sensitive_attrs=["sex"],
            methods=["gradcam"],
            n_samples=4,
        )
        return [row["image_path"] for row in manifest]

    assert _paths("arm_a", preds) == _paths("arm_b", shuffled)


def test_explain_image_model_rejects_unknown_methods(tmp_path: Path, monkeypatch) -> None:
    preds = pd.DataFrame(
        {"image_path": ["x.png"], "y_true": [1], "y_pred": [1], "y_proba": [0.9], "sex": ["Male"]}
    )
    monkeypatch.setattr(xai, "_load_model", lambda path, device: (_tiny_cnn(), {}))
    out = xai.explain_image_model(
        tmp_path / "run",
        "k",
        tmp_path / "fake.pt",
        preds,
        image_col="image_path",
        sensitive_attrs=["sex"],
        methods=["bogus"],
    )
    assert out == []
