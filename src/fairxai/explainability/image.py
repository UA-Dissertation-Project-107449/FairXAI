"""Post-hoc explainability for trained dermatology CNNs (SHAP, LIME, Grad-CAM).

Parallel to :mod:`fairxai.explainability.tabular`, but for image models: the
tabular wrappers operate on feature matrices and do **not** work on images, so
these are separate. All three methods are post-hoc — they reload a saved
``baseline/models/<dataset>_<model>.pt`` checkpoint and explain individual test
images. None are training-time callbacks; no retraining happens here.

Layering:
- Pure heatmap functions (``gradcam_heatmap`` / ``lime_heatmap`` / ``shap_heatmap``)
  take a model (or predict fn) and an input, and return a normalised ``[0, 1]``
  2-D saliency map. They are framework-light and unit-testable on a tiny CNN.
- The driver (``explain_image_model``) loads the checkpoint, selects a small set
  of test images stratified by sensitive group and outcome (TP/FP/TN/FN), runs
  the enabled methods, and writes overlay PNGs + a manifest that links each
  explanation back to its group and outcome.

Runtime toggles (which methods, how many images) come from the ``xai`` section of
the pipeline config via the caller script.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METHODS = ("shap", "lime", "gradcam")  # canonical order (matches tabular nomenclature)
_OUTCOMES = ("TP", "FP", "TN", "FN")


# --------------------------------------------------------------------------- #
# Pure heatmap functions
# --------------------------------------------------------------------------- #
def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max a 2-D map to ``[0, 1]``; flat maps become all-zeros."""
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _find_last_conv(model: Any) -> Any:
    """Return the last ``Conv2d`` module in the network (Grad-CAM target layer)."""
    import torch.nn as nn  # type: ignore

    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("No Conv2d layer found; Grad-CAM needs a convolutional model.")
    return last


def gradcam_heatmap(
    model: Any,
    input_tensor: Any,
    target_class: int,
    conv_layer: Optional[Any] = None,
) -> np.ndarray:
    """Grad-CAM saliency for one image (1xCxHxW tensor) wrt ``target_class``.

    Hooks the last conv layer's activations and gradients, weights channels by the
    mean gradient, ReLUs the combination, and upsamples to the input resolution.
    """
    import torch  # type: ignore
    import torch.nn.functional as functional  # type: ignore

    model.eval()
    conv_layer = conv_layer or _find_last_conv(model)

    activations: dict[str, Any] = {}
    gradients: dict[str, Any] = {}
    fwd = conv_layer.register_forward_hook(lambda _m, _i, out: activations.__setitem__("v", out))
    bwd = conv_layer.register_full_backward_hook(
        lambda _m, _gi, gout: gradients.__setitem__("v", gout[0])
    )
    try:
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = model(input_tensor)
        model.zero_grad(set_to_none=True)
        logits[0, target_class].backward()

        act = activations["v"][0]  # (C, h, w)
        grad = gradients["v"][0]  # (C, h, w)
        weights = grad.mean(dim=(1, 2))  # (C,)
        cam = torch.relu((weights[:, None, None] * act).sum(dim=0))  # (h, w)
        cam = functional.interpolate(
            cam[None, None], size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        return _normalize(cam.detach().cpu().numpy())
    finally:
        fwd.remove()
        bwd.remove()


def lime_heatmap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    target_class: int,
    *,
    num_samples: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """LIME image saliency: per-superpixel weight for ``target_class``.

    ``image`` is HxWx3 in ``[0, 1]``; ``predict_fn`` maps a batch of such images to
    class-probability rows. The returned map assigns each superpixel its local
    linear weight (clamped at 0), normalised to ``[0, 1]``.
    """
    from lime import lime_image  # type: ignore

    explainer = lime_image.LimeImageExplainer(random_state=random_state)
    explanation = explainer.explain_instance(
        image.astype(np.float64),
        predict_fn,
        labels=(target_class,),
        hide_color=0,
        num_samples=num_samples,
        random_seed=random_state,
    )
    segments = explanation.segments
    weights = dict(explanation.local_exp[target_class])
    heat = np.zeros(segments.shape, dtype=np.float64)
    for seg_id, weight in weights.items():
        heat[segments == seg_id] = max(weight, 0.0)
    return _normalize(heat)


def shap_heatmap(
    model: Any,
    input_tensor: Any,
    background: Any,
    target_class: int,
) -> np.ndarray:
    """SHAP saliency via GradientExplainer; abs-sum over channels, normalised."""
    import shap  # type: ignore

    model.eval()
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(input_tensor)
    # shap_values: list (per class) of (N, C, H, W) — or a single array on newer shap.
    if isinstance(shap_values, list):
        values = shap_values[target_class]
    else:
        values = shap_values[..., target_class] if shap_values.ndim == 5 else shap_values
    values = np.asarray(values)[0]  # (C, H, W)
    return _normalize(np.abs(values).sum(axis=0))


# --------------------------------------------------------------------------- #
# Image selection (stratified by sensitive group x outcome)
# --------------------------------------------------------------------------- #
def _outcome(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    return "FN"


def select_images(
    df: pd.DataFrame,
    sensitive_attrs: Iterable[str],
    *,
    n_samples: int,
    per_cell: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Pick up to ``n_samples`` rows stratified by (sensitive group, outcome).

    Round-robins across available sensitive attributes so every attribute and
    every outcome class gets some coverage before the budget is spent.
    """
    work = df.copy()
    work["_outcome"] = [_outcome(int(t), int(p)) for t, p in zip(work["y_true"], work["y_pred"])]
    attrs = [a for a in sensitive_attrs if a in work.columns]

    picked: list[int] = []
    seen: set[int] = set()
    # Build (attr, group, outcome) cells; draw round-robin until budget hit.
    cells: list[pd.DataFrame] = []
    for attr in attrs:
        for (_group, _oc), cell in work.groupby([attr, "_outcome"], dropna=False):
            cells.append(cell.sample(min(per_cell, len(cell)), random_state=random_state))
    if not cells:  # no sensitive attrs present — fall back to a plain sample
        cells = [work.sample(min(n_samples, len(work)), random_state=random_state)]

    for cell in cells:
        for idx in cell.index:
            if idx in seen:
                continue
            picked.append(idx)
            seen.add(idx)
            if len(picked) >= n_samples:
                return (
                    work.loc[picked]
                    .drop(columns="_outcome", errors="ignore")
                    .assign(outcome=[work.at[i, "_outcome"] for i in picked])
                )
    out = work.loc[picked]
    return out.drop(columns="_outcome").assign(outcome=[work.at[i, "_outcome"] for i in picked])


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _load_model(checkpoint_path: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    """Rebuild the architecture and load saved weights (no pretrained download)."""
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torchvision.models as tv_models  # type: ignore

    from fairxai.training.vision import _build_image_model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, _, _ = _build_image_model(
        tv_models,
        nn,
        ckpt["model_name"],
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def _build_transform(ckpt: dict[str, Any]) -> Any:
    import torchvision.transforms as transforms  # type: ignore

    meta = ckpt.get("transform", {})
    size = meta.get("resize", [ckpt.get("image_size", 224)] * 2)
    mean = meta.get("normalize_mean", [0.485, 0.456, 0.406])
    std = meta.get("normalize_std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize((size[0], size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _save_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, out_path: Path) -> None:
    """Write a side-by-side original + heatmap-overlay PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(image_rgb)
    axes[0].set_title("input")
    axes[1].imshow(image_rgb)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title("saliency")
    for ax in axes:
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def explain_image_model(
    run_root: Path,
    run_key: str,
    checkpoint_path: Path,
    predictions_df: pd.DataFrame,
    *,
    image_col: str,
    sensitive_attrs: Iterable[str],
    methods: Iterable[str] = METHODS,
    n_samples: int = 12,
    per_cell: int = 1,
    num_samples_lime: int = 1000,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Explain a small stratified sample of test images for one trained model.

    Writes ``baseline/explanations/<run_key>/<method>/<image_id>.png`` and returns
    manifest rows (image id, group values, outcome, method, png path).
    """
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    methods = [m for m in methods if m in METHODS]
    if not methods:
        logger.warning("No valid methods for %s; skipping.", run_key)
        return []

    torch_device = torch.device(device)
    model, ckpt = _load_model(checkpoint_path, torch_device)
    transform = _build_transform(ckpt)

    selected = select_images(
        predictions_df, sensitive_attrs, n_samples=n_samples, per_cell=per_cell
    )
    out_root = run_root / "baseline" / "explanations" / run_key
    attrs = [a for a in sensitive_attrs if a in selected.columns]

    # Shared background batch for SHAP (a few transformed images).
    background = None
    if "shap" in methods:
        bg_imgs = [
            transform(Image.open(p).convert("RGB"))
            for p in selected[image_col].head(min(4, len(selected)))
        ]
        background = torch.stack(bg_imgs).to(torch_device) if bg_imgs else None

    manifest: list[dict[str, Any]] = []
    for pos, (_, row) in enumerate(selected.iterrows()):
        pil = Image.open(row[image_col]).convert("RGB")
        tensor = transform(pil).unsqueeze(0).to(torch_device)
        target = int(row["y_pred"])
        resized = np.asarray(pil.resize((tensor.shape[-1], tensor.shape[-2]))) / 255.0
        image_id = f"{pos:03d}_{Path(str(row[image_col])).stem}"

        for method in methods:
            try:
                if method == "gradcam":
                    heat = gradcam_heatmap(model, tensor, target)
                elif method == "lime":
                    heat = lime_heatmap(
                        _lime_predict_fn(model, transform, torch_device),
                        resized,
                        target,
                        num_samples=num_samples_lime,
                    )
                else:  # shap
                    if background is None:
                        continue
                    heat = shap_heatmap(model, tensor, background, target)
            except Exception as exc:  # noqa: BLE001 - one bad image must not kill the run
                logger.warning("%s/%s failed on %s: %s", run_key, method, image_id, exc)
                continue

            png = out_root / method / f"{image_id}.png"
            _save_overlay(resized, heat, png)
            manifest.append(
                {
                    "run_key": run_key,
                    "image_id": image_id,
                    "image_path": str(row[image_col]),
                    "method": method,
                    "y_true": int(row["y_true"]),
                    "y_pred": target,
                    "outcome": row.get("outcome"),
                    **{attr: row[attr] for attr in attrs},
                    "png_path": str(png),
                }
            )

    if manifest:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manifest).to_csv(out_root / "manifest.csv", index=False)
    logger.info("Wrote %d explanations for %s", len(manifest), run_key)
    return manifest


def _lime_predict_fn(model: Any, transform: Any, device: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return a batched HxWx3[0,1] -> class-probability function for LIME."""
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    def predict(images: np.ndarray) -> np.ndarray:
        batch = torch.stack(
            [transform(Image.fromarray((img * 255).astype(np.uint8))) for img in images]
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1)
        return probs.cpu().numpy()

    return predict
