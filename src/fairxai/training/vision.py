"""PyTorch image baseline training for dermatology datasets."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from fairxai.utils.gpu import detect_accelerator


def _require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from PIL import Image  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
        from torchvision import models, transforms  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyTorch vision dependencies missing. Install a platform-specific torch build "
            "and project extra, e.g. pip install -e '.[vision]' after installing torch/torchvision "
            "for CUDA, ROCm, or CPU."
        ) from exc
    return torch, nn, Image, DataLoader, Dataset, models, transforms


try:  # torch is optional at import time; only required when training actually runs.
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore
except Exception:  # pragma: no cover - exercised only in torch-less environments
    _TorchDataset = object  # type: ignore


class _CsvImageDataset(_TorchDataset):
    """CSV-backed image dataset.

    Defined at module level (not as a closure) so DataLoader workers using the
    ``forkserver``/``spawn`` start methods can pickle it. ``image_col`` and
    ``target_col`` are passed in rather than captured from an enclosing scope.
    """

    def __init__(self, csv_path, transform, image_col: str, target_col: str):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_col = image_col
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torch  # type: ignore
        from PIL import Image  # type: ignore

        row = self.df.iloc[idx]
        image = Image.open(row[self.image_col]).convert("RGB")
        label = torch.tensor(int(row[self.target_col]), dtype=torch.long)
        return self.transform(image), label, idx


# model_name -> (torchvision factory attr, weights enum attr, head replacement strategy).
# Head strategies:
#   "fc"              ResNet family, final layer is ``model.fc``.
#   "classifier_last" MobileNetV3 / EfficientNet, head is the last Linear in a Sequential.
#   "classifier"      DenseNet, head is a single Linear at ``model.classifier``.
_MODEL_REGISTRY: dict[str, tuple[str, str, str]] = {
    "resnet18": ("resnet18", "ResNet18_Weights", "fc"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", "classifier_last"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", "classifier_last"),
    "densenet121": ("densenet121", "DenseNet121_Weights", "classifier"),
}


def _build_image_model(
    models: Any,
    nn: Any,
    model_name: str,
    *,
    pretrained: bool,
    freeze_backbone: bool,
    num_classes: int = 2,
) -> tuple[Any, str, str | None]:
    """Build a torchvision model with a generic classifier head swap.

    Returns ``(model, weights_enum_name, weights_name)`` where ``weights_name`` is
    the resolved weights identifier (e.g. ``IMAGENET1K_V1``) or ``None`` when
    ``pretrained`` is False. The backbone is frozen *before* the head is replaced
    so the fresh head keeps ``requires_grad=True``.
    """
    if model_name not in _MODEL_REGISTRY:
        supported = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unsupported model_name={model_name!r}. Supported: {supported}.")

    factory_attr, weights_enum_attr, head_strategy = _MODEL_REGISTRY[model_name]
    weights = getattr(models, weights_enum_attr).DEFAULT if pretrained else None
    model = getattr(models, factory_attr)(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if head_strategy == "fc":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif head_strategy == "classifier_last":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif head_strategy == "classifier":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:  # pragma: no cover - registry is the single source of truth
        raise ValueError(f"Unknown head strategy {head_strategy!r} for model {model_name!r}.")

    weights_name = getattr(weights, "name", None)
    return model, weights_enum_attr, weights_name


def _metrics(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict[str, Any]:
    y_pred = [1 if prob >= threshold else 0 for prob in y_prob]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = None
    return metrics


def _build_predictions_df(
    y_true: list[int],
    y_prob: list[float],
    row_indices: list[int],
    csv_path: Path,
    image_col: str,
    sensitive_cols: list[str],
) -> tuple[Any, dict[str, Any]]:
    """Assemble the prediction CSV frame (+ metrics) shared by both training paths."""
    base = pd.read_csv(csv_path).iloc[row_indices].reset_index(drop=True)
    preds = pd.DataFrame(
        {
            "y_true": y_true,
            "y_proba": y_prob,
            "y_pred": [1 if p >= 0.5 else 0 for p in y_prob],
            "threshold": 0.5,
            "confidence": [abs(p - 0.5) for p in y_prob],
            "near_threshold": [abs(p - 0.5) < 0.1 for p in y_prob],
        }
    )
    metadata_cols = [
        col
        for col in [image_col, "patient_id", "lesion_id", *sensitive_cols]
        if col in base.columns
    ]
    preds = pd.concat([base[metadata_cols].reset_index(drop=True), preds], axis=1)
    return preds, _metrics(y_true, y_prob)


def _detach_head(model: Any, strategy: str, nn: Any) -> Any:
    """Replace the classifier head with ``Identity`` in place and return the head module.

    Lets a frozen backbone be used as a fixed feature extractor (``model(x)`` then
    yields pooled features). Use :func:`_reattach_head` to restore the full model.
    """
    if strategy == "fc":
        head = model.fc
        model.fc = nn.Identity()
    elif strategy == "classifier_last":
        head = model.classifier[-1]
        model.classifier[-1] = nn.Identity()
    elif strategy == "classifier":
        head = model.classifier
        model.classifier = nn.Identity()
    else:  # pragma: no cover - registry is the single source of truth
        raise ValueError(f"Unknown head strategy {strategy!r}.")
    return head


def _reattach_head(model: Any, strategy: str, head: Any) -> None:
    """Inverse of :func:`_detach_head`: restore ``head`` so the full model is saveable."""
    if strategy == "fc":
        model.fc = head
    elif strategy == "classifier_last":
        model.classifier[-1] = head
    elif strategy == "classifier":
        model.classifier = head
    else:  # pragma: no cover - registry is the single source of truth
        raise ValueError(f"Unknown head strategy {strategy!r}.")


def _extract_features(
    model: Any, loader: Any, device: Any, torch: Any
) -> tuple[Any, list[int], list[int]]:
    """Run the (eval-mode, no-grad) backbone over a loader once, returning cached features.

    Returns ``(features_cpu, labels, row_indices)``. The backbone must already be in
    eval mode so BatchNorm running stats are NOT updated and dropout is disabled — the
    cached features are then identical every head-training epoch.
    """
    model.eval()
    feats: list[Any] = []
    labels: list[int] = []
    row_indices: list[int] = []
    with torch.no_grad():
        for images, lbl, indices in loader:
            out = model(images.to(device))
            feats.append(out.detach().cpu())
            labels.extend(int(v) for v in lbl.tolist())
            row_indices.extend(int(v) for v in indices.tolist())
    return torch.cat(feats, dim=0), labels, row_indices


def _train_head(
    head: Any,
    features: Any,
    labels: Any,
    criterion: Any,
    optimizer: Any,
    *,
    epochs: int,
    batch_size: int,
    device: Any,
    torch: Any,
    random_state: int,
) -> tuple[list[dict[str, Any]], float]:
    """Train a linear head over cached features. No image decode, no backbone forward."""
    head.train()
    n = features.size(0)
    generator = torch.Generator().manual_seed(random_state)
    history: list[dict[str, Any]] = []
    train_start = time.perf_counter()
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        perm = torch.randperm(n, generator=generator)
        total_loss = 0.0
        total_seen = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = features[idx].to(device)
            yb = labels[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(yb.size(0))
            total_seen += int(yb.size(0))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_seen, 1),
                "epoch_time_seconds": time.perf_counter() - epoch_start,
            }
        )
    return history, time.perf_counter() - train_start


def _head_scores(
    head: Any, features: Any, device: Any, torch: Any, *, batch_size: int = 256
) -> list[float]:
    """Positive-class probabilities from a trained head over cached features.

    Scored in batches so the full feature tensor is never moved to the device at once,
    which would risk OOM on larger datasets.
    """
    head.eval()
    scores: list[float] = []
    n = features.size(0)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = features[start : start + batch_size].to(device)
            logits = head(chunk)
            scores.extend(float(p) for p in torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return scores


def train_image_baseline(
    *,
    train_csv: Path,
    test_csv: Path,
    output_root: Path,
    dataset_name: str,
    target_col: str,
    sensitive_cols: list[str],
    image_col: str = "image_path",
    model_name: str = "resnet18",
    device_request: str = "auto",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.0003,
    image_size: int = 224,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    num_workers: int = 0,
    random_state: int = 42,
    cache_frozen_features: bool = False,
) -> dict[str, Any]:
    """Train one image baseline and persist model, predictions, metrics.

    When ``cache_frozen_features`` is True *and* ``freeze_backbone`` is True, the
    backbone is run once in eval mode to cache pooled features, and only the linear
    head is trained over those cached vectors. This skips per-epoch image decode and
    backbone forwards, so cost becomes near-independent of epoch count. Note: this
    uses the pretrained backbone's frozen BatchNorm stats with dropout disabled, so
    results differ from the default path, which keeps the backbone in train mode and
    lets its BatchNorm running stats adapt across epochs.
    """
    torch, nn, Image, DataLoader, Dataset, models, transforms = _require_torch()

    resolved = detect_accelerator(device_request)
    torch_device_name = "cuda" if resolved in {"cuda", "rocm"} else "cpu"
    device = torch.device(torch_device_name)
    torch.manual_seed(random_state)
    if torch_device_name == "cuda":
        torch.cuda.manual_seed_all(random_state)

    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )
    transform_meta = {
        "resize": [image_size, image_size],
        "normalize_mean": normalize_mean,
        "normalize_std": normalize_std,
    }
    train_dataset = _CsvImageDataset(train_csv, transform, image_col, target_col)
    test_dataset = _CsvImageDataset(test_csv, transform, image_col, target_col)
    # forkserver avoids the Python 3.12 "fork() in a multi-threaded process" deadlock
    # warning (torch spins up threads). persistent_workers keeps workers alive across
    # epochs so they are spawned once per loader instead of re-spawned every epoch.
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch_device_name == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["multiprocessing_context"] = "forkserver"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model, weights_enum, weights_name = _build_image_model(
        models,
        nn,
        model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        num_classes=2,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if cache_frozen_features and not freeze_backbone:
        logging.warning(
            "cache_frozen_features=True ignored: feature caching requires a frozen backbone "
            "(freeze_backbone=False). Falling back to the standard per-epoch training path."
        )
    feature_cache = bool(cache_frozen_features and freeze_backbone)

    if feature_cache:
        # Frozen backbone -> features are identical every epoch, so compute them once
        # (eval mode) and train only the linear head over the cached vectors.
        head_strategy = _MODEL_REGISTRY[model_name][2]
        head = _detach_head(model, head_strategy, nn)
        model.eval()
        logging.info("  feature-cache: extracting frozen-backbone features once")
        extract_start = time.perf_counter()
        train_features, train_labels, train_row_idx = _extract_features(
            model, train_loader, device, torch
        )
        test_features, test_labels, test_row_idx = _extract_features(
            model, test_loader, device, torch
        )
        feature_extraction_time_seconds = time.perf_counter() - extract_start
        head = head.to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
        history, head_train_time_seconds = _train_head(
            head,
            train_features,
            torch.tensor(train_labels, dtype=torch.long),
            criterion,
            optimizer,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            torch=torch,
            random_state=random_state,
        )
        # Total wall time = one-off feature extraction + head training, so cache-path
        # speedup numbers do not under-report the real cost.
        train_time_seconds = feature_extraction_time_seconds + head_train_time_seconds
        for entry in history:
            logging.info(
                "  epoch=%d/%d train_loss=%.4f epoch_time=%.2fs",
                entry["epoch"],
                epochs,
                entry["train_loss"],
                entry["epoch_time_seconds"],
            )
        train_predictions, train_metrics = _build_predictions_df(
            train_labels,
            _head_scores(head, train_features, device, torch, batch_size=batch_size),
            train_row_idx,
            train_csv,
            image_col,
            sensitive_cols,
        )
        test_predictions, test_metrics = _build_predictions_df(
            test_labels,
            _head_scores(head, test_features, device, torch, batch_size=batch_size),
            test_row_idx,
            test_csv,
            image_col,
            sensitive_cols,
        )
        _reattach_head(model, head_strategy, head)  # restore full model for checkpoint
        model = model.to(device)
    else:
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=learning_rate,
        )

        history = []
        train_start = time.perf_counter()
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            model.train()
            total_loss = 0.0
            total_seen = 0
            for images, labels, _ in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * int(labels.size(0))
                total_seen += int(labels.size(0))
            epoch_loss = total_loss / max(total_seen, 1)
            epoch_time = time.perf_counter() - epoch_start
            history.append(
                {"epoch": epoch + 1, "train_loss": epoch_loss, "epoch_time_seconds": epoch_time}
            )
            logging.info(
                "  epoch=%d/%d train_loss=%.4f epoch_time=%.2fs",
                epoch + 1,
                epochs,
                epoch_loss,
                epoch_time,
            )
        train_time_seconds = time.perf_counter() - train_start
        # No separate feature-extraction phase in the standard path; the backbone runs
        # inside every epoch, so all wall time is head/backbone training.
        feature_extraction_time_seconds = 0.0
        head_train_time_seconds = train_time_seconds

        def predict(loader, csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
            model.eval()
            y_true: list[int] = []
            y_prob: list[float] = []
            row_indices: list[int] = []
            with torch.no_grad():
                for images, labels, indices in loader:
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
                    y_prob.extend(float(p) for p in probs)
                    y_true.extend(int(v) for v in labels.tolist())
                    row_indices.extend(int(v) for v in indices.tolist())
            return _build_predictions_df(
                y_true, y_prob, row_indices, csv_path, image_col, sensitive_cols
            )

        train_predictions, train_metrics = predict(train_loader, train_csv)
        test_predictions, test_metrics = predict(test_loader, test_csv)

    models_dir = output_root / "models"
    predictions_dir = output_root / "results" / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{dataset_name}_{model_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "architecture": model_name,
            "weights_enum": weights_enum,
            "weights_name": weights_name,
            "transform": transform_meta,
            "target_col": target_col,
            "image_size": image_size,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "feature_cache": feature_cache,
            "device_requested": device_request,
            "device_resolved": resolved,
        },
        model_path,
    )
    train_pred_path = predictions_dir / f"{dataset_name}_{model_name}_train.csv"
    test_pred_path = predictions_dir / f"{dataset_name}_{model_name}_test.csv"
    train_predictions.to_csv(train_pred_path, index=False)
    test_predictions.to_csv(test_pred_path, index=False)

    result = {
        "status": "success",
        "model_type": model_name,
        "architecture": model_name,
        "weights_enum": weights_enum,
        "weights_name": weights_name,
        "transform": transform_meta,
        "feature_cache": feature_cache,
        "train_time_seconds": train_time_seconds,
        "feature_extraction_time_seconds": feature_extraction_time_seconds,
        "head_train_time_seconds": head_train_time_seconds,
        "model_file": str(model_path),
        "train_predictions": str(train_pred_path),
        "test_predictions": str(test_pred_path),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "device_requested": device_request,
        "device_resolved": resolved,
        "torch_device": torch_device_name,
        "n_train": len(train_dataset),
        "n_test": len(test_dataset),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "image_size": image_size,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "num_workers": num_workers,
            "cache_frozen_features": cache_frozen_features,
        },
    }
    metrics_path = output_root / "results" / f"{dataset_name}_{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    result["metrics_file"] = str(metrics_path)
    return result
