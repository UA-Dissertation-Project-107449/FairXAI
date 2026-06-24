"""PyTorch image baseline training for dermatology datasets."""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from fairxai.utils.gpu import detect_accelerator

logger = logging.getLogger(__name__)


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


def _fmt_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


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


class _EarlyStopper:
    """Patience-based early stopping with best-weight restoration.

    Monitors a single "higher is better" score: validation AUC when defined, else
    ``-val_loss`` (so a degenerate single-class validation slice falls back to loss
    without changing the comparison direction). The caller supplies a no-arg
    ``state_provider`` returning the module ``state_dict`` to snapshot at each new best.
    """

    def __init__(self, patience: int, min_delta: float, enabled: bool):
        self.patience = patience
        self.min_delta = min_delta
        self.enabled = enabled
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.best_state: Any = None
        self.num_bad = 0
        self.stopped = False

    @staticmethod
    def score(val_loss: Any, val_auc: Any) -> Any:
        if val_auc is not None:
            return float(val_auc)
        if val_loss is not None:
            return -float(val_loss)
        return None

    def update(self, epoch: int, val_loss: Any, val_auc: Any, state_provider: Any) -> bool:
        """Record this epoch's validation score; return True when training should stop."""
        if not self.enabled:
            return False
        current = self.score(val_loss, val_auc)
        if current is None:
            return False
        if current > self.best_score + self.min_delta:
            self.best_score = current
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(state_provider())
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.stopped = True
        return self.stopped


def _stratified_split_indices(
    labels: Any, val_fraction: float, torch: Any, seed: int
) -> tuple[Any, Any]:
    """Split row indices into (train, val) keeping per-class proportions.

    Row-stratified only (not patient-grouped): this internal validation slice drives
    early stopping during fit and never feeds the reported test metric, which keeps
    the upstream patient-grouped split.
    """
    gen = torch.Generator().manual_seed(seed)
    label_list = [int(v) for v in labels.tolist()]
    train_parts: list[Any] = []
    val_parts: list[Any] = []
    for cls in sorted(set(label_list)):
        cls_idx = torch.tensor([i for i, v in enumerate(label_list) if v == cls], dtype=torch.long)
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=gen)]
        total = perm.numel()
        if total <= 1:
            train_parts.append(perm)  # too few to validate on; keep for fitting
            continue
        n_val = int(round(total * val_fraction))
        n_val = min(max(n_val, 1), total - 1)  # keep both sides non-empty
        val_parts.append(perm[:n_val])
        train_parts.append(perm[n_val:])
    empty = torch.tensor([], dtype=torch.long)
    train_idx = torch.cat(train_parts) if train_parts else empty
    val_idx = torch.cat(val_parts) if val_parts else empty
    return train_idx.long(), val_idx.long()


def _evaluate_head(
    head: Any, features: Any, labels: Any, criterion: Any, device: Any, torch: Any, batch_size: int
) -> tuple[float, Any]:
    """Validation loss + AUC for a linear head over cached features (AUC None if degenerate)."""
    head.eval()
    total_loss = 0.0
    total_seen = 0
    probs: list[float] = []
    with torch.no_grad():
        for start in range(0, features.size(0), batch_size):
            xb = features[start : start + batch_size].to(device)
            yb = labels[start : start + batch_size].to(device)
            logits = head(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * int(yb.size(0))
            total_seen += int(yb.size(0))
            probs.extend(float(p) for p in torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    val_loss = total_loss / max(total_seen, 1)
    y_true = [int(v) for v in labels.tolist()]
    try:
        val_auc: Any = float(roc_auc_score(y_true, probs))
    except ValueError:
        val_auc = None
    return val_loss, val_auc


def _evaluate_model(
    model: Any, loader: Any, criterion: Any, device: Any, torch: Any
) -> tuple[float, Any]:
    """Validation loss + AUC for the full model over a loader (AUC None if degenerate)."""
    model.eval()
    total_loss = 0.0
    total_seen = 0
    probs: list[float] = []
    y_true: list[int] = []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * int(labels.size(0))
            total_seen += int(labels.size(0))
            probs.extend(float(p) for p in torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            y_true.extend(int(v) for v in labels.cpu().tolist())
    val_loss = total_loss / max(total_seen, 1)
    try:
        val_auc: Any = float(roc_auc_score(y_true, probs))
    except ValueError:
        val_auc = None
    return val_loss, val_auc


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
    early_stopping: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    """Train a linear head over cached features. No image decode, no backbone forward.

    When ``early_stopping['enabled']`` is True, a stratified validation slice is carved
    from the cached features; ``epochs`` becomes a cap and the best-AUC head weights are
    restored on stop. Returns ``(history, train_time_seconds, summary)`` where ``summary``
    carries ``epochs_run``/``best_epoch``/``early_stopped``.
    """
    es_cfg = early_stopping or {}
    es_enabled = bool(es_cfg.get("enabled", False))
    patience = int(es_cfg.get("patience", 5))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    val_fraction = float(es_cfg.get("val_fraction", 0.15))

    val_features = val_labels = None
    if es_enabled and val_fraction > 0:
        train_idx, val_idx = _stratified_split_indices(labels, val_fraction, torch, random_state)
        if val_idx.numel() > 0 and train_idx.numel() > 0:
            val_features = features[val_idx]
            val_labels = labels[val_idx]
            fit_features = features[train_idx]
            fit_labels = labels[train_idx]
        else:  # too few rows to validate on; fall back to plain training
            es_enabled = False
            fit_features, fit_labels = features, labels
    else:
        es_enabled = False
        fit_features, fit_labels = features, labels

    stopper = _EarlyStopper(patience, min_delta, es_enabled)
    generator = torch.Generator().manual_seed(random_state)
    history: list[dict[str, Any]] = []
    train_start = time.perf_counter()
    m = fit_features.size(0)
    epochs_run = 0
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        head.train()
        perm = torch.randperm(m, generator=generator)
        total_loss = 0.0
        total_seen = 0
        for start in range(0, m, batch_size):
            idx = perm[start : start + batch_size]
            xb = fit_features[idx].to(device)
            yb = fit_labels[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(yb.size(0))
            total_seen += int(yb.size(0))
        val_loss = val_auc = None
        if val_features is not None:
            val_loss, val_auc = _evaluate_head(
                head, val_features, val_labels, criterion, device, torch, batch_size
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_seen, 1),
                "val_loss": val_loss,
                "val_auc": val_auc,
                "epoch_time_seconds": time.perf_counter() - epoch_start,
            }
        )
        epochs_run = epoch + 1
        if stopper.update(epoch + 1, val_loss, val_auc, head.state_dict):
            break

    if stopper.best_state is not None:
        head.load_state_dict(stopper.best_state)  # restore best-AUC weights
    summary = {
        "epochs_run": epochs_run,
        "best_epoch": stopper.best_epoch if stopper.best_state is not None else epochs_run,
        "early_stopped": stopper.stopped,
    }
    return history, time.perf_counter() - train_start, summary


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
    cache_frozen_features: bool = True,
    early_stopping: dict[str, Any] | None = None,
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

    if cache_frozen_features and not freeze_backbone:
        logger.warning(
            "cache_frozen_features=True ignored: feature caching requires a frozen backbone "
            "(freeze_backbone=False). Falling back to the standard per-epoch training path."
        )
    feature_cache = bool(cache_frozen_features and freeze_backbone)
    logger.info(
        "[PHASE] Image model start dataset=%s model=%s device_requested=%s "
        "device_resolved=%s torch_device=%s pretrained=%s freeze_backbone=%s "
        "feature_cache=%s epochs=%s batch_size=%s image_size=%s",
        dataset_name,
        model_name,
        device_request,
        resolved,
        torch_device_name,
        pretrained,
        freeze_backbone,
        feature_cache,
        epochs,
        batch_size,
        image_size,
    )
    logger.info(
        "[PHASE] Image data rows dataset=%s model=%s train=%d test=%d",
        dataset_name,
        model_name,
        len(train_dataset),
        len(test_dataset),
    )

    model, weights_enum, weights_name = _build_image_model(
        models,
        nn,
        model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        num_classes=2,
    )
    model = model.to(device)
    logger.info(
        "  weights_enum=%s weights_name=%s num_workers=%s", weights_enum, weights_name, num_workers
    )

    criterion = nn.CrossEntropyLoss()

    if feature_cache:
        # Frozen backbone -> features are identical every epoch, so compute them once
        # (eval mode) and train only the linear head over the cached vectors.
        head_strategy = _MODEL_REGISTRY[model_name][2]
        head = _detach_head(model, head_strategy, nn)
        model.eval()
        logger.info("  feature-cache: extracting frozen-backbone features once")
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
        history, head_train_time_seconds, train_summary = _train_head(
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
            early_stopping=early_stopping,
        )
        # Total wall time = one-off feature extraction + head training, so cache-path
        # speedup numbers do not under-report the real cost.
        train_time_seconds = feature_extraction_time_seconds + head_train_time_seconds
        for entry in history:
            logger.info(
                "  epoch=%d/%d train_loss=%.4f val_loss=%s val_auc=%s epoch_time=%.2fs",
                entry["epoch"],
                epochs,
                entry["train_loss"],
                _fmt_metric(entry.get("val_loss")),
                _fmt_metric(entry.get("val_auc")),
                entry["epoch_time_seconds"],
            )
        if train_summary["early_stopped"]:
            logger.info(
                "  early-stopped at epoch=%d/%d best_epoch=%d",
                train_summary["epochs_run"],
                epochs,
                train_summary["best_epoch"],
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

        es_cfg = early_stopping or {}
        es_enabled = bool(es_cfg.get("enabled", False))
        es_patience = int(es_cfg.get("patience", 5))
        es_min_delta = float(es_cfg.get("min_delta", 0.0))
        es_val_fraction = float(es_cfg.get("val_fraction", 0.15))

        # Carve a stratified validation slice for monitoring; the full train_loader is
        # still used afterwards to emit the train-prediction CSV downstream stages need.
        fit_loader = train_loader
        val_loader_es = None
        if es_enabled and es_val_fraction > 0:
            from torch.utils.data import Subset  # type: ignore

            targets = torch.tensor(train_dataset.df[target_col].astype(int).tolist())
            tr_idx, va_idx = _stratified_split_indices(
                targets, es_val_fraction, torch, random_state
            )
            if va_idx.numel() > 0 and tr_idx.numel() > 0:
                fit_loader = DataLoader(
                    Subset(train_dataset, tr_idx.tolist()),
                    batch_size=batch_size,
                    shuffle=True,
                    **loader_kwargs,
                )
                val_loader_es = DataLoader(
                    Subset(train_dataset, va_idx.tolist()),
                    batch_size=batch_size,
                    shuffle=False,
                    **loader_kwargs,
                )
            else:
                es_enabled = False

        stopper = _EarlyStopper(es_patience, es_min_delta, es_enabled)
        history = []
        train_start = time.perf_counter()
        epochs_run = 0
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            model.train()
            total_loss = 0.0
            total_seen = 0
            for images, labels, _ in fit_loader:
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
            val_loss = val_auc = None
            if val_loader_es is not None:
                val_loss, val_auc = _evaluate_model(model, val_loader_es, criterion, device, torch)
            epoch_time = time.perf_counter() - epoch_start
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "epoch_time_seconds": epoch_time,
                }
            )
            logger.info(
                "  epoch=%d/%d train_loss=%.4f val_loss=%s val_auc=%s epoch_time=%.2fs",
                epoch + 1,
                epochs,
                epoch_loss,
                _fmt_metric(val_loss),
                _fmt_metric(val_auc),
                epoch_time,
            )
            epochs_run = epoch + 1
            if stopper.update(epoch + 1, val_loss, val_auc, model.state_dict):
                break
        if stopper.best_state is not None:
            model.load_state_dict(stopper.best_state)  # restore best-AUC weights
            logger.info(
                "  early-stopped at epoch=%d/%d best_epoch=%d",
                epochs_run,
                epochs,
                stopper.best_epoch,
            )
        train_summary = {
            "epochs_run": epochs_run,
            "best_epoch": stopper.best_epoch if stopper.best_state is not None else epochs_run,
            "early_stopped": stopper.stopped,
        }
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
        "epochs_run": train_summary["epochs_run"],
        "best_epoch": train_summary["best_epoch"],
        "early_stopped": train_summary["early_stopped"],
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
            "early_stopping": early_stopping or {"enabled": False},
        },
    }
    metrics_path = output_root / "results" / f"{dataset_name}_{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    result["metrics_file"] = str(metrics_path)
    logger.info(
        "[SUCCESS] Image model complete dataset=%s model=%s accuracy=%s f1=%s auc=%s train_s=%.2f",
        dataset_name,
        model_name,
        _fmt_metric(test_metrics.get("accuracy")),
        _fmt_metric(test_metrics.get("f1_score")),
        _fmt_metric(test_metrics.get("auc_roc")),
        train_time_seconds,
    )
    logger.info(
        "[SUCCESS] Image model artifacts dataset=%s model=%s checkpoint=%s metrics=%s "
        "train_predictions=%s test_predictions=%s",
        dataset_name,
        model_name,
        model_path,
        metrics_path,
        train_pred_path,
        test_pred_path,
    )
    return result
