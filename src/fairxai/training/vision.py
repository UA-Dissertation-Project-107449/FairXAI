"""PyTorch image baseline training for dermatology datasets."""

from __future__ import annotations

import json
import logging
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
) -> dict[str, Any]:
    """Train one image baseline and persist model, predictions, metrics."""
    torch, nn, Image, DataLoader, Dataset, models, transforms = _require_torch()

    resolved = detect_accelerator(device_request)
    torch_device_name = "cuda" if resolved in {"cuda", "rocm"} else "cpu"
    device = torch.device(torch_device_name)
    torch.manual_seed(random_state)
    if torch_device_name == "cuda":
        torch.cuda.manual_seed_all(random_state)

    class CsvImageDataset(Dataset):
        def __init__(self, csv_path: Path, transform):
            self.df = pd.read_csv(csv_path)
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image = Image.open(row[image_col]).convert("RGB")
            return self.transform(image), torch.tensor(int(row[target_col]), dtype=torch.long), idx

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = CsvImageDataset(train_csv, transform)
    test_dataset = CsvImageDataset(test_csv, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch_device_name == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device_name == "cuda",
    )

    if model_name != "resnet18":
        raise ValueError("Only model_name='resnet18' is supported for dermatology v1.")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
    )

    history = []
    for epoch in range(epochs):
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
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss})
        logging.info("  epoch=%d/%d train_loss=%.4f", epoch + 1, epochs, epoch_loss)

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
        metadata_cols = [col for col in [image_col, "patient_id", "lesion_id", *sensitive_cols] if col in base.columns]
        preds = pd.concat([base[metadata_cols].reset_index(drop=True), preds], axis=1)
        return preds, _metrics(y_true, y_prob)

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
            "target_col": target_col,
            "image_size": image_size,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
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
        },
    }
    metrics_path = output_root / "results" / f"{dataset_name}_{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    result["metrics_file"] = str(metrics_path)
    return result
