"""Unit tests for early-stopping head training (stage 7).

Tiny synthetic feature tensors only — no image decode, no checkpoints. Torch is an
optional extra, so the whole module is skipped when it is not installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from fairxai.training.vision import (  # noqa: E402
    _EarlyStopper,
    _grouped_split_indices,
    _stratified_split_indices,
    _train_head,
)


def _separable(n: int, dim: int = 4, seed: int = 0):
    """Two linearly separable classes so the head converges within a few epochs."""
    gen = torch.Generator().manual_seed(seed)
    half = n // 2
    x0 = torch.randn(half, dim, generator=gen) - 2.0
    x1 = torch.randn(n - half, dim, generator=gen) + 2.0
    feats = torch.cat([x0, x1], dim=0)
    labels = torch.tensor([0] * half + [1] * (n - half), dtype=torch.long)
    return feats, labels


def test_early_stopping_stops_before_cap_and_logs_val():
    feats, labels = _separable(200)
    head = nn.Linear(feats.size(1), 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.05)
    history, _, summary = _train_head(
        head,
        feats,
        labels,
        nn.CrossEntropyLoss(),
        optimizer,
        epochs=100,
        batch_size=32,
        device=torch.device("cpu"),
        torch=torch,
        random_state=0,
        early_stopping={"enabled": True, "patience": 3, "min_delta": 0.0, "val_fraction": 0.25},
    )
    assert summary["early_stopped"] is True
    assert summary["epochs_run"] < 100
    assert 1 <= summary["best_epoch"] <= summary["epochs_run"]
    assert all("val_auc" in h and "val_loss" in h for h in history)


def test_disabled_runs_full_epochs_without_validation():
    feats, labels = _separable(80)
    head = nn.Linear(feats.size(1), 2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.05)
    history, _, summary = _train_head(
        head,
        feats,
        labels,
        nn.CrossEntropyLoss(),
        optimizer,
        epochs=5,
        batch_size=16,
        device=torch.device("cpu"),
        torch=torch,
        random_state=0,
        early_stopping={"enabled": False},
    )
    assert summary["early_stopped"] is False
    assert summary["epochs_run"] == 5
    assert all(h["val_auc"] is None and h["val_loss"] is None for h in history)


def test_stratified_split_keeps_both_classes_in_validation():
    labels = torch.tensor([0] * 40 + [1] * 10)
    train_idx, val_idx = _stratified_split_indices(labels, 0.2, torch, 0)
    assert train_idx.numel() + val_idx.numel() == 50
    val_labels = labels[val_idx].tolist()
    assert 0 in val_labels and 1 in val_labels


def test_grouped_split_keeps_each_patient_on_one_side():
    # 20 patients, 3 rows each; a patient's rows must never straddle train/val.
    n_patients = 20
    labels = torch.tensor([i % 2 for i in range(n_patients) for _ in range(3)])
    groups = [f"p{i}" for i in range(n_patients) for _ in range(3)]
    train_idx, val_idx = _stratified_split_indices(labels, 0.25, torch, 0, groups=groups)
    assert train_idx.numel() and val_idx.numel()
    train_patients = {groups[i] for i in train_idx.tolist()}
    val_patients = {groups[i] for i in val_idx.tolist()}
    assert train_patients.isdisjoint(val_patients)  # no patient leakage
    val_labels = labels[val_idx].tolist()
    assert 0 in val_labels and 1 in val_labels


def test_grouped_split_falls_back_to_row_stratified_when_single_group():
    # One patient cannot be split by group; returns None so the caller row-splits.
    labels = torch.tensor([0, 1, 0, 1])
    assert _grouped_split_indices(labels, ["p0"] * 4, 0.25, torch, 0) is None
    train_idx, val_idx = _stratified_split_indices(labels, 0.25, torch, 0, groups=["p0"] * 4)
    assert train_idx.numel() + val_idx.numel() == 4


def test_early_stopper_tracks_best_and_restores():
    stopper = _EarlyStopper(patience=2, min_delta=0.0, enabled=True)
    states = [{"w": 0.5}, {"w": 0.9}, {"w": 0.4}, {"w": 0.3}]
    aucs = [0.5, 0.9, 0.4, 0.3]
    stopped_at = None
    for epoch, (auc, st) in enumerate(zip(aucs, states), start=1):
        if stopper.update(epoch, 0.1, auc, lambda st=st: st):
            stopped_at = epoch
            break
    assert stopper.best_epoch == 2
    assert stopper.best_state == {"w": 0.9}
    assert stopped_at == 4  # two non-improving epochs after the best
