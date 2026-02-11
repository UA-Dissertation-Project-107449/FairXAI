"""Complexity and overlap metrics for profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - optional dependency
    LogisticRegression = None


def _select_numeric(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if target not in df.columns:
        return None, None
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target in numeric.columns:
        numeric = numeric.drop(columns=[target])
    numeric = numeric.dropna(axis=0)
    if numeric.empty:
        return None, None
    X = numeric.to_numpy(dtype=float)
    y = df.loc[numeric.index, target].to_numpy()
    return X, y


def _ensure_binary(y: np.ndarray) -> np.ndarray | None:
    values = pd.Series(y).dropna().unique()
    if len(values) != 2:
        return None
    return y


def _feature_overlap_ratio(feature: np.ndarray, y: np.ndarray) -> float | None:
    classes = np.unique(y)
    if len(classes) != 2:
        return None
    f0 = feature[y == classes[0]]
    f1 = feature[y == classes[1]]
    if f0.size == 0 or f1.size == 0:
        return None
    min0, max0 = np.nanmin(f0), np.nanmax(f0)
    min1, max1 = np.nanmin(f1), np.nanmax(f1)
    overlap = max(0.0, min(max0, max1) - max(min0, min1))
    span = max(max0, max1) - min(min0, min1)
    if span <= 0:
        return None
    return float(overlap / span)


def f2_overlap(X: np.ndarray, y: np.ndarray) -> float | None:
    ratios = []
    for col in range(X.shape[1]):
        ratio = _feature_overlap_ratio(X[:, col], y)
        if ratio is not None:
            ratios.append(ratio)
    if not ratios:
        return None
    return float(np.prod(ratios))


def f3_overlap(X: np.ndarray, y: np.ndarray) -> float | None:
    ratios = []
    for col in range(X.shape[1]):
        ratio = _feature_overlap_ratio(X[:, col], y)
        if ratio is not None:
            ratios.append((ratio, col))
    if not ratios:
        return None
    ratios.sort(key=lambda x: x[0])
    _, best_col = ratios[0]
    classes = np.unique(y)
    f0 = X[:, best_col][y == classes[0]]
    f1 = X[:, best_col][y == classes[1]]
    overlap_min = max(np.nanmin(f0), np.nanmin(f1))
    overlap_max = min(np.nanmax(f0), np.nanmax(f1))
    if overlap_max <= overlap_min:
        return 0.0
    in_overlap = (X[:, best_col] >= overlap_min) & (X[:, best_col] <= overlap_max)
    return float(np.mean(in_overlap))


def _sample_indices(n: int, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_samples:
        return np.arange(n)
    return rng.choice(n, size=max_samples, replace=False)


def n3_error(X: np.ndarray, y: np.ndarray, max_samples: int = 2000) -> float | None:
    n = X.shape[0]
    if n < 2:
        return None
    rng = np.random.default_rng(42)
    idx = _sample_indices(n, max_samples, rng)
    Xs = X[idx]
    ys = y[idx]
    diffs = Xs[:, None, :] - Xs[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    nn = np.argmin(dists, axis=1)
    opp = ys[nn] != ys
    return float(np.mean(opp))


def raug_overlap(X: np.ndarray, y: np.ndarray, k: int = 5, max_samples: int = 2000) -> float | None:
    n = X.shape[0]
    if n <= 1:
        return None
    rng = np.random.default_rng(42)
    idx = _sample_indices(n, max_samples, rng)
    Xs = X[idx]
    ys = y[idx]
    k = min(k, len(Xs) - 1)
    diffs = Xs[:, None, :] - Xs[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argpartition(dists, kth=k, axis=1)[:, :k]
    opp_counts = (ys[nn_idx] != ys[:, None]).sum(axis=1)
    in_overlap = opp_counts > 0
    return float(np.mean(in_overlap))


def l2_linear_error(X: np.ndarray, y: np.ndarray) -> float | None:
    if LogisticRegression is None:
        return None
    try:
        model = LogisticRegression(max_iter=1000, solver="liblinear")
        model.fit(X, y)
        acc = model.score(X, y)
        return float(1.0 - acc)
    except Exception:
        return None


def bayes_imbalance(y: np.ndarray) -> float | None:
    if y.size == 0:
        return None
    p_pos = float(np.mean(y))
    return float(abs(p_pos - 0.5) / 0.5)


def compute_complexity_metrics(
    df: pd.DataFrame,
    target: str = "heart_disease",
    max_samples: int = 2000,
) -> dict:
    X, y = _select_numeric(df, target)
    if X is None or y is None:
        return {}
    y = _ensure_binary(y)
    if y is None:
        return {}
    metrics = {
        "F2": f2_overlap(X, y),
        "F3": f3_overlap(X, y),
        "N3": n3_error(X, y, max_samples=max_samples),
        "Raug": raug_overlap(X, y, k=5, max_samples=max_samples),
        "L2": l2_linear_error(X, y),
        "BayesImbalance": bayes_imbalance(y),
        "max_samples": max_samples,
    }
    return metrics
