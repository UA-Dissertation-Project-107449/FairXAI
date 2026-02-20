"""Complexity and overlap metrics for profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - optional dependency
    LogisticRegression = None


SUPPORTED_COMPLEXITY_METRICS = [
    "F2",
    "F3",
    "F4",
    "N2",
    "N3",
    "N4",
    "Raug",
    "L1",
    "L2",
    "L3",
    "T1",
    "BayesImbalance",
]

METRIC_IMBALANCE_ALIASES = {
    "F2": "F2Imbalance",
    "F3": "F3Imbalance",
    "F4": "F4Imbalance",
    "N2": "N2Imbalance",
    "N3": "N3Imbalance",
    "N4": "N4Imbalance",
    "Raug": "RaugImbalance",
    "L1": "L1Imbalance",
    "L2": "L2Imbalance",
    "L3": "L3Imbalance",
    "T1": "T1Imbalance",
}


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


def get_supported_complexity_metrics(include_aliases: bool = False) -> list[str]:
    if not include_aliases:
        return list(SUPPORTED_COMPLEXITY_METRICS)
    aliases = [METRIC_IMBALANCE_ALIASES[m] for m in SUPPORTED_COMPLEXITY_METRICS if m in METRIC_IMBALANCE_ALIASES]
    return list(SUPPORTED_COMPLEXITY_METRICS) + aliases


def is_primary_complexity_metric(metric_name: str) -> bool:
    return metric_name in SUPPORTED_COMPLEXITY_METRICS


def is_complexity_metric_key(metric_name: str) -> bool:
    return metric_name in SUPPORTED_COMPLEXITY_METRICS or metric_name in METRIC_IMBALANCE_ALIASES.values()


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


def _overlap_interval(feature: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    classes = np.unique(y)
    if len(classes) != 2:
        return None
    f0 = feature[y == classes[0]]
    f1 = feature[y == classes[1]]
    if f0.size == 0 or f1.size == 0:
        return None
    overlap_min = max(np.nanmin(f0), np.nanmin(f1))
    overlap_max = min(np.nanmax(f0), np.nanmax(f1))
    if overlap_max <= overlap_min:
        return overlap_min, overlap_min
    return overlap_min, overlap_max


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances without O(n²·features) intermediate."""
    sq = np.einsum("ij,ij->i", X, X)          # (n,) — squared norms
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)      # numerical guard
    return np.sqrt(dist_sq)


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


def f4_overlap(X: np.ndarray, y: np.ndarray) -> float | None:
    if X.shape[0] == 0 or X.shape[1] == 0:
        return None
    mask = np.ones(X.shape[0], dtype=bool)
    last_ratio: float | None = None

    for _ in range(X.shape[1]):
        if mask.sum() < 2:
            break
        current_X = X[mask]
        current_y = y[mask]
        if len(np.unique(current_y)) != 2:
            break

        candidates: list[tuple[float, int, np.ndarray]] = []
        for col in range(current_X.shape[1]):
            interval = _overlap_interval(current_X[:, col], current_y)
            if interval is None:
                continue
            overlap_min, overlap_max = interval
            col_mask = (current_X[:, col] >= overlap_min) & (current_X[:, col] <= overlap_max)
            ratio = float(np.mean(col_mask))
            candidates.append((ratio, col, col_mask))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[0])
        best_ratio, _, best_mask = candidates[0]
        last_ratio = best_ratio

        selected_idx = np.where(mask)[0]
        keep_idx = selected_idx[best_mask]
        new_mask = np.zeros_like(mask)
        new_mask[keep_idx] = True
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    if last_ratio is None:
        return None
    return float(mask.sum() / X.shape[0])


def _sample_indices(n: int, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_samples:
        return np.arange(n)
    return rng.choice(n, size=max_samples, replace=False)


def n3_error(X: np.ndarray, y: np.ndarray, max_samples: int = 1000) -> float | None:
    n = X.shape[0]
    if n < 2:
        return None
    rng = np.random.default_rng(42)
    idx = _sample_indices(n, max_samples, rng)
    Xs = X[idx]
    ys = y[idx]
    dists = _pairwise_distances(Xs)            # (m, m) — no 3-D intermediate
    np.fill_diagonal(dists, np.inf)
    nn = np.argmin(dists, axis=1)
    opp = ys[nn] != ys
    return float(np.mean(opp))


def n2_ratio(X: np.ndarray, y: np.ndarray, max_samples: int = 1000) -> float | None:
    n = X.shape[0]
    if n < 3:
        return None
    rng = np.random.default_rng(42)
    idx = _sample_indices(n, max_samples, rng)
    Xs = X[idx]
    ys = y[idx]
    if len(np.unique(ys)) != 2:
        return None

    dists = _pairwise_distances(Xs)
    np.fill_diagonal(dists, np.inf)
    ratios = []
    for i in range(len(Xs)):
        same_mask = ys == ys[i]
        same_mask[i] = False
        opp_mask = ys != ys[i]
        if not np.any(same_mask) or not np.any(opp_mask):
            continue
        same_dist = np.min(dists[i, same_mask])
        opp_dist = np.min(dists[i, opp_mask])
        ratios.append(float(same_dist / (opp_dist + 1e-12)))

    if not ratios:
        return None
    return float(np.mean(ratios))


def raug_overlap(X: np.ndarray, y: np.ndarray, k: int = 5, max_samples: int = 1000) -> float | None:
    n = X.shape[0]
    if n <= 1:
        return None
    rng = np.random.default_rng(42)
    idx = _sample_indices(n, max_samples, rng)
    Xs = X[idx]
    ys = y[idx]
    k = min(k, len(Xs) - 1)
    dists = _pairwise_distances(Xs)            # (m, m) — no 3-D intermediate
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argpartition(dists, kth=k, axis=1)[:, :k]
    opp_counts = (ys[nn_idx] != ys[:, None]).sum(axis=1)
    in_overlap = opp_counts > 0
    return float(np.mean(in_overlap))


def _synthetic_interpolation(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    max_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if X.shape[0] < 3:
        return None, None

    idx = _sample_indices(X.shape[0], max_samples, rng)
    Xs = X[idx]
    ys = y[idx]
    classes = np.unique(ys)
    synth_rows: list[np.ndarray] = []
    synth_labels: list[object] = []

    for cls in classes:
        class_idx = np.where(ys == cls)[0]
        if len(class_idx) < 2:
            continue
        class_X = Xs[class_idx]
        dists = _pairwise_distances(class_X)
        np.fill_diagonal(dists, np.inf)
        nn = np.argmin(dists, axis=1)
        lam = rng.random(len(class_idx))
        base = class_X
        neigh = class_X[nn]
        synthetic = base + lam[:, None] * (neigh - base)
        synth_rows.extend(synthetic)
        synth_labels.extend([cls] * len(synthetic))

    if not synth_rows:
        return None, None
    return np.asarray(synth_rows, dtype=float), np.asarray(synth_labels)


def _nearest_label(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray) -> np.ndarray:
    """Nearest-neighbour labels using the squared-distance trick (no 3-D tensor)."""
    sq_train = np.einsum("ij,ij->i", train_X, train_X)
    sq_test  = np.einsum("ij,ij->i", test_X,  test_X)
    dist_sq  = sq_test[:, None] + sq_train[None, :] - 2.0 * (test_X @ train_X.T)
    np.maximum(dist_sq, 0.0, out=dist_sq)
    nn = np.argmin(dist_sq, axis=1)            # argmin of squared == argmin of distance
    return train_y[nn]


def n4_error(X: np.ndarray, y: np.ndarray, max_samples: int = 1000) -> float | None:
    rng = np.random.default_rng(42)
    synth_X, synth_y = _synthetic_interpolation(X, y, rng=rng, max_samples=max_samples)
    if synth_X is None or synth_y is None:
        return None

    pred = _nearest_label(X, y, synth_X)
    return float(np.mean(pred != synth_y))


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


def l1_linear_boundary_error(X: np.ndarray, y: np.ndarray) -> float | None:
    if LogisticRegression is None:
        return None
    try:
        model = LogisticRegression(max_iter=1000, solver="liblinear")
        model.fit(X, y)
        pred = model.predict(X)
        misclassified = pred != y
        if not np.any(misclassified):
            return 0.0
        if hasattr(model, "decision_function"):
            margins = np.abs(model.decision_function(X))
        else:
            proba = model.predict_proba(X)
            margins = np.abs(proba.max(axis=1) - 0.5)
        penalty = margins[misclassified]
        return float(np.mean(penalty) * np.mean(misclassified))
    except Exception:
        return None


def l3_linear_synth_error(X: np.ndarray, y: np.ndarray, max_samples: int = 1000) -> float | None:
    if LogisticRegression is None:
        return None
    try:
        model = LogisticRegression(max_iter=1000, solver="liblinear")
        model.fit(X, y)
    except Exception:
        return None

    rng = np.random.default_rng(42)
    synth_X, synth_y = _synthetic_interpolation(X, y, rng=rng, max_samples=max_samples)
    if synth_X is None or synth_y is None:
        return None

    pred = model.predict(synth_X)
    return float(np.mean(pred != synth_y))


def t1_structural_overlap(X: np.ndarray, y: np.ndarray, max_samples: int = 600) -> float | None:
    if X.shape[0] < 3:
        return None
    rng = np.random.default_rng(42)
    idx = _sample_indices(X.shape[0], max_samples, rng)
    Xs = X[idx]
    ys = y[idx]
    if len(np.unique(ys)) != 2:
        return None

    dists = _pairwise_distances(Xs)
    np.fill_diagonal(dists, np.inf)
    radii = np.zeros(len(Xs), dtype=float)

    for i in range(len(Xs)):
        opp_mask = ys != ys[i]
        if not np.any(opp_mask):
            return None
        radii[i] = np.min(dists[i, opp_mask])

    keep = np.ones(len(Xs), dtype=bool)
    for i, j in combinations(range(len(Xs)), 2):
        if ys[i] != ys[j]:
            continue
        center_dist = dists[i, j]
        if center_dist + radii[i] <= radii[j]:
            keep[i] = False
        elif center_dist + radii[j] <= radii[i]:
            keep[j] = False

    return float(np.sum(keep) / len(Xs))


def bayes_imbalance(y: np.ndarray) -> float | None:
    if y.size == 0:
        return None
    p_pos = float(np.mean(y))
    return float(abs(p_pos - 0.5) / 0.5)


def _with_aliases(metrics: dict[str, float | None]) -> dict[str, float | None]:
    enriched = dict(metrics)
    for base_name, alias_name in METRIC_IMBALANCE_ALIASES.items():
        if base_name in metrics:
            enriched[alias_name] = metrics[base_name]
    return enriched


def compute_complexity_metrics(
    df: pd.DataFrame,
    target: str = "heart_disease",
    max_samples: int = 1000,
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
        "F4": f4_overlap(X, y),
        "N2": n2_ratio(X, y, max_samples=max_samples),
        "N3": n3_error(X, y, max_samples=max_samples),
        "N4": n4_error(X, y, max_samples=max_samples),
        "Raug": raug_overlap(X, y, k=5, max_samples=max_samples),
        "L1": l1_linear_boundary_error(X, y),
        "L2": l2_linear_error(X, y),
        "L3": l3_linear_synth_error(X, y, max_samples=max_samples),
        "T1": t1_structural_overlap(X, y, max_samples=max_samples),
        "BayesImbalance": bayes_imbalance(y),
    }
    output = _with_aliases(metrics)
    return output
