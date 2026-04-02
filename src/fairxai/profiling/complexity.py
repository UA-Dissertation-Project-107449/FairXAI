"""Complexity and overlap metrics for profiling.

This module intentionally follows Domain_characterization metric behavior for all
implemented metrics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import ComplexityConfig

try:
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NearestNeighbors
    from sklearn.svm import LinearSVC
except Exception:  # pragma: no cover - optional dependency
    pairwise_distances = None
    NearestNeighbors = None
    LinearSVC = None

logger = logging.getLogger(__name__)


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


def _select_numeric(
    df: pd.DataFrame, target: str
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if target not in df.columns:
        return None, None

    numeric = df.select_dtypes(include=[np.number]).copy()
    if target in numeric.columns:
        numeric = numeric.drop(columns=[target])

    if numeric.empty:
        return None, None

    y = pd.Series(df[target]).loc[numeric.index]
    non_nan_target = y.notna()
    numeric = numeric.loc[non_nan_target]
    y = y.loc[non_nan_target]

    if numeric.empty:
        return None, None

    X = numeric.to_numpy(dtype=np.float32)
    return X, y.to_numpy()


def get_supported_complexity_metrics(include_aliases: bool = False) -> list[str]:
    if not include_aliases:
        return list(SUPPORTED_COMPLEXITY_METRICS)
    aliases = [
        METRIC_IMBALANCE_ALIASES[m]
        for m in SUPPORTED_COMPLEXITY_METRICS
        if m in METRIC_IMBALANCE_ALIASES
    ]
    return list(SUPPORTED_COMPLEXITY_METRICS) + aliases


def is_primary_complexity_metric(metric_name: str) -> bool:
    return metric_name in SUPPORTED_COMPLEXITY_METRICS


def is_complexity_metric_key(metric_name: str) -> bool:
    return (
        metric_name in SUPPORTED_COMPLEXITY_METRICS
        or metric_name in METRIC_IMBALANCE_ALIASES.values()
    )


def _encode_binary_labels(y: np.ndarray) -> np.ndarray | None:
    values = pd.Series(y).dropna().unique()
    if len(values) != 2:
        return None
    return (y == values[1]).astype(np.int16)


def _with_aliases(metrics: dict[str, float | None]) -> dict[str, float | None]:
    enriched = dict(metrics)
    for base_name, alias_name in METRIC_IMBALANCE_ALIASES.items():
        if base_name in metrics:
            enriched[alias_name] = metrics[base_name]
    return enriched


def _stable_subsample(
    X: np.ndarray, y: np.ndarray, max_samples: int, random_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(y) <= max_samples:
        return X, y

    rng = np.random.RandomState(random_seed)
    idx = np.sort(rng.choice(len(y), size=max_samples, replace=False))
    return X[idx], y[idx]


def f2_overlap(X: np.ndarray, y: np.ndarray) -> float | None:
    non_binary_features = np.array([len(np.unique(col)) > 3 for col in X.T])
    if not np.any(non_binary_features):
        return None

    Xf = X[:, non_binary_features]
    c0 = Xf[y == 0]
    c1 = Xf[y == 1]
    if c0.size == 0 or c1.size == 0:
        return None

    max_c0 = np.max(c0, axis=0)
    max_c1 = np.max(c1, axis=0)
    min_c0 = np.min(c0, axis=0)
    min_c1 = np.min(c1, axis=0)

    overlap = np.maximum(0.0, np.minimum(max_c0, max_c1) - np.maximum(min_c0, min_c1))
    range_c0 = np.maximum(max_c0 - min_c0, 1e-12)
    range_c1 = np.maximum(max_c1 - min_c1, 1e-12)

    f_c0 = overlap / range_c0
    f_c1 = overlap / range_c1
    return float(np.mean([np.prod(f_c0), np.prod(f_c1)]))


def f3_overlap(X: np.ndarray, y: np.ndarray) -> float | None:
    c0 = X[y == 0]
    c1 = X[y == 1]
    if c0.size == 0 or c1.size == 0:
        return None

    max_c0 = np.max(c0, axis=0)
    max_c1 = np.max(c1, axis=0)
    min_c0 = np.min(c0, axis=0)
    min_c1 = np.min(c1, axis=0)

    minmax = np.minimum(max_c0, max_c1)
    maxmin = np.maximum(min_c0, min_c1)

    overlap_count_c0 = np.sum((c0 >= maxmin) & (c0 <= minmax), axis=0)
    overlap_count_c1 = np.sum((c1 >= maxmin) & (c1 <= minmax), axis=0)
    ratio_c0 = overlap_count_c0 / max(c0.shape[0], 1)
    ratio_c1 = overlap_count_c1 / max(c1.shape[0], 1)

    non_binary_features = np.array([len(np.unique(col)) > 3 for col in X.T])
    if not np.any(non_binary_features):
        return None

    min_ratio_c0 = np.min(ratio_c0[non_binary_features])
    min_ratio_c1 = np.min(ratio_c1[non_binary_features])
    return float((min_ratio_c0 + min_ratio_c1) / 2.0)


def f4_overlap(X: np.ndarray, y: np.ndarray) -> float | None:
    non_binary_features = np.array([len(np.unique(col)) > 2 for col in X.T])
    if not np.any(non_binary_features):
        return None

    Xf = X[:, non_binary_features]
    c0 = Xf[y == 0]
    c1 = Xf[y == 1]
    if c0.size == 0 or c1.size == 0:
        return None

    max_c0 = np.max(c0, axis=0)
    max_c1 = np.max(c1, axis=0)
    min_c0 = np.min(c0, axis=0)
    min_c1 = np.min(c1, axis=0)

    minmax = np.minimum(max_c0, max_c1)
    maxmin = np.maximum(min_c0, min_c1)

    overlap_matrix = (Xf > minmax) | (Xf < maxmin)
    overlap_counts = np.sum(overlap_matrix, axis=0)
    overlap_matrix = overlap_matrix[:, np.flip(np.argsort(overlap_counts))]

    classified = np.zeros(Xf.shape[0], dtype=bool)
    for i in range(Xf.shape[1]):
        classified = np.logical_or(classified, overlap_matrix[:, i])
        if np.sum(classified) == Xf.shape[0]:
            break

    labels_c0 = y == 0
    labels_c1 = y == 1
    f4_c0 = np.sum(np.logical_and(labels_c0, ~classified)) / max(np.sum(labels_c0), 1)
    f4_c1 = np.sum(np.logical_and(labels_c1, ~classified)) / max(np.sum(labels_c1), 1)
    return float((f4_c0 + f4_c1) / 2.0)


def l1_linear_boundary_error(
    X: np.ndarray, y: np.ndarray, svc_max_iter: int = 1000
) -> float | None:
    if LinearSVC is None:
        return None
    try:
        clf = LinearSVC(max_iter=svc_max_iter)
        clf.fit(X, y)
        pred = clf.predict(X)
        incorrect = pred != y

        labels_c0 = y == 0
        labels_c1 = y == 1
        if np.sum(labels_c0) == 0 or np.sum(labels_c1) == 0:
            return None

        margins = np.abs(clf.decision_function(X))
        l1_c0 = np.sum(margins[np.logical_and(labels_c0, incorrect)]) / np.sum(labels_c0)
        l1_c1 = np.sum(margins[np.logical_and(labels_c1, incorrect)]) / np.sum(labels_c1)
        return float((l1_c0 + l1_c1) / 2.0)
    except Exception:
        return None


def l2_linear_error(X: np.ndarray, y: np.ndarray, svc_max_iter: int = 1000) -> float | None:
    if LinearSVC is None:
        return None
    try:
        clf = LinearSVC(max_iter=svc_max_iter)
        clf.fit(X, y)
        pred = clf.predict(X)

        labels_c0 = y == 0
        labels_c1 = y == 1
        if np.sum(labels_c0) == 0 or np.sum(labels_c1) == 0:
            return None

        err_c0 = np.mean(pred[labels_c0] != y[labels_c0])
        err_c1 = np.mean(pred[labels_c1] != y[labels_c1])
        return float((err_c0 + err_c1) / 2.0)
    except Exception:
        return None


def l3_linear_synth_error(
    X: np.ndarray, y: np.ndarray, random_seed: int = 42, svc_max_iter: int = 1000
) -> float | None:
    if LinearSVC is None:
        return None
    try:
        rng = np.random.RandomState(random_seed)
        clf = LinearSVC(max_iter=svc_max_iter)
        clf.fit(X, y)

        test_X = np.zeros_like(X, dtype=np.float32)
        test_y = np.zeros((len(y),), dtype=np.int16)

        for label in [0, 1]:
            class_idx = np.where(y == label)[0]
            if len(class_idx) < 2:
                continue

            n_syn = len(class_idx)
            s1 = rng.choice(class_idx, n_syn, replace=True)
            s2 = rng.choice(class_idx, n_syn, replace=True)
            alphas = rng.rand(n_syn)
            synthetic = alphas[:, None] * X[s1] + (1.0 - alphas[:, None]) * X[s2]

            start = 0 if label == 0 else -n_syn
            end = n_syn if label == 0 else None
            test_X[start:end] = synthetic
            test_y[start:end] = label

        pred = clf.predict(test_X)

        labels_c0 = test_y == 0
        labels_c1 = test_y == 1
        if np.sum(labels_c0) == 0 or np.sum(labels_c1) == 0:
            return None

        err_c0 = np.mean(pred[labels_c0] != test_y[labels_c0])
        err_c1 = np.mean(pred[labels_c1] != test_y[labels_c1])
        return float((err_c0 + err_c1) / 2.0)
    except Exception:
        return None


def n3_error(X: np.ndarray, y: np.ndarray) -> float | None:
    if NearestNeighbors is None or len(X) < 2:
        return None

    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(X)
    _, idx = knn.kneighbors(X, 2)
    nn = idx[:, -1]
    errors = y != y[nn]

    labels_c0 = y == 0
    labels_c1 = y == 1
    if np.sum(labels_c0) == 0 or np.sum(labels_c1) == 0:
        return None

    n3_c0 = np.sum(errors[labels_c0]) / np.sum(labels_c0)
    n3_c1 = np.sum(errors[labels_c1]) / np.sum(labels_c1)
    return float((n3_c0 + n3_c1) / 2.0)


def n2_ratio(X: np.ndarray, y: np.ndarray) -> float | None:
    if NearestNeighbors is None:
        return None

    X0 = X[y == 0]
    X1 = X[y == 1]
    if X0.shape[0] < 2 or X1.shape[0] < 2:
        return None

    knn0 = NearestNeighbors(n_neighbors=2)
    knn0.fit(X0)
    closest0, _ = knn0.kneighbors(X0, 2)
    furthest1, _ = knn0.kneighbors(X1, 2)

    knn1 = NearestNeighbors(n_neighbors=2)
    knn1.fit(X1)
    closest1, _ = knn1.kneighbors(X1, 2)
    furthest0, _ = knn1.kneighbors(X0, 2)

    n2_c0_raw = np.sum(closest0[:, -1]) / (np.sum(furthest0[:, 0]) + 1e-12)
    n2_c1_raw = np.sum(closest1[:, -1]) / (np.sum(furthest1[:, 0]) + 1e-12)

    n2_c0 = n2_c0_raw / (1.0 + n2_c0_raw)
    n2_c1 = n2_c1_raw / (1.0 + n2_c1_raw)
    return float((n2_c0 + n2_c1) / 2.0)


def n4_error(X: np.ndarray, y: np.ndarray, random_seed: int = 42) -> float | None:
    if NearestNeighbors is None:
        return None

    rng = np.random.RandomState(random_seed)
    test_X = np.zeros_like(X, dtype=np.float32)
    test_y = np.zeros((len(y),), dtype=np.int16)

    for label in [0, 1]:
        class_idx = np.where(y == label)[0]
        if len(class_idx) < 2:
            continue

        n_syn = len(class_idx)
        s1 = rng.choice(class_idx, n_syn, replace=False)
        s2 = rng.choice(class_idx, n_syn, replace=False)
        alphas = rng.rand(n_syn)
        synthetic = alphas[:, None] * X[s1] + (1.0 - alphas[:, None]) * X[s2]

        start = 0 if label == 0 else -n_syn
        end = n_syn if label == 0 else None
        test_X[start:end] = synthetic
        test_y[start:end] = label

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(X)
    _, idx = knn.kneighbors(test_X, 1)
    nn = idx[:, 0]

    errors = y[nn] != test_y
    labels_c0 = test_y == 0
    labels_c1 = test_y == 1
    if np.sum(labels_c0) == 0 or np.sum(labels_c1) == 0:
        return None

    n4_c0 = np.sum(errors[labels_c0]) / np.sum(labels_c0)
    n4_c1 = np.sum(errors[labels_c1]) / np.sum(labels_c1)
    return float((n4_c0 + n4_c1) / 2.0)


def _t1_class_loop(class_data: np.ndarray, sorted_indices: np.ndarray, radii: np.ndarray) -> float:
    if pairwise_distances is None:
        return float("nan")

    is_remaining = np.ones((class_data.shape[0],), dtype=bool)
    n_hyperspheres = 0
    history: list[int] = []

    for idx in sorted_indices:
        if not np.any(is_remaining):
            break
        if not is_remaining[idx]:
            continue

        n_hyperspheres += 1
        dist_row = pairwise_distances(
            class_data[idx, :].reshape(1, -1), class_data[is_remaining, :]
        )
        mask = dist_row > radii[idx]
        is_remaining[is_remaining] = np.logical_and(mask.flatten(), is_remaining[is_remaining])
        is_remaining[idx] = False

        history.append(dist_row.shape[1])
        history = history[-100:]
        if len(history) == 100 and np.mean(np.abs(np.diff(np.array(history)))) < 1.5:
            n_hyperspheres += int(np.sum(is_remaining))
            break

    return float(n_hyperspheres / class_data.shape[0])


def t1_structural_overlap(X: np.ndarray, y: np.ndarray) -> float | None:
    if NearestNeighbors is None or pairwise_distances is None or X.shape[0] < 3:
        return None

    X0 = X[y == 0]
    X1 = X[y == 1]
    if X0.shape[0] == 0 or X1.shape[0] == 0:
        return None

    knn1 = NearestNeighbors(n_neighbors=1)
    knn1.fit(X1)
    radii0, _ = knn1.kneighbors(X0, 1)
    radii0 = radii0[:, 0]

    knn0 = NearestNeighbors(n_neighbors=1)
    knn0.fit(X0)
    radii1, _ = knn0.kneighbors(X1, 1)
    radii1 = radii1[:, 0]

    idx0 = np.flip(np.argsort(radii0))
    idx1 = np.flip(np.argsort(radii1))

    t1_c0 = _t1_class_loop(X0, idx0, radii0)
    t1_c1 = _t1_class_loop(X1, idx1, radii1)

    if np.isnan(t1_c0) or np.isnan(t1_c1):
        return None
    return float((t1_c0 + t1_c1) / 2.0)


def raug_components(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    delta: int = 2,
) -> tuple[float | None, float | None, float | None]:
    if NearestNeighbors is None or len(X) < 2:
        return None, None, None

    labels_c0 = y == 0
    labels_c1 = y == 1
    if np.sum(labels_c0) == 0 or np.sum(labels_c1) == 0:
        return None, None, None

    def _raug_for_mask(mask: np.ndarray) -> float:
        knn = NearestNeighbors(n_neighbors=min(k + 1, len(X)))
        knn.fit(X)
        _, idx = knn.kneighbors(X[mask, :], min(k + 1, len(X)))
        if idx.shape[1] > 1:
            idx = idx[:, 1:]
        counts = np.sum(y[idx] != y[mask][0], axis=1)
        return float(np.sum(counts > delta))

    r_maj = _raug_for_mask(labels_c0)
    r_min = _raug_for_mask(labels_c1)

    n0 = float(np.sum(labels_c0))
    n1 = float(np.sum(labels_c1))
    ir = n0 / max(n1, 1.0)

    r_maj_norm = float(r_maj / max(n0, 1.0))
    r_min_norm = float(r_min / max(n1, 1.0))
    raug_final = float((r_maj + ir * r_min) / (ir + 1.0))

    return r_maj_norm, r_min_norm, raug_final


def raug_overlap(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    delta: int = 2,
    output_variant: str = "minority_normalized",
) -> float | None:
    r_maj_norm, r_min_norm, raug_final = raug_components(X, y, k=k, delta=delta)
    if r_maj_norm is None or r_min_norm is None or raug_final is None:
        return None

    if output_variant == "final_weighted_count":
        return raug_final
    if output_variant == "majority_normalized":
        return r_maj_norm
    # Domain_characterization JSON writer stores the minority-normalized component.
    return r_min_norm


def bayes_imbalance(
    X: np.ndarray, y: np.ndarray, k: int = 5, search_depth: int = 100
) -> float | None:
    if NearestNeighbors is None:
        return None

    if X.shape[0] < 3:
        return None

    search_depth = min(search_depth, X.shape[0] - 1)
    if search_depth <= 1:
        return None

    minority_data = X[y == 1, :]
    if minority_data.shape[0] == 0:
        return None

    knn = NearestNeighbors(n_neighbors=search_depth)
    knn.fit(X)
    _, closest = knn.kneighbors(minority_data, search_depth)
    closest = closest[:, 1:]
    if closest.shape[1] == 0:
        return None

    k_eff = min(k, closest.shape[1])
    cumulative = np.cumsum(y[closest] == 1, axis=1)
    m_values = cumulative[:, k_eff - 1].astype(float)

    expansion_needed = m_values == k_eff
    if np.any(expansion_needed) and cumulative.shape[1] > 1:
        diffs = np.diff(cumulative[expansion_needed, :], axis=1)
        m_values[expansion_needed] = np.argmin(diffs, axis=1)

    m_values[np.logical_and(expansion_needed, m_values == 0)] = float(search_depth)

    counts = np.zeros_like(m_values, dtype=float)
    counts[expansion_needed] = m_values[expansion_needed] + 1.0 - float(k_eff)

    denom = float(k_eff) + counts
    fp_values = ((float(k_eff) + counts) - m_values) / (denom + 1e-12)

    n0 = float(np.sum(y == 0))
    n1 = float(np.sum(y == 1))
    if n1 == 0:
        return None

    fp_balanced = (n0 / n1) * fp_values
    fn_values = m_values / (denom + 1e-12)

    ibi = (fp_balanced / (fp_balanced + fn_values + 1e-12)) - (
        fp_values / (fp_values + fn_values + 1e-12)
    )
    return float(np.mean(ibi))


def compute_complexity_metrics(
    df: pd.DataFrame,
    target: str | None = None,
    max_samples: int | None = None,
    config: ComplexityConfig | None = None,
) -> dict[str, float | None]:
    """Compute all supported complexity metrics for *df*."""
    if config is None:
        config = ComplexityConfig()

    target = target or config.default_target
    seed = config.random_seed
    svc_max_iter = config.linear_svc_max_iter

    X, y_raw = _select_numeric(df, target)
    if X is None or y_raw is None:
        return {}

    y = _encode_binary_labels(y_raw)
    if y is None:
        return {}

    metric_max_samples = max_samples if max_samples is not None else config.max_samples
    X_main, y_main = _stable_subsample(X, y, max_samples=metric_max_samples, random_seed=seed)
    X_t1, y_t1 = _stable_subsample(X, y, max_samples=config.t1_max_samples, random_seed=seed + 1)

    metrics = {
        "F2": f2_overlap(X_main, y_main),
        "F3": f3_overlap(X_main, y_main),
        "F4": f4_overlap(X_main, y_main),
        "N2": n2_ratio(X_main, y_main),
        "N3": n3_error(X_main, y_main),
        "N4": n4_error(X_main, y_main, random_seed=seed),
        "Raug": raug_overlap(
            X_main,
            y_main,
            k=config.raug_k,
            delta=config.raug_delta,
            output_variant=config.raug_output_variant,
        ),
        "L1": l1_linear_boundary_error(X_main, y_main, svc_max_iter=svc_max_iter),
        "L2": l2_linear_error(X_main, y_main, svc_max_iter=svc_max_iter),
        "L3": l3_linear_synth_error(X_main, y_main, random_seed=seed, svc_max_iter=svc_max_iter),
        "T1": t1_structural_overlap(X_t1, y_t1),
        "BayesImbalance": bayes_imbalance(
            X_main, y_main, k=config.bayes_k, search_depth=config.bayes_search_depth
        ),
    }

    return _with_aliases(metrics)
