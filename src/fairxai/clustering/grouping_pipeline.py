"""Leakage-safe pre-train clustering shared by the pipeline and the study.

The standalone study (``scripts/studies/run_grouping_analysis.py``) fits
clustering on the *concatenated* train+test frame — fine for post-hoc
reporting.  When clusters are meant to **affect training**, that is data
leakage: the cluster boundaries must not see test rows.

:func:`cluster_and_persist` closes that gap.  It fits the
:class:`~fairxai.clustering.engine.ClusteringEngine` on the **train split
only**, assigns test rows by nearest centroid in the train-scaled feature
space (method-agnostic — works for KMeans/Hierarchical/DBSCAN/GMM alike), and
writes ``group_cluster`` back into the canonical train/test split CSVs that the
trainer reads.  It is **idempotent**: if both splits already carry
``group_cluster`` it skips re-clustering, so a later post-assess study pass
cannot overwrite the train-only labels with leaky ones.

This module does NOT change any public ``ClusteringEngine`` / ``ClusterProfiler``
signature, so the WebApp adapters remain unaffected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fairxai.clustering.engine import ClusteringEngine, ClusteringError
from fairxai.clustering.models import ClusterResult
from fairxai.clustering.profiles import ClusterProfiler
from fairxai.experiments.data_io import resolve_dataset_dir

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_EXCLUDE = [
    "heart_disease",
    "age_group",
    "sex",
    "ethnicity",
    "group_cluster",
]

GROUP_CLUSTER_COL = "group_cluster"


def assign_clusters_nearest_centroid(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    train_labels: np.ndarray,
) -> np.ndarray:
    """Assign test rows to train-derived clusters by nearest centroid.

    Scaling is fit on the **train split only** (leakage guard), matching the
    ``StandardScaler`` the engine applies internally.  Centroids are the mean
    scaled position of each train cluster; each test row takes the label of the
    closest centroid (Euclidean).

    Returns an integer label array aligned with ``test_df`` rows.
    """
    cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
    if not cols:
        raise ClusteringError("No shared feature columns for test-cluster assignment.")

    scaler = StandardScaler().fit(train_df[cols].to_numpy(dtype=float))
    x_train = scaler.transform(train_df[cols].to_numpy(dtype=float))
    x_test = scaler.transform(test_df[cols].fillna(0.0).to_numpy(dtype=float))

    labels = np.asarray(train_labels)
    unique = np.unique(labels)
    centroids = np.vstack([x_train[labels == c].mean(axis=0) for c in unique])

    # Euclidean distance from each test row to each centroid → argmin.
    dists = np.linalg.norm(x_test[:, None, :] - centroids[None, :, :], axis=2)
    nearest = unique[dists.argmin(axis=1)]
    return nearest.astype(int)


# Split-file variants written by preprocess, in the order we try to cluster on.
# The TRAINER reads the *scaled* variant, so group_cluster MUST land there;
# the plain variant is written too for the study / inspection. All variants
# hold the same rows in the same order (same preprocess split).
_SPLIT_VARIANTS = ("_train_scaled.csv", "_train.csv")


def _variant_paths(canonical_dir: Path, dataset: str) -> list[tuple[Path, Path]]:
    """Return existing (train, test) path pairs across known split variants."""
    pairs = []
    for suffix in _SPLIT_VARIANTS:
        train_path = canonical_dir / f"{dataset}{suffix}"
        test_path = canonical_dir / f"{dataset}{suffix.replace('_train', '_test')}"
        if train_path.exists() and test_path.exists():
            pairs.append((train_path, test_path))
    return pairs


def _load_splits(
    processed_dir: Path, dataset: str, binning: str
) -> Optional[tuple[pd.DataFrame, pd.DataFrame, list[tuple[Path, Path]]]]:
    """Resolve the canonical split CSVs; cluster on the first available variant.

    Returns the train/test frames to cluster on plus *all* existing
    (train, test) path pairs so the labels can be persisted into every variant
    (notably the scaled files the trainer consumes).
    """
    canonical_dir = resolve_dataset_dir(processed_dir, dataset, binning)
    pairs = _variant_paths(canonical_dir, dataset)
    if not pairs:
        logger.warning(
            "[WARNING] cluster: no train/test split for '%s' under %s — skipping.",
            dataset,
            canonical_dir,
        )
        return None
    train_path, test_path = pairs[0]
    return pd.read_csv(train_path), pd.read_csv(test_path), pairs


def cluster_and_persist(
    dataset: str,
    processed_dir: Path,
    binning: str,
    method_cfg: Optional[dict[str, Any]],
    feature_exclude: Optional[list[str]],
    out_dir: Path,
    target_col: str = "heart_disease",
    min_clusters: int = 2,
    min_cluster_size_abs: int = 1,
    min_cluster_size_frac: float = 0.0,
    min_silhouette: Optional[float] = None,
) -> Optional[ClusterResult]:
    """Fit clustering on the train split, label test, persist ``group_cluster``.

    Args:
        dataset: Dataset name (e.g. ``"cleveland"``).
        processed_dir: ``data/processed/<pipeline>`` root.
        binning: Canonical binning subdir key (e.g. ``"fixed_10yr"``).
        method_cfg: ``clustering_methods`` config subset; ``None`` → engine
            defaults (all four methods).
        feature_exclude: Columns to strip before clustering;
            ``None`` → :data:`DEFAULT_FEATURE_EXCLUDE`.
        out_dir: Where diagnostics / assignments / profiles are written.
        target_col: Outcome column for cluster profiling.

    Returns:
        The :class:`ClusterResult`, or ``None`` if skipped (idempotent) or no
        splits were found.
    """
    loaded = _load_splits(processed_dir, dataset, binning)
    if loaded is None:
        return None
    train_df, test_df, variant_pairs = loaded

    # Idempotency guard: never re-cluster (and never overwrite train-only labels
    # with a leaky re-fit) when the column is already present in the cluster-source
    # split. The source is the first variant — the scaled file the trainer reads.
    if GROUP_CLUSTER_COL in train_df.columns and GROUP_CLUSTER_COL in test_df.columns:
        logger.info(
            "[INFO] cluster: '%s' already has %s in its splits — skipping (idempotent).",
            dataset,
            GROUP_CLUSTER_COL,
        )
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = (
        list(feature_exclude) if feature_exclude is not None else list(DEFAULT_FEATURE_EXCLUDE)
    )

    logger.info(
        "[PHASE] cluster (train-only fit) dataset=%s train=%d test=%d",
        dataset,
        len(train_df),
        len(test_df),
    )
    try:
        engine = ClusteringEngine(
            config=method_cfg,
            feature_exclude=exclude,
            min_clusters=min_clusters,
            min_cluster_size_abs=min_cluster_size_abs,
            min_cluster_size_frac=min_cluster_size_frac,
            min_silhouette=min_silhouette,
        )
        result = engine.fit(train_df)  # TRAIN ONLY — leakage guard
    except ClusteringError as exc:
        # Includes the validity gate: degenerate-only solutions → skip grouping
        # for this dataset (group_cluster simply not injected; pipeline proceeds).
        logger.warning(
            "[INFO] cluster: no usable clustering for %s (%s) — group_cluster not injected.",
            dataset,
            exc,
        )
        return None

    train_labels = result.group_cluster.to_numpy()
    test_labels = assign_clusters_nearest_centroid(
        train_df, test_df, result.feature_cols, train_labels
    )

    # Persist group_cluster into EVERY split variant (scaled + plain). The
    # trainer reads the scaled file, so the labels must land there; the plain
    # file is updated for the study / inspection. All variants share row order.
    written = _persist_to_variants(variant_pairs, train_labels, test_labels)
    logger.info(
        "[SUCCESS] cluster: method=%s n_clusters=%d silhouette=%.4f → %s",
        result.method,
        result.n_clusters,
        result.silhouette,
        ", ".join(written),
    )

    # Diagnostics + assignments + profiles (train split).
    train_df[GROUP_CLUSTER_COL] = train_labels
    engine.save_diagnostics(result, out_dir)
    result.to_assignments_df().to_csv(out_dir / "cluster_assignments.csv")
    profile_target = target_col if target_col in train_df.columns else _guess_target(train_df)
    if profile_target is not None:
        profiler = ClusterProfiler(target_col=profile_target)
        report = profiler.compute(train_df, cluster_col=GROUP_CLUSTER_COL)
        profiler.save_report(report, out_dir / "subgroup_profiles.md")

    return result


def _persist_to_variants(
    variant_pairs: list[tuple[Path, Path]],
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> list[str]:
    """Write group_cluster into each (train, test) variant by row position.

    Variants share the same rows/order, so labels are attached positionally. A
    variant whose row count does not match is skipped with a warning (defensive;
    should not happen for files from the same preprocess split).
    """
    written: list[str] = []
    for train_path, test_path in variant_pairs:
        tr = pd.read_csv(train_path)
        te = pd.read_csv(test_path)
        if len(tr) != len(train_labels) or len(te) != len(test_labels):
            logger.warning(
                "[WARNING] cluster: row-count mismatch for %s/%s — skipping this variant.",
                train_path.name,
                test_path.name,
            )
            continue
        tr[GROUP_CLUSTER_COL] = train_labels
        te[GROUP_CLUSTER_COL] = test_labels
        tr.to_csv(train_path, index=False)
        te.to_csv(test_path, index=False)
        written.extend([train_path.name, test_path.name])
    return written


def _guess_target(df: pd.DataFrame) -> Optional[str]:
    return next(
        (c for c in df.columns if "disease" in c.lower() or "target" in c.lower()),
        df.columns[-1] if len(df.columns) else None,
    )
