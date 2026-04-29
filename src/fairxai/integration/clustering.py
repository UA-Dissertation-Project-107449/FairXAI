"""WebApp adapter for cluster-based subgroup discovery."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

from fairxai.clustering.engine import ClusteringEngine
from fairxai.clustering.profiles import ClusterProfiler

logger = logging.getLogger(__name__)

_DISPARITY_P0_THRESHOLD = 3.0
_DISPARITY_P1_THRESHOLD = 1.5
_TOP_N_DOMINANT_FEATURES = 3


def run_clustering(
    csv_path: str | Path,
    target_column: str,
    pca2d: list[list] | None = None,
) -> dict[str, Any]:
    """Discover subgroups via unsupervised clustering and compute per-cluster statistics.

    Tries KMeans, Hierarchical, DBSCAN, and GMM; selects the solution with the
    highest silhouette score.

    Args:
        csv_path: Absolute path to the dataset CSV file.
        target_column: Name of the binary target column.
        pca2d: Optional existing PCA 2D coords ``[[x, y, class_label], ...]``
            from a prior characterization run.  When supplied, cluster labels are
            overlaid on these coords so PCA is not recomputed.

    Returns:
        JSON-serializable dict with cluster profiles and recommendations.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset")

    df = df.dropna(subset=[target_column]).copy()
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column])

    engine = ClusteringEngine()
    result = engine.fit(df, feature_cols=None)

    df["group_cluster"] = result.group_cluster.values

    profiler = ClusterProfiler(target_col=target_column)
    report = profiler.compute(df, cluster_col="group_cluster", feature_cols=result.feature_cols)

    clusters = _build_cluster_list(result, report, df, target_column)
    pca2d_clusters = _build_pca_clusters(df, result, pca2d)
    recommendations = _generate_recommendations(clusters)

    return {
        "method": result.method,
        "n_clusters": result.n_clusters,
        "silhouette": round(result.silhouette, 4),
        "clusters": clusters,
        "pca2d_clusters": pca2d_clusters,
        "recommendations": recommendations,
    }


def _build_cluster_list(
    result: Any,
    report: Any,
    df: pd.DataFrame,
    target_column: str,
) -> list[dict[str, Any]]:
    total = len(df)
    clusters = []
    for cid in sorted(df["group_cluster"].dropna().unique()):
        cid = int(cid)
        grp = df[df["group_cluster"] == cid]
        count = len(grp)
        pct = round(count / total * 100, 1) if total > 0 else 0.0
        target_vals = pd.to_numeric(grp[target_column], errors="coerce").dropna()
        target_rate = round(float(target_vals.mean()), 4) if len(target_vals) > 0 else None
        narrative = report.narratives.get(cid, "")
        dominant = _dominant_features(report, cid)
        clusters.append(
            {
                "id": cid,
                "size": count,
                "pct": pct,
                "target_rate": target_rate,
                "narrative": narrative,
                "dominant_features": dominant,
            }
        )
    return clusters


def _dominant_features(report: Any, cid: int) -> dict[str, float]:
    if report.feature_means.empty or cid not in report.feature_means.index:
        return {}
    row = report.feature_means.loc[cid]
    top = row.nlargest(_TOP_N_DOMINANT_FEATURES)
    return {k: round(float(v), 3) for k, v in top.items()}


def _build_pca_clusters(
    df: pd.DataFrame,
    result: Any,
    pca2d: list[list] | None,
) -> list[list]:
    cluster_labels = df["group_cluster"].values

    if pca2d is not None and len(pca2d) == len(df):
        # Reuse existing PCA coords, replace class label with cluster id
        return [[float(pt[0]), float(pt[1]), int(cid)] for pt, cid in zip(pca2d, cluster_labels)]

    # Recompute PCA from numeric features
    numeric_cols = df[result.feature_cols].select_dtypes(include=[np.number])
    if numeric_cols.shape[1] < 2:
        return []
    X = StandardScaler().fit_transform(numeric_cols.fillna(0).values)
    n_components = min(2, X.shape[0], X.shape[1])
    coords = SklearnPCA(n_components=n_components, random_state=42).fit_transform(X)
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    return [[float(row[0]), float(row[1]), int(cid)] for row, cid in zip(coords, cluster_labels)]


def _generate_recommendations(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rates = [c["target_rate"] for c in clusters if c["target_rate"] is not None]
    if len(rates) < 2:
        return []

    max_rate = max(rates)
    min_rate = min(rates)
    if min_rate <= 0:
        return []

    ratio = max_rate / min_rate
    max_cluster = next(c for c in clusters if c["target_rate"] == max_rate)
    min_cluster = next(c for c in clusters if c["target_rate"] == min_rate)

    if ratio >= _DISPARITY_P0_THRESHOLD:
        priority = "P0"
        title = "Severe outcome disparity across discovered clusters"
        action = (
            f"Cluster {max_cluster['id']} ({max_cluster['pct']:.1f}% of data) has a "
            f"{max_rate:.0%} positive rate versus {min_rate:.0%} in Cluster "
            f"{min_cluster['id']} — a {ratio:.1f}× gap. "
            f"Investigate whether cluster membership correlates with sensitive attributes. "
            f"Apply stratified sampling or fairness constraints before training."
        )
        outcome = "Reduced risk of model learning spurious cluster-based patterns."
    elif ratio >= _DISPARITY_P1_THRESHOLD:
        priority = "P1"
        title = "Moderate outcome disparity across discovered clusters"
        action = (
            f"Cluster {max_cluster['id']} shows a {ratio:.1f}× higher positive rate than "
            f"Cluster {min_cluster['id']}. Monitor subgroup performance metrics after training."
        )
        outcome = "Improved subgroup fairness awareness during model evaluation."
    else:
        return []

    return [{"priority": priority, "title": title, "action": action, "expected_outcome": outcome}]
