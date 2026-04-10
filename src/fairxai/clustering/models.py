"""Data models for the clustering-based subgroup discovery module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cluster assignment result
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """Output of ClusteringEngine.fit()."""

    group_cluster: pd.Series
    """Integer cluster labels (0..k-1), indexed like the input DataFrame."""

    method: str
    """Name of the winning method, e.g. 'kmeans'."""

    n_clusters: int
    """Number of clusters in the winning solution."""

    silhouette: float
    """Silhouette score of the winning solution."""

    feature_cols: List[str]
    """Feature columns used for clustering (after exclusion)."""

    diagnostics: List["ClusterDiagnostics"] = field(default_factory=list)
    """Per-method diagnostics from the full grid search."""

    def to_assignments_df(self) -> pd.DataFrame:
        """Return a DataFrame suitable for writing as cluster_assignments.csv."""
        return pd.DataFrame(
            {
                "group_cluster": self.group_cluster.values,
                "method_used": self.method,
                "n_clusters": self.n_clusters,
                "silhouette": round(self.silhouette, 4),
            },
            index=self.group_cluster.index,
        )


# ---------------------------------------------------------------------------
# Per-method grid diagnostics
# ---------------------------------------------------------------------------


@dataclass
class ClusterDiagnostics:
    """Diagnostics for one method + parameter combination."""

    method: str
    params: Dict[str, Any]
    n_clusters: int
    silhouette: Optional[float]
    bic: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_clusters": self.n_clusters,
            "silhouette": round(self.silhouette, 4) if self.silhouette is not None else None,
            "bic": round(self.bic, 4) if self.bic is not None else None,
            "note": self.note,
            **{f"param_{k}": v for k, v in self.params.items()},
        }


# ---------------------------------------------------------------------------
# Cluster report (output of ClusterProfiler)
# ---------------------------------------------------------------------------


@dataclass
class ClusterReport:
    """Statistical + narrative cluster characterization."""

    feature_means: pd.DataFrame
    """Rows = cluster_id, columns = feature. Cell = mean value."""

    feature_stds: pd.DataFrame
    """Rows = cluster_id, columns = feature. Cell = std dev."""

    global_means: pd.Series
    """Global mean per feature (for narrative generation)."""

    chi_square_results: List[Dict[str, Any]] = field(default_factory=list)
    """Chi-square test results: cluster vs outcome independence."""

    mann_whitney_results: List[Dict[str, Any]] = field(default_factory=list)
    """Mann-Whitney U pairwise results: outcome distributions between clusters."""

    narratives: Dict[int, str] = field(default_factory=dict)
    """Cluster-id → clinical narrative string."""

    outcome_rates: pd.Series = field(default_factory=pd.Series)
    """Positive outcome rate per cluster (for risk stratification)."""
