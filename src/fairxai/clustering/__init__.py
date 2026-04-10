"""Clustering-based subgroup discovery for fairness analysis.

Discovers latent patient subgroups via unsupervised learning and evaluates
per-cluster fairness metrics.  Outputs flow back into ``group_cluster`` column
so downstream stages (train, assess) include clusters as a sensitive attribute.
"""

from .engine import ClusteringEngine, ClusteringError
from .fairness import FairnessPerCluster
from .models import ClusterDiagnostics, ClusterReport, ClusterResult
from .profiles import ClusterProfiler

__all__ = [
    "ClusteringEngine",
    "ClusteringError",
    "FairnessPerCluster",
    "ClusterProfiler",
    "ClusterResult",
    "ClusterDiagnostics",
    "ClusterReport",
]
