"""Similarity-based individual fairness analysis.

Evaluates whether similar patients (in feature space) receive similar
predictions, using k-nearest neighbour consistency as the fairness metric.
"""

from .density import ViolationDensityMapper
from .engine import SimilarityEngine
from .models import SimilarityResult, SimilarityRow, ViolationMapResult

__all__ = [
    "SimilarityEngine",
    "ViolationDensityMapper",
    "SimilarityResult",
    "SimilarityRow",
    "ViolationMapResult",
]
