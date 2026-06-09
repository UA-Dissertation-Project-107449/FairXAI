"""Similarity-based individual fairness analysis.

Evaluates whether similar patients (in feature space) receive similar
predictions, using k-nearest neighbour consistency as the fairness metric.
"""

from .density import ViolationDensityMapper
from .engine import SimilarityEngine
from .models import SimilarityResult, SimilarityRow, ViolationMapResult
from .similarity_pipeline import (
    load_all_model_predictions,
    resolve_feature_cols,
    run_similarity,
    run_similarity_for_predictions,
)

__all__ = [
    "SimilarityEngine",
    "ViolationDensityMapper",
    "SimilarityResult",
    "SimilarityRow",
    "ViolationMapResult",
    "run_similarity",
    "run_similarity_for_predictions",
    "load_all_model_predictions",
    "resolve_feature_cols",
]
