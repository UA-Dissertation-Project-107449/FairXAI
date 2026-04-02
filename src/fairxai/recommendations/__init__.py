"""Fairness triage recommendation engine.

Provides dataset-level fairness and explainability triage based on
profiling metrics and, when available, historical experiment evidence.
"""

from .engine import RecommendationEngine
from .models import DatasetIngestion, ReadinessStatus, Recommendation, TriageReport

__all__ = [
    "RecommendationEngine",
    "TriageReport",
    "Recommendation",
    "ReadinessStatus",
    "DatasetIngestion",
]
