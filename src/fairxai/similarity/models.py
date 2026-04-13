"""Data models for the similarity-based individual fairness module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SimilarityResult:
    """k-NN consistency scores for one or more k values."""

    rows: List["SimilarityRow"] = field(default_factory=list)

    def to_df_rows(self) -> List[dict]:
        return [r.to_dict() for r in self.rows]


@dataclass
class SimilarityRow:
    """Consistency statistics for a single k."""

    k: int
    mean_consistency: float
    std_consistency: float
    min_consistency: float
    median_consistency: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "mean_consistency": round(self.mean_consistency, 4),
            "std_consistency": round(self.std_consistency, 4),
            "min_consistency": round(self.min_consistency, 4),
            "median_consistency": round(self.median_consistency, 4),
            "n_samples": self.n_samples,
        }


@dataclass
class ViolationMapResult:
    """Result of the PCA-based violation density map."""

    output_file: Optional[Path]
    """Path to the saved PNG, or None if generation failed."""

    n_samples: int = 0
    k_used: int = 5
