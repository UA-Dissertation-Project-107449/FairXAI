"""SimilarityEngine: k-NN individual fairness consistency across multiple k values."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from fairxai.fairness.metrics import FairnessMetrics

from .models import SimilarityResult, SimilarityRow

logger = logging.getLogger(__name__)


class SimilarityEngine:
    """Compute individual fairness via k-NN prediction consistency.

    Wraps :meth:`fairxai.fairness.metrics.FairnessMetrics.individual_fairness_consistency`
    and runs it for each k in *k_values*.

    Args:
        k_values: List of neighbourhood sizes to evaluate.
        pred_col: Binary prediction column name.
    """

    def __init__(
        self,
        k_values: Optional[List[int]] = None,
        pred_col: str = "y_pred",
        standardize: bool = True,
    ) -> None:
        self.k_values = k_values or [5, 10, 20]
        self.pred_col = pred_col
        self.standardize = standardize
        self._fm = FairnessMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> SimilarityResult:
        """Run k-NN consistency for all configured k values.

        Args:
            df: DataFrame with *feature_cols* and *pred_col*.
            feature_cols: Numeric columns to use for distance computation.

        Returns:
            :class:`SimilarityResult` with one :class:`SimilarityRow` per k.

        Raises:
            ValueError: If *pred_col* is missing or no valid feature columns.
        """
        if self.pred_col not in df.columns:
            raise ValueError(f"Prediction column '{self.pred_col}' not found in DataFrame.")

        valid_cols = [c for c in feature_cols if c in df.columns]
        if not valid_cols:
            raise ValueError("No valid feature columns found in DataFrame.")

        n_samples = len(df)
        rows = []

        for k in self.k_values:
            if k >= n_samples:
                logger.warning("[WARNING] k=%d >= n_samples=%d, skipping", k, n_samples)
                continue
            try:
                result = self._fm.individual_fairness_consistency(
                    df,
                    feature_cols=valid_cols,
                    pred_col=self.pred_col,
                    k=k,
                    standardize=self.standardize,
                )
                rows.append(
                    SimilarityRow(
                        k=k,
                        mean_consistency=result["mean_consistency"],
                        std_consistency=result["std_consistency"],
                        min_consistency=result["min_consistency"],
                        median_consistency=result["median_consistency"],
                        n_samples=n_samples,
                    )
                )
                logger.info(
                    "[SUCCESS] similarity k=%d: mean_consistency=%.4f",
                    k,
                    result["mean_consistency"],
                )
            except Exception as exc:
                logger.warning("[WARNING] similarity k=%d failed: %s", k, exc)

        return SimilarityResult(rows=rows)

    def save_scores(self, result: SimilarityResult, output_dir: Path) -> Path:
        """Write similarity_fairness_scores.csv to *output_dir*."""
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(result.to_df_rows())
        out = output_dir / "similarity_fairness_scores.csv"
        df.to_csv(out, index=False)
        logger.info("[SUCCESS] similarity_fairness_scores saved to %s", out)
        return out

    # ------------------------------------------------------------------
    # Per-sample scores (needed by density mapper)
    # ------------------------------------------------------------------

    def per_sample_consistency(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        k: int,
    ) -> np.ndarray:
        """Return per-sample consistency scores for a single k.

        Used by :class:`~fairxai.similarity.density.ViolationDensityMapper`.
        Delegates to :class:`FairnessMetrics` so the same (optionally scaled)
        distance is used here and in :meth:`compute`.

        Returns:
            1-D float array of length ``len(df)``, values in [0, 1].
        """
        valid_cols = [c for c in feature_cols if c in df.columns]
        return self._fm._per_sample_consistency(df, valid_cols, self.pred_col, k, self.standardize)

    def per_group_consistency(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        group_cols: List[str],
        k: int = 5,
    ) -> dict:
        """Per-sensitive-group k-NN consistency, one block per group column.

        Neighbours are global; scores are aggregated per group value. A low score
        for a group flags an individual-fairness gap for that subgroup.

        Returns:
            ``{group_col: {group_value: {"mean_consistency", "min_consistency",
            "n"}}}`` for each present, non-constant-free group column.
        """
        valid_cols = [c for c in feature_cols if c in df.columns]
        out: dict = {}
        for gcol in group_cols:
            if gcol not in df.columns:
                continue
            out[gcol] = self._fm.individual_fairness_by_group(
                df,
                feature_cols=valid_cols,
                group_col=gcol,
                pred_col=self.pred_col,
                k=k,
                standardize=self.standardize,
            )
        return out
