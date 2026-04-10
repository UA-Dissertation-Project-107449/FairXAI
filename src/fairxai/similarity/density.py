"""ViolationDensityMapper: PCA 2D scatter coloured by k-NN consistency score."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .models import ViolationMapResult

logger = logging.getLogger(__name__)


class ViolationDensityMapper:
    """Map fairness violation hotspots on a PCA 2D projection.

    Low k-NN consistency scores (near 0) indicate regions where similar
    patients receive inconsistent predictions — potential individual fairness
    violations.  These are shown in red; high-consistency regions in blue.

    Args:
        k: Number of neighbours to use for consistency scoring.
        sample_size: Max samples to plot (random subsample if exceeded).
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        k: int = 5,
        sample_size: int = 1500,
        random_state: int = 42,
    ) -> None:
        self.k = k
        self.sample_size = sample_size
        self.random_state = random_state

    def compute(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        pred_col: str,
        output_file: Path,
        similarity_engine: Optional[object] = None,
    ) -> ViolationMapResult:
        """Generate and save the violation density map.

        Args:
            df: DataFrame with *feature_cols* and *pred_col*.
            feature_cols: Numeric columns for PCA + k-NN distance.
            pred_col: Binary prediction column.
            output_file: Where to write the PNG.
            similarity_engine: Pre-built :class:`~fairxai.similarity.engine.SimilarityEngine`
                instance.  When ``None``, a new one is created with ``k=self.k``.

        Returns:
            :class:`ViolationMapResult`.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("[WARNING] matplotlib not available; skipping violation density map")
            return ViolationMapResult(output_file=None)

        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("[WARNING] scikit-learn not available; skipping violation density map")
            return ViolationMapResult(output_file=None)

        valid_cols = [c for c in feature_cols if c in df.columns]
        if not valid_cols or pred_col not in df.columns:
            logger.warning("[WARNING] violation_density_map: missing columns, skipping")
            return ViolationMapResult(output_file=None)

        # Subsample if large
        plot_df = df.copy()
        if len(plot_df) > self.sample_size:
            plot_df = plot_df.sample(self.sample_size, random_state=self.random_state)

        n = len(plot_df)
        if n < max(self.k + 1, 10):
            logger.warning(
                "[WARNING] violation_density_map: only %d samples (need >%d), skipping",
                n,
                self.k + 1,
            )
            return ViolationMapResult(output_file=None)

        try:
            # Scale + PCA projection (reuse pattern from viz/comparisons.py:119-120)
            X_scaled = StandardScaler().fit_transform(plot_df[valid_cols].values)
            pca = PCA(n_components=2, random_state=self.random_state)
            X_2d = pca.fit_transform(X_scaled)

            # Per-sample consistency scores
            if similarity_engine is None:
                from .engine import SimilarityEngine
                similarity_engine = SimilarityEngine(k_values=[self.k], pred_col=pred_col)

            consistencies = similarity_engine.per_sample_consistency(
                plot_df, valid_cols, k=self.k
            )

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(
                X_2d[:, 0],
                X_2d[:, 1],
                c=consistencies,
                cmap="RdYlBu",
                vmin=0.0,
                vmax=1.0,
                s=20,
                alpha=0.7,
                edgecolors="none",
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("k-NN Consistency (1=fair, 0=violation)", fontsize=10)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=10)
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=10)
            ax.set_title(f"Individual Fairness Violation Map (k={self.k}, n={n})", fontsize=12)
            fig.tight_layout()

            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info("[SUCCESS] violation_density_map.png → %s", output_file)
            return ViolationMapResult(output_file=output_file, n_samples=n, k_used=self.k)

        except Exception as exc:
            logger.warning("[WARNING] violation_density_map failed: %s", exc)
            return ViolationMapResult(output_file=None, n_samples=n, k_used=self.k)
