"""FairnessPerCluster: per-cluster fairness metrics and Cramér's V analysis."""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import pandas as pd
from scipy.stats import chi2_contingency

from fairxai.fairness.metrics import FairnessMetrics

logger = logging.getLogger(__name__)


class FairnessPerCluster:
    """Compute group fairness metrics stratified by cluster assignment.

    Reuses :class:`fairxai.fairness.metrics.FairnessMetrics` for each cluster
    subset, so metrics are computed identically to the main pipeline.

    Args:
        sensitive_attrs: Sensitive attribute column names.  When ``None``,
            ``["age_group", "sex"]`` is used.
        pred_col: Name of the binary prediction column.
        min_group_size: Clusters smaller than this are skipped (too few
            samples to compute reliable fairness metrics).
    """

    def __init__(
        self,
        sensitive_attrs: Optional[List[str]] = None,
        pred_col: str = "y_pred",
        min_group_size: int = 5,
    ) -> None:
        self.sensitive_attrs = sensitive_attrs or ["age_group", "sex"]
        self.pred_col = pred_col
        self.min_group_size = min_group_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
        cluster_col: str = "group_cluster",
    ) -> pd.DataFrame:
        """Compute demographic parity and equalized odds for each cluster.

        Args:
            df: DataFrame that contains *cluster_col*, prediction column,
                and all sensitive attribute columns.
            cluster_col: Name of the cluster assignment column.

        Returns:
            DataFrame with columns:
            ``cluster_id``, ``sensitive_attr``, ``dp_max_diff``,
            ``eo_tpr_diff``, ``eo_fpr_diff``, ``n_samples``, ``is_fair``
        """
        rows = []
        cluster_ids = sorted(df[cluster_col].dropna().unique())

        for cid in cluster_ids:
            subset = df[df[cluster_col] == cid].copy()
            n = len(subset)
            if n < self.min_group_size:
                logger.debug("cluster %s: only %d samples, skipping fairness metrics", cid, n)
                continue

            for attr in self.sensitive_attrs:
                if attr not in subset.columns:
                    continue
                if self.pred_col not in subset.columns:
                    continue

                fm = FairnessMetrics(sensitive_attributes=[attr])

                try:
                    dp = fm.demographic_parity(subset, attr, pred_col=self.pred_col)
                    dp_diff = dp.get("max_difference", float("nan"))
                except Exception:
                    dp_diff = float("nan")

                try:
                    eo = fm.equalized_odds(subset, attr, pred_col=self.pred_col)
                    eo_tpr_diff = eo.get("tpr_max_difference", float("nan"))
                    eo_fpr_diff = eo.get("fpr_max_difference", float("nan"))
                except Exception:
                    eo_tpr_diff = float("nan")
                    eo_fpr_diff = float("nan")

                is_fair = all(
                    v <= 0.10 for v in [dp_diff, eo_tpr_diff, eo_fpr_diff] if not math.isnan(v)
                )

                rows.append(
                    {
                        "cluster_id": int(cid),
                        "sensitive_attr": attr,
                        "dp_max_diff": round(dp_diff, 4),
                        "eo_tpr_diff": round(eo_tpr_diff, 4),
                        "eo_fpr_diff": round(eo_fpr_diff, 4),
                        "n_samples": n,
                        "is_fair": is_fair,
                    }
                )

        result = pd.DataFrame(rows)
        if result.empty:
            result = pd.DataFrame(
                columns=[
                    "cluster_id",
                    "sensitive_attr",
                    "dp_max_diff",
                    "eo_tpr_diff",
                    "eo_fpr_diff",
                    "n_samples",
                    "is_fair",
                ]
            )
        logger.info("[SUCCESS] fairness_by_cluster: %d rows", len(result))
        return result

    def cramers_v_matrix(
        self,
        df: pd.DataFrame,
        cluster_col: str = "group_cluster",
    ) -> pd.DataFrame:
        """Compute Cramér's V between cluster assignment and each sensitive attr.

        A high V (close to 1.0) indicates the cluster assignment is largely
        recovering an existing sensitive attribute split rather than discovering
        new latent structure.

        Returns:
            DataFrame with columns: ``sensitive_attr``, ``cramers_v``, ``chi2_pvalue``
        """
        rows = []
        for attr in self.sensitive_attrs:
            if attr not in df.columns:
                continue
            try:
                contingency = pd.crosstab(df[cluster_col], df[attr])
                chi2, p_value, _, _ = chi2_contingency(contingency)
                n = contingency.values.sum()
                k = min(contingency.shape)
                v = math.sqrt(chi2 / (n * max(k - 1, 1)))
                rows.append(
                    {
                        "sensitive_attr": attr,
                        "cramers_v": round(min(v, 1.0), 4),
                        "chi2_pvalue": round(p_value, 6),
                    }
                )
                if v > 0.7:
                    logger.warning(
                        "[WARNING] cluster vs %s: Cramers_V=%.3f - cluster may be "
                        "recovering existing attribute split",
                        attr,
                        v,
                    )
            except Exception as exc:
                logger.debug("cramers_v for %s failed: %s", attr, exc)

        return (
            pd.DataFrame(rows)
            if rows
            else pd.DataFrame(columns=["sensitive_attr", "cramers_v", "chi2_pvalue"])
        )
