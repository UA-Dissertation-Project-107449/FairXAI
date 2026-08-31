"""Subgroup-resolved SHAP attribution summaries.

Global SHAP summaries answer "which features drive this model?".  They cannot
answer "does the model explain its decisions the same way for every group?",
which is the question a fairness audit actually needs: two cohorts can receive
the same accuracy from attributions built on completely different features.

This module takes an already-computed absolute SHAP matrix (``|phi|``, one row
per explained sample, one column per feature) plus the sensitive-attribute
values of those same rows, and reports:

* ``per_group``   — the usual mean/std/percentile summary, one block per group.
* ``disparity``   — per feature, the spread of attribution across groups.
* ``agreement``   — per group, how far its feature ranking sits from the
  cohort-wide ranking (Spearman, top-k overlap, worst rank shift).

Two magnitudes are reported for every group, and they answer different
questions.  ``mean_abs_shap`` is the raw attribution mass, which moves with how
confident the model is on that group, so a uniformly less-confident group looks
"less explained" on every feature at once.  ``share`` normalises each group's
vector to sum to one, so it isolates *which* features carry the explanation
from *how much* explanation there is.  A disparity that survives in ``share`` is
a structural difference in how the model reasons about the groups; one visible
only in ``mean_abs_shap`` is a confidence difference.

No SHAP values are computed here — the caller has already paid that cost.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_MIN_GROUP_SIZE = 30
DEFAULT_TOP_K = 5

# Percentiles reported per group, mirroring the global summary written by the
# training scripts so the two files can be read side by side.
_PERCENTILES = (25, 50, 75)


@dataclass
class SubgroupShapSummary:
    """Long-format subgroup attribution tables, all carrying an ``attribute`` column.

    Attributes:
        per_group: One row per (attribute, group, feature).
        disparity: One row per (attribute, feature).
        agreement: One row per (attribute, group).
        skipped: Group labels dropped for being smaller than the size floor,
            mapped to their row count. Kept so a caller can log *why* a group is
            absent instead of silently reporting a partial cohort.
    """

    per_group: pd.DataFrame
    disparity: pd.DataFrame
    agreement: pd.DataFrame
    skipped: dict[str, dict[str, int]]

    @property
    def is_empty(self) -> bool:
        return self.per_group.empty


def _group_labels(values: pd.Series) -> pd.Series:
    """Normalise a sensitive column to clean string labels.

    Group keys end up as CSV cell values and as figure legends, so they are
    stringified once here rather than at every consumer. Missing values become
    NA and are excluded downstream — an unknown group is not a group.
    """
    labels = values.astype("object").where(values.notna(), other=pd.NA)
    return labels.map(lambda v: v if v is pd.NA else str(v))


def _rank_agreement(
    group_means: np.ndarray,
    overall_means: np.ndarray,
    top_k: int,
) -> tuple[float, float, float]:
    """Compare one group's feature ordering against the cohort-wide ordering.

    Returns ``(spearman, top_k_overlap, max_rank_shift)``. Spearman is computed
    on the attribution magnitudes directly (a rank correlation of the ranks they
    induce); top-k overlap is the Jaccard-style fraction of the cohort's top-k
    features that the group also ranks in its own top-k; max rank shift is the
    largest position change any single feature undergoes, which catches the case
    where the orderings agree broadly but one clinically meaningful feature moves
    a long way.
    """
    n_features = len(overall_means)
    if n_features < 2:
        return float("nan"), float("nan"), 0.0

    # A constant vector — most plausibly a group with no attribution at all —
    # induces no ordering, so rank correlation is undefined rather than zero.
    # Checked here so the undefined case is a deliberate NaN, not a warning from
    # inside pandas.
    if np.ptp(group_means) == 0 or np.ptp(overall_means) == 0:
        spearman = float("nan")
    else:
        spearman = pd.Series(group_means).corr(pd.Series(overall_means), method="spearman")

    k = min(top_k, n_features)
    group_top = set(np.argsort(-group_means)[:k].tolist())
    overall_top = set(np.argsort(-overall_means)[:k].tolist())
    overlap = len(group_top & overall_top) / float(k)

    # Rank 0 = strongest attribution. argsort of the negated means gives the
    # ordering; argsort of that ordering gives each feature its position.
    group_rank = np.argsort(np.argsort(-group_means))
    overall_rank = np.argsort(np.argsort(-overall_means))
    max_shift = float(np.max(np.abs(group_rank - overall_rank)))

    return float(spearman) if pd.notna(spearman) else float("nan"), float(overlap), max_shift


def summarise_subgroup_shap(
    shap_abs: np.ndarray,
    feature_names: Sequence[str],
    sensitive: pd.DataFrame,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    top_k: int = DEFAULT_TOP_K,
) -> Optional[SubgroupShapSummary]:
    """Summarise an absolute SHAP matrix by each sensitive attribute.

    Args:
        shap_abs: ``|phi|`` matrix, shape ``(n_samples, n_features)``. Already
            absolute — this function does not re-take the absolute value, so a
            signed matrix would be summarised wrongly.
        feature_names: Column names for ``shap_abs``.
        sensitive: Sensitive-attribute frame with exactly ``n_samples`` rows, in
            the same row order as ``shap_abs``. Row alignment is the caller's
            responsibility and cannot be checked here beyond the row count.
        min_group_size: Groups smaller than this are excluded. Per-feature
            percentiles over a handful of rows are noise, and a disparity driven
            by a five-person group is not a finding.
        top_k: Size of the top-k overlap window in the agreement table.

    Returns:
        A :class:`SubgroupShapSummary`, or ``None`` when no attribute retains at
        least two groups above the size floor (nothing to compare).

    Raises:
        ValueError: If the SHAP matrix, feature names, and sensitive frame do
            not agree on their shapes.
    """
    shap_abs = np.asarray(shap_abs)
    if shap_abs.ndim != 2:
        raise ValueError(f"shap_abs must be 2D [n_samples, n_features], got shape {shap_abs.shape}")
    if shap_abs.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but shap_abs has "
            f"{shap_abs.shape[1]} columns"
        )
    if len(sensitive) != shap_abs.shape[0]:
        raise ValueError(
            f"sensitive has {len(sensitive)} rows but shap_abs has {shap_abs.shape[0]}"
        )

    features = list(feature_names)
    overall_means = shap_abs.mean(axis=0)
    overall_total = float(overall_means.sum())
    overall_share = (
        overall_means / overall_total if overall_total > 0 else np.zeros_like(overall_means)
    )

    per_group_rows: list[dict] = []
    disparity_rows: list[dict] = []
    agreement_rows: list[dict] = []
    skipped: dict[str, dict[str, int]] = {}

    for attribute in sensitive.columns:
        labels = _group_labels(sensitive[attribute]).reset_index(drop=True)

        kept: dict[str, np.ndarray] = {}
        attr_skipped: dict[str, int] = {}
        for label, count in labels.value_counts(dropna=True).items():
            if int(count) < min_group_size:
                attr_skipped[str(label)] = int(count)
                continue
            kept[str(label)] = (labels == label).to_numpy()

        if attr_skipped:
            skipped[str(attribute)] = attr_skipped

        # One group cannot be disparate against itself.
        if len(kept) < 2:
            continue

        group_means: dict[str, np.ndarray] = {}
        group_shares: dict[str, np.ndarray] = {}

        for label, mask in kept.items():
            block = shap_abs[mask]
            means = block.mean(axis=0)
            total = float(means.sum())
            shares = means / total if total > 0 else np.zeros_like(means)
            group_means[label] = means
            group_shares[label] = shares

            p25, p50, p75 = (np.percentile(block, p, axis=0) for p in _PERCENTILES)
            order = np.argsort(np.argsort(-means))
            for j, feature in enumerate(features):
                per_group_rows.append(
                    {
                        "attribute": str(attribute),
                        "group": label,
                        "n": int(block.shape[0]),
                        "feature": feature,
                        "mean_abs_shap": float(means[j]),
                        "std_abs_shap": float(block[:, j].std()),
                        "p25": float(p25[j]),
                        "p50": float(p50[j]),
                        "p75": float(p75[j]),
                        "share": float(shares[j]),
                        "rank": int(order[j]) + 1,
                    }
                )

            spearman, overlap, max_shift = _rank_agreement(means, overall_means, top_k)
            agreement_rows.append(
                {
                    "attribute": str(attribute),
                    "group": label,
                    "n": int(block.shape[0]),
                    "spearman_vs_overall": spearman,
                    f"top{top_k}_overlap_vs_overall": overlap,
                    "max_rank_shift_vs_overall": max_shift,
                }
            )

        labels_kept = list(group_means.keys())
        means_matrix = np.vstack([group_means[label] for label in labels_kept])
        shares_matrix = np.vstack([group_shares[label] for label in labels_kept])

        for j, feature in enumerate(features):
            col_means = means_matrix[:, j]
            col_shares = shares_matrix[:, j]
            hi, lo = int(np.argmax(col_means)), int(np.argmin(col_means))
            disparity_rows.append(
                {
                    "attribute": str(attribute),
                    "feature": feature,
                    "n_groups": len(labels_kept),
                    "mean_abs_shap_overall": float(overall_means[j]),
                    "share_overall": float(overall_share[j]),
                    "min_group_mean_abs_shap": float(col_means[lo]),
                    "max_group_mean_abs_shap": float(col_means[hi]),
                    "mean_abs_shap_gap": float(col_means[hi] - col_means[lo]),
                    "min_group_share": float(col_shares.min()),
                    "max_group_share": float(col_shares.max()),
                    "share_gap": float(col_shares.max() - col_shares.min()),
                    "max_group": labels_kept[hi],
                    "min_group": labels_kept[lo],
                }
            )

    if not per_group_rows:
        return None

    per_group = pd.DataFrame(per_group_rows).sort_values(
        ["attribute", "group", "mean_abs_shap"], ascending=[True, True, False]
    )
    disparity = pd.DataFrame(disparity_rows).sort_values(
        ["attribute", "share_gap"], ascending=[True, False]
    )
    agreement = pd.DataFrame(agreement_rows).sort_values(["attribute", "group"])

    return SubgroupShapSummary(
        per_group=per_group.reset_index(drop=True),
        disparity=disparity.reset_index(drop=True),
        agreement=agreement.reset_index(drop=True),
        skipped=skipped,
    )
