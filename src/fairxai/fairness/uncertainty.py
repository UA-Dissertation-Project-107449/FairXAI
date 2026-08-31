"""Bootstrap confidence intervals for group fairness metrics and parity gaps.

A disparity reported as a bare number invites a comparison it cannot support.
On a 300-row cohort split into six age-by-sex cells, a 12-point TPR gap and a
2-point TPR gap can be the same finding: sampling noise. Everything downstream
of the assessment stage — model comparison, mitigation "improved it" claims,
per-cluster subgroup results — rests on gaps being distinguishable from zero,
which a point estimate cannot establish.

This module resamples the prediction frame and re-runs the *whole* metric
computation on each replicate, so every scalar the assessment already reports
gains an interval without any metric having to be reimplemented. Adding a new
fairness metric to :class:`~fairxai.fairness.metrics.FairnessMetrics` gives it a
confidence interval for free.

**Method.** Nonparametric bootstrap with percentile intervals. Resampling is
stratified by sensitive group crossed with outcome by default: group sizes and
per-group prevalence are properties of the cohort design, not quantities being
estimated, and letting them wander makes intervals reflect the resampling scheme
rather than the uncertainty in the metric. Where a stratum is too small to
resample, the scheme degrades (group×outcome → group → unstratified) and records
which scheme actually ran, so the choice is reportable rather than hidden.

**Percentile intervals** are used rather than BCa: they are the convention in
the fairness literature this work compares against, and the bias correction BCa
adds requires a jackknife pass over the full cohort per quantity, which is not
affordable at 68k rows.

**A max-gap interval is not a test.** Most parity metrics report their
disparity as ``max(rate) - min(rate)`` over groups. That statistic is bounded
below by zero and biased upward under resampling — with three groups drawn from
one identical population it still returns a visible positive gap in every
replicate — so its interval essentially never contains zero and "the interval
excludes zero" is not evidence of disparity for it. Those intervals are reported
here as *descriptive* spread only.

The inferential instrument is the ``pairwise`` table: the signed difference
between two named groups on a given quantity, which is centred at zero under the
null. Each row carries an interval, a two-sided bootstrap p-value, and a
Benjamini-Hochberg adjusted p-value taken across every comparison in the run —
an assessment reporting four parity metrics over several groups asks dozens of
questions, and some clear an unadjusted threshold by chance.

**Individual fairness is deliberately excluded.** Its k-NN consistency is
O(n^2) in cohort size, so bootstrapping it is not affordable, and a resampled
cohort has duplicate rows at distance zero from each other, which would inflate
consistency by construction rather than measure it.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import FairnessMetrics

DEFAULT_N_BOOTSTRAP = 1000
# Replicate counts used by adaptive_bootstrap_replicates(). A percentile interval
# needs enough replicates that its endpoints are resolved rather than pinned to
# an extreme order statistic; 1000 is the usual floor for a 95% interval. Large
# cohorts are thinned because each replicate re-runs every metric over every row,
# and their intervals are narrow enough that endpoint noise stops mattering.
BOOTSTRAP_BASE_REPLICATES = 1000
BOOTSTRAP_LARGE_REPLICATES = 200
BOOTSTRAP_LARGE_COHORT_ROWS = 10_000
DEFAULT_ALPHA = 0.05
DEFAULT_MIN_STRATUM = 2

STRATIFY_GROUP_OUTCOME = "group_outcome"
STRATIFY_GROUP = "group"
STRATIFY_NONE = "none"
_STRATIFY_FALLBACK = {
    STRATIFY_GROUP_OUTCOME: STRATIFY_GROUP,
    STRATIFY_GROUP: STRATIFY_NONE,
}

# Keys inside a metric block that hold per-group sub-dictionaries. Their keys are
# group labels, not quantity names, so they need one extra level of unpacking.
_GROUP_CONTAINERS = {
    "group_rates",
    "group_metrics",
    "group_tpr",
    "group_precision",
    "group_calibration",
}

# Sample counts and configuration echoes. They are either fixed by the
# stratification or not statistics at all, so an interval on them is noise in the
# output rather than information.
_NON_STATISTIC_KEYS = {
    "count",
    "n",
    "n_bins",
    "positive_count",
    "negative_count",
    "true_positive_count",
    "predicted_positive_count",
}

_TABLE_COLUMNS = [
    "scope",
    "attribute",
    "metric",
    "group",
    "quantity",
    "point",
    "ci_low",
    "ci_high",
    "se",
    "bootstrap_mean",
    "includes_zero",
    "n_valid",
    "n_boot",
    "alpha",
    "method",
    "stratify",
]

_PAIRWISE_COLUMNS = [
    "scope",
    "attribute",
    "metric",
    "quantity",
    "group_a",
    "group_b",
    "point_a",
    "point_b",
    "difference",
    "ci_low",
    "ci_high",
    "se",
    "bootstrap_mean",
    "excludes_zero",
    "p_value",
    "p_value_bh",
    "significant",
    "n_valid",
    "n_comparisons",
    "n_boot",
    "alpha",
    "method",
    "stratify",
]

MetricKey = Tuple[str, str, str, str, str]


def adaptive_bootstrap_replicates(
    n_rows: int,
    base: int = BOOTSTRAP_BASE_REPLICATES,
    large: int = BOOTSTRAP_LARGE_REPLICATES,
    threshold: int = BOOTSTRAP_LARGE_COHORT_ROWS,
) -> int:
    """Replicate count appropriate to cohort size.

    Bootstrap cost is replicates times cohort size, and the cohorts in this work
    span two orders of magnitude. Holding 1000 replicates on a 68k-row cohort
    costs minutes per model for intervals that are already tight; holding 200 on
    a 300-row cohort leaves the endpoints visibly noisy on exactly the cohorts
    where the interval carries the argument. This mirrors
    :func:`~fairxai.explainability.tabular.adaptive_shap_sample_cap`.
    """
    return large if n_rows > threshold else base


@dataclass
class BootstrapResult:
    """Confidence intervals for every scalar the fairness assessment reports.

    Attributes:
        table: One row per (scope, attribute, metric, group, quantity), carrying
            the point estimate from the full sample alongside its interval.
            ``group`` is empty for cohort-level quantities such as a parity gap.
            Descriptive: see the module docstring on why a max-gap interval is
            not a test.
        pairwise: Signed differences between named group pairs, one row per
            (quantity, group_a, group_b). This is the table that supports a
            claim that two groups are treated differently.
        metadata: What actually ran — replicate count, level, stratification
            scheme after any degradation, and the seed. Reported alongside the
            numbers so an interval can be reproduced and defended.
    """

    table: pd.DataFrame
    pairwise: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_PAIRWISE_COLUMNS))
    metadata: Dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.table.empty

    def significant_differences(self, adjusted: bool = True) -> pd.DataFrame:
        """Group differences that survive as findings.

        The shortlist an audit acts on. ``adjusted`` uses the Benjamini-Hochberg
        adjusted p-value across every comparison in the run, which is the honest
        default: one assessment asks the same question of every group pair on
        every metric, and some of those will clear an unadjusted threshold by
        chance. Pass ``False`` only when a single comparison was predeclared as
        the question being asked.
        """
        if self.pairwise.empty:
            return self.pairwise
        if adjusted:
            keep = self.pairwise["significant"]
        else:
            keep = self.pairwise["p_value"] < self.pairwise["alpha"]
        return self.pairwise[keep].reset_index(drop=True)


def _is_statistic(key: str, value: object) -> bool:
    """True when a leaf is a number worth putting an interval around.

    Booleans are excluded first: ``is_fair`` is a thresholded verdict, and Python
    treats it as an int, so an unguarded numeric check would silently produce a
    confidence interval for a yes/no answer.
    """
    if isinstance(value, bool):
        return False
    if key in _NON_STATISTIC_KEYS:
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _flatten_metric_block(
    block: Dict,
    scope: str,
    attribute: str,
    metric: str,
    out: Dict[MetricKey, float],
) -> None:
    """Pull the scalars out of one metric dict, keyed by group where applicable."""
    for key, value in block.items():
        if key in _GROUP_CONTAINERS and isinstance(value, dict):
            for group, group_block in value.items():
                if not isinstance(group_block, dict):
                    continue
                for quantity, leaf in group_block.items():
                    if _is_statistic(quantity, leaf):
                        out[(scope, attribute, metric, str(group), quantity)] = float(leaf)
        elif _is_statistic(key, value):
            out[(scope, attribute, metric, "", key)] = float(value)


def flatten_fairness_metrics(results: Dict) -> Dict[MetricKey, float]:
    """Flatten a ``calculate_all_metrics`` result to addressable scalars.

    The key is ``(scope, attribute, metric, group, quantity)``; ``group`` is the
    empty string for cohort-level quantities. Keying structurally rather than by
    a dotted string keeps the pieces available as table columns later, and makes
    a replicate that is missing a group detectable as a missing key rather than
    as a silently shifted row.

    Nested per-bin calibration detail is dropped: which bins are populated varies
    between replicates, so bin-level values are not the same quantity across the
    bootstrap and cannot be pooled into one interval.
    """
    out: Dict[MetricKey, float] = {}

    for attribute, metrics in (results.get("group_fairness") or {}).items():
        for metric, block in metrics.items():
            if isinstance(block, dict):
                _flatten_metric_block(block, "group_fairness", str(attribute), str(metric), out)

    for attribute, block in (results.get("calibration") or {}).items():
        if isinstance(block, dict):
            _flatten_metric_block(block, "calibration", str(attribute), "calibration", out)

    return out


def _strata_labels(
    df: pd.DataFrame,
    sensitive_attributes: Sequence[str],
    stratify: str,
    true_col: str,
) -> Optional[pd.Series]:
    """Build the stratum key for the requested scheme, or ``None`` for i.i.d."""
    if stratify == STRATIFY_NONE:
        return None

    cols = [c for c in sensitive_attributes if c in df.columns]
    if stratify == STRATIFY_GROUP_OUTCOME and true_col in df.columns:
        cols = cols + [true_col]
    if not cols:
        return None

    return df[cols].astype(str).agg("|".join, axis=1)


def _resolve_stratification(
    df: pd.DataFrame,
    sensitive_attributes: Sequence[str],
    stratify: str,
    true_col: str,
    min_stratum: int,
) -> Tuple[str, Optional[pd.Series]]:
    """Degrade the stratification scheme until every stratum can be resampled.

    A stratum of one row resamples to that same row in every replicate, which
    reports zero variance for a cell that in truth carries almost none of the
    information. Falling back a level is preferable to reporting a confidently
    wrong interval.
    """
    scheme = stratify
    while True:
        labels = _strata_labels(df, sensitive_attributes, scheme, true_col)
        if labels is None:
            return STRATIFY_NONE, None

        smallest = int(labels.value_counts().min())
        if smallest >= min_stratum:
            return scheme, labels

        fallback = _STRATIFY_FALLBACK.get(scheme, STRATIFY_NONE)
        logging.warning(
            "Bootstrap stratification '%s' has a stratum of %d row(s) (floor=%d); "
            "falling back to '%s'.",
            scheme,
            smallest,
            min_stratum,
            fallback,
        )
        if scheme == STRATIFY_NONE:
            return STRATIFY_NONE, None
        scheme = fallback


def _stratum_blocks(strata: Optional[pd.Series], n_rows: int) -> Optional[List[np.ndarray]]:
    """Positional row indices per stratum, computed once for all replicates."""
    if strata is None:
        return None

    values = strata.to_numpy()
    positions = np.arange(n_rows)
    return [positions[values == value] for value in pd.unique(values)]


def _replicate_indices(
    n_rows: int,
    blocks: Optional[List[np.ndarray]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one bootstrap replicate as positional indices."""
    if blocks is None:
        return rng.integers(0, n_rows, size=n_rows)
    return np.concatenate([rng.choice(b, size=b.size, replace=True) for b in blocks])


def _bootstrap_p_value(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value for the null that a difference is zero.

    The proportion of replicates falling on the far side of zero, doubled. The
    replicate count bounds the resolution: with ``B`` replicates nothing can be
    reported below ``2/(B+1)``, so the value is floored there rather than
    reported as zero, which would claim a precision the resampling never had.
    """
    if diffs.size == 0:
        return float("nan")

    below = float(np.mean(diffs <= 0.0))
    above = float(np.mean(diffs >= 0.0))
    p = 2.0 * min(below, above)
    return float(min(1.0, max(p, 2.0 / (diffs.size + 1))))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values (monotone, clipped to 1).

    Bonferroni is the wrong instrument here even though it is the more familiar
    one: fairness metrics computed on the same cohort are strongly dependent —
    ``fnr`` is ``1 - tpr``, and ``tpr`` is reported by both equalized odds and
    equal opportunity — so controlling the family-wise error rate over dozens of
    near-duplicate comparisons removes real findings along with the spurious
    ones. Controlling the false discovery rate keeps the shortlist usable while
    still bounding how much of it is noise.
    """
    n = p_values.size
    if n == 0:
        return p_values

    order = np.argsort(p_values)
    ranked = p_values[order] * n / np.arange(1, n + 1)
    # Step-up: an adjusted p-value can never exceed a larger one further up.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]

    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def _pairwise_differences(
    point: Dict[MetricKey, float],
    replicates: Dict[MetricKey, List[float]],
    alpha: float,
    n_boot: int,
    scheme: str,
) -> pd.DataFrame:
    """Bootstrap the signed difference between every pair of groups.

    Differences are taken *within* a replicate, so the correlation between two
    groups' estimates on the same resampled cohort is preserved. Differencing
    two independently summarised intervals instead would overstate the
    uncertainty and hide real disparities.

    Multiplicity is controlled across every comparison the run produces, not
    within one metric: an assessment that reports four parity metrics over three
    age bands and two sexes has asked dozens of questions, and the correction has
    to cover all of them for "this gap is a finding" to mean anything.
    """
    families: Dict[Tuple[str, str, str, str], List[str]] = {}
    for scope, attribute, metric, group, quantity in point:
        if not group:
            continue
        families.setdefault((scope, attribute, metric, quantity), []).append(group)

    bounds = [100 * (alpha / 2), 100 * (1 - alpha / 2)]
    rows = []
    for (scope, attribute, metric, quantity), groups in families.items():
        ordered = sorted(set(groups))
        for i, group_a in enumerate(ordered):
            for group_b in ordered[i + 1 :]:
                key_a = (scope, attribute, metric, group_a, quantity)
                key_b = (scope, attribute, metric, group_b, quantity)
                diffs = np.asarray(replicates[key_a], dtype=float) - np.asarray(
                    replicates[key_b], dtype=float
                )
                valid = diffs[np.isfinite(diffs)]

                if valid.size == 0:
                    ci_low = ci_high = se = boot_mean = p_value = np.nan
                    excludes = False
                else:
                    ci_low, ci_high = (float(v) for v in np.percentile(valid, bounds))
                    se = float(valid.std(ddof=1)) if valid.size > 1 else np.nan
                    boot_mean = float(valid.mean())
                    p_value = _bootstrap_p_value(valid)
                    excludes = not (ci_low <= 0.0 <= ci_high)

                rows.append(
                    {
                        "scope": scope,
                        "attribute": attribute,
                        "metric": metric,
                        "quantity": quantity,
                        "group_a": group_a,
                        "group_b": group_b,
                        "point_a": point[key_a],
                        "point_b": point[key_b],
                        "difference": point[key_a] - point[key_b],
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "se": se,
                        "bootstrap_mean": boot_mean,
                        "excludes_zero": bool(excludes),
                        "p_value": p_value,
                        "p_value_bh": np.nan,
                        "significant": False,
                        "n_valid": int(valid.size),
                        "n_comparisons": 0,
                        "n_boot": int(n_boot),
                        "alpha": float(alpha),
                        "method": "percentile",
                        "stratify": scheme,
                    }
                )

    frame = pd.DataFrame(rows, columns=_PAIRWISE_COLUMNS)
    if frame.empty:
        return frame

    testable = frame["p_value"].notna()
    frame.loc[testable, "p_value_bh"] = _benjamini_hochberg(
        frame.loc[testable, "p_value"].to_numpy()
    )
    frame["significant"] = frame["p_value_bh"] < alpha
    frame["n_comparisons"] = int(testable.sum())

    return frame.sort_values(
        ["scope", "attribute", "metric", "quantity", "group_a", "group_b"]
    ).reset_index(drop=True)


def bootstrap_fairness_metrics(
    df: pd.DataFrame,
    sensitive_attributes: Sequence[str],
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    stratify: str = STRATIFY_GROUP_OUTCOME,
    random_state: int = 42,
    true_col: str = "y_true",
    min_stratum: int = DEFAULT_MIN_STRATUM,
    metrics_calculator: Optional[FairnessMetrics] = None,
) -> BootstrapResult:
    """Attach percentile confidence intervals to every reported fairness scalar.

    Args:
        df: Prediction frame — the same one the assessment stage passes to
            ``calculate_all_metrics`` (``y_true``, ``y_pred``, optionally
            ``y_proba``, plus the sensitive columns).
        sensitive_attributes: Columns to compute fairness across. These are the
            resolved column names present in ``df``, not the configured names.
        n_boot: Replicate count. 200 is enough for a 95% percentile interval;
            pushing it higher narrows Monte Carlo noise in the endpoints, not the
            interval itself.
        alpha: Two-sided level. 0.05 gives a 95% interval.
        stratify: ``group_outcome`` (default), ``group``, or ``none``. May be
            degraded automatically; the scheme that ran is in the metadata.
        random_state: Seed, so a reported interval is reproducible.
        true_col: Outcome column, used for the outcome half of the stratum key.
        min_stratum: Smallest stratum that may be resampled before degrading.
        metrics_calculator: Optional pre-configured calculator. Built from
            ``sensitive_attributes`` when omitted.

    Returns:
        A :class:`BootstrapResult`. The table is empty when the frame yields no
        metrics at all (no usable sensitive column, or an empty frame).

    Raises:
        ValueError: If ``n_boot`` is below 2 or ``alpha`` is not in (0, 1).
    """
    if n_boot < 2:
        raise ValueError(f"n_boot must be at least 2, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")

    usable = [c for c in sensitive_attributes if c in df.columns]

    if not usable or df.empty:
        logging.warning(
            "Bootstrap skipped: no usable sensitive attribute in the frame "
            "(requested=%s, available=%d rows)",
            list(sensitive_attributes),
            len(df),
        )
        return BootstrapResult(
            table=pd.DataFrame(columns=_TABLE_COLUMNS),
            pairwise=pd.DataFrame(columns=_PAIRWISE_COLUMNS),
            metadata={},
        )

    calculator = metrics_calculator or FairnessMetrics(sensitive_attributes=list(usable))
    frame = df.reset_index(drop=True)

    # feature_cols is deliberately omitted: individual fairness is not bootstrapped.
    point = flatten_fairness_metrics(calculator.calculate_all_metrics(frame))
    if not point:
        logging.warning("Bootstrap skipped: the fairness calculator produced no scalars")
        return BootstrapResult(
            table=pd.DataFrame(columns=_TABLE_COLUMNS),
            pairwise=pd.DataFrame(columns=_PAIRWISE_COLUMNS),
            metadata={},
        )

    scheme, strata = _resolve_stratification(
        frame, usable, stratify, true_col, min_stratum=min_stratum
    )
    blocks = _stratum_blocks(strata, len(frame))
    rng = np.random.default_rng(random_state)

    replicates: Dict[MetricKey, List[float]] = {key: [] for key in point}
    for _ in range(n_boot):
        sample = frame.take(_replicate_indices(len(frame), blocks, rng))
        drawn = flatten_fairness_metrics(calculator.calculate_all_metrics(sample))
        for key in point:
            # A group absent from this replicate contributes no value rather than
            # a zero, which would drag its interval toward an outcome that never
            # happened.
            replicates[key].append(drawn.get(key, np.nan))

    lo_pct, hi_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    rows = []
    for key, values in replicates.items():
        scope, attribute, metric, group, quantity = key
        arr = np.asarray(values, dtype=float)
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            ci_low = ci_high = se = boot_mean = np.nan
        else:
            ci_low, ci_high = (float(v) for v in np.percentile(valid, [lo_pct, hi_pct]))
            se = float(valid.std(ddof=1)) if valid.size > 1 else np.nan
            boot_mean = float(valid.mean())

        rows.append(
            {
                "scope": scope,
                "attribute": attribute,
                "metric": metric,
                "group": group,
                "quantity": quantity,
                "point": point[key],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "se": se,
                "bootstrap_mean": boot_mean,
                "includes_zero": bool(ci_low <= 0.0 <= ci_high) if valid.size else False,
                "n_valid": int(valid.size),
                "n_boot": int(n_boot),
                "alpha": float(alpha),
                "method": "percentile",
                "stratify": scheme,
            }
        )

    table = pd.DataFrame(rows, columns=_TABLE_COLUMNS).sort_values(
        ["scope", "attribute", "metric", "group", "quantity"]
    )
    pairwise = _pairwise_differences(point, replicates, alpha, n_boot, scheme)

    # A bootstrap p-value cannot resolve below 2/(B+1), so with too few
    # replicates a real difference is reported at a p-value the resampling
    # invented. Flagged rather than silently trusted.
    p_floor = 2.0 / (n_boot + 1)
    p_resolved = p_floor <= alpha / max(int(len(pairwise)), 1)
    if not p_resolved:
        logging.warning(
            "Bootstrap p-values are under-resolved for multiplicity control: %d replicates "
            "cannot report below p=%.4f, while %d comparisons need roughly p=%.4f. Raise "
            "n_boot to at least %d before reading the adjusted column as a finding.",
            n_boot,
            p_floor,
            len(pairwise),
            alpha / max(int(len(pairwise)), 1),
            int(np.ceil(2 * max(int(len(pairwise)), 1) / alpha)),
        )

    metadata = {
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "method": "percentile",
        "stratify_requested": stratify,
        "stratify_used": scheme,
        "random_state": int(random_state),
        "n_rows": int(len(frame)),
        "sensitive_attributes": list(usable),
        "individual_fairness_bootstrapped": False,
        "n_pairwise_comparisons": int(len(pairwise)),
        "p_value_resolution": float(p_floor),
        "p_values_resolved_for_multiplicity": bool(p_resolved),
    }
    return BootstrapResult(
        table=table.reset_index(drop=True),
        pairwise=pairwise,
        metadata=metadata,
    )
