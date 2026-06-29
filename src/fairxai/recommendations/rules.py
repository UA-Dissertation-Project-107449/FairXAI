"""Triage rule functions: one per TRIAGE_PLAN category (A–G).

Each public function accepts a profiling dict, a ``TriageConfig``, and an
optional ``HistoricalReference``, and returns ``list[Recommendation]``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from . import evidence as ev
from .config import TriageConfig
from .history import HistoricalReference
from .models import (
    Confidence,
    DatasetIngestion,
    Priority,
    ReadinessStatus,
    Recommendation,
    TriageCategory,
)

logger = logging.getLogger(__name__)


# ===================================================================
# A — Task framing readiness (binary vs multiclass)
# ===================================================================


def check_task_framing(
    profile: Dict,
    config: TriageConfig,
    ref: Optional[HistoricalReference] = None,
) -> List[Recommendation]:
    """Category A: determine if the label setup is suitable for fair benchmarking."""
    recs: List[Recommendation] = []
    n_classes = ev.get_n_classes(profile)

    if n_classes <= 2:
        return recs  # binary — no framing concern

    # --- Multiclass: check subgroup support per class ---
    sensitive_attrs = ev.get_sensitive_attrs(profile)
    low_support_groups: List[str] = []

    for attr in sensitive_attrs:
        class_support = ev.get_group_class_support(profile, attr)
        for group, counts in class_support.items():
            for cls_label, cnt in counts.items():
                if cnt < config.multiclass_minority_support:
                    low_support_groups.append(f"{attr}={group}, class={cls_label} (n={cnt})")

    # --- Complexity warning ---
    high_complexity_metrics: Dict[str, float] = {}
    for m in config.complexity_warning_metrics:
        val = ev.get_complexity_metric(profile, m)
        if val is not None and val > config.complexity_high_threshold:
            high_complexity_metrics[m] = val

    if low_support_groups or high_complexity_metrics:
        evidence = {
            "n_classes": n_classes,
            "low_support_slices": low_support_groups[:10],  # cap for readability
            "high_complexity_metrics": high_complexity_metrics,
        }
        priority = Priority.P1 if low_support_groups else Priority.P2

        recs.append(
            Recommendation(
                category=TriageCategory.A_TASK_FRAMING,
                priority=priority,
                title="Consider binary framing for fairness benchmark",
                evidence=evidence,
                fairness_relevance=(
                    "Low-support multiclass slices make per-class fairness metrics unreliable."
                ),
                explainability_relevance=(
                    "Class overlap in multiclass settings destabilises explanations."
                ),
                action=(
                    "Collapse the target to binary for the initial benchmark unless "
                    "per-slice support is ≥{}.".format(config.multiclass_minority_support)
                ),
                expected_outcome="More reliable early fairness diagnostics.",
                confidence=Confidence.HIGH if low_support_groups else Confidence.MEDIUM,
            )
        )

    return recs


# ===================================================================
# B — Sensitive-attribute adequacy
# ===================================================================


def check_sensitive_adequacy(
    profile: Dict,
    ingestion: DatasetIngestion,
    config: TriageConfig,
) -> List[Recommendation]:
    """Category B: verify fairness can be evaluated at all."""
    recs: List[Recommendation] = []

    # --- No sensitive columns declared / detected ---
    if not ingestion.sensitive_columns:
        recs.append(
            Recommendation(
                category=TriageCategory.B_SENSITIVE_ADEQUACY,
                priority=Priority.P0,
                title="No sensitive attributes identified",
                evidence={"declared_sensitive_columns": []},
                fairness_relevance=(
                    "Fairness needs at least one sensitive attribute (e.g. sex, age, ethnicity)."
                ),
                explainability_relevance=(
                    "Explanations can't be audited for differential treatment without one."
                ),
                action="Declare at least one sensitive attribute before running fairness analysis.",
                expected_outcome="Fairness metrics become computable and interpretable.",
                confidence=Confidence.HIGH,
            )
        )
        return recs  # everything else depends on having sensitive attrs

    # --- Check each declared sensitive column ---
    for attr in ingestion.sensitive_columns:
        # Missing / null fraction
        missing_frac = ev.get_missing_fraction(profile, attr)
        if missing_frac > config.max_null_fraction:
            recs.append(
                Recommendation(
                    category=TriageCategory.B_SENSITIVE_ADEQUACY,
                    priority=Priority.P0,
                    title=f"High null rate in sensitive attr '{attr}'",
                    evidence={
                        "attribute": attr,
                        "missing_fraction": round(missing_frac, 4),
                        "threshold": config.max_null_fraction,
                    },
                    fairness_relevance=(
                        f"'{attr}' is {missing_frac:.1%} missing; fairness metrics on "
                        "incomplete group labels are unreliable."
                    ),
                    explainability_relevance=(
                        "Group-conditional explanations would rest on a biased subset."
                    ),
                    action=f"Investigate and impute '{attr}', or flag limited fairness validity.",
                    expected_outcome="Fairness metrics become interpretable for this attribute.",
                    confidence=Confidence.HIGH,
                )
            )

        # Too few unique groups
        counts = ev.get_group_counts(profile, attr)
        n_groups = len(counts)
        if n_groups < config.min_unique_groups:
            recs.append(
                Recommendation(
                    category=TriageCategory.B_SENSITIVE_ADEQUACY,
                    priority=Priority.P0,
                    title=f"Insufficient groups in '{attr}' ({n_groups})",
                    evidence={
                        "attribute": attr,
                        "n_groups": n_groups,
                        "threshold": config.min_unique_groups,
                    },
                    fairness_relevance=(
                        f"'{attr}' has only {n_groups} group(s); "
                        f"{config.min_unique_groups} are needed for comparison."
                    ),
                    explainability_relevance=(
                        "Group-contrastive explanations need at least two groups."
                    ),
                    action=f"Verify the encoding of '{attr}'; re-process if wrongly collapsed.",
                    expected_outcome="Group fairness metrics become well-defined.",
                    confidence=Confidence.HIGH,
                )
            )

    return recs


# ===================================================================
# C — Representation and subgroup support risk
# ===================================================================


def check_representation_risk(
    profile: Dict,
    config: TriageConfig,
    ref: Optional[HistoricalReference] = None,
) -> List[Recommendation]:
    """Category C: detect under-represented groups that may bias fairness conclusions."""
    recs: List[Recommendation] = []

    for attr in ev.get_sensitive_attrs(profile):
        # --- Size ratio imbalance ---
        ratio = ev.get_size_ratio(profile, attr)
        if ratio is not None and ratio > config.size_ratio_warning:
            recs.append(
                Recommendation(
                    category=TriageCategory.C_REPRESENTATION,
                    priority=Priority.P1,
                    title=f"High representation imbalance in '{attr}'",
                    evidence={
                        "attribute": attr,
                        "size_ratio": round(ratio, 2),
                        "threshold": config.size_ratio_warning,
                        "group_counts": ev.get_group_counts(profile, attr),
                    },
                    fairness_relevance=(
                        f"Group sizes in '{attr}' differ by {ratio:.1f}x, widening the "
                        "minority group's confidence intervals."
                    ),
                    explainability_relevance=(
                        "Explanations may be dominated by the majority group."
                    ),
                    action="Mark low-support groups low-confidence; consider resampling or more data.",
                    expected_outcome="Lower risk of misleading subgroup fairness claims.",
                    confidence=Confidence.HIGH,
                )
            )

        # --- Statistical parity violation (pre-model) ---
        spd = ev.get_statistical_parity_diff(profile, attr)
        if spd is not None and spd > config.statistical_parity_warning:
            recs.append(
                Recommendation(
                    category=TriageCategory.C_REPRESENTATION,
                    priority=Priority.P1,
                    title=f"Label imbalance across '{attr}' groups",
                    evidence={
                        "attribute": attr,
                        "statistical_parity_difference": round(spd, 4),
                        "threshold": config.statistical_parity_warning,
                        "positive_rates": ev.get_positive_rates(profile, attr),
                    },
                    fairness_relevance=(
                        f"Statistical parity difference of {spd:.3f} in '{attr}' shows the "
                        "positive-class base rate varies across groups."
                    ),
                    explainability_relevance=(
                        "Feature importances may reflect base-rate gaps, not real signal."
                    ),
                    action=(
                        "Check whether the imbalance is real prevalence or collection bias, "
                        "and document it."
                    ),
                    expected_outcome="Better-calibrated interpretation of detected disparities.",
                    confidence=Confidence.HIGH if spd > 0.25 else Confidence.MEDIUM,
                )
            )

        # --- Small absolute group size ---
        min_size = ev.get_min_group_size(profile, attr)
        if min_size is not None and min_size < config.min_group_samples:
            counts = ev.get_group_counts(profile, attr)
            recs.append(
                Recommendation(
                    category=TriageCategory.C_REPRESENTATION,
                    priority=Priority.P1,
                    title=f"Very small group(s) in '{attr}'",
                    evidence={
                        "attribute": attr,
                        "min_group_size": min_size,
                        "threshold": config.min_group_samples,
                        "group_counts": counts,
                    },
                    fairness_relevance=(
                        f"A group in '{attr}' has only {min_size} samples (min "
                        f"{config.min_group_samples}), making per-group metrics unreliable."
                    ),
                    explainability_relevance=(
                        "Explanations from very few samples may not generalise."
                    ),
                    action=(
                        "Collect more data for under-represented groups, or mark them "
                        "low-confidence."
                    ),
                    expected_outcome="More robust per-group estimates.",
                    confidence=Confidence.HIGH,
                )
            )

    # --- Binning sensitivity (multi-group attributes with extreme imbalance) ---
    for attr in ev.get_sensitive_attrs(profile):
        counts = ev.get_group_counts(profile, attr)
        ratio = ev.get_size_ratio(profile, attr)
        if counts is not None and len(counts) > 2 and ratio is not None:
            if ratio > config.binning_size_ratio_warning:
                recs.append(
                    Recommendation(
                        category=TriageCategory.C_REPRESENTATION,
                        priority=Priority.P2,
                        title=f"Binning imbalance in '{attr}' — consider rebinning",
                        evidence={
                            "attribute": attr,
                            "n_groups": len(counts),
                            "size_ratio": round(ratio, 2),
                            "threshold": config.binning_size_ratio_warning,
                            "group_counts": counts,
                        },
                        fairness_relevance=(
                            f"'{attr}' has {len(counts)} groups at a {ratio:.1f}x size ratio "
                            "(threshold "
                            f"{config.binning_size_ratio_warning:.1f}x), starving the smallest bins."
                        ),
                        explainability_relevance=(
                            "Tiny bins yield unreliable, majority-dominated explanations."
                        ),
                        action=(
                            "Try alternative binning (quantile, equal-width, fewer bins) to "
                            "balance group sizes."
                        ),
                        expected_outcome="More balanced groups and comparable per-group estimates.",
                        confidence=Confidence.MEDIUM,
                    )
                )

    # --- Intersectional low-support slices ---
    low_intersections = ev.get_low_support_intersections(
        profile, min_samples=config.intersectional_min_samples
    )
    if low_intersections:
        slice_details = [
            {"pair": pair, "slice": slc, "n_samples": n} for pair, slc, n in low_intersections[:15]
        ]
        recs.append(
            Recommendation(
                category=TriageCategory.C_REPRESENTATION,
                priority=Priority.P2,
                title="Low-support intersectional slices detected",
                evidence={
                    "n_low_support_slices": len(low_intersections),
                    "threshold": config.intersectional_min_samples,
                    "examples": slice_details,
                },
                fairness_relevance=(
                    f"{len(low_intersections)} intersectional slice(s) have <"
                    f"{config.intersectional_min_samples} samples, making their analysis unreliable."
                ),
                explainability_relevance=(
                    "Low-support intersectional explanations may be driven by noise."
                ),
                action="Flag these low-support intersections; avoid strong fairness claims for them.",
                expected_outcome="Transparent reporting with explicit confidence qualifications.",
                confidence=Confidence.MEDIUM,
            )
        )

    return recs


# ===================================================================
# D — Overlap and local ambiguity risk
# ===================================================================


def check_overlap_ambiguity(
    profile: Dict,
    config: TriageConfig,
    ref: Optional[HistoricalReference] = None,
) -> List[Recommendation]:
    """Category D: identify data regions where fairness/explainability may degrade."""
    recs: List[Recommendation] = []

    # --- Global elevated overlap metrics ---
    elevated: Dict[str, Dict] = {}
    for metric_name in config.elevated_metrics:
        val = ev.get_complexity_metric(profile, metric_name)
        if val is None:
            continue

        ref_stats = ref.get_complexity_reference(metric_name) if ref else None
        if ref_stats:
            comparison = ev.compare_to_reference(val, ref_stats.to_dict())
            if comparison and comparison.get("above_p75"):
                elevated[metric_name] = {
                    "value": round(val, 4),
                    "reference_median": round(ref_stats.median, 4),
                    "reference_p75": round(ref_stats.p75, 4),
                    "percentile_approx": comparison["percentile_approx"],
                }
        else:
            # No reference; use a simple absolute threshold
            if val > config.complexity_high_threshold:
                elevated[metric_name] = {"value": round(val, 4)}

    if elevated:
        recs.append(
            Recommendation(
                category=TriageCategory.D_OVERLAP_AMBIGUITY,
                priority=Priority.P1,
                title="Elevated class-overlap / ambiguity metrics",
                evidence={"elevated_metrics": elevated},
                fairness_relevance=(
                    "Intrinsically hard-to-classify samples amplify fairness gaps when "
                    "concentrated in subgroups."
                ),
                explainability_relevance=("High overlap makes SHAP/LIME values unstable."),
                action="Flag the dataset as high-ambiguity; review subgroup overlap before comparing.",
                expected_outcome="Clearer attribution of fairness gaps to overlap vs. representation.",
                confidence=Confidence.HIGH if len(elevated) >= 3 else Confidence.MEDIUM,
            )
        )

    # --- Subgroup complexity divergence ---
    divergent_groups: List[Dict] = []
    for attr in ev.get_sensitive_attrs(profile):
        for metric_name in config.elevated_metrics:
            global_val = ev.get_complexity_metric(profile, metric_name)
            if global_val is None or global_val == 0:
                continue

            counts = ev.get_group_counts(profile, attr)
            for group in counts:
                grp_val = ev.get_group_complexity(profile, attr, group, metric_name)
                if grp_val is None:
                    continue
                rel_diff = abs(grp_val - global_val) / max(abs(global_val), 1e-9)
                if rel_diff > config.group_divergence_threshold:
                    divergent_groups.append(
                        {
                            "attribute": attr,
                            "group": group,
                            "metric": metric_name,
                            "group_value": round(grp_val, 4),
                            "global_value": round(global_val, 4),
                            "relative_difference": round(rel_diff, 4),
                        }
                    )

    if divergent_groups:
        recs.append(
            Recommendation(
                category=TriageCategory.D_OVERLAP_AMBIGUITY,
                priority=Priority.P2,
                title="Subgroup complexity diverges from global",
                evidence={
                    "n_divergent_pairs": len(divergent_groups),
                    "divergence_threshold": config.group_divergence_threshold,
                    "examples": divergent_groups[:10],
                },
                fairness_relevance=(
                    "Uneven subgroup difficulty produces fairness gaps even with an unbiased model."
                ),
                explainability_relevance=(
                    "Explanation reliability varies across groups by complexity."
                ),
                action="Document elevated-complexity subgroups as a confounder for fairness gaps.",
                expected_outcome="More accurate attribution of gaps to difficulty vs. bias.",
                confidence=Confidence.MEDIUM,
            )
        )

    return recs


# ===================================================================
# E — Explainability suitability (pre-model proxy)
# ===================================================================


def check_explainability_suitability(
    profile: Dict,
    config: TriageConfig,
    ref: Optional[HistoricalReference] = None,
) -> List[Recommendation]:
    """Category E: set realistic expectations for explanation quality."""
    recs: List[Recommendation] = []

    # --- Linear complexity ---
    high_linear: Dict[str, float] = {}
    for m in config.linear_complexity_metrics:
        val = ev.get_complexity_metric(profile, m)
        if val is not None and val > config.explainability_high_threshold:
            high_linear[m] = round(val, 4)

    t1_val = ev.get_complexity_metric(profile, config.structural_overlap_metric)
    structural_high = t1_val is not None and t1_val > config.explainability_high_threshold

    if high_linear or structural_high:
        evidence: Dict = {"high_linear_metrics": high_linear}
        if t1_val is not None:
            evidence["T1_structural_overlap"] = round(t1_val, 4)

        recs.append(
            Recommendation(
                category=TriageCategory.E_EXPLAINABILITY,
                priority=Priority.P2,
                title="Linear explanations may be unreliable",
                evidence=evidence,
                fairness_relevance=(
                    "Poor linear separability makes performance-tied fairness metrics "
                    "fluctuate with the threshold."
                ),
                explainability_relevance=(
                    "Simple linear attributions capture only part of the decision logic."
                ),
                action=(
                    "Prefer robust, uncertainty-aware explainers (e.g. SHAP TreeExplainer) "
                    "and report explanation stability."
                ),
                expected_outcome="Realistic explanation-quality expectations.",
                confidence=Confidence.MEDIUM,
            )
        )

    return recs


# ===================================================================
# G — Data quality (missing values, duplicate rows)
# ===================================================================


def _format_columns(names: List[str], limit: int = 5) -> str:
    """Render a capped, human-readable column list ('a, b, c and 4 more')."""
    shown = [f"'{n}'" for n in names[:limit]]
    extra = len(names) - limit
    text = ", ".join(shown)
    if extra > 0:
        text += f" and {extra} more"
    return text


def check_data_quality(
    profile: Dict,
    ingestion: DatasetIngestion,
    config: TriageConfig,
) -> List[Recommendation]:
    """Category G: missing values and duplicate rows.

    Severity depends on the column role. Missing values in the target or an
    index column, or duplicate index values, are hard failures (P0 → Not ready).
    Missing feature values or whole-row duplicates are caveats (P1).
    """
    recs: List[Recommendation] = []

    columns_with_missing: Dict[str, int] = profile.get("missing_value_analysis", {}).get(
        "columns_with_missing", {}
    )
    dup = profile.get("duplicate_analysis", {})
    row_duplicates = int(dup.get("duplicate_count", 0) or 0)
    index_duplicates = int(dup.get("index_duplicate_count", 0) or 0)

    label = ingestion.label_column
    index_cols = set(ingestion.identifier_columns or [])

    # --- P0: target missing ---
    if config.flag_any_missing and label and columns_with_missing.get(label, 0) > 0:
        recs.append(
            Recommendation(
                category=TriageCategory.G_DATA_QUALITY,
                priority=Priority.P0,
                title=f"Missing values in target column '{label}'",
                evidence={"column": label, "missing_count": int(columns_with_missing[label])},
                fairness_relevance="Rows with a missing label cannot be evaluated for fairness.",
                explainability_relevance="Explanations require a defined target for every row.",
                action=f"Remove or correct rows where '{label}' is missing before benchmarking.",
                expected_outcome="Every row has a usable label.",
                confidence=Confidence.HIGH,
            )
        )

    # --- P0: index missing or duplicate ---
    if config.flag_any_missing:
        missing_index = [c for c in index_cols if columns_with_missing.get(c, 0) > 0]
        if missing_index:
            recs.append(
                Recommendation(
                    category=TriageCategory.G_DATA_QUALITY,
                    priority=Priority.P0,
                    title=f"Missing values in index column {_format_columns(missing_index)}",
                    evidence={"columns": missing_index},
                    fairness_relevance="A missing identifier breaks row traceability.",
                    explainability_relevance="Per-row explanations need a stable identifier.",
                    action=f"Populate the index column {_format_columns(missing_index)}.",
                    expected_outcome="Every row is uniquely identifiable.",
                    confidence=Confidence.HIGH,
                )
            )

    if config.flag_any_duplicates and index_duplicates > 0:
        recs.append(
            Recommendation(
                category=TriageCategory.G_DATA_QUALITY,
                priority=Priority.P0,
                title="Duplicate values in the index column",
                evidence={"index_duplicate_count": index_duplicates},
                fairness_relevance="A non-unique identifier indicates corrupted or merged records.",
                explainability_relevance="Duplicate IDs make per-row explanations ambiguous.",
                action="Ensure the index column holds unique values.",
                expected_outcome="The index uniquely identifies each row.",
                confidence=Confidence.HIGH,
            )
        )

    # --- P1: feature missing ---
    if config.flag_any_missing:
        feature_missing = [
            c
            for c, n in columns_with_missing.items()
            if n > 0 and c != label and c not in index_cols
        ]
        if feature_missing:
            recs.append(
                Recommendation(
                    category=TriageCategory.G_DATA_QUALITY,
                    priority=Priority.P1,
                    title=f"Missing values in {_format_columns(feature_missing)}",
                    evidence={"columns": feature_missing},
                    fairness_relevance="Missing features bias complexity and subgroup estimates.",
                    explainability_relevance="Imputed or dropped values shift feature attributions.",
                    action=f"Impute or document the gaps in {_format_columns(feature_missing)}.",
                    expected_outcome="Profiling metrics reflect complete feature data.",
                    confidence=Confidence.HIGH,
                )
            )

    # --- P1: whole-row duplicates ---
    if config.flag_any_duplicates and row_duplicates > 0:
        recs.append(
            Recommendation(
                category=TriageCategory.G_DATA_QUALITY,
                priority=Priority.P1,
                title=f"{row_duplicates} duplicate row(s)",
                evidence={"duplicate_count": row_duplicates},
                fairness_relevance="Duplicate rows over-weight some records and skew group balance.",
                explainability_relevance="Repeated rows inflate apparent support for their patterns.",
                action="Deduplicate the dataset, or confirm the repeats are intentional.",
                expected_outcome="Each record is counted once.",
                confidence=Confidence.HIGH,
            )
        )

    return recs


# ===================================================================
# F — Fairness benchmark readiness status
# ===================================================================


def check_readiness(
    recommendations: List[Recommendation],
    config: TriageConfig,
    profile: Optional[Dict] = None,
) -> Recommendation:
    """Category F: summarise readiness as a single recommendation.

    Returns exactly one ``Recommendation`` with the readiness verdict.
    """
    p0_count = sum(1 for r in recommendations if r.priority == Priority.P0)
    p1_count = sum(1 for r in recommendations if r.priority == Priority.P1)

    if config.p0_makes_not_ready and p0_count > 0:
        status = ReadinessStatus.NOT_READY
        priority = Priority.P0
    elif p1_count >= config.p1_caveat_threshold:
        status = ReadinessStatus.READY_WITH_CAVEATS
        priority = Priority.P1
    else:
        status = ReadinessStatus.READY
        priority = Priority.P3

    # Defensive cap: any missing data or duplicate rows must prevent a green
    # "Ready" verdict, regardless of how the priority thresholds are configured.
    cap_reason: Optional[str] = None
    if status == ReadinessStatus.READY and profile is not None:
        total_missing = profile.get("missing_value_analysis", {}).get("total_missing", 0) or 0
        dup = profile.get("duplicate_analysis", {})
        total_dupes = (dup.get("duplicate_count", 0) or 0) + (
            dup.get("index_duplicate_count", 0) or 0
        )
        if total_missing > 0 or total_dupes > 0:
            status = ReadinessStatus.READY_WITH_CAVEATS
            priority = Priority.P1
            cap_reason = "Missing values or duplicate rows present; capped below 'Ready'."

    top_actions = []
    for r in sorted(recommendations, key=lambda r: r.priority.value):
        if r.priority in (Priority.P0, Priority.P1):
            top_actions.append(f"[{r.priority.value}] {r.title}")
        if len(top_actions) >= 5:
            break

    return Recommendation(
        category=TriageCategory.F_READINESS,
        priority=priority,
        title=f"Readiness: {status.value}",
        evidence={
            "readiness_status": status.value,
            "p0_count": p0_count,
            "p1_count": p1_count,
            "total_recommendations": len(recommendations),
            **({"cap_reason": cap_reason} if cap_reason else {}),
        },
        fairness_relevance=(f"Overall readiness for fairness benchmarking: **{status.value}**."),
        explainability_relevance=(
            "Explanation audit reliability correlates with benchmark readiness."
        ),
        action=(
            "Address top-priority issues before relying on fairness results:\n"
            + "\n".join(f"  - {a}" for a in top_actions)
            if top_actions
            else "No blocking issues found. Proceed with fairness benchmarking."
        ),
        expected_outcome=("A well-scoped, trustworthy fairness and explainability assessment."),
        confidence=Confidence.HIGH,
    )


# ===================================================================
# Public convenience: run all rule checks
# ===================================================================


def run_all_checks(
    profile: Dict,
    ingestion: DatasetIngestion,
    config: TriageConfig,
    ref: Optional[HistoricalReference] = None,
) -> List[Recommendation]:
    """Execute categories A–E plus G (data quality), then derive F (readiness).

    Returns the full sorted list including the readiness recommendation.
    """
    recs: List[Recommendation] = []
    recs.extend(check_task_framing(profile, config, ref))
    recs.extend(check_sensitive_adequacy(profile, ingestion, config))
    recs.extend(check_representation_risk(profile, config, ref))
    recs.extend(check_overlap_ambiguity(profile, config, ref))
    recs.extend(check_explainability_suitability(profile, config, ref))
    recs.extend(check_data_quality(profile, ingestion, config))

    readiness = check_readiness(recs, config, profile=profile)
    recs.append(readiness)

    # Sort: P0 first, then P1, P2, P3
    priority_order = {Priority.P0: 0, Priority.P1: 1, Priority.P2: 2, Priority.P3: 3}
    recs.sort(key=lambda r: priority_order.get(r.priority, 99))

    return recs
