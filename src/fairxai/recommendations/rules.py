"""Triage rule functions: one per TRIAGE_PLAN category (A–F).

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
                    "Multiclass subgroup slices with very low support make per-class "
                    "fairness metrics unreliable and statistically fragile."
                ),
                explainability_relevance=(
                    "High overlap complexity in multiclass settings makes explanations "
                    "less stable, as decision boundaries between similar classes blur."
                ),
                action=(
                    "Consider collapsing the target to a binary framing for the "
                    "initial fairness benchmark. Keep the multiclass formulation only "
                    "when subgroup-by-class support is adequate (≥{} per slice).".format(
                        config.multiclass_minority_support
                    )
                ),
                expected_outcome=(
                    "More reliable fairness diagnostics and clearer explainability "
                    "narratives in the early profiling phase."
                ),
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
    n_samples = ev.get_n_samples(profile) or 1

    # --- No sensitive columns declared / detected ---
    if not ingestion.sensitive_columns:
        recs.append(
            Recommendation(
                category=TriageCategory.B_SENSITIVE_ADEQUACY,
                priority=Priority.P0,
                title="No sensitive attributes identified",
                evidence={"declared_sensitive_columns": []},
                fairness_relevance=(
                    "Fairness cannot be assessed without at least one sensitive / "
                    "protected attribute (e.g., sex, age group, ethnicity)."
                ),
                explainability_relevance=(
                    "Without sensitive attributes, explanations cannot be audited "
                    "for differential treatment across demographic groups."
                ),
                action=(
                    "Identify and declare at least one sensitive attribute before "
                    "running any fairness analysis."
                ),
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
                        f"Attribute '{attr}' has {missing_frac:.1%} missing values. "
                        "Fairness metrics computed on incomplete group labels are unreliable."
                    ),
                    explainability_relevance=(
                        "Group-conditional explanations will be based on a biased subset "
                        "of the data if many group labels are missing."
                    ),
                    action=(
                        f"Investigate why '{attr}' has high missingness. Consider "
                        "imputation, data augmentation, or flagging this benchmark "
                        "as 'limited fairness validity'."
                    ),
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
                        f"Attribute '{attr}' has only {n_groups} group(s). At least "
                        f"{config.min_unique_groups} are needed for meaningful fairness comparison."
                    ),
                    explainability_relevance=(
                        "Group-contrastive explanations require at least two groups to "
                        "highlight differential model behaviour."
                    ),
                    action=(
                        f"Verify the encoding of '{attr}'. If the attribute has been "
                        "incorrectly collapsed, re-process it."
                    ),
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
                        f"Group sizes in '{attr}' differ by {ratio:.1f}x. The minority "
                        "group's fairness metrics will have wide confidence intervals."
                    ),
                    explainability_relevance=(
                        "Explanations may be dominated by patterns in the majority group, "
                        "under-representing minority-group decision factors."
                    ),
                    action=(
                        "Mark low-support groups as low-confidence for fairness "
                        "conclusions. Consider resampling or collecting additional data."
                    ),
                    expected_outcome=(
                        "Reduced risk of misleading fairness claims from unstable "
                        "subgroup estimates."
                    ),
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
                        f"Statistical parity difference of {spd:.3f} in '{attr}' "
                        "indicates the positive-class base rate varies substantially "
                        "across groups, which will propagate into model predictions."
                    ),
                    explainability_relevance=(
                        "Feature importance rankings may reflect base-rate differences "
                        "rather than genuine predictive signal when label imbalance is high."
                    ),
                    action=(
                        "Investigate whether the label imbalance reflects real prevalence "
                        "differences or data collection bias. Document the finding and "
                        "consider adjusting fairness metric interpretation accordingly."
                    ),
                    expected_outcome=(
                        "Better-calibrated expectations for fairness metric values and "
                        "more nuanced interpretation of any detected disparities."
                    ),
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
                        f"At least one group in '{attr}' has only {min_size} samples, "
                        f"below the minimum threshold of {config.min_group_samples}. "
                        "Per-group metrics will be statistically unreliable."
                    ),
                    explainability_relevance=(
                        "Group-level explanations based on very few samples may not "
                        "generalize to the broader population."
                    ),
                    action=(
                        "Request additional data for under-represented groups, or "
                        "mark their fairness results as low-confidence."
                    ),
                    expected_outcome=(
                        "More robust per-group fairness and explainability estimates."
                    ),
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
                            f"Attribute '{attr}' has {len(counts)} groups with a "
                            f"max/min size ratio of {ratio:.1f}x (threshold: "
                            f"{config.binning_size_ratio_warning:.1f}x). "
                            "Extremely unequal bins reduce statistical power for "
                            "the smallest groups and may distort fairness metrics."
                        ),
                        explainability_relevance=(
                            "Explanations for tiny bins may be unreliable; dominant bins "
                            "can overshadow minority-bin effects in feature-importance."
                        ),
                        action=(
                            "Try alternative binning strategies (quantile, equal-width, or "
                            "fewer bins) to improve group balance. Compare fairness metrics "
                            "across strategies using the age-binning experiment module."
                        ),
                        expected_outcome=(
                            "More balanced group sizes leading to more reliable and "
                            "comparable per-group fairness estimates."
                        ),
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
                    f"{len(low_intersections)} intersectional slice(s) have fewer than "
                    f"{config.intersectional_min_samples} samples. Intersectional fairness "
                    "analysis for these groups will be unreliable."
                ),
                explainability_relevance=(
                    "Intersectional explanations (e.g., older females) may be driven by "
                    "noise rather than signal in low-support slices."
                ),
                action=(
                    "Acknowledge low-support intersections in the fairness report. "
                    "Avoid strong claims about fairness for these subgroups."
                ),
                expected_outcome=(
                    "Transparent fairness reporting with explicit confidence qualifications."
                ),
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
                    "High overlap means some samples are intrinsically hard to classify "
                    "correctly. If these overlap regions are concentrated in specific "
                    "subgroups, fairness gaps will be amplified."
                ),
                explainability_relevance=(
                    "In high-overlap regions, model explanations become less decisive — "
                    "small changes in features flip predictions, making SHAP / LIME "
                    "values unstable."
                ),
                action=(
                    "Flag the dataset as high ambiguity. Require subgroup-level overlap "
                    "review before comparing fairness metrics across groups."
                ),
                expected_outcome=(
                    "Better interpretation of whether observed fairness gaps originate "
                    "from representation issues or intrinsic data overlap."
                ),
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
                    "Some subgroups face higher intrinsic classification difficulty than "
                    "others. This can produce fairness gaps even with an unbiased model."
                ),
                explainability_relevance=(
                    "Explanations will differ in reliability across groups: stable in "
                    "low-complexity subgroups, noisy in high-complexity ones."
                ),
                action=(
                    "Investigate which subgroups have elevated complexity and document "
                    "this as a confounding factor when interpreting fairness gaps."
                ),
                expected_outcome=(
                    "More accurate attribution of fairness gaps to data difficulty "
                    "vs. model bias."
                ),
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
                    "When linear decision boundaries poorly separate the classes, "
                    "fairness metrics tied to model performance (e.g., equalized odds) "
                    "may fluctuate with threshold choice."
                ),
                explainability_relevance=(
                    "High linear complexity means simple feature-attribution explanations "
                    "(e.g., logistic-regression coefficients, linear SHAP) will capture "
                    "only part of the decision logic. Users may be misled by seemingly "
                    "clear but incomplete explanations."
                ),
                action=(
                    "Prefer robust, uncertainty-aware explanation methods (e.g., SHAP "
                    "with TreeExplainer on ensemble models). Report explanation "
                    "stability alongside feature importance."
                ),
                expected_outcome=(
                    "More realistic expectations for explanation quality and reduced "
                    "risk of stakeholders over-trusting simple attributions."
                ),
                confidence=Confidence.MEDIUM,
            )
        )

    return recs


# ===================================================================
# F — Fairness benchmark readiness status
# ===================================================================


def check_readiness(
    recommendations: List[Recommendation],
    config: TriageConfig,
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
    """Execute categories A–E and then derive F (readiness).

    Returns the full sorted list including the readiness recommendation.
    """
    recs: List[Recommendation] = []
    recs.extend(check_task_framing(profile, config, ref))
    recs.extend(check_sensitive_adequacy(profile, ingestion, config))
    recs.extend(check_representation_risk(profile, config, ref))
    recs.extend(check_overlap_ambiguity(profile, config, ref))
    recs.extend(check_explainability_suitability(profile, config, ref))

    readiness = check_readiness(recs, config)
    recs.append(readiness)

    # Sort: P0 first, then P1, P2, P3
    priority_order = {Priority.P0: 0, Priority.P1: 1, Priority.P2: 2, Priority.P3: 3}
    recs.sort(key=lambda r: priority_order.get(r.priority, 99))

    return recs
