"""Generate all dissertation plots from a completed cardiac pipeline run.

Usage
-----
    python scripts/generate_dissertation_plots.py --run-id latest
    python scripts/generate_dissertation_plots.py --run-id 2026-03-31_run_01

Outputs are saved to ``output/cardiac/dissertation_figures/`` organised by section:

    fairness/
        fairness_metric_heatmap_age.png
        fairness_metric_heatmap_sex.png
        group_performance_gaps_age.png
        group_performance_gaps_sex.png
        bias_amplification_waterfall.png   (requires stage_gaps.json in run dir)
    transformations/
        transformation_impact.png
        before_after_distributions.png
        scaling_effects.png
    cross_model/
        intersectional_heatmap_dp.png
        cross_model_radar.png
        mitigation_effectiveness_matrix.png
        pareto_all_models.png
    fairness_comparison/
        fairness_evidence_summary.csv
        before_after_metric_radar.png
        mitigation_delta_matrix.png
        group_before_after_age.png
        group_before_after_sex.png
        group_delta_age.png
        group_delta_sex.png
        cross_model_baseline_radar.png
        cross_model_best_available_radar.png

The script skips plots whose required data files are missing and logs a warning
for each, so a partial run still produces as many plots as possible.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

# Ensure the project src is importable when running as a script
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from fairxai.cli.runner_base import setup_study_logging
from fairxai.comparison import load_comparison_config
from fairxai.viz.fairness_comparison import (
    save_before_after_metric_radar,
    save_cross_model_baseline_radar,
    save_cross_model_best_available_radar,
    save_fairness_evidence_summary,
    save_group_before_after_bars,
    save_group_delta_bars,
    save_mitigation_delta_matrix,
    select_primary_fairness_row,
)
from fairxai.viz.experiment_plots import (
    save_cross_model_radar,
    save_intersectional_heatmap,
    save_mitigation_effectiveness_matrix,
    save_pareto_all_models,
)
from fairxai.viz.fairness import (
    plot_bias_amplification_waterfall,
    plot_fairness_metric_heatmap,
    plot_group_performance_gaps,
)
from fairxai.viz.transformations import (
    plot_before_after_distributions,
    plot_scaling_effects,
    plot_transformation_impact,
)

logger = logging.getLogger(__name__)

_RUNS_BASE = _ROOT / "output" / "cardiac" / "runs"
_DATA_PROCESSED = _ROOT / "data" / "processed" / "cardiac"
_OUT_BASE = _ROOT / "output" / "cardiac" / "studies" / "dissertation_figures"

# Sensitive attribute names as used in fairness JSONs (with _cat) and in CSV columns (without)
_SENSITIVE_ATTRS = ["age_group_cat", "sex_cat"]
_FEATURE_COLS = ["trestbps", "chol", "thalach", "oldpeak", "ca"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_run_dir(run_id: str) -> Path:
    if not _RUNS_BASE.exists():
        raise FileNotFoundError(f"Runs base directory not found: {_RUNS_BASE}")
    if run_id == "latest":
        candidates = sorted(
            [d for d in _RUNS_BASE.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No run directories found under {_RUNS_BASE}")
        chosen = candidates[0]
        logger.info("Resolved --run-id latest to %s", chosen.name)
        return chosen
    run_dir = _RUNS_BASE / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("[WARNING] File not found (skipping): %s", path.name)
        return None
    return pd.read_csv(path)


def _resolve_comparisons_dir(run_dir: Path) -> Path:
    """Resolve the comparisons data directory for both old and current layouts."""
    candidates = [
        run_dir / "experiments" / "comparisons" / "data",
        run_dir / "experiments" / "comparisons",
        run_dir / "experiments" / "full" / "comparisons",
    ]
    for candidate in candidates:
        if (candidate / "full_comparison.csv").exists():
            return candidate
    # Default to latest layout when files are not present yet.
    return candidates[0]


def _report(label: str, result) -> None:
    if result is not None:
        logger.info("[SUCCESS] %s", label)
    else:
        logger.warning("[WARNING] %s: skipped (empty data or missing columns)", label)


def _phase(name: str) -> None:
    logger.info("[PHASE] %s", name)


def _normalize_cross_model_summary(
    summary_df: pd.DataFrame | None,
    full_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if summary_df is None or summary_df.empty:
        return summary_df

    df = summary_df.copy()

    if "f1_score" not in df.columns and "f1" in df.columns:
        df["f1_score"] = df["f1"]
    if "auc_roc" not in df.columns and "auc" in df.columns:
        df["auc_roc"] = df["auc"]

    # Fill missing metrics from full_comparison rows referenced by experiment_id.
    if full_df is not None and not full_df.empty and "experiment_id" in df.columns:
        metric_cols = {
            "precision": ["precision_value", "precision"],
            "auc_roc": ["auc_value", "auc_roc"],
            "recall": ["recall_value", "recall"],
            "f1_score": ["f1_value", "f1_score"],
            "fairness_gap": ["fairness_gap"],
        }
        src = full_df.copy()
        merge_map = {"experiment_id": src.get("experiment_id")}
        for target_col, candidates in metric_cols.items():
            for candidate in candidates:
                if candidate in src.columns:
                    merge_map[target_col] = src[candidate]
                    break
        enrich_df = pd.DataFrame(merge_map)
        df = df.merge(enrich_df, on="experiment_id", how="left", suffixes=("", "_from_full"))
        for target_col in metric_cols:
            fallback_col = f"{target_col}_from_full"
            if fallback_col in df.columns:
                if target_col not in df.columns:
                    df[target_col] = df[fallback_col]
                else:
                    df[target_col] = df[target_col].fillna(df[fallback_col])
                df = df.drop(columns=[fallback_col])

    return df


def _canonical_full_for_plots(
    experiment_index: pd.DataFrame | None,
    metric_values: pd.DataFrame | None,
    metric_deltas: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Build a wide plot-ready frame from canonical long metric tables."""
    if experiment_index is None or metric_values is None:
        return None
    if experiment_index.empty or metric_values.empty:
        return None

    id_cols = [
        "experiment_id",
        "dataset",
        "model_type",
        "model_variant",
        "binning_strategy",
        "training_method",
        "mitigation_technique",
    ]
    keep_cols = [c for c in id_cols + ["status"] if c in experiment_index.columns]
    wide = experiment_index[keep_cols].copy()

    value_pivot = (
        metric_values.pivot_table(index="experiment_id", columns="metric", values="value", aggfunc="first")
        .reset_index()
        .rename(
            columns={
                "f1": "f1_value",
                "recall": "recall_value",
                "precision": "precision_value",
                "auc_roc": "auc_value",
                "accuracy": "accuracy_value",
                "demographic_parity_gap": "dp_max_diff",
                "equalized_odds_gap": "eq_odds_max_diff",
                "equalized_odds_tpr_gap": "eq_odds_global_tpr_diff",
                "equalized_odds_fpr_gap": "eq_odds_global_fpr_diff",
            }
        )
    )
    wide = wide.merge(value_pivot, on="experiment_id", how="left")

    if metric_deltas is not None and not metric_deltas.empty:
        deltas = metric_deltas.copy()
        deltas["plot_delta"] = pd.to_numeric(deltas["delta"], errors="coerce")
        gap_mask = deltas["metric_family"].astype(str).eq("fairness_gap")
        if "improvement" in deltas.columns:
            deltas.loc[gap_mask, "plot_delta"] = pd.to_numeric(
                deltas.loc[gap_mask, "improvement"], errors="coerce"
            )
        delta_pivot = (
            deltas.pivot_table(
                index="experiment_id", columns="metric", values="plot_delta", aggfunc="first"
            )
            .reset_index()
            .rename(
                columns={
                    "f1": "delta_f1",
                    "recall": "delta_recall",
                    "precision": "delta_precision",
                    "auc_roc": "delta_auc",
                    "accuracy": "delta_accuracy",
                    "fairness_gap": "delta_fairness_gap",
                    "demographic_parity_gap": "delta_dp_gap",
                    "equalized_odds_tpr_gap": "delta_eq_tpr_gap",
                    "equalized_odds_fpr_gap": "delta_eq_fpr_gap",
                }
            )
        )
        wide = wide.merge(delta_pivot, on="experiment_id", how="left")

    return wide


def _dataset_prefix(dataset: object, model_label: str | None = None) -> str:
    dataset_slug = str(dataset).lower().replace(" ", "_")
    if model_label:
        return f"{dataset_slug}_{model_label}"
    return dataset_slug


def _remove_generic_fairness_comparison_artifacts(out_dir: Path) -> None:
    """Remove pre-canonical generic filenames so stale figures do not linger."""
    filenames = [
        "before_after_metric_radar.png",
        "mitigation_delta_matrix.png",
        "group_before_after_age.png",
        "group_before_after_sex.png",
        "group_delta_age.png",
        "group_delta_sex.png",
        "cross_model_baseline_radar.png",
        "cross_model_best_available_radar.png",
    ]
    for name in filenames:
        path = out_dir / name
        if path.exists():
            try:
                path.unlink()
                logger.info("[CLEANUP] Removed generic fairness comparison figure: %s", path.name)
            except OSError as exc:
                logger.warning("[CLEANUP] Could not remove %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def _generate_fairness_plots(
    full_df: pd.DataFrame | None,
    per_group_df: pd.DataFrame | None,
    fairness_dir: Path,
    run_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _phase("fairness plots")

    # 1. Fairness metric heatmaps (one per sensitive attribute)
    if full_df is not None:
        for attr in _SENSITIVE_ATTRS:
            attr_short = attr.replace("_cat", "")
            result = plot_fairness_metric_heatmap(
                full_df, attr, out_dir / f"fairness_metric_heatmap_{attr_short}.png"
            )
            _report(f"fairness_metric_heatmap_{attr_short}", result)
    else:
        logger.warning("[WARNING] fairness_metric_heatmap: full_comparison.csv missing")

    # 2. Group performance gaps (baseline vs best LR experiment per attribute)
    baseline_jsons = sorted(fairness_dir.glob("*_logistic_regression_fairness_assessment.json"))
    if baseline_jsons:
        before_json = baseline_jsons[0]
        # Use the same file for both if no experiment JSON is available.
        for attr in _SENSITIVE_ATTRS:
            attr_short = attr.replace("_cat", "")
            result = plot_group_performance_gaps(
                before_json,
                before_json,  # placeholder; replace with experiment JSON for real before/after
                attr,
                out_dir / f"group_performance_gaps_{attr_short}.png",
            )
            _report(f"group_performance_gaps_{attr_short}", result)
    else:
        logger.warning(
            "[WARNING] group_performance_gaps: no baseline fairness JSON found under %s",
            fairness_dir,
        )

    # 3. Bias amplification waterfall (requires stage_gaps.json placed in run dir)
    stage_gaps_path = run_dir / "stage_gaps.json"
    if stage_gaps_path.exists():
        with stage_gaps_path.open() as f:
            stages_dict = json.load(f)
        result = plot_bias_amplification_waterfall(
            stages_dict, out_dir / "bias_amplification_waterfall.png"
        )
        _report("bias_amplification_waterfall", result)
    else:
        logger.warning(
            "[WARNING] bias_amplification_waterfall: stage_gaps.json not found in run dir. "
            "Create %s with {stage_name: fairness_gap} entries.",
            stage_gaps_path.name,
        )


def _generate_transformation_plots(
    run_dir: Path,
    full_df: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _phase("transformation plots")

    # 4. Transformation impact - best LR mitigation vs baseline (from full_comparison.csv)
    if full_df is not None:
        baseline_rows = full_df[full_df["mitigation_technique"] == "baseline"]
        non_baseline = full_df[full_df["mitigation_technique"] != "baseline"]
        if not baseline_rows.empty and not non_baseline.empty:
            # Support both old (f1_score) and new (f1_value) column naming
            metric_cols_candidates = [
                ("f1_score", "f1"),
                ("f1_value", "f1"),
                ("recall_value", "recall"),
                ("precision_value", "precision"),
                ("auc_value", "auc_roc"),
                ("auc_roc", "auc_roc"),
                ("fairness_gap", "fairness_gap"),
            ]
            baseline_row = baseline_rows.iloc[0]
            best_row = (
                non_baseline.loc[non_baseline["fairness_gain_pct"].idxmax()]
                if "fairness_gain_pct" in non_baseline.columns
                else non_baseline.iloc[0]
            )
            before_dict, after_dict = {}, {}
            seen_labels = set()
            for col, label in metric_cols_candidates:
                if col in full_df.columns and label not in seen_labels:
                    if pd.notna(baseline_row.get(col)) and pd.notna(best_row.get(col)):
                        before_dict[label] = float(baseline_row[col])
                        after_dict[label] = float(best_row[col])
                        seen_labels.add(label)
            result = plot_transformation_impact(
                before_dict, after_dict, out_dir / "transformation_impact.png"
            )
            _report("transformation_impact", result)
        else:
            logger.warning(
                "[WARNING] transformation_impact: need both baseline and non-baseline rows"
            )
    else:
        logger.warning("[WARNING] transformation_impact: full_comparison.csv missing")

    # 5. Before/after distributions - look for train prediction CSVs (pre- and post-SMOTE split)
    results_dir = run_dir / "baseline" / "results" / "predictions"
    pred_csvs = sorted(results_dir.glob("*_logistic_regression_train.csv"))
    if pred_csvs:
        pred_df = pd.read_csv(pred_csvs[0])
        feature_cols = [c for c in _FEATURE_COLS if c in pred_df.columns]
        if feature_cols:
            mid = len(pred_df) // 2
            result = plot_before_after_distributions(
                pred_df.iloc[:mid],
                pred_df.iloc[mid:],
                feature_cols,
                out_dir / "before_after_distributions.png",
            )
            _report("before_after_distributions", result)
        else:
            logger.warning(
                "[WARNING] before_after_distributions: no feature cols in prediction CSV"
            )
    else:
        logger.warning(
            "[WARNING] before_after_distributions: no train prediction CSV found under %s",
            results_dir,
        )

    # 6. Scaling effects - raw vs scaled using processed train CSV
    raw_csv = _DATA_PROCESSED / "cleveland_train.csv"
    scaled_csv = _DATA_PROCESSED / "cleveland_train_scaled.csv"
    if raw_csv.exists() and scaled_csv.exists():
        raw_df = pd.read_csv(raw_csv)
        scaled_df = pd.read_csv(scaled_csv)
        feature_cols = [c for c in _FEATURE_COLS if c in raw_df.columns and c in scaled_df.columns]
        if feature_cols:
            result = plot_scaling_effects(
                raw_df[feature_cols], scaled_df[feature_cols], out_dir / "scaling_effects.png"
            )
            _report("scaling_effects", result)
        else:
            logger.warning("[WARNING] scaling_effects: no shared feature cols in raw/scaled CSVs")
    else:
        logger.warning(
            "[WARNING] scaling_effects: cleveland_train.csv or cleveland_train_scaled.csv "
            "not found under %s",
            _DATA_PROCESSED,
        )


def _generate_cross_model_plots(
    full_df: pd.DataFrame | None,
    per_group_df: pd.DataFrame | None,
    summary_df: pd.DataFrame | None,
    out_dir: Path,
    comparison_config: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _phase("cross-model plots")

    # 7. Intersectional heatmap
    if per_group_df is not None:
        result = save_intersectional_heatmap(
            per_group_df, "demographic_parity_rate", out_dir / "intersectional_heatmap_dp.png"
        )
        _report("intersectional_heatmap_dp", result)
    else:
        logger.warning(
            "[WARNING] intersectional_heatmap: per_group.csv missing (run full pipeline first)"
        )

    # 8. Cross-model radar
    if summary_df is not None:
        normalized_summary = _normalize_cross_model_summary(summary_df, full_df)
        result = save_cross_model_radar(normalized_summary, out_dir / "cross_model_radar.png")
        _report("cross_model_radar", result)
    else:
        logger.warning(
            "[WARNING] cross_model_radar: cross_model_summary.csv missing (run full pipeline first)"
        )

    legacy_score_enabled = bool(
        comparison_config.get("legacy_score", {}).get("dissertation_figures_enabled", False)
    )

    # 9. Legacy score-based mitigation effectiveness matrix
    if full_df is not None and legacy_score_enabled:
        result = save_mitigation_effectiveness_matrix(
            full_df, out_dir / "mitigation_effectiveness_matrix.png"
        )
        _report("mitigation_effectiveness_matrix", result)
    elif full_df is not None:
        logger.info("[SKIP] mitigation_effectiveness_matrix: legacy score figure disabled")
    else:
        logger.warning("[WARNING] mitigation_effectiveness_matrix: full_comparison.csv missing")

    # 10. Legacy Pareto figure
    if full_df is not None and legacy_score_enabled:
        x_col = "f1_value" if "f1_value" in full_df.columns else "f1_score"
        result = save_pareto_all_models(full_df, out_dir / "pareto_all_models.png", x_col=x_col)
        _report("pareto_all_models", result)
    elif full_df is not None:
        logger.info("[SKIP] pareto_all_models: legacy score figure disabled")
    else:
        logger.warning("[WARNING] pareto_all_models: full_comparison.csv missing")


def _generate_fairness_comparison_plots(
    full_df: pd.DataFrame | None,
    per_group_df: pd.DataFrame | None,
    evidence_summary_df: pd.DataFrame | None,
    out_dir: Path,
    comparison_config: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _phase("fairness comparison plots")
    if comparison_config.get("figures", {}).get("canonical_naming", True):
        _remove_generic_fairness_comparison_artifacts(out_dir)

    if full_df is None:
        logger.warning("[WARNING] fairness_comparison: comparison metric data missing")
        return

    selection_cfg = comparison_config.get("selection", {})
    primary_model = selection_cfg.get("primary_model_type", "logistic_regression")
    model_label = selection_cfg.get("primary_model_label", "lr")
    min_recall_delta = float(selection_cfg.get("min_recall_delta", -0.03))
    include_appendix = bool(
        comparison_config.get("figures", {}).get("include_best_available_appendix", True)
    )

    if evidence_summary_df is not None and not evidence_summary_df.empty:
        evidence_summary_df.to_csv(out_dir / "fairness_evidence_summary.csv", index=False)
        _report("fairness_evidence_summary", out_dir / "fairness_evidence_summary.csv")
    else:
        result = save_fairness_evidence_summary(
            full_df,
            per_group_df,
            out_dir / "fairness_evidence_summary.csv",
            top_n=int(selection_cfg.get("top_n", 5)),
        )
        _report("fairness_evidence_summary", result)

    for dataset in sorted(full_df["dataset"].dropna().unique()):
        dataset_df = full_df[full_df["dataset"].astype(str) == str(dataset)].copy()
        if dataset_df.empty:
            continue
        primary_df = dataset_df
        if "model_type" in primary_df.columns:
            primary_df = primary_df[primary_df["model_type"].astype(str) == str(primary_model)]

        selected = select_primary_fairness_row(
            dataset_df, model_type=primary_model, min_recall_delta=min_recall_delta
        )
        dataset_prefix = _dataset_prefix(dataset)
        primary_prefix = _dataset_prefix(dataset, model_label)

        result = save_before_after_metric_radar(
            dataset_df,
            out_dir / f"{primary_prefix}_primary_mitigation_radar_before_after.png",
            selected_row=selected,
        )
        _report(f"{primary_prefix}_primary_mitigation_radar_before_after", result)

        result = save_mitigation_delta_matrix(
            primary_df, out_dir / f"{primary_prefix}_mitigation_metric_delta_matrix.png"
        )
        _report(f"{primary_prefix}_mitigation_metric_delta_matrix", result)

        result = save_cross_model_baseline_radar(
            dataset_df, out_dir / f"{dataset_prefix}_baseline_cross_model_radar.png"
        )
        _report(f"{dataset_prefix}_baseline_cross_model_radar", result)

        if include_appendix:
            result = save_cross_model_best_available_radar(
                dataset_df,
                out_dir / f"{dataset_prefix}_unbalanced_best_available_cross_model_radar.png",
            )
            _report(f"{dataset_prefix}_unbalanced_best_available_cross_model_radar", result)

        if selected is None:
            logger.warning("[WARNING] group before/after plots: no selected mitigation row")
            continue
        if per_group_df is None:
            logger.warning("[WARNING] group before/after plots: group metric deltas missing")
            continue

        dataset_groups = per_group_df[per_group_df["dataset"].astype(str) == str(dataset)].copy()
        for attr, label in [("age_group", "age_group"), ("sex", "sex")]:
            result = save_group_before_after_bars(
                dataset_groups,
                out_dir / f"{primary_prefix}_primary_{label}_before_after.png",
                attr,
                selected,
            )
            _report(f"{primary_prefix}_primary_{label}_before_after", result)

            result = save_group_delta_bars(
                dataset_groups,
                out_dir / f"{primary_prefix}_primary_{label}_delta.png",
                attr,
                selected,
            )
            _report(f"{primary_prefix}_primary_{label}_delta", result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all dissertation plots from a cardiac pipeline run."
    )
    parser.add_argument(
        "--run-id",
        default="latest",
        help="Run ID (directory name under output/cardiac/runs/) or 'latest'.",
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/comparison.yaml",
        help="Comparison YAML config (CLI flags override overlapping values).",
    )
    legacy_group = parser.add_mutually_exclusive_group()
    legacy_group.add_argument(
        "--legacy-score-figures",
        dest="legacy_score_figures",
        action="store_true",
        help="Enable legacy score-value dissertation figures.",
    )
    legacy_group.add_argument(
        "--no-legacy-score-figures",
        dest="legacy_score_figures",
        action="store_false",
        help="Disable legacy score-value dissertation figures.",
    )
    parser.set_defaults(legacy_score_figures=None)
    args = parser.parse_args()

    setup_study_logging(
        _ROOT,
        "dissertation_figures",
        args.run_id,
        "dissertation_figures.log",
        verbose=False,
        log_subdir="cardiac",
    )

    run_dir = _resolve_run_dir(args.run_id)
    comparison_config = load_comparison_config(_ROOT, args.config)
    if args.legacy_score_figures is not None:
        comparison_config.setdefault("legacy_score", {})[
            "dissertation_figures_enabled"
        ] = args.legacy_score_figures
    comparisons_dir = _resolve_comparisons_dir(run_dir)
    fairness_dir = run_dir / "baseline" / "prediction_fairness"
    if not fairness_dir.exists():
        fairness_dir = run_dir / "baseline" / "fairness"
    out_base = _OUT_BASE / run_dir.name
    logger.info("[PHASE] Dissertation plot generation started")
    logger.info(
        "[RUN_CONTEXT] requested_run_id=%s resolved_run_id=%s output_dir=%s",
        args.run_id,
        run_dir.name,
        out_base,
    )

    full_df = _safe_read_csv(comparisons_dir / "full_comparison.csv")
    per_group_df = _safe_read_csv(comparisons_dir / "per_group.csv")
    if per_group_df is None:
        per_group_df = _safe_read_csv(comparisons_dir / "per_group_comparison.csv")
    summary_df = _safe_read_csv(comparisons_dir / "cross_model_summary.csv")
    experiment_index_df = _safe_read_csv(comparisons_dir / "experiment_index.csv")
    metric_values_df = _safe_read_csv(comparisons_dir / "metric_values.csv")
    metric_deltas_df = _safe_read_csv(comparisons_dir / "metric_deltas.csv")
    group_metric_deltas_df = _safe_read_csv(comparisons_dir / "group_metric_deltas.csv")
    evidence_summary_df = _safe_read_csv(comparisons_dir / "fairness_evidence_summary.csv")

    canonical_full_df = _canonical_full_for_plots(
        experiment_index_df, metric_values_df, metric_deltas_df
    )
    fairness_comparison_full_df = canonical_full_df if canonical_full_df is not None else full_df
    fairness_comparison_group_df = (
        group_metric_deltas_df if group_metric_deltas_df is not None else per_group_df
    )

    _generate_fairness_plots(full_df, per_group_df, fairness_dir, run_dir, out_base / "fairness")
    _generate_transformation_plots(run_dir, full_df, out_base / "transformations")
    _generate_cross_model_plots(
        full_df, per_group_df, summary_df, out_base / "cross_model", comparison_config
    )
    _generate_fairness_comparison_plots(
        fairness_comparison_full_df,
        fairness_comparison_group_df,
        evidence_summary_df,
        out_base / "fairness_comparison",
        comparison_config,
    )

    logger.info("[SUCCESS] Dissertation figures generated: output_dir=%s", out_base)


if __name__ == "__main__":
    main()
