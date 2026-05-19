"""Generate all dissertation plots from a completed cardiac pipeline run.

Usage
-----
    python scripts/generate_dissertation_plots.py --run-id latest
    python scripts/generate_dissertation_plots.py --run-id 2026-03-31_run_01

Outputs are saved to ``output/cardiac/studies/dissertation_figures/<run_id>/``
organised by section:

    fairness/
        cleveland_age_group_fairness_metric_heatmap.png
        cleveland_sex_fairness_metric_heatmap.png
        bias_amplification_waterfall.png   (requires stage_gaps.json in run dir)
    transformations/
        transformation_impact.png
        before_after_distributions.png
        scaling_effects.png
    cross_model/
        cleveland_demographic_parity_intersectional_heatmap.png
    fairness_comparison/
        data/fairness_evidence_summary.csv
        plots/cleveland_lr_primary_mitigation_radar_before_after.png
        plots/cleveland_lr_mitigation_metric_delta_matrix.png
        plots/cleveland_lr_primary_age_group_performance_gaps.png
        plots/cleveland_lr_primary_sex_performance_gaps.png
        plots/cleveland_lr_primary_age_group_before_after.png
        plots/cleveland_lr_primary_sex_before_after.png
        plots/cleveland_lr_primary_age_group_delta.png
        plots/cleveland_lr_primary_sex_delta.png
        plots/cleveland_baseline_cross_model_radar.png

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
from fairxai.comparison import build_metric_plot_frame, figure_filename, load_comparison_config
from fairxai.comparison.metric_tables import build_fairness_evidence_summary
from fairxai.viz.fairness import (
    plot_bias_amplification_waterfall,
    plot_fairness_metric_heatmap,
)
from fairxai.viz.fairness_comparison import (
    save_before_after_metric_radar,
    save_binning_strategy_delta_matrix,
    save_cross_model_baseline_radar,
    save_cross_model_best_available_radar,
    save_group_before_after_bars,
    save_group_delta_bars,
    save_group_error_consequence_bars,
    save_group_performance_gap_bars,
    save_intersectional_heatmap,
    save_mitigation_delta_matrix,
    save_model_overfit_gap_bars,
    save_top_n_binning_strategy_age_group_small_multiples,
    save_top_n_binning_strategy_summary,
    select_primary_fairness_row,
)
from fairxai.viz.clustering import (
    save_cluster_fairness_heatmap,
    save_cluster_profile_bars,
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


def _find_processed_csv(dataset: str, suffix: str) -> Path | None:
    """Locate a processed CSV for a dataset, trying subdirectory then flat layout."""
    subdir = _DATA_PROCESSED / dataset / f"{dataset}_{suffix}.csv"
    flat = _DATA_PROCESSED / f"{dataset}_{suffix}.csv"
    if subdir.exists():
        return subdir
    if flat.exists():
        return flat
    return None


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


def _resolve_comparisons_dir(run_dir: Path, comparison_config: dict | None = None) -> Path:
    """Resolve the comparisons data directory for both old and current layouts."""
    data_subdir = (comparison_config or {}).get("outputs", {}).get("comparison_data_dir", "data")
    candidates = [
        run_dir / "experiments" / "comparisons" / data_subdir,
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


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


def _generate_fairness_plots(
    full_df: pd.DataFrame | None,
    run_dir: Path,
    out_dir: Path,
    comparison_config: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _phase("fairness plots")

    # 1. Fairness metric heatmaps (one per sensitive attribute)
    if full_df is not None:
        datasets = (
            sorted(full_df["dataset"].dropna().unique())
            if "dataset" in full_df.columns
            else ["dataset"]
        )
        for dataset in datasets:
            dataset_df = (
                full_df[full_df["dataset"].astype(str) == str(dataset)].copy()
                if "dataset" in full_df.columns
                else full_df
            )
            for attr in _SENSITIVE_ATTRS:
                filename = figure_filename(
                    comparison_config,
                    "fairness_metric_heatmap",
                    dataset=dataset,
                    sensitive_attr=attr,
                )
                result = plot_fairness_metric_heatmap(dataset_df, attr, out_dir / filename)
                _report(filename.removesuffix(".png"), result)
    else:
        logger.warning("[WARNING] fairness_metric_heatmap: full_comparison.csv missing")

    # 2. Bias amplification waterfall (requires stage_gaps.json placed in run dir)
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

    # 6. Scaling effects - raw vs scaled using processed train CSV (one plot per dataset)
    scaling_datasets = (
        sorted(full_df["dataset"].dropna().unique())
        if full_df is not None and "dataset" in full_df.columns
        else ["cleveland"]
    )
    for dataset in scaling_datasets:
        raw_csv = _find_processed_csv(dataset, "train")
        scaled_csv = _find_processed_csv(dataset, "train_scaled")
        if raw_csv is None or scaled_csv is None:
            logger.warning(
                "[WARNING] scaling_effects: %s_train[_scaled].csv not found under %s",
                dataset,
                _DATA_PROCESSED,
            )
            continue
        raw_df = pd.read_csv(raw_csv)
        scaled_df = pd.read_csv(scaled_csv)
        feature_cols = [c for c in _FEATURE_COLS if c in raw_df.columns and c in scaled_df.columns]
        if feature_cols:
            result = plot_scaling_effects(
                raw_df[feature_cols],
                scaled_df[feature_cols],
                out_dir / f"{dataset}_scaling_effects.png",
            )
            _report(f"{dataset}_scaling_effects", result)
        else:
            logger.warning(
                "[WARNING] scaling_effects: no shared feature cols in raw/scaled CSVs for %s",
                dataset,
            )


def _generate_cross_model_plots(
    per_group_df: pd.DataFrame | None,
    out_dir: Path,
    comparison_config: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _phase("cross-model plots")

    if per_group_df is not None and "dataset" in per_group_df.columns:
        for dataset in sorted(per_group_df["dataset"].dropna().unique()):
            dataset_df = per_group_df[per_group_df["dataset"].astype(str) == str(dataset)].copy()
            filename = figure_filename(
                comparison_config,
                "intersectional_heatmap",
                dataset=dataset,
                metric="demographic_parity",
            )
            result = save_intersectional_heatmap(
                dataset_df, "demographic_parity_rate", out_dir / filename
            )
            _report(filename.removesuffix(".png"), result)
    elif per_group_df is not None:
        result = save_intersectional_heatmap(
            per_group_df,
            "demographic_parity_rate",
            out_dir / "dataset_demographic_parity_intersectional_heatmap.png",
        )
        _report("dataset_demographic_parity_intersectional_heatmap", result)
    else:
        logger.warning(
            "[WARNING] intersectional_heatmap: per_group.csv missing (run full pipeline first)"
        )


def _generate_fairness_comparison_plots(
    full_df: pd.DataFrame | None,
    per_group_df: pd.DataFrame | None,
    evidence_summary_df: pd.DataFrame | None,
    out_dir: Path,
    comparison_config: dict,
) -> None:
    output_cfg = comparison_config.get("outputs", {})
    plots_dir = out_dir / output_cfg.get("dissertation_plot_dir", "plots")
    data_dir = out_dir / output_cfg.get("dissertation_data_dir", "data")
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    _phase("fairness comparison plots")

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
        summary_file = data_dir / "fairness_evidence_summary.csv"
        evidence_summary_df.to_csv(summary_file, index=False)
        _report("fairness_evidence_summary", summary_file)
    else:
        summary_df = build_fairness_evidence_summary(full_df, per_group_df, comparison_config)
        if not summary_df.empty:
            summary_file = data_dir / "fairness_evidence_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            _report("fairness_evidence_summary", summary_file)
        else:
            _report("fairness_evidence_summary", None)

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

        radar_name = figure_filename(
            comparison_config,
            "primary_mitigation_radar",
            dataset=dataset,
            model_label=model_label,
        )
        result = save_before_after_metric_radar(
            dataset_df,
            plots_dir / radar_name,
            selected_row=selected,
        )
        _report(radar_name.removesuffix(".png"), result)

        matrix_name = figure_filename(
            comparison_config,
            "mitigation_delta_matrix",
            dataset=dataset,
            model_label=model_label,
        )
        result = save_mitigation_delta_matrix(primary_df, plots_dir / matrix_name)
        _report(matrix_name.removesuffix(".png"), result)

        baseline_radar_name = figure_filename(
            comparison_config,
            "baseline_cross_model_radar",
            dataset=dataset,
        )
        result = save_cross_model_baseline_radar(dataset_df, plots_dir / baseline_radar_name)
        _report(baseline_radar_name.removesuffix(".png"), result)

        if include_appendix:
            best_name = figure_filename(
                comparison_config,
                "best_available_cross_model_radar",
                dataset=dataset,
            )
            result = save_cross_model_best_available_radar(
                dataset_df,
                plots_dir / best_name,
            )
            _report(best_name.removesuffix(".png"), result)

        if selected is None:
            logger.warning("[WARNING] group before/after plots: no selected mitigation row")
            continue
        if per_group_df is None:
            logger.warning("[WARNING] group before/after plots: group metric deltas missing")
            continue

        dataset_groups = (
            per_group_df[per_group_df["dataset"].astype(str) == str(dataset)].copy()
            if "dataset" in per_group_df.columns
            else per_group_df.copy()
        )
        for attr in ["age_group", "sex"]:
            performance_name = figure_filename(
                comparison_config,
                "group_performance_gaps",
                dataset=dataset,
                model_label=model_label,
                sensitive_attr=attr,
            )
            result = save_group_performance_gap_bars(
                dataset_groups,
                plots_dir / performance_name,
                attr,
                selected,
            )
            _report(performance_name.removesuffix(".png"), result)

            before_after_name = figure_filename(
                comparison_config,
                "group_before_after",
                dataset=dataset,
                model_label=model_label,
                sensitive_attr=attr,
            )
            result = save_group_before_after_bars(
                dataset_groups,
                plots_dir / before_after_name,
                attr,
                selected,
            )
            _report(before_after_name.removesuffix(".png"), result)

            delta_name = figure_filename(
                comparison_config,
                "group_delta",
                dataset=dataset,
                model_label=model_label,
                sensitive_attr=attr,
            )
            result = save_group_delta_bars(
                dataset_groups,
                plots_dir / delta_name,
                attr,
                selected,
            )
            _report(delta_name.removesuffix(".png"), result)

            consequence_name = figure_filename(
                comparison_config,
                "group_error_consequences",
                dataset=dataset,
                model_label=model_label,
                sensitive_attr=attr,
            )
            result = save_group_error_consequence_bars(
                dataset_groups,
                plots_dir / consequence_name,
                attr,
                selected,
            )
            _report(consequence_name.removesuffix(".png"), result)


# ---------------------------------------------------------------------------
# Binning sensitivity plots
# ---------------------------------------------------------------------------


def _generate_binning_sensitivity_plots(
    full_df,
    per_group_df,
    out_base,
    comparison_config: dict,
) -> None:
    """Generate binning strategy sensitivity figures.

    Three outputs per dataset:
    - binning_strategy_delta_matrix: heatmap of metric deltas by strategy
    - binning_strategy_summary: horizontal bar ranking of top-N strategies
    - binning_strategy_age_group_small_multiples: per-strategy age-group deltas
    """
    if full_df is None or full_df.empty:
        logger.warning("[WARNING] binning sensitivity: full_df missing, skipping")
        return

    selection = (comparison_config or {}).get("selection", {})
    model_type = selection.get("primary_model_type", "logistic_regression")
    model_label = selection.get("primary_model_label", "lr")
    top_n = int(selection.get("top_n", 5))
    min_recall_delta = float(selection.get("min_recall_delta", -0.03))

    plots_dir = out_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(full_df["dataset"].dropna().unique()):
        dataset_df = full_df[full_df["dataset"].astype(str) == str(dataset)].copy()
        if "model_type" in dataset_df.columns:
            dataset_df = dataset_df[dataset_df["model_type"].astype(str) == str(model_type)]

        dataset_groups = (
            per_group_df[per_group_df["dataset"].astype(str) == str(dataset)].copy()
            if per_group_df is not None and "dataset" in per_group_df.columns
            else per_group_df
        )

        matrix_name = figure_filename(
            comparison_config,
            "binning_strategy_delta_matrix",
            dataset=dataset,
            model_label=model_label,
        )
        result = save_binning_strategy_delta_matrix(
            dataset_df, plots_dir / matrix_name, model_type, min_recall_delta
        )
        _report(matrix_name.removesuffix(".png"), result)

        summary_name = figure_filename(
            comparison_config,
            "binning_strategy_summary",
            dataset=dataset,
            model_label=model_label,
            n=top_n,
        )
        result = save_top_n_binning_strategy_summary(
            dataset_df, plots_dir / summary_name, model_type, top_n, min_recall_delta
        )
        _report(summary_name.removesuffix(".png"), result)

        multiples_name = figure_filename(
            comparison_config,
            "binning_strategy_age_group_small_multiples",
            dataset=dataset,
            model_label=model_label,
            n=top_n,
        )
        result = save_top_n_binning_strategy_age_group_small_multiples(
            dataset_df,
            dataset_groups,
            plots_dir / multiples_name,
            model_type,
            top_n,
            min_recall_delta,
        )
        _report(multiples_name.removesuffix(".png"), result)


# ---------------------------------------------------------------------------
# Model stability / overfit gap
# ---------------------------------------------------------------------------


def _generate_model_stability_plots(
    run_dir: Path,
    full_df: "pd.DataFrame | None",
    out_dir: Path,
    comparison_config: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _phase("model stability plots")

    # --- overfit gap visualisation ---
    overfit_csv = run_dir / "baseline" / "prediction_fairness" / "overfit_gap_table.csv"
    if overfit_csv.exists():
        overfit_df = _safe_read_csv(overfit_csv)
        if overfit_df is not None and not overfit_df.empty:
            result = save_model_overfit_gap_bars(
                overfit_df, out_dir / "model_overfit_gap_bars.png"
            )
            _report("model_overfit_gap_bars", result)
        else:
            logger.warning("[WARNING] model_overfit_gap_bars: overfit_gap_table.csv is empty")
    else:
        logger.warning(
            "[WARNING] model_overfit_gap_bars: overfit_gap_table.csv not found at %s",
            overfit_csv,
        )

    # --- baseline-only comparison table ---
    if full_df is None or full_df.empty:
        logger.warning("[WARNING] baseline_model_comparison: full_df missing")
        return

    mit_col = "mitigation_technique" if "mitigation_technique" in full_df.columns else None
    if mit_col is None:
        logger.warning("[WARNING] baseline_model_comparison: no mitigation_technique column")
        return

    baseline_df = full_df[full_df[mit_col].astype(str).str.lower() == "baseline"].copy()
    if baseline_df.empty:
        logger.warning("[WARNING] baseline_model_comparison: no baseline rows in full_df")
        return

    keep_cols = [
        c
        for c in [
            "dataset",
            "model_type",
            "f1_value",
            "recall_value",
            "precision_value",
            "auc_value",
            "fairness_gap",
            "dem_parity_age_group_max_diff",
            "dem_parity_sex_max_diff",
        ]
        if c in baseline_df.columns
    ]
    best_rows = (
        baseline_df.sort_values("f1_value", ascending=False)
        .groupby(["dataset", "model_type"], as_index=False)
        .first()
    )
    table = best_rows[keep_cols].copy()

    # Merge overfit_risk if overfit_gap_table was loaded
    if overfit_csv.exists():
        odf = _safe_read_csv(overfit_csv)
        if odf is not None and "overfit_risk" in odf.columns:
            merge_key_left = "model_type"
            merge_key_right = "model"
            if merge_key_right in odf.columns:
                table = table.merge(
                    odf[["model", "overfit_risk"]].rename(
                        columns={"model": merge_key_left}
                    ),
                    on=merge_key_left,
                    how="left",
                )

    out_csv = out_dir / "baseline_model_comparison.csv"
    table.to_csv(out_csv, index=False)
    _report("baseline_model_comparison", out_csv)


# ---------------------------------------------------------------------------
# Cluster evidence plots (Task 5)
# ---------------------------------------------------------------------------

_GROUPING_STUDIES_BASE = _ROOT / "output" / "cardiac" / "studies" / "grouping"


def _resolve_latest_grouping_dir() -> Path | None:
    """Return the most recent grouping study directory, or None if absent."""
    if not _GROUPING_STUDIES_BASE.exists():
        return None
    latest_txt = _GROUPING_STUDIES_BASE / "latest.txt"
    if latest_txt.exists():
        study_id = latest_txt.read_text().strip()
        candidate = _GROUPING_STUDIES_BASE / study_id
        if candidate.is_dir():
            return candidate
    candidates = sorted(
        [p for p in _GROUPING_STUDIES_BASE.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _generate_cluster_evidence_plots(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _phase("cluster evidence plots")

    grouping_dir = _resolve_latest_grouping_dir()
    if grouping_dir is None:
        logger.warning(
            "[WARNING] cluster_evidence: no grouping study found under %s — "
            "run scripts/studies/run_grouping_analysis.py first",
            _GROUPING_STUDIES_BASE,
        )
        return

    for ds_dir in sorted(grouping_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name

        fairness_csv = ds_dir / "fairness_by_cluster.csv"
        if not fairness_csv.exists():
            logger.warning(
                "[WARNING] cluster_evidence: fairness_by_cluster.csv missing: %s", fairness_csv
            )
            continue

        fairness_df = _safe_read_csv(fairness_csv)
        if fairness_df is None or fairness_df.empty:
            logger.warning("[WARNING] cluster_evidence: fairness_by_cluster.csv is empty")
            continue

        result = save_cluster_profile_bars(
            fairness_df, out_dir / f"{dataset}_cluster_profile.png"
        )
        _report(f"{dataset}_cluster_profile", result)

        result = save_cluster_fairness_heatmap(
            fairness_df, out_dir / f"{dataset}_cluster_fairness.png"
        )
        _report(f"{dataset}_cluster_fairness", result)


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
    comparisons_dir = _resolve_comparisons_dir(run_dir, comparison_config)
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
    experiment_index_df = _safe_read_csv(comparisons_dir / "experiment_index.csv")
    metric_values_df = _safe_read_csv(comparisons_dir / "metric_values.csv")
    metric_deltas_df = _safe_read_csv(comparisons_dir / "metric_deltas.csv")
    group_metric_deltas_df = _safe_read_csv(comparisons_dir / "group_metric_deltas.csv")
    evidence_summary_df = _safe_read_csv(comparisons_dir / "fairness_evidence_summary.csv")

    canonical_full_df = build_metric_plot_frame(
        experiment_index_df, metric_values_df, metric_deltas_df
    )
    fairness_comparison_full_df = canonical_full_df if canonical_full_df is not None else full_df
    fairness_comparison_group_df = (
        group_metric_deltas_df if group_metric_deltas_df is not None else per_group_df
    )

    _generate_fairness_plots(full_df, run_dir, out_base / "fairness", comparison_config)
    _generate_transformation_plots(run_dir, full_df, out_base / "transformations")
    _generate_cross_model_plots(
        fairness_comparison_group_df, out_base / "cross_model", comparison_config
    )
    _generate_fairness_comparison_plots(
        fairness_comparison_full_df,
        fairness_comparison_group_df,
        evidence_summary_df,
        out_base / "fairness_comparison",
        comparison_config,
    )
    _generate_binning_sensitivity_plots(
        fairness_comparison_full_df,
        fairness_comparison_group_df,
        out_base / "fairness_comparison",
        comparison_config,
    )
    _generate_model_stability_plots(
        run_dir,
        fairness_comparison_full_df,
        out_base / "model_stability",
        comparison_config,
    )
    _generate_cluster_evidence_plots(out_base / "cluster_evidence")

    logger.info("[SUCCESS] Dissertation figures generated: output_dir=%s", out_base)


if __name__ == "__main__":
    main()
