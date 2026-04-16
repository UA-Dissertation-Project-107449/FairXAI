"""Compare results across combinatorial experiments."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
# Ensure local experiment helpers (e.g., _gates.py) are importable from wrapper entrypoints.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gates import evaluate_recall_gate, load_gate_thresholds

from fairxai.cli.runner_base import get_project_root, setup_phase_logging
from fairxai.cli.runner_utils import resolve_latest_run_dir, resolve_run_id
from fairxai.experiments.versioning import ExperimentVersioning
from fairxai.viz.experiment_plots import (
    save_comparison_heatmap,
    save_pareto_frontier,
    save_tradeoff_scatter,
)

# Composite score weights (must sum to 1.0)
SCORE_WEIGHTS = {"f1_value": 0.40, "recall_value": 0.30, "accuracy_value": 0.20, "auc_value": 0.10}


def load_all_results(versioning: ExperimentVersioning) -> pd.DataFrame:
    """
    Load all experiment results and combine into DataFrame.

    Args:
        versioning: Versioning system instance

    Returns:
        DataFrame with all experiment results
    """
    experiments = versioning.list_experiments()

    if not experiments:
        logging.warning("No experiments found in latest_run")
        return pd.DataFrame()

    all_results = []

    for exp_summary in experiments:
        exp_id = exp_summary["experiment_id"]

        try:
            # Load full experiment data
            exp_data = versioning.load_experiment(exp_id)

            if exp_data["results"] is None:
                logging.warning(f"No results for experiment {exp_id}, skipping")
                continue

            config = exp_data["manifest"]["configuration"]
            results = exp_data["results"]

            # Extract metrics
            row = {
                "experiment_id": exp_id,
                "dataset": config["dataset"],
                "binning_strategy": config["binning_strategy"],
                "mitigation_technique": config["mitigation_technique"],
                "training_method": config["training_method"],
                "model_type": config.get("model_type", "logistic_regression"),
                "model_variant": config.get("model_variant", "default"),
                "status": results["execution"]["status"],
                "error": results["execution"].get("error"),
            }

            # Add performance metrics
            if results["execution"]["status"] == "success":
                if config["training_method"] == "kfold_cv":
                    # CV results
                    cv_results = results.get("cv_results", {})
                    for metric_name in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
                        if metric_name in cv_results:
                            row[f"{metric_name}_mean"] = cv_results[metric_name]["mean"]
                            row[f"{metric_name}_std"] = cv_results[metric_name]["std"]
                    # Unified metric columns for comparisons
                    if "f1_score" in cv_results:
                        row["f1_value"] = cv_results["f1_score"]["mean"]
                    if "accuracy" in cv_results:
                        row["accuracy_value"] = cv_results["accuracy"]["mean"]
                    if "recall" in cv_results:
                        row["recall_value"] = cv_results["recall"]["mean"]
                    if "precision" in cv_results:
                        row["precision_value"] = cv_results["precision"]["mean"]
                    if "auc_roc" in cv_results:
                        row["auc_value"] = cv_results["auc_roc"]["mean"]
                else:
                    # Single split results
                    test_metrics = results.get("test_metrics", {})
                    for metric_name, value in test_metrics.items():
                        row[metric_name] = value
                    # Unified metric columns for comparisons
                    row["f1_value"] = test_metrics.get("f1_score")
                    row["accuracy_value"] = test_metrics.get("accuracy")
                    row["recall_value"] = test_metrics.get("recall")
                    row["precision_value"] = test_metrics.get("precision")
                    row["auc_value"] = test_metrics.get("auc_roc")

                # Standardize metric columns to reduce missing values
                row["accuracy"] = row.get("accuracy_value", row.get("accuracy"))
                row["precision"] = row.get("precision_value", row.get("precision"))
                row["recall"] = row.get("recall_value", row.get("recall"))
                row["f1_score"] = row.get("f1_value", row.get("f1_score"))
                row["auc_roc"] = row.get("auc_value", row.get("auc_roc"))

                # Add fairness metrics
                fairness = results.get("fairness_metrics", {})
                if fairness:
                    dp_diffs = []
                    eq_diffs = []

                    # New structure: group_fairness to {attr} to {demographic_parity, equalized_odds}
                    group_fairness = fairness.get("group_fairness", {})
                    if group_fairness:
                        for attr, metrics in group_fairness.items():
                            dp = (
                                metrics.get("demographic_parity")
                                if isinstance(metrics, dict)
                                else None
                            )
                            if isinstance(dp, dict):
                                max_diff = dp.get("max_difference", np.nan)
                                row[f"dem_parity_{attr}_max_diff"] = max_diff
                                if pd.notna(max_diff):
                                    dp_diffs.append(max_diff)

                            eq = (
                                metrics.get("equalized_odds") if isinstance(metrics, dict) else None
                            )
                            if isinstance(eq, dict):
                                tpr_diff = eq.get(
                                    "tpr_max_difference", eq.get("tpr_difference", np.nan)
                                )
                                fpr_diff = eq.get(
                                    "fpr_max_difference", eq.get("fpr_difference", np.nan)
                                )
                                row[f"eq_odds_{attr}_tpr_diff"] = tpr_diff
                                row[f"eq_odds_{attr}_fpr_diff"] = fpr_diff
                                if pd.notna(tpr_diff):
                                    eq_diffs.append(tpr_diff)
                                if pd.notna(fpr_diff):
                                    eq_diffs.append(fpr_diff)

                    # Legacy structure: fairness_metrics with demographic_parity / equalized_odds
                    if "demographic_parity" in fairness:
                        for attr, metrics in fairness["demographic_parity"].items():
                            max_diff = metrics.get("max_difference", np.nan)
                            row[f"dem_parity_{attr}_max_diff"] = max_diff
                            if pd.notna(max_diff):
                                dp_diffs.append(max_diff)

                    if "equalized_odds" in fairness:
                        for attr, metrics in fairness["equalized_odds"].items():
                            tpr_diff = metrics.get("tpr_difference", np.nan)
                            fpr_diff = metrics.get("fpr_difference", np.nan)
                            row[f"eq_odds_{attr}_tpr_diff"] = tpr_diff
                            row[f"eq_odds_{attr}_fpr_diff"] = fpr_diff
                            if pd.notna(tpr_diff):
                                eq_diffs.append(tpr_diff)
                            if pd.notna(fpr_diff):
                                eq_diffs.append(fpr_diff)

                    if dp_diffs:
                        row["dp_max_diff"] = float(np.nanmax(dp_diffs))
                    if eq_diffs:
                        row["eq_odds_max_diff"] = float(np.nanmax(eq_diffs))

            all_results.append(row)

        except Exception as e:
            logging.error(f"Failed to load experiment {exp_id}: {e}")
            continue

    return pd.DataFrame(all_results)


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics grouped by key factors."""
    if df.empty:
        return pd.DataFrame()

    # Group by mitigation technique
    summary = df.groupby("mitigation_technique").agg(
        {
            "accuracy_value": ["mean", "std", "min", "max"],
            "recall_value": ["mean", "std", "min", "max"],
            "f1_value": ["mean", "std", "min", "max"],
            "auc_value": ["mean", "std", "min", "max"],
            "score_value": ["mean", "std", "min", "max"],
        }
    )

    return summary


def compare_binning_strategies(df: pd.DataFrame, output_dir: Path, plots_dir: Path = None):
    """
    Create binning strategy comparison table.

    Pivot table: binning_strategy x dataset with performance metrics
    """
    if df.empty:
        logging.warning("No data for binning comparison")
        return

    # Use appropriate metric columns based on training method
    metric_col = "score_value"

    # Pivot: rows=binning, cols=dataset, values=avg F1
    pivot = df.pivot_table(
        index="binning_strategy", columns="dataset", values=metric_col, aggfunc="mean"
    )

    # Save CSV in data/, heatmap PNG in plots/
    output_file = output_dir / "binning_summary.csv"
    pivot.to_csv(output_file)
    logging.info(f"[SUCCESS] Saved binning summary: {output_file}")

    heatmap_dir = plots_dir if plots_dir else output_dir
    heatmap_file = heatmap_dir / "binning_comparison_heatmap.png"
    save_comparison_heatmap(
        pivot, title="Binning Strategy Comparison (Composite Score)", output_file=heatmap_file
    )
    logging.info(f"[SUCCESS] Saved binning heatmap: {heatmap_file}")

    return pivot


def compare_mitigation_techniques(df: pd.DataFrame, output_dir: Path, plots_dir: Path = None):
    """
    Create mitigation technique comparison table.

    Pivot table: mitigation_technique x dataset with performance metrics
    """
    if df.empty:
        logging.warning("No data for mitigation comparison")
        return

    metric_col = "score_value"

    # Pivot: rows=mitigation, cols=dataset, values=avg F1
    pivot = df.pivot_table(
        index="mitigation_technique", columns="dataset", values=metric_col, aggfunc="mean"
    )

    # Save CSV in data/, heatmap PNG in plots/
    output_file = output_dir / "mitigation_summary.csv"
    pivot.to_csv(output_file)
    logging.info(f"[SUCCESS] Saved mitigation summary: {output_file}")

    heatmap_dir = plots_dir if plots_dir else output_dir
    heatmap_file = heatmap_dir / "mitigation_comparison_heatmap.png"
    save_comparison_heatmap(
        pivot, title="Mitigation Technique Comparison (Composite Score)", output_file=heatmap_file
    )
    logging.info(f"[SUCCESS] Saved mitigation heatmap: {heatmap_file}")

    return pivot


def filter_best_configurations(
    df: pd.DataFrame,
    output_dir: Path,
    gate_thresholds: dict,
    performance_threshold: float = 0.15,
):
    """Filter configurations using fairness-first gate logic (consistent with combinatorial runner).

    Gates (applied as hard exclusions, config-driven):
     1. Recall hard floor: exclude when recall < ``recall_hard_floor``
     2. Fairness gate: exclude when ``fairness_gap > max_fairness_violation``
     3. Performance drop: exclude when
         ``(baseline_score - score) / baseline_score > performance_threshold``

    When strict gates filter everything, a fallback CSV is emitted (ranked by
    fairness_gap asc, score desc) tagged with ``selection_mode='fallback'``.
    The two output files are kept strictly separate.
    """
    if df.empty:
        logging.warning("No data for filtering")
        return pd.DataFrame()

    recall_hard_floor = gate_thresholds["recall_hard_floor"]
    min_recall = gate_thresholds["min_recall"]
    max_fairness_violation = gate_thresholds["max_fairness_violation"]

    baseline_df = df[df["mitigation_technique"] == "baseline"]
    best_configs = []

    key_pairs = df[["dataset", "model_type"]].drop_duplicates().itertuples(index=False, name=None)

    for dataset, model_type in key_pairs:
        dataset_df = df[(df["dataset"] == dataset) & (df["model_type"] == model_type)].copy()
        baseline_row = baseline_df[
            (baseline_df["dataset"] == dataset) & (baseline_df["model_type"] == model_type)
        ]

        if baseline_row.empty:
            logging.warning(f"No baseline for {dataset}/{model_type}, skipping")
            continue

        baseline_score = baseline_row["score_value"].values[0]
        mitigated_df = dataset_df[dataset_df["mitigation_technique"] != "baseline"]

        for _, row in mitigated_df.iterrows():
            score = row["score_value"]
            fairness_gap = row.get("fairness_gap")
            recall = row.get("recall_value")

            if baseline_score == 0 or pd.isna(baseline_score) or pd.isna(score):
                continue

            performance_drop = (baseline_score - score) / baseline_score

            # Gate 1: recall hard floor (two-tier: record tier, hard-exclude below floor)
            recall_gate = evaluate_recall_gate(recall, recall_hard_floor, min_recall)
            if not recall_gate.passed:
                continue  # hard exclusion

            # Gate 2: fairness gate
            if fairness_gap is None or pd.isna(fairness_gap):
                continue
            if fairness_gap > max_fairness_violation:
                continue

            # Gate 3: performance threshold
            if performance_drop > performance_threshold:
                continue

            best_configs.append(
                {
                    "dataset": dataset,
                    "model_type": model_type,
                    "model_variant": row.get("model_variant", "default"),
                    "binning": row["binning_strategy"],
                    "mitigation": row["mitigation_technique"],
                    "training_method": row["training_method"],
                    "score": score,
                    "baseline_score": baseline_score,
                    "score_drop_pct": performance_drop * 100,
                    "fairness_gap": fairness_gap,
                    "baseline_fairness_gap": row.get("baseline_fairness_gap"),
                    "fairness_gain_score": row.get("fairness_gain_score"),
                    "fairness_gain_pct": row.get("fairness_gain_pct"),
                    "f1_score": row.get("f1_value"),
                    "recall": recall,
                    "accuracy": row.get("accuracy_value"),
                    "auc_roc": row.get("auc_value"),
                    "recall_tier": recall_gate.tier,
                    "selection_mode": "strict",
                    "experiment_id": row["experiment_id"],
                }
            )

    best_df = pd.DataFrame(best_configs)

    if not best_df.empty:
        best_df = best_df.sort_values(["dataset", "score"], ascending=[True, False])
        best_df["is_fallback"] = False
        output_file = output_dir / "top_configs.csv"
        best_df.to_csv(output_file, index=False)
        logging.info(f"[SUCCESS] Saved top configurations: {output_file}")
        logging.info(f"  Found {len(best_df)} configurations meeting strict gates")
        return best_df

    # Fallback: no configs passed strict gates; emit fallback shortlist merged into top_configs
    logging.warning(
        "No configurations passed strict gates "
        f"(recall_hard_floor={recall_hard_floor:.2f}, "
        f"max_fairness_violation={max_fairness_violation:.2f}, "
        f"performance_threshold={performance_threshold:.2f}). "
        "Emitting fallback shortlist ranked by fairness_gap."
    )
    fallback_rows = []
    non_baseline = df[df["mitigation_technique"] != "baseline"].copy()
    for _, row in non_baseline.iterrows():
        recall = row.get("recall_value")
        fairness_gap = row.get("fairness_gap")
        score = row.get("score_value")
        if pd.isna(score):
            continue
        fallback_rows.append(
            {
                "dataset": row["dataset"],
                "model_type": row.get("model_type", ""),
                "model_variant": row.get("model_variant", "default"),
                "binning": row["binning_strategy"],
                "mitigation": row["mitigation_technique"],
                "training_method": row["training_method"],
                "score": score,
                "fairness_gap": fairness_gap,
                "recall": recall,
                "f1_score": row.get("f1_value"),
                "selection_mode": "fallback",
                "is_fallback": True,
                "experiment_id": row["experiment_id"],
            }
        )

    fallback_df = pd.DataFrame(fallback_rows)
    if not fallback_df.empty:
        fallback_df = fallback_df.sort_values(
            ["dataset", "fairness_gap", "score"],
            ascending=[True, True, False],
        )
        # Write merged top_configs with is_fallback=True (no separate file)
        top_configs_file = output_dir / "top_configs.csv"
        fallback_df.to_csv(top_configs_file, index=False)
        logging.warning(f"[FALLBACK] Saved fallback configs as top_configs: {top_configs_file}")

    return best_df  # empty DataFrame (strict set was empty)


def _extract_per_group_fairness(fairness_metrics: dict) -> list:
    """Flatten per-group fairness data from a calculate_all_metrics() result dict.

    Returns a list of records:
        {sensitive_attr, group, metric, value}

    Covers group_fairness to {attr} to {demographic_parity, equalized_odds,
    equal_opportunity, predictive_parity} to group_rates / group_metrics.
    """
    records = []
    group_fairness = fairness_metrics.get("group_fairness", {}) if fairness_metrics else {}
    for attr, attr_metrics in group_fairness.items():
        if not isinstance(attr_metrics, dict):
            continue

        dp = attr_metrics.get("demographic_parity", {})
        if isinstance(dp, dict):
            for group, gdata in dp.get("group_rates", {}).items():
                if isinstance(gdata, dict):
                    records.append(
                        {
                            "sensitive_attr": attr,
                            "group": str(group),
                            "metric": "demographic_parity_rate",
                            "value": gdata.get("positive_rate"),
                        }
                    )

        eq = attr_metrics.get("equalized_odds", {})
        if isinstance(eq, dict):
            for group, gdata in eq.get("group_metrics", {}).items():
                if isinstance(gdata, dict):
                    records.append(
                        {
                            "sensitive_attr": attr,
                            "group": str(group),
                            "metric": "tpr",
                            "value": gdata.get("tpr"),
                        }
                    )
                    records.append(
                        {
                            "sensitive_attr": attr,
                            "group": str(group),
                            "metric": "fpr",
                            "value": gdata.get("fpr"),
                        }
                    )

        eqopp = attr_metrics.get("equal_opportunity", {})
        if isinstance(eqopp, dict):
            for group, tpr_val in eqopp.get("group_tpr", {}).items():
                records.append(
                    {
                        "sensitive_attr": attr,
                        "group": str(group),
                        "metric": "equal_opportunity_tpr",
                        "value": tpr_val if not isinstance(tpr_val, dict) else tpr_val.get("tpr"),
                    }
                )

        pp = attr_metrics.get("predictive_parity", {})
        if isinstance(pp, dict):
            for group, prec_val in pp.get("group_precision", {}).items():
                records.append(
                    {
                        "sensitive_attr": attr,
                        "group": str(group),
                        "metric": "predictive_parity_precision",
                        "value": (
                            prec_val
                            if not isinstance(prec_val, dict)
                            else prec_val.get("precision")
                        ),
                    }
                )
    return records


def _load_baseline_per_group(run_root: Path, dataset: str, model_type: str) -> list:
    """Load per-group fairness records from the stage-6 baseline assessment JSON.

    Returns records in the same format as _extract_per_group_fairness(), with an
    extra source='baseline_assess' field so they can be distinguished.
    Falls back to an empty list if the JSON is absent or malformed.
    """
    # Stage 6 writes to: {run_root}/baseline/prediction_fairness/fairness_report.json
    # nested as {dataset: {model_type: {train_metrics: ..., test_metrics: ...}}}
    json_path = run_root / "baseline" / "prediction_fairness" / "fairness_report.json"
    if not json_path.exists():
        logging.debug(f"Baseline fairness report not found: {json_path}")
        return []
    try:
        with open(json_path) as f:
            data = json.load(f)
        # Navigate nested structure: data[dataset][model_type]
        model_data = data.get(dataset, {}).get(model_type, {})
        if not model_data:
            logging.debug(f"No fairness data for {dataset}/{model_type} in {json_path}")
            return []
        # Use test_metrics as the reference
        test_metrics = model_data.get("test_metrics", {})
        records = _extract_per_group_fairness(test_metrics)
        for r in records:
            r["source"] = "baseline_assess"
        return records
    except Exception as exc:
        logging.warning(f"Failed to load baseline per-group JSON {json_path}: {exc}")
        return []


def _build_per_group_comparison(
    df_success: pd.DataFrame,
    versioning: "ExperimentVersioning",
    run_root: Path,
    output_dir: Path,
) -> None:
    """Build and save per_group_comparison.csv.

    For each experiment, loads per-group fairness from its result JSON and pairs it
    with the stage-6 baseline per-group JSON for the same (dataset, model_type).
    Output columns:
        dataset, model_type, binning_strategy, training_method, mitigation_technique,
        model_variant, sensitive_attr, group, metric,
        baseline_value, experiment_value, delta
    """
    # Pre-load baseline per-group data keyed by (dataset, model_type)
    baseline_pg: dict = {}
    for (dataset, model_type), _ in df_success.groupby(["dataset", "model_type"], dropna=False):
        key = (dataset, model_type)
        if key not in baseline_pg:
            records = _load_baseline_per_group(run_root, dataset, model_type)
            baseline_pg[key] = {
                (r["sensitive_attr"], r["group"], r["metric"]): r["value"] for r in records
            }

    rows = []
    for _, exp_row in df_success.iterrows():
        if exp_row.get("status") != "success":
            continue
        exp_id = exp_row["experiment_id"]
        dataset = exp_row["dataset"]
        model_type = exp_row.get("model_type", "logistic_regression")
        baseline_lookup_key = (dataset, model_type)

        try:
            exp_data = versioning.load_experiment(exp_id)
            fairness_metrics = (
                exp_data["results"].get("fairness_metrics", {}) if exp_data.get("results") else {}
            )
        except Exception:
            continue

        pg_records = _extract_per_group_fairness(fairness_metrics)
        baseline_map = baseline_pg.get(baseline_lookup_key, {})

        for rec in pg_records:
            baseline_val = baseline_map.get((rec["sensitive_attr"], rec["group"], rec["metric"]))
            exp_val = rec["value"]
            delta = (
                (exp_val - baseline_val)
                if (exp_val is not None and baseline_val is not None)
                else None
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model_type": model_type,
                    "binning_strategy": exp_row["binning_strategy"],
                    "training_method": exp_row["training_method"],
                    "mitigation_technique": exp_row["mitigation_technique"],
                    "model_variant": exp_row.get("model_variant", ""),
                    "sensitive_attr": rec["sensitive_attr"],
                    "group": rec["group"],
                    "metric": rec["metric"],
                    "baseline_value": baseline_val,
                    "experiment_value": exp_val,
                    "delta": delta,
                }
            )

    if not rows:
        logging.warning(
            "No per-group comparison rows generated (stage-6 baseline JSONs may be absent)"
        )
        return

    pg_df = pd.DataFrame(rows)
    pg_csv = output_dir / "per_group.csv"
    pg_df.to_csv(pg_csv, index=False)
    logging.info(f"[SUCCESS] Saved per-group comparison: {pg_csv} ({len(pg_df)} rows)")


def _promote_top_n_models(versioning, df_success: "pd.DataFrame", save_top_n: int = 10) -> None:
    """Move top-N experiment model PKLs from models/_temp/ to models/ and write index."""
    import json
    import shutil

    if save_top_n <= 0:
        return

    temp_dir = versioning.latest_dir / "models" / "_temp"
    models_dir = versioning.latest_dir / "models"

    if not temp_dir.exists():
        logging.debug("[TOP_N] No _temp/ dir found — model saving may not be enabled")
        return

    # Rank by composite score, take top N
    rank_df = df_success[
        [
            "experiment_id",
            "score_value",
            "dataset",
            "model_type",
            "binning_strategy",
            "mitigation_technique",
        ]
    ].copy()
    rank_df = rank_df.dropna(subset=["score_value"]).sort_values("score_value", ascending=False)
    top_ids = rank_df.head(save_top_n)["experiment_id"].tolist()

    index_entries = []
    promoted = 0
    for rank, exp_id in enumerate(top_ids, start=1):
        src = temp_dir / f"{exp_id}.pkl"
        if not src.exists():
            logging.debug(f"[TOP_N] No temp model for {exp_id}, skipping")
            continue
        dst = models_dir / f"{exp_id}.pkl"
        shutil.move(str(src), str(dst))
        row = rank_df[rank_df["experiment_id"] == exp_id].iloc[0]
        index_entries.append(
            {
                "rank": rank,
                "experiment_id": exp_id,
                "composite_score": float(row["score_value"]),
                "dataset": row.get("dataset", ""),
                "model_type": row.get("model_type", ""),
                "binning": row.get("binning_strategy", ""),
                "mitigation": row.get("mitigation_technique", ""),
            }
        )
        promoted += 1

    # Clean up remaining temp models
    shutil.rmtree(str(temp_dir), ignore_errors=True)

    if index_entries:
        index_path = models_dir / "top_models.json"
        with open(index_path, "w") as f:
            json.dump(index_entries, f, indent=2, default=str)
        logging.info(f"[TOP_N] Promoted {promoted}/{save_top_n} models to {models_dir}")
        logging.info(f"[TOP_N] Index saved: {index_path}")
    else:
        logging.warning("[TOP_N] No temp models found to promote")


def run_comparison_analysis(
    results_dir: str = None,
    pipeline: str = "cardiac",
    fairness_threshold: float = None,
    performance_threshold: float = 0.15,
    no_plots: bool = False,
    verbose: int = 0,
    run_id: str = None,
    output_root: str = None,
    save_top_n: int = 10,
):
    """Main comparison script."""
    project_root = get_project_root(Path(__file__))
    use_run_id = bool(run_id or os.getenv("RUN_ID") or os.getenv("PREFECT__RUNTIME__FLOW_RUN_ID"))
    run_id = resolve_run_id(run_id) if use_run_id else None
    setup_phase_logging(
        project_root,
        "experiment_comparison.log",
        verbose=verbose,
        run_id=run_id,
        stage_name="compare",
    )

    # Load canonical gate thresholds; CLI --fairness-threshold overrides max_fairness_violation.
    gate_thresholds = load_gate_thresholds({}, project_root)
    if fairness_threshold is not None:
        gate_thresholds["max_fairness_violation"] = float(fairness_threshold)

    logging.info("[PHASE] Comparison started")
    logging.info(
        f"Gate thresholds: recall_hard_floor={gate_thresholds['recall_hard_floor']:.2f}, "
        f"min_recall={gate_thresholds['min_recall']:.2f}, "
        f"max_fairness_violation={gate_thresholds['max_fairness_violation']:.2f}"
    )

    base_output_dir = Path(output_root) if output_root else (project_root / f"output/{pipeline}")
    if results_dir:
        candidate = Path(results_dir)
        if (candidate / "manifests").exists() or (candidate / "results").exists():
            run_dir = candidate
            base_output_dir = candidate.parent
        else:
            base_output_dir = candidate
            run_dir = resolve_latest_run_dir(base_output_dir)
    elif run_id:
        run_dir = base_output_dir / "runs" / run_id / "experiments"
    else:
        run_dir = resolve_latest_run_dir(base_output_dir)

    if run_dir is not None and not (run_dir / "manifests").exists():
        candidate = run_dir / "experiments"
        if (candidate / "manifests").exists() or (candidate / "results").exists():
            run_dir = candidate

    logging.info(
        f"[RUN_CONTEXT] pipeline={pipeline} run_id={run_id or 'none'} "
        f"base_output_dir={base_output_dir} run_dir={run_dir if run_dir else 'not_found'}"
    )

    if run_dir is None or not run_dir.exists():
        logging.error(f"No run directory found under {base_output_dir}")
        logging.error("Run combinatorial experiments first")
        return

    # Initialize versioning
    versioning = ExperimentVersioning(base_output_dir, run_dir=run_dir)

    # Derive run_root: the directory that contains baseline/, experiments/, etc.
    # run_dir is typically {run_root}/experiments.
    run_root = run_dir.parent if run_dir else None

    # Load all results
    logging.info("Loading experiment results...")
    df = load_all_results(versioning)

    if df.empty:
        logging.error("No results loaded")
        return

    logging.info(
        f"Loaded experiments: total={len(df)} "
        f"successful={(df['status'] == 'success').sum()} "
        f"failed={(df['status'] == 'failed').sum()}"
    )

    # Filter successful experiments
    df_success = df[df["status"] == "success"].copy()

    if df_success.empty:
        logging.error("No successful experiments to analyze")
        return

    # Create output directories
    output_dir = versioning.latest_dir / "comparisons"
    output_dir.mkdir(exist_ok=True)
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # Compute composite score for ranking
    for metric, weight in SCORE_WEIGHTS.items():
        if metric not in df_success.columns:
            df_success[metric] = np.nan

    df_success["score_value"] = (
        df_success["f1_value"] * SCORE_WEIGHTS["f1_value"]
        + df_success["recall_value"] * SCORE_WEIGHTS["recall_value"]
        + df_success["accuracy_value"] * SCORE_WEIGHTS["accuracy_value"]
        + df_success["auc_value"] * SCORE_WEIGHTS["auc_value"]
    )

    # Compute fairness gap for trade-off analysis
    fairness_cols = ["dp_max_diff", "eq_odds_max_diff"]
    for col in fairness_cols:
        if col not in df_success.columns:
            df_success[col] = np.nan
    df_success["fairness_gap"] = df_success[fairness_cols].max(axis=1, skipna=True)

    # Compute fairness gains per metric vs baseline
    # Key must include model_type so LR baseline is not confused with RF baseline.
    fairness_metric_cols = [
        c for c in df_success.columns if c.startswith("dem_parity_") or c.startswith("eq_odds_")
    ]

    baseline_lookup = {}
    for _, row in df_success[df_success["mitigation_technique"] == "baseline"].iterrows():
        key = (
            row["dataset"],
            row.get("model_type", "logistic_regression"),
            row["binning_strategy"],
            row["training_method"],
        )
        baseline_lookup[key] = row

    for col in fairness_metric_cols:
        gain_col = f"gain_{col}"
        df_success[gain_col] = np.nan

    df_success["baseline_fairness_gap"] = np.nan
    df_success["fairness_gain_score"] = np.nan
    df_success["fairness_gain_pct"] = np.nan

    for idx, row in df_success.iterrows():
        key = (
            row["dataset"],
            row.get("model_type", "logistic_regression"),
            row["binning_strategy"],
            row["training_method"],
        )
        baseline = baseline_lookup.get(key)
        if baseline is None:
            continue

        gains = []
        for col in fairness_metric_cols:
            base_val = baseline.get(col)
            curr_val = row.get(col)
            if pd.isna(base_val) or pd.isna(curr_val):
                continue
            gain = base_val - curr_val
            df_success.at[idx, f"gain_{col}"] = gain
            gains.append(gain)

        base_gap = baseline.get("fairness_gap")
        df_success.at[idx, "baseline_fairness_gap"] = base_gap

        if gains:
            gain_score = float(np.mean(gains))
            df_success.at[idx, "fairness_gain_score"] = gain_score
            if base_gap and base_gap > 0:
                df_success.at[idx, "fairness_gain_pct"] = gain_score / base_gap

    # Mark pareto-optimal experiments per dataset (non-dominated on score_value vs fairness_gap)
    df_success["is_pareto"] = False
    for dataset in df_success["dataset"].unique():
        mask = df_success["dataset"] == dataset
        sub = df_success.loc[mask, ["score_value", "fairness_gap"]].copy()
        for idx in sub.index:
            sv, fg = sub.at[idx, "score_value"], sub.at[idx, "fairness_gap"]
            if pd.isna(sv) or pd.isna(fg):
                continue
            dominated = sub[
                (sub["score_value"] >= sv)
                & (sub["fairness_gap"] <= fg)
                & ~((sub["score_value"] == sv) & (sub["fairness_gap"] == fg))
            ]
            if dominated.empty:
                df_success.at[idx, "is_pareto"] = True

    # Save full results table (in data/ subdir)
    full_results_file = data_dir / "full_comparison.csv"
    df_success.to_csv(full_results_file, index=False)
    logging.info(f"[SUCCESS] Saved full results: {full_results_file}")

    # Create comparison tables
    logging.info("Generating comparison tables...")

    compare_binning_strategies(df_success, data_dir, plots_dir=plots_dir)
    compare_mitigation_techniques(df_success, data_dir, plots_dir=plots_dir)

    # Filter best configurations (writes to data_dir now)
    logging.info("Filtering best configurations...")
    best_configs = filter_best_configurations(
        df_success,
        data_dir,
        gate_thresholds,
        performance_threshold,
    )

    # Trade-off visuals (per dataset, then per dataset x model_type)
    if not no_plots:
        for dataset in sorted(df_success["dataset"].unique()):
            subset = df_success[df_success["dataset"] == dataset].copy()
            if subset.empty:
                continue

            # Tradeoff: CSV in data/, PNG in plots/
            tradeoff_csv = data_dir / f"tradeoff_{dataset}.csv"
            subset.to_csv(tradeoff_csv, index=False)

            tradeoff_png = plots_dir / f"tradeoff_{dataset}.png"
            tradeoff_result = save_tradeoff_scatter(
                subset,
                x_col="score_value",
                y_col="fairness_gap",
                hue_col="mitigation_technique",
                style_col="training_method",
                title=f"{dataset} - Performance vs Fairness",
                output_file=tradeoff_png,
            )
            if tradeoff_result:
                logging.info(f"[SUCCESS] Saved tradeoff plot: {tradeoff_png}")
            else:
                logging.warning(f"Tradeoff plot skipped (no data): {tradeoff_png}")

            # Pareto: subset to pareto-flagged rows; CSV in data/, PNG in plots/
            pareto_subset = (
                subset[subset["is_pareto"]].copy() if "is_pareto" in subset.columns else subset
            )
            pareto_csv = data_dir / f"pareto_{dataset}.csv"
            pareto_subset.to_csv(pareto_csv, index=False)

            pareto_png = plots_dir / f"pareto_{dataset}.png"
            pareto_result = save_pareto_frontier(
                subset,
                x_col="score_value",
                y_col="fairness_gap",
                title=f"{dataset} - Pareto Frontier",
                output_file=pareto_png,
            )
            if pareto_result:
                logging.info(f"[SUCCESS] Saved pareto plot: {pareto_png}")
            else:
                logging.warning(f"Pareto plot skipped (no data): {pareto_png}")

            # Per-model Pareto plots: one per (dataset, model_type) — PNG only in plots/
            for model_type in sorted(subset["model_type"].dropna().unique()):
                model_subset = subset[subset["model_type"] == model_type].copy()
                if model_subset.empty or model_subset["fairness_gap"].isna().all():
                    continue
                pareto_model_png = plots_dir / f"pareto_{dataset}_{model_type}.png"
                pareto_model_result = save_pareto_frontier(
                    model_subset,
                    x_col="score_value",
                    y_col="fairness_gap",
                    title=f"{dataset} / {model_type} - Pareto Frontier",
                    output_file=pareto_model_png,
                )
                if pareto_model_result:
                    logging.info(f"[SUCCESS] Saved per-model pareto: {pareto_model_png}")
                else:
                    logging.debug(f"Per-model pareto skipped (no data): {pareto_model_png}")

    # Summary outputs
    summary_rows = []
    for dataset in sorted(df_success["dataset"].unique()):
        subset = df_success[df_success["dataset"] == dataset].copy()
        if subset.empty:
            continue
        best_score = subset.sort_values("score_value", ascending=False).head(1)
        best_gain = subset.sort_values("fairness_gain_score", ascending=False).head(1)
        summary_rows.append(
            {
                "dataset": dataset,
                "best_score_experiment": best_score["experiment_id"].values[0],
                "best_score": best_score["score_value"].values[0],
                "best_score_fairness_gap": best_score["fairness_gap"].values[0],
                "best_gain_experiment": best_gain["experiment_id"].values[0],
                "best_fairness_gain": best_gain["fairness_gain_score"].values[0],
                "best_gain_score": best_gain["score_value"].values[0],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = data_dir / "top_configs.csv"
    summary_df.to_csv(summary_csv, index=False)
    logging.info(f"[SUCCESS] Saved summary: {summary_csv}")

    # Cross-model summary: best config per (dataset, model_type)
    cross_model_rows = []
    for (dataset, model_type), group in df_success.groupby(["dataset", "model_type"], dropna=False):
        if group.empty:
            continue
        best = group.sort_values("score_value", ascending=False).iloc[0]
        cross_model_rows.append(
            {
                "dataset": dataset,
                "model_type": model_type,
                "best_mitigation": best["mitigation_technique"],
                "best_binning": best["binning_strategy"],
                "best_training_method": best["training_method"],
                "best_model_variant": best.get("model_variant", ""),
                "score": best["score_value"],
                "accuracy": best.get("accuracy_value", best.get("accuracy")),
                "f1": best.get("f1_value"),
                "f1_score": best.get("f1_value"),
                "recall": best.get("recall_value"),
                "precision": best.get("precision_value", best.get("precision")),
                "auc_roc": best.get("auc_value", best.get("auc_roc")),
                "fairness_gap": best.get("fairness_gap"),
                "fairness_gain_pct": best.get("fairness_gain_pct"),
                "experiment_id": best["experiment_id"],
            }
        )
    cross_model_df = pd.DataFrame(cross_model_rows)
    cross_model_csv = data_dir / "cross_model_summary.csv"
    cross_model_df.to_csv(cross_model_csv, index=False)
    logging.info(f"[SUCCESS] Saved cross-model summary: {cross_model_csv}")

    # Per-subgroup before/after comparison (requires stage-6 baseline JSONs)
    if run_root is not None and run_root.exists():
        _build_per_group_comparison(df_success, versioning, run_root, data_dir)
    else:
        logging.debug("Skipping per-group comparison: run_root not available")

    # Promote top N models from _temp/ to models/
    _promote_top_n_models(versioning, df_success, save_top_n=save_top_n)

    # Print summary
    logging.info("[PHASE] Comparison complete")
    logging.info(
        f"[SUMMARY] output_dir={output_dir} successful_experiments={len(df_success)} "
        f"best_configs={0 if best_configs is None else len(best_configs)}"
    )
    if best_configs is not None and not best_configs.empty:
        logging.info(f"Best configurations saved: count={len(best_configs)}")


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(description="Compare combinatorial experiment results")
    parser.add_argument("--results-dir", type=str, default=None, help="Base results directory")
    parser.add_argument(
        "--run-id", type=str, default=os.getenv("RUN_ID"), help="Run identifier to compare"
    )
    parser.add_argument(
        "--output-root", type=str, default=None, help="Base output directory for run outputs"
    )
    parser.add_argument(
        "--pipeline", type=str, default="cardiac", help="Pipeline name (e.g., cardiac, dermatology)"
    )
    parser.add_argument(
        "--fairness-threshold",
        type=float,
        default=None,
        help="Override max_fairness_violation gate (default: from thresholds.yaml)",
    )
    parser.add_argument(
        "--performance-threshold",
        type=float,
        default=0.15,
        help="Maximum performance drop (15%% default)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument(
        "--save-top-n",
        type=int,
        default=10,
        help="Number of top-ranked experiment models to promote from _temp/ (0 = skip)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug"
    )

    args = parser.parse_args()

    run_comparison_analysis(
        results_dir=args.results_dir,
        pipeline=args.pipeline,
        fairness_threshold=args.fairness_threshold,
        performance_threshold=args.performance_threshold,
        no_plots=args.no_plots,
        verbose=args.verbose,
        run_id=args.run_id,
        output_root=args.output_root,
        save_top_n=args.save_top_n,
    )


if __name__ == "__main__":
    main()
