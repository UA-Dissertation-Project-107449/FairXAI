"""Adapters from canonical comparison tables to plot-ready frames."""

from __future__ import annotations

import pandas as pd


def build_metric_plot_frame(
    experiment_index: pd.DataFrame | None,
    metric_values: pd.DataFrame | None,
    metric_deltas: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Build a wide frame used by plotting functions.

    Canonical storage remains long-form. This adapter is the only place where
    dissertation plotting gets a wide, full_comparison-like table.
    """
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
        metric_values.pivot_table(
            index="experiment_id", columns="metric", values="value", aggfunc="first"
        )
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

    if "dp_max_diff" in wide.columns:
        wide["fairness_gap"] = pd.to_numeric(wide["dp_max_diff"], errors="coerce")
        if "eq_odds_max_diff" in wide.columns:
            wide["fairness_gap"] = wide[["fairness_gap", "eq_odds_max_diff"]].max(
                axis=1, skipna=True
            )

    return wide
