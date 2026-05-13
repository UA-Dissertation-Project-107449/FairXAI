"""
Experiment comparison plots — heatmaps, scatter trade-offs, Pareto frontiers,
cross-model radar, intersectional fairness, and mitigation effectiveness.

Migrated from the legacy ``fairxai.visualization.plots`` module.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Colour palette per model type (colour-blind friendly)
PALETTE_MODEL = {
    "logistic_regression": "#0072B2",
    "random_forest": "#009E73",
    "svm": "#D55E00",
    "xgboost": "#CC79A7",
}

MITIGATION_LABELS = {
    "baseline": "Baseline",
    "adasyn": "ADASYN",
    "adasyn+exponentiated_gradient": "ADASYN + EG",
    "adasyn+exponentiated_gradient+threshold_optimizer": "ADASYN + EG + TO",
    "exponentiated_gradient": "Exp. Gradient",
    "grid_search": "Fairlearn Grid",
    "reweighting": "Reweighting",
    "reweighting+threshold_optimizer": "Reweighting + TO",
    "smote": "SMOTE",
    "smote+exponentiated_gradient": "SMOTE + EG",
    "smote+exponentiated_gradient+threshold_optimizer": "SMOTE + EG + TO",
    "smote+threshold_optimizer": "SMOTE + TO",
    "threshold_optimizer": "Threshold Opt.",
}


def _display_mitigation(value: object) -> str:
    value = str(value)
    if value in MITIGATION_LABELS:
        return MITIGATION_LABELS[value]
    return value.replace("_", " ").replace("+", " + ").title()


def _normalize_sensitive_attr(attr: object) -> str:
    attr_str = str(attr)
    if attr_str.endswith("_cat"):
        return attr_str[: -len("_cat")]
    return attr_str


def _pretty_group_label(attr: object, group: object) -> str:
    group_str = str(group)
    if _normalize_sensitive_attr(attr) == "sex":
        return {"0": "Female", "1": "Male"}.get(group_str, group_str)
    return group_str


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _max_from_columns(row: pd.Series, columns: list[str]) -> float:
    vals = [row.get(col) for col in columns if col in row.index]
    vals = [float(v) for v in vals if pd.notna(v)]
    if not vals:
        return float("nan")
    return float(np.nanmax(vals))


def _gap_specs(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    dp_cols = [c for c in df.columns if c.startswith("dem_parity_") and c.endswith("_max_diff")]
    eq_tpr_cols = [c for c in df.columns if c.startswith("eq_odds_") and "_tpr_" in c]
    eq_fpr_cols = [c for c in df.columns if c.startswith("eq_odds_") and "_fpr_" in c]
    return dp_cols, eq_tpr_cols, eq_fpr_cols


def _baseline_match(full_df: pd.DataFrame, row: pd.Series) -> pd.Series | None:
    if full_df is None or full_df.empty:
        return None
    required = ["dataset", "model_type", "binning_strategy", "training_method"]
    if not all(col in full_df.columns and col in row.index for col in required):
        return None

    mask = full_df["mitigation_technique"].astype(str).eq("baseline")
    for col in required:
        mask &= full_df[col].astype(str).eq(str(row.get(col)))
    if "model_variant" in full_df.columns and "model_variant" in row.index:
        exact = mask & full_df["model_variant"].astype(str).eq(str(row.get("model_variant")))
        if exact.any():
            return full_df[exact].iloc[0]
    if mask.any():
        return full_df[mask].iloc[0]
    return None


def select_primary_fairness_row(
    full_df: pd.DataFrame,
    model_type: str = "logistic_regression",
    min_recall_delta: float = -0.03,
) -> pd.Series | None:
    """Select primary mitigation row for before/after evidence plots."""
    if full_df is None or full_df.empty or "mitigation_technique" not in full_df.columns:
        return None

    df = full_df.copy()
    if "model_type" in df.columns:
        df = df[df["model_type"] == model_type]
    df = df[df["mitigation_technique"] != "baseline"].copy()
    if df.empty:
        return None

    for col in ["delta_fairness_gap", "delta_recall", "delta_f1"]:
        df[col] = _numeric_series(df, col)

    eligible = df[(df["delta_fairness_gap"] > 0) & (df["delta_recall"] >= min_recall_delta)]
    if eligible.empty:
        eligible = df[df["delta_fairness_gap"] > 0]
    if eligible.empty:
        eligible = df

    eligible = eligible.sort_values(
        ["delta_fairness_gap", "delta_recall", "delta_f1"],
        ascending=[False, False, False],
        na_position="last",
    )
    return eligible.iloc[0] if not eligible.empty else None


def _radar_values(row: pd.Series, df_columns: list[str]) -> list[float]:
    dp_cols, eq_tpr_cols, eq_fpr_cols = _gap_specs(pd.DataFrame(columns=df_columns))
    dp_gap = row.get("dp_max_diff", _max_from_columns(row, dp_cols))
    if pd.isna(dp_gap):
        dp_gap = _max_from_columns(row, dp_cols)
    eq_tpr_gap = _max_from_columns(row, eq_tpr_cols)
    eq_fpr_gap = _max_from_columns(row, eq_fpr_cols)
    values = [
        row.get("f1_value", row.get("f1_score", row.get("f1"))),
        row.get("recall_value", row.get("recall")),
        row.get("precision_value", row.get("precision")),
        row.get("auc_value", row.get("auc_roc")),
        1 - dp_gap if pd.notna(dp_gap) else np.nan,
        1 - eq_tpr_gap if pd.notna(eq_tpr_gap) else np.nan,
        1 - eq_fpr_gap if pd.notna(eq_fpr_gap) else np.nan,
    ]
    return [float(np.clip(v, 0, 1)) if pd.notna(v) else np.nan for v in values]


def save_before_after_metric_radar(
    full_df: pd.DataFrame,
    output_file,
    selected_row: pd.Series | None = None,
):
    """Landscape radar: matched baseline vs selected mitigation, metric-by-metric."""
    if full_df is None or full_df.empty:
        return None
    selected = selected_row if selected_row is not None else select_primary_fairness_row(full_df)
    if selected is None:
        return None
    baseline = _baseline_match(full_df, selected)
    if baseline is None:
        return None

    categories = [
        "F1",
        "Recall",
        "Precision",
        "AUC-ROC",
        "DP Fairness",
        "EO TPR Fairness",
        "EO FPR Fairness",
    ]
    base_values = _radar_values(baseline, list(full_df.columns))
    after_values = _radar_values(selected, list(full_df.columns))
    if np.isnan(base_values).all() or np.isnan(after_values).all():
        return None

    n = len(categories)
    angles = [i * 2 * np.pi / n for i in range(n)] + [0]
    base_closed = base_values + [base_values[0]]
    after_closed = after_values + [after_values[0]]

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, polar=True)
    info_ax = fig.add_subplot(1, 2, 2)
    info_ax.axis("off")

    ax.plot(angles, base_closed, color="#666666", linewidth=2, label="Baseline")
    ax.fill(angles, base_closed, color="#666666", alpha=0.12)
    ax.plot(angles, after_closed, color="#0072B2", linewidth=2.5, label="After Mitigation")
    ax.fill(angles, after_closed, color="#0072B2", alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)

    deltas = np.array(after_values) - np.array(base_values)
    table_rows = [
        [label, f"{base:.3f}", f"{after:.3f}", f"{delta * 100:+.1f} pp"]
        for label, base, after, delta in zip(categories, base_values, after_values, deltas)
    ]
    table = info_ax.table(
        cellText=table_rows,
        colLabels=["Metric", "Baseline", "After", "Delta"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    title = (
        "Baseline vs Mitigation Metric Profile\n"
        f"{_display_mitigation(selected.get('mitigation_technique'))} / "
        f"{selected.get('binning_strategy')} / {selected.get('training_method')}"
    )
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def save_mitigation_delta_matrix(full_df: pd.DataFrame, output_file):
    """Wide heatmap of metric deltas vs matched baseline. Positive = better."""
    if full_df is None or full_df.empty or "mitigation_technique" not in full_df.columns:
        return None

    delta_specs = [
        ("delta_f1", "F1"),
        ("delta_recall", "Recall"),
        ("delta_precision", "Precision"),
        ("delta_auc", "AUC-ROC"),
        ("delta_accuracy", "Accuracy"),
        ("delta_fairness_gap", "Fairness Gap"),
        ("delta_dp_gap", "DP Gap"),
        ("delta_eq_tpr_gap", "EO TPR Gap"),
        ("delta_eq_fpr_gap", "EO FPR Gap"),
    ]
    available = [(col, label) for col, label in delta_specs if col in full_df.columns]
    if len(available) < 2:
        return None

    plot_df = full_df[full_df["mitigation_technique"] != "baseline"].copy()
    if plot_df.empty:
        return None
    for col, _ in available:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    agg = plot_df.groupby("mitigation_technique")[[col for col, _ in available]].mean()
    agg = agg.dropna(how="all")
    if agg.empty:
        return None
    if "delta_fairness_gap" in agg.columns:
        agg = agg.sort_values("delta_fairness_gap", ascending=False)

    plot_values = agg.rename(columns=dict(available)) * 100
    plot_values.index = [_display_mitigation(idx) for idx in plot_values.index]
    max_abs = float(np.nanmax(np.abs(plot_values.to_numpy(dtype=float))))
    max_abs = max(max_abs, 1.0)

    fig, ax = plt.subplots(figsize=(16, max(5, len(plot_values) * 0.35 + 2)))
    sns.heatmap(
        plot_values,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        linewidths=0.5,
        cbar_kws={"label": "Delta vs baseline (percentage points)"},
        ax=ax,
    )
    ax.set_title("Mitigation Effects by Metric (positive = improvement)", fontsize=12)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mitigation")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def _filter_per_group_for_selected(
    per_group_df: pd.DataFrame,
    selected_row: pd.Series,
    sensitive_attr: str,
) -> pd.DataFrame:
    if per_group_df is None or per_group_df.empty:
        return pd.DataFrame()
    df = per_group_df.copy()
    df["sensitive_attr_norm"] = df["sensitive_attr"].map(_normalize_sensitive_attr)
    df = df[df["sensitive_attr_norm"] == _normalize_sensitive_attr(sensitive_attr)]

    if "experiment_id" in df.columns and "experiment_id" in selected_row.index:
        exp_id = selected_row.get("experiment_id")
        if pd.notna(exp_id):
            exp_df = df[df["experiment_id"].astype(str) == str(exp_id)]
            if not exp_df.empty:
                df = exp_df
    else:
        for col in [
            "dataset",
            "model_type",
            "binning_strategy",
            "training_method",
            "mitigation_technique",
            "model_variant",
        ]:
            if col in df.columns and col in selected_row.index:
                df = df[df[col].astype(str) == str(selected_row.get(col))]

    for col in [
        "baseline_value",
        "experiment_value",
        "delta",
        "baseline_overall_value",
        "experiment_overall_value",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["baseline_value", "experiment_value"])


def save_group_before_after_bars(
    per_group_df: pd.DataFrame,
    output_file,
    sensitive_attr: str,
    selected_row: pd.Series,
):
    """Landscape grouped bars: baseline vs selected mitigation by subgroup."""
    df = _filter_per_group_for_selected(per_group_df, selected_row, sensitive_attr)
    if df.empty:
        return None

    metric_order = [
        ("tpr", "TPR"),
        ("fpr", "FPR"),
        ("predictive_parity_precision", "Precision"),
        ("demographic_parity_rate", "DP Rate"),
    ]
    available = [(m, label) for m, label in metric_order if m in set(df["metric"])]
    if not available:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(14, 5), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    palette = {"Baseline": "#8C8C8C", "After Mitigation": "#0072B2"}

    for idx, (metric, label) in enumerate(available):
        ax = axes[idx]
        sub = df[df["metric"] == metric].copy()
        sub["group_label"] = [
            _pretty_group_label(attr, group)
            for attr, group in zip(sub["sensitive_attr"], sub["group"])
        ]
        records = []
        for _, row in sub.iterrows():
            records.append(
                {
                    "group": row["group_label"],
                    "condition": "Baseline",
                    "value": row["baseline_value"],
                }
            )
            records.append(
                {
                    "group": row["group_label"],
                    "condition": "After Mitigation",
                    "value": row["experiment_value"],
                }
            )
        plot_df = pd.DataFrame(records)
        sns.barplot(data=plot_df, x="group", y="value", hue="condition", palette=palette, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("Group")
        ax.set_ylabel("Rate" if idx == 0 else "")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
        if idx > 0 and ax.get_legend():
            ax.get_legend().remove()

    attr_label = _normalize_sensitive_attr(sensitive_attr).replace("_", " ").title()
    fig.suptitle(
        f"Per-Group Metrics Before/After - {attr_label} - "
        f"{_display_mitigation(selected_row.get('mitigation_technique'))}",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def _group_improvement(row: pd.Series) -> float:
    base = row.get("baseline_value")
    exp = row.get("experiment_value")
    if pd.isna(base) or pd.isna(exp):
        return np.nan
    metric = row.get("metric")
    if metric in {"tpr", "equal_opportunity_tpr", "predictive_parity_precision"}:
        return exp - base
    if metric == "fpr":
        return base - exp
    if metric == "demographic_parity_rate":
        base_overall = row.get("baseline_overall_value")
        exp_overall = row.get("experiment_overall_value")
        if pd.isna(base_overall) or pd.isna(exp_overall):
            return np.nan
        return abs(base - base_overall) - abs(exp - exp_overall)
    return exp - base


def save_group_delta_bars(
    per_group_df: pd.DataFrame,
    output_file,
    sensitive_attr: str,
    selected_row: pd.Series,
):
    """Landscape delta bars. Positive means subgroup-level improvement."""
    df = _filter_per_group_for_selected(per_group_df, selected_row, sensitive_attr)
    if df.empty:
        return None
    df["improvement"] = df.apply(_group_improvement, axis=1)
    df = df.dropna(subset=["improvement"])
    if df.empty:
        return None

    metric_order = [
        ("tpr", "TPR"),
        ("fpr", "FPR"),
        ("predictive_parity_precision", "Precision"),
        ("demographic_parity_rate", "DP Distance"),
    ]
    available = [(m, label) for m, label in metric_order if m in set(df["metric"])]
    if not available:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(14, 5), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    max_abs = max(float(np.nanmax(np.abs(df["improvement"]))), 0.02)

    for idx, (metric, label) in enumerate(available):
        ax = axes[idx]
        sub = df[df["metric"] == metric].copy()
        sub["group_label"] = [
            _pretty_group_label(attr, group)
            for attr, group in zip(sub["sensitive_attr"], sub["group"])
        ]
        sub = sub.groupby("group_label", as_index=False)["improvement"].mean()
        colors = ["#2E8B57" if value >= 0 else "#B22222" for value in sub["improvement"]]
        ax.bar(sub["group_label"], sub["improvement"] * 100, color=colors, edgecolor="white")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Group")
        ax.set_ylabel("Improvement (pp)" if idx == 0 else "")
        ax.set_ylim(-max_abs * 110, max_abs * 110)
        ax.tick_params(axis="x", rotation=30)

    attr_label = _normalize_sensitive_attr(sensitive_attr).replace("_", " ").title()
    fig.suptitle(
        f"Subgroup Improvement Deltas - {attr_label} - "
        f"{_display_mitigation(selected_row.get('mitigation_technique'))}",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def _model_radar_values(row: pd.Series) -> list[float]:
    fairness_gap = row.get("fairness_gap")
    return [
        row.get("f1_value", row.get("f1_score", row.get("f1"))),
        row.get("recall_value", row.get("recall")),
        row.get("precision_value", row.get("precision")),
        row.get("auc_value", row.get("auc_roc")),
        1 - fairness_gap if pd.notna(fairness_gap) else np.nan,
    ]


def _save_model_radar(rows_df: pd.DataFrame, output_file, title: str, note: str | None = None):
    if rows_df is None or rows_df.empty or "model_type" not in rows_df.columns:
        return None
    categories = ["F1", "Recall", "Precision", "AUC-ROC", "Fairness\n(1-gap)"]
    n = len(categories)
    angles = [i * 2 * np.pi / n for i in range(n)] + [0]

    fig, ax = plt.subplots(figsize=(14, 6), subplot_kw={"polar": True})
    for _, row in rows_df.iterrows():
        model = row["model_type"]
        values = _model_radar_values(row)
        if any(pd.isna(v) for v in values):
            continue
        values = [float(np.clip(v, 0, 1)) for v in values]
        values_closed = values + [values[0]]
        color = PALETTE_MODEL.get(model, "#333333")
        ax.plot(angles, values_closed, color=color, linewidth=2, label=model.replace("_", " ").title())
        ax.fill(angles, values_closed, color=color, alpha=0.14)

    if not ax.lines:
        plt.close(fig)
        return None
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.set_title(title, fontsize=13, pad=20)
    ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=9)
    if note:
        fig.text(0.5, 0.03, note, ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def _best_rows_by_model(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby("model_type", dropna=False):
        sort_cols = [col for col in ["f1_value", "recall_value"] if col in group.columns]
        group = group.copy()
        for col in sort_cols + ["fairness_gap"]:
            if col in group.columns:
                group[col] = pd.to_numeric(group[col], errors="coerce")
        if "fairness_gap" in group.columns:
            group = group.sort_values(sort_cols + ["fairness_gap"], ascending=[False] * len(sort_cols) + [True])
        elif sort_cols:
            group = group.sort_values(sort_cols, ascending=False)
        rows.append(group.iloc[0])
    return pd.DataFrame(rows)


def save_cross_model_baseline_radar(full_df: pd.DataFrame, output_file):
    """Baseline-only radar across model families."""
    if full_df is None or full_df.empty or "mitigation_technique" not in full_df.columns:
        return None
    base_df = full_df[full_df["mitigation_technique"] == "baseline"].copy()
    if base_df.empty or "model_type" not in base_df.columns:
        return None
    rows_df = _best_rows_by_model(base_df)
    return _save_model_radar(
        rows_df,
        output_file,
        "Cross-Model Baseline Radar",
        note="Baseline-only comparison; mitigated LR configs are excluded.",
    )


def save_cross_model_best_available_radar(full_df: pd.DataFrame, output_file):
    """Appendix radar: best available row per model; LR may include mitigation."""
    if full_df is None or full_df.empty or "model_type" not in full_df.columns:
        return None
    rows_df = _best_rows_by_model(full_df.copy())
    return _save_model_radar(
        rows_df,
        output_file,
        "Cross-Model Best Available Radar",
        note="Appendix only: logistic regression has mitigation candidates; other models are baseline-only in current runs.",
    )


def _count_group_improvements(per_group_df: pd.DataFrame | None, selected_row: pd.Series) -> tuple[int, int]:
    if per_group_df is None or per_group_df.empty:
        return 0, 0
    df = per_group_df.copy()
    if "experiment_id" in df.columns and "experiment_id" in selected_row.index:
        df = df[df["experiment_id"].astype(str) == str(selected_row.get("experiment_id"))]
    else:
        for col in [
            "dataset",
            "model_type",
            "binning_strategy",
            "training_method",
            "mitigation_technique",
            "model_variant",
        ]:
            if col in df.columns and col in selected_row.index:
                df = df[df[col].astype(str) == str(selected_row.get(col))]
    if df.empty:
        return 0, 0
    for col in ["baseline_value", "experiment_value", "baseline_overall_value", "experiment_overall_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["baseline_value", "experiment_value"])
    if df.empty:
        return 0, 0
    df["improvement"] = df.apply(_group_improvement, axis=1)
    df = df.dropna(subset=["improvement"])
    if df.empty:
        return 0, 0
    df["group_key"] = df["sensitive_attr"].map(_normalize_sensitive_attr) + ":" + df["group"].astype(str)
    group_scores = df.groupby("group_key")["improvement"].mean()
    return int((group_scores > 1e-12).sum()), int((group_scores < -1e-12).sum())


def build_fairness_evidence_summary(
    full_df: pd.DataFrame,
    per_group_df: pd.DataFrame | None = None,
    top_n: int = 5,
    min_recall_delta: float = -0.03,
) -> pd.DataFrame:
    """Top mitigation evidence rows, ranked by metric-level fairness improvement."""
    if full_df is None or full_df.empty or "mitigation_technique" not in full_df.columns:
        return pd.DataFrame()
    df = full_df[full_df["mitigation_technique"] != "baseline"].copy()
    if "model_type" in df.columns:
        lr_df = df[df["model_type"] == "logistic_regression"].copy()
        if not lr_df.empty:
            df = lr_df
    if df.empty:
        return pd.DataFrame()
    for col in [
        "delta_fairness_gap",
        "delta_recall",
        "delta_f1",
        "delta_dp_gap",
        "delta_eq_tpr_gap",
        "delta_eq_fpr_gap",
        "f1_value",
        "recall_value",
        "fairness_gap",
    ]:
        df[col] = _numeric_series(df, col)

    eligible = df[(df["delta_fairness_gap"] > 0) & (df["delta_recall"] >= min_recall_delta)]
    if eligible.empty:
        eligible = df[df["delta_fairness_gap"] > 0]
    if eligible.empty:
        eligible = df
    eligible = eligible.sort_values(
        ["delta_fairness_gap", "delta_recall", "delta_f1"],
        ascending=[False, False, False],
        na_position="last",
    )

    rows = []
    seen_mitigations = set()
    for _, row in eligible.iterrows():
        mitigation = row.get("mitigation_technique")
        if mitigation in seen_mitigations:
            continue
        seen_mitigations.add(mitigation)
        groups_improved, groups_worsened = _count_group_improvements(per_group_df, row)
        rows.append(
            {
                "dataset": row.get("dataset"),
                "model_type": row.get("model_type"),
                "model_variant": row.get("model_variant"),
                "binning_strategy": row.get("binning_strategy"),
                "training_method": row.get("training_method"),
                "mitigation_technique": mitigation,
                "display_mitigation": _display_mitigation(mitigation),
                "f1": row.get("f1_value"),
                "recall": row.get("recall_value"),
                "fairness_gap": row.get("fairness_gap"),
                "delta_fairness_gap": row.get("delta_fairness_gap"),
                "delta_recall": row.get("delta_recall"),
                "delta_f1": row.get("delta_f1"),
                "delta_dp_gap": row.get("delta_dp_gap"),
                "delta_eq_tpr_gap": row.get("delta_eq_tpr_gap"),
                "delta_eq_fpr_gap": row.get("delta_eq_fpr_gap"),
                "groups_improved": groups_improved,
                "groups_worsened": groups_worsened,
                "experiment_id": row.get("experiment_id"),
            }
        )
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def save_fairness_evidence_summary(
    full_df: pd.DataFrame,
    per_group_df: pd.DataFrame | None,
    output_file,
    top_n: int = 5,
):
    summary = build_fairness_evidence_summary(full_df, per_group_df, top_n=top_n)
    if summary.empty:
        return None
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return output_file


def save_comparison_heatmap(
    pivot: pd.DataFrame,
    title: str,
    output_file,
    fmt: str = ".3f",
    cmap: str = "viridis",
):
    """Save a heatmap for a comparison pivot table."""
    if pivot is None or pivot.empty:
        return None

    fig = plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_tradeoff_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    style_col: str,
    title: str,
    output_file,
):
    """Save a scatter plot showing performance vs fairness trade-offs."""
    if df is None or df.empty:
        return None

    plot_df = df[[x_col, y_col, hue_col, style_col]].dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return None

    use_hue = hue_col in plot_df.columns and plot_df[hue_col].notna().any()
    use_style = style_col in plot_df.columns and plot_df[style_col].notna().any()

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue=hue_col if use_hue else None,
        style=style_col if use_style else None,
        alpha=0.8,
    )
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if use_hue or use_style:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_file,
):
    """Save a Pareto frontier plot (maximize x, minimize y)."""
    if df is None or df.empty:
        return None

    data = df[[x_col, y_col]].dropna().copy()
    if data.empty:
        return None

    # Compute Pareto frontier: maximize x, minimize y
    data = data.sort_values([x_col, y_col], ascending=[False, True])
    pareto = []
    best_y = None
    for _, row in data.iterrows():
        y_val = row[y_col]
        if best_y is None or y_val <= best_y:
            pareto.append(row)
            best_y = y_val

    pareto_df = pd.DataFrame(pareto)

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6)
    sns.lineplot(data=pareto_df, x=x_col, y=y_col, color="red", marker="o")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_intersectional_heatmap(
    per_group_df: pd.DataFrame,
    metric: str,
    output_file,
):
    """Heatmap of per-subgroup fairness delta — mitigation × demographic group.

    Rows = mitigation technique, columns = subgroup label (sensitive_attr + group),
    cell value = mean ``delta`` (experiment − baseline).  Green = improvement,
    red = worsening.

    Parameters
    ----------
    per_group_df : pd.DataFrame
        ``per_group_comparison.csv`` content.  Required columns:
        ``mitigation_technique``, ``sensitive_attr``, ``group``, ``metric``, ``delta``.
    metric : str
        Metric name to filter on, e.g. ``"demographic_parity_rate"`` or ``"tpr"``.
    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """
    sub = per_group_df[per_group_df["metric"] == metric].copy()
    if sub.empty:
        logger.warning("save_intersectional_heatmap: no rows for metric '%s'", metric)
        return None

    value_col = "delta"

    # Guard against non-numeric/empty deltas in partially populated outputs.
    if "delta" in sub.columns:
        sub["delta"] = pd.to_numeric(sub["delta"], errors="coerce")
    else:
        sub["delta"] = np.nan

    # Fallback 1: derive delta from experiment_value and baseline_value.
    if sub["delta"].dropna().empty and {"experiment_value", "baseline_value"}.issubset(sub.columns):
        exp_val = pd.to_numeric(sub["experiment_value"], errors="coerce")
        base_val = pd.to_numeric(sub["baseline_value"], errors="coerce")
        sub["delta"] = exp_val - base_val

    # Fallback 2: if no baseline pairing is available, visualize experiment values directly.
    if sub["delta"].dropna().empty and "experiment_value" in sub.columns:
        sub["experiment_value"] = pd.to_numeric(sub["experiment_value"], errors="coerce")
        value_col = "experiment_value"

    sub = sub.dropna(subset=[value_col])
    if sub.empty:
        logger.warning("save_intersectional_heatmap: no numeric values for metric '%s'", metric)
        return None

    sub["group_label"] = (
        sub["sensitive_attr"].str.replace("_cat", "").str.title() + "\n" + sub["group"].astype(str)
    )

    agg = sub.groupby(["mitigation_technique", "group_label"])[value_col].mean().reset_index()
    pivot = agg.pivot(index="mitigation_technique", columns="group_label", values=value_col)

    if pivot.empty:
        return None

    values = pivot.to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        logger.warning(
            "save_intersectional_heatmap: all delta values are NaN for metric '%s'", metric
        )
        return None
    abs_max = max(float(np.abs(values[valid]).max()), 0.01)

    fig, ax = plt.subplots(
        figsize=(max(10, pivot.shape[1] * 1.5), max(5, pivot.shape[0] * 0.6 + 2))
    )
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-abs_max,
        vmax=abs_max,
        linewidths=0.5,
        ax=ax,
    )
    metric_label = metric.replace("_", " ").title()
    value_label = (
        "Delta (Experiment − Baseline)"
        if value_col == "delta"
        else "Experiment Value (baseline unavailable)"
    )
    ax.set_title(
        f"Intersectional Fairness — {metric_label}\n{value_label}",
        fontsize=12,
    )
    ax.set_xlabel("Subgroup")
    ax.set_ylabel("Mitigation Technique")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_cross_model_radar(summary_df: pd.DataFrame, output_file):
    """Spider / radar chart comparing model types across 5 performance + fairness axes.

    Axes: F1, Recall, Precision, AUC-ROC, Fairness (= 1 − fairness_gap).
    One filled polygon per model type.

    Parameters
    ----------
    summary_df : pd.DataFrame
        ``cross_model_summary.csv`` content.  Required columns:
        ``model_type``, ``f1_score``, ``recall``, ``precision``,
        ``auc_roc``, ``fairness_gap``.
    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """
    if summary_df is None or summary_df.empty:
        logger.warning("save_cross_model_radar: empty DataFrame")
        return None

    if "model_type" not in summary_df.columns:
        logger.warning("save_cross_model_radar: missing required column model_type")
        return None

    df = summary_df.copy()

    aliases = {
        "f1_score": ["f1_score", "f1", "f1_value"],
        "recall": ["recall", "recall_value"],
        "precision": ["precision", "precision_value"],
        "auc_roc": ["auc_roc", "auc", "auc_value"],
    }

    def _resolve_alias(candidates: list[str]) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    metric_axes = []
    for canonical, candidates in aliases.items():
        src = _resolve_alias(candidates)
        if src is None:
            continue
        df[canonical] = pd.to_numeric(df[src], errors="coerce")
        metric_axes.append((canonical, canonical.replace("_", " ").upper()))

    if "fairness_gap" not in df.columns:
        if {"dp_max_diff", "eq_odds_max_diff"}.issubset(df.columns):
            df["fairness_gap"] = df[["dp_max_diff", "eq_odds_max_diff"]].max(axis=1, skipna=True)
        else:
            logger.warning("save_cross_model_radar: fairness_gap missing and cannot be derived")
            return None

    df["fairness_gap"] = pd.to_numeric(df["fairness_gap"], errors="coerce")
    df["fairness_score"] = 1 - df["fairness_gap"].clip(0, 1)

    # Keep a meaningful radar even when some metrics are unavailable.
    axes_specs = metric_axes + [("fairness_score", "Fairness\n(1−gap)")]
    if len(axes_specs) < 3:
        logger.warning("save_cross_model_radar: insufficient metric axes after normalization")
        return None

    df = df.groupby("model_type", as_index=False)[[col for col, _ in axes_specs]].mean()
    df = df.dropna(subset=[col for col, _ in axes_specs])
    if df.empty:
        logger.warning("save_cross_model_radar: no complete rows after normalization")
        return None

    pretty_labels = {
        "f1_score": "F1",
        "recall": "Recall",
        "precision": "Precision",
        "auc_roc": "AUC-ROC",
        "fairness_score": "Fairness\n(1−gap)",
    }

    categories = [pretty_labels.get(col, label) for col, label in axes_specs]
    n = len(categories)
    angles = [i * 2 * np.pi / n for i in range(n)] + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for _, row in df.iterrows():
        model = row["model_type"]
        values = [row[col] for col, _ in axes_specs]
        values_closed = values + [values[0]]
        color = PALETTE_MODEL.get(model, "#333333")
        label = model.replace("_", " ").title()
        ax.plot(angles, values_closed, color=color, linewidth=2, label=label)
        ax.fill(angles, values_closed, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Cross-Model Performance + Fairness Radar", fontsize=12, pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.1), fontsize=9)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_mitigation_effectiveness_matrix(full_df: pd.DataFrame, output_file):
    """Side-by-side heatmaps: fairness gain % and performance cost % per mitigation.

    Rows = mitigation technique (means over all binnings / training methods).
    Left panel = fairness_gain_pct (green = high), right = performance cost (red = high).

    Parameters
    ----------
    full_df : pd.DataFrame
        ``full_comparison.csv`` content.  Required columns:
        ``mitigation_technique``, ``fairness_gain_pct``.
        ``performance_cost_pct`` is used when present. Older ``score_drop_pct``
        remains supported as a fallback.
    output_file : path-like
        Destination path for the saved PNG.

    Returns
    -------
    path-like or None
    """
    if "fairness_gain_pct" not in full_df.columns:
        logger.warning("save_mitigation_effectiveness_matrix: fairness_gain_pct not in DataFrame")
        return None

    plot_df = full_df[full_df["mitigation_technique"] != "baseline"].copy()
    if plot_df.empty:
        return None

    plot_df["fairness_gain_pct"] = pd.to_numeric(plot_df["fairness_gain_pct"], errors="coerce")
    # Stored values are fractions in current comparison outputs; display true percent.
    if plot_df["fairness_gain_pct"].abs().max(skipna=True) <= 1.5:
        plot_df["fairness_gain_pct"] = plot_df["fairness_gain_pct"] * 100

    if "performance_cost_pct" in plot_df.columns:
        plot_df["score_drop_pct"] = pd.to_numeric(plot_df["performance_cost_pct"], errors="coerce")
    elif "score_drop_pct" in plot_df.columns:
        plot_df["score_drop_pct"] = pd.to_numeric(plot_df["score_drop_pct"], errors="coerce")
    else:
        if "score_value" in plot_df.columns and "baseline_score" in plot_df.columns:
            plot_df["score_drop_pct"] = (
                (plot_df["baseline_score"] - plot_df["score_value"])
                / plot_df["baseline_score"].clip(lower=1e-9)
                * 100
            ).clip(lower=0)
        else:
            plot_df["score_drop_pct"] = float("nan")

    agg = plot_df.groupby("mitigation_technique")[["fairness_gain_pct", "score_drop_pct"]].mean()
    if agg.empty:
        return None
    agg.index = [_display_mitigation(idx) for idx in agg.index]

    has_cost = not agg["score_drop_pct"].isna().all()
    n_panels = 2 if has_cost else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels + 1, max(5, len(agg) * 0.45 + 2)))
    if n_panels == 1:
        axes = [axes]

    sns.heatmap(
        agg[["fairness_gain_pct"]],
        annot=True,
        fmt=".1f",
        cmap="Greens",
        linewidths=0.5,
        vmin=0,
        ax=axes[0],
    )
    axes[0].set_title("Fairness Gain %", fontsize=11)
    axes[0].set_ylabel("Mitigation Technique")
    axes[0].set_xlabel("")

    if has_cost:
        sns.heatmap(
            agg[["score_drop_pct"]],
            annot=True,
            fmt=".1f",
            cmap="Reds",
            linewidths=0.5,
            vmin=0,
            ax=axes[1],
        )
        axes[1].set_title("Performance Cost %", fontsize=11)
        axes[1].set_ylabel("")
        axes[1].set_xlabel("")

    fig.suptitle("Mitigation Effectiveness: Fairness Gain vs Metric-Level Cost", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def save_pareto_all_models(
    full_df: pd.DataFrame,
    output_file,
    x_col: str = "f1_value",
    y_col: str = "fairness_gap",
):
    """Pareto frontier scatter with one coloured point cloud per model type.

    Parameters
    ----------
    full_df : pd.DataFrame
        ``full_comparison.csv`` content.  Required columns:
        ``model_type``, plus whichever columns are passed as ``x_col`` / ``y_col``.
    output_file : path-like
        Destination path for the saved PNG.
    x_col : str
        Column to maximise (x-axis).  Default ``"f1_value"``.
    y_col : str
        Column to minimise (y-axis).  Default ``"fairness_gap"``.

    Returns
    -------
    path-like or None
    """
    needed = {"model_type", x_col, y_col}
    if not needed.issubset(full_df.columns):
        logger.warning("save_pareto_all_models: missing columns %s", needed - set(full_df.columns))
        return None

    data = full_df[["model_type", x_col, y_col]].dropna(subset=[x_col, y_col]).copy()
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_type, group in data.groupby("model_type"):
        color = PALETTE_MODEL.get(model_type, "#333333")
        label = model_type.replace("_", " ").title()

        ax.scatter(group[x_col], group[y_col], alpha=0.45, color=color, s=30, label=label)

        sorted_g = group.sort_values([x_col, y_col], ascending=[False, True])
        pareto = []
        best_y = None
        for _, row in sorted_g.iterrows():
            if best_y is None or row[y_col] <= best_y:
                pareto.append(row)
                best_y = row[y_col]

        if len(pareto) >= 2:
            pf = pd.DataFrame(pareto)
            ax.plot(pf[x_col], pf[y_col], color=color, linewidth=2, marker="o", markersize=5)

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title("Pareto Frontier — All Model Types")
    ax.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file
