"""Metric-level fairness comparison plots for dissertation evidence."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fairxai.comparison.baseline_matching import build_baseline_lookups, find_matching_baseline
from fairxai.viz.labels import display_mitigation, normalize_sensitive_attr, pretty_group_label
from fairxai.viz.save_utils import heatmap_size, save_figure

logger = logging.getLogger(__name__)

PALETTE_MODEL = {
    "logistic_regression": "#0072B2",
    "random_forest": "#009E73",
    "svm": "#D55E00",
    "xgboost": "#CC79A7",
}


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


def _select_best_per_strategy(
    full_df: pd.DataFrame,
    model_type: str = "logistic_regression",
    min_recall_delta: float = -0.03,
) -> pd.DataFrame:
    """One best-mitigation row per binning_strategy, sorted by delta_fairness_gap DESC."""
    if full_df is None or full_df.empty or "mitigation_technique" not in full_df.columns:
        return pd.DataFrame()
    df = full_df.copy()
    if "model_type" in df.columns:
        df = df[df["model_type"].astype(str) == str(model_type)]
    df = df[df["mitigation_technique"].astype(str) != "baseline"]
    if df.empty or "binning_strategy" not in df.columns:
        return pd.DataFrame()
    for col in ["delta_fairness_gap", "delta_recall", "delta_f1"]:
        df[col] = _numeric_series(df, col)
    best_rows = []
    for _strategy, group in df.groupby("binning_strategy", dropna=False):
        eligible = group[
            (group["delta_fairness_gap"] > 0) & (group["delta_recall"] >= min_recall_delta)
        ]
        if eligible.empty:
            eligible = group[group["delta_fairness_gap"] > 0]
        if eligible.empty:
            eligible = group
        best_rows.append(
            eligible.sort_values(
                ["delta_fairness_gap", "delta_recall", "delta_f1"],
                ascending=[False, False, False],
                na_position="last",
            ).iloc[0]
        )
    if not best_rows:
        return pd.DataFrame()
    result = pd.DataFrame(best_rows)
    result = result.sort_values("delta_fairness_gap", ascending=False, na_position="last")
    return result.reset_index(drop=True)


def _baseline_match(full_df: pd.DataFrame, row: pd.Series) -> pd.Series | None:
    if full_df is None or full_df.empty:
        return None
    exact, no_variant = build_baseline_lookups(full_df)
    baseline, _source = find_matching_baseline(row, exact, no_variant)
    return baseline


def select_primary_fairness_row(
    full_df: pd.DataFrame,
    model_type: str = "logistic_regression",
    min_recall_delta: float = -0.03,
) -> pd.Series | None:
    """Select the primary mitigation row for before/after evidence plots."""
    if full_df is None or full_df.empty or "mitigation_technique" not in full_df.columns:
        return None

    df = full_df.copy()
    if "model_type" in df.columns:
        df = df[df["model_type"].astype(str) == str(model_type)]
    df = df[df["mitigation_technique"].astype(str) != "baseline"].copy()
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
    """Landscape radar: matched baseline vs selected mitigation."""
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
        f"{display_mitigation(selected.get('mitigation_technique'))} / "
        f"{selected.get('binning_strategy')} / {selected.get('training_method')}"
    )
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    save_figure(fig, output_file, dpi=300)
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

    plot_df = full_df[full_df["mitigation_technique"].astype(str) != "baseline"].copy()
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
    plot_values.index = [display_mitigation(idx) for idx in plot_values.index]
    max_abs = float(np.nanmax(np.abs(plot_values.to_numpy(dtype=float))))
    max_abs = max(max_abs, 1.0)

    width, height = heatmap_size(plot_values.index, len(plot_values.columns), 16, 5)
    fig, ax = plt.subplots(figsize=(width, height))
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
    save_figure(fig, output_file, dpi=300)
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
    df["sensitive_attr_norm"] = df["sensitive_attr"].map(normalize_sensitive_attr)
    df = df[df["sensitive_attr_norm"] == normalize_sensitive_attr(sensitive_attr)]

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
        "improvement",
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
            pretty_group_label(attr, group)
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

    attr_label = normalize_sensitive_attr(sensitive_attr).replace("_", " ").title()
    fig.suptitle(
        f"Per-Group Metrics Before/After - {attr_label} - "
        f"{display_mitigation(selected_row.get('mitigation_technique'))}",
        fontsize=12,
    )
    plt.tight_layout()
    save_figure(fig, output_file, dpi=300)
    plt.close(fig)
    return output_file


def save_group_performance_gap_bars(
    per_group_df: pd.DataFrame,
    output_file,
    sensitive_attr: str,
    selected_row: pd.Series,
):
    """Before/after per-group performance plot using canonical paired rows.

    This is the real replacement for the old JSON-based ``group_performance_gaps``
    figure. It requires matched baseline/experiment values from the comparison
    stage and avoids pairing a report with itself.
    """
    df = _filter_per_group_for_selected(per_group_df, selected_row, sensitive_attr)
    if df.empty:
        return None

    metric_order = [
        ("tpr", "TPR"),
        ("fpr", "FPR"),
        ("predictive_parity_precision", "Precision"),
    ]
    available = [(m, label) for m, label in metric_order if m in set(df["metric"])]
    if not available:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(4.6 * len(available), 4.8))
    axes = np.atleast_1d(axes).ravel()
    palette = {"Baseline": "#8C8C8C", "After Mitigation": "#0072B2"}

    for idx, (metric, label) in enumerate(available):
        ax = axes[idx]
        sub = df[df["metric"] == metric].copy()
        sub["group_label"] = [
            pretty_group_label(attr, group)
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

    attr_label = normalize_sensitive_attr(sensitive_attr).replace("_", " ").title()
    fig.suptitle(
        f"Per-Group Performance Gaps - {attr_label} - "
        f"{display_mitigation(selected_row.get('mitigation_technique'))}",
        fontsize=12,
    )
    plt.tight_layout()
    save_figure(fig, output_file, dpi=300)
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
    if "improvement" in df.columns:
        df["improvement"] = pd.to_numeric(df["improvement"], errors="coerce")
        missing = df["improvement"].isna()
        if missing.any():
            df.loc[missing, "improvement"] = df[missing].apply(_group_improvement, axis=1)
    else:
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
            pretty_group_label(attr, group)
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

    attr_label = normalize_sensitive_attr(sensitive_attr).replace("_", " ").title()
    fig.suptitle(
        f"Subgroup Improvement Deltas - {attr_label} - "
        f"{display_mitigation(selected_row.get('mitigation_technique'))}",
        fontsize=12,
    )
    plt.tight_layout()
    save_figure(fig, output_file, dpi=300)
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
        ax.plot(
            angles, values_closed, color=color, linewidth=2, label=model.replace("_", " ").title()
        )
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
    save_figure(fig, output_file, dpi=300)
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
            group = group.sort_values(
                sort_cols + ["fairness_gap"], ascending=[False] * len(sort_cols) + [True]
            )
        elif sort_cols:
            group = group.sort_values(sort_cols, ascending=False)
        rows.append(group.iloc[0])
    return pd.DataFrame(rows)


def save_cross_model_baseline_radar(full_df: pd.DataFrame, output_file):
    """Baseline-only radar across model families."""
    if full_df is None or full_df.empty or "mitigation_technique" not in full_df.columns:
        return None
    base_df = full_df[full_df["mitigation_technique"].astype(str) == "baseline"].copy()
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
        note=(
            "Appendix only: logistic regression has mitigation candidates; "
            "other models are baseline-only in current runs."
        ),
    )


def save_intersectional_heatmap(
    per_group_df: pd.DataFrame,
    metric: str,
    output_file,
):
    """Heatmap of subgroup improvement by mitigation and demographic group."""
    if per_group_df is None or per_group_df.empty:
        return None
    sub = per_group_df[per_group_df["metric"].astype(str) == str(metric)].copy()
    if sub.empty:
        logger.warning("save_intersectional_heatmap: no rows for metric '%s'", metric)
        return None

    value_col = "improvement" if "improvement" in sub.columns else "delta"
    if value_col in sub.columns:
        sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    else:
        sub[value_col] = np.nan

    if sub[value_col].dropna().empty and {"experiment_value", "baseline_value"}.issubset(
        sub.columns
    ):
        exp_val = pd.to_numeric(sub["experiment_value"], errors="coerce")
        base_val = pd.to_numeric(sub["baseline_value"], errors="coerce")
        sub["delta"] = exp_val - base_val
        value_col = "delta"

    if sub[value_col].dropna().empty and "experiment_value" in sub.columns:
        sub["experiment_value"] = pd.to_numeric(sub["experiment_value"], errors="coerce")
        value_col = "experiment_value"

    sub = sub.dropna(subset=[value_col])
    if sub.empty:
        logger.warning("save_intersectional_heatmap: no numeric values for metric '%s'", metric)
        return None

    sub["group_label"] = (
        sub["sensitive_attr"].map(normalize_sensitive_attr).astype(str).str.title()
        + "\n"
        + sub["group"].astype(str)
    )
    sub["mitigation_label"] = sub["mitigation_technique"].map(display_mitigation)

    agg = sub.groupby(["mitigation_label", "group_label"])[value_col].mean().reset_index()
    pivot = agg.pivot(index="mitigation_label", columns="group_label", values=value_col)
    if pivot.empty:
        return None

    values = pivot.to_numpy(dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        logger.warning("save_intersectional_heatmap: all values are NaN for metric '%s'", metric)
        return None
    abs_max = max(float(np.abs(values[valid]).max()), 0.01)

    width, height = heatmap_size(pivot.index, pivot.shape[1], min_width=12, min_height=5)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-abs_max,
        vmax=abs_max,
        linewidths=0.5,
        cbar_kws={
            "label": (
                "Improvement (positive = better)"
                if value_col == "improvement"
                else "Delta (experiment - baseline)"
            )
        },
        ax=ax,
    )
    metric_label = metric.replace("_", " ").title()
    value_label = (
        "Improvement (positive = better)"
        if value_col == "improvement"
        else (
            "Delta (Experiment - Baseline)"
            if value_col == "delta"
            else "Experiment Value (baseline unavailable)"
        )
    )
    ax.set_title(f"Intersectional Fairness - {metric_label}\n{value_label}", fontsize=12)
    ax.set_xlabel("Subgroup")
    ax.set_ylabel("Mitigation")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    save_figure(fig, output_file, dpi=300)
    plt.close(fig)
    return output_file


def save_group_error_consequence_bars(
    per_group_df: pd.DataFrame,
    output_file,
    sensitive_attr: str,
    selected_row: pd.Series,
):
    """FNR + FPR before/after per subgroup — clinical consequence framing.

    FNR (missed-diagnosis risk) and FPR (false-alarm risk) are shown as
    absolute rates per group, not deltas, so the reader sees actual harm
    levels rather than only changes. Group counts are annotated so small-bin
    instability is visible.
    """
    df = _filter_per_group_for_selected(per_group_df, selected_row, sensitive_attr)
    if df.empty:
        return None

    consequence_metrics = [
        ("fnr", "FNR — Missed Diagnosis Risk"),
        ("fpr", "FPR — False Alarm Risk"),
    ]
    available = [(m, label) for m, label in consequence_metrics if m in set(df["metric"])]
    if not available:
        return None

    count_col = {"fnr": "positive_count", "fpr": "negative_count"}

    fig, axes = plt.subplots(1, len(available), figsize=(6.5 * len(available), 5.2), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    palette = {"Baseline": "#8C8C8C", "After Mitigation": "#0072B2"}

    for idx, (metric, label) in enumerate(available):
        ax = axes[idx]
        sub = df[df["metric"] == metric].copy()
        sub["group_label"] = [
            pretty_group_label(attr, group)
            for attr, group in zip(sub["sensitive_attr"], sub["group"])
        ]

        records = []
        for _, row in sub.iterrows():
            records.append(
                {
                    "group": row["group_label"],
                    "condition": "Baseline",
                    "value": row["baseline_value"],
                    "n": row.get(count_col[metric]),
                }
            )
            records.append(
                {
                    "group": row["group_label"],
                    "condition": "After Mitigation",
                    "value": row["experiment_value"],
                    "n": row.get(count_col[metric]),
                }
            )
        plot_df = pd.DataFrame(records)

        sns.barplot(
            data=plot_df,
            x="group",
            y="value",
            hue="condition",
            palette=palette,
            ax=ax,
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Group")
        ax.set_ylabel("Rate" if idx == 0 else "")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
        if idx > 0 and ax.get_legend():
            ax.get_legend().remove()

        # Annotate n= counts once per group (above the taller bar)
        groups_ordered = plot_df["group"].unique()
        n_per_group = (
            sub.set_index("group_label")[count_col[metric]]
            if count_col[metric] in sub.columns
            else pd.Series(dtype=float)
        )
        for g_idx, group_label in enumerate(groups_ordered):
            n_val = n_per_group.get(group_label)
            if n_val is None or (isinstance(n_val, float) and np.isnan(n_val)):
                continue
            n_int = int(n_val)
            annotation = f"n={n_int}" if n_int >= 5 else f"n={n_int}*"
            group_vals = plot_df[plot_df["group"] == group_label]["value"].dropna()
            y_top = float(group_vals.max()) if not group_vals.empty else 0.0
            ax.text(
                g_idx,
                min(y_top + 0.04, 1.0),
                annotation,
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )

    attr_label = normalize_sensitive_attr(sensitive_attr).replace("_", " ").title()
    fig.suptitle(
        f"Group Error Consequences Before/After — {attr_label} — "
        f"{display_mitigation(selected_row.get('mitigation_technique'))}\n"
        f"* n < 5: estimate unreliable",
        fontsize=11,
    )
    plt.tight_layout()
    save_figure(fig, output_file, dpi=300)
    plt.close(fig)
    return output_file


def save_binning_strategy_delta_matrix(
    full_df: pd.DataFrame,
    output_file,
    model_type: str = "logistic_regression",
    min_recall_delta: float = -0.03,
):
    """Heatmap of metric deltas by binning strategy (best mitigation per strategy).

    Rows = binning strategies, cols = metric deltas. Positive = improvement vs baseline.
    """
    best = _select_best_per_strategy(full_df, model_type, min_recall_delta)
    if best.empty:
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
    available = [(col, label) for col, label in delta_specs if col in best.columns]
    if len(available) < 2:
        return None

    for col, _ in available:
        best[col] = pd.to_numeric(best[col], errors="coerce")

    plot_values = (
        best.set_index("binning_strategy")[[col for col, _ in available]].rename(
            columns=dict(available)
        )
        * 100
    )
    plot_values = plot_values.dropna(how="all")
    if plot_values.empty:
        return None

    max_abs = float(np.nanmax(np.abs(plot_values.to_numpy(dtype=float))))
    max_abs = max(max_abs, 1.0)

    width, height = heatmap_size(plot_values.index, len(plot_values.columns), 16, 5)
    fig, ax = plt.subplots(figsize=(width, height))
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
    ax.set_title(
        "Binning Strategy Effects by Metric\n(best mitigation per strategy; positive = improvement)",
        fontsize=12,
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Binning Strategy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    save_figure(fig, output_file, dpi=300)
    plt.close(fig)
    return output_file


def save_top_n_binning_strategy_summary(
    full_df: pd.DataFrame,
    output_file,
    model_type: str = "logistic_regression",
    top_n: int = 5,
    min_recall_delta: float = -0.03,
):
    """Horizontal bar chart: top-N strategies ranked by fairness-gap improvement.

    Each bar = fairness-gap delta for the best mitigation of that strategy.
    Recall delta is annotated on each bar.
    """
    best = _select_best_per_strategy(full_df, model_type, min_recall_delta)
    if best.empty or "delta_fairness_gap" not in best.columns:
        return None

    best["delta_fairness_gap"] = pd.to_numeric(best["delta_fairness_gap"], errors="coerce")
    best = best.dropna(subset=["delta_fairness_gap"]).head(top_n)
    if best.empty:
        return None

    strategies = list(best["binning_strategy"].astype(str))
    values = list(best["delta_fairness_gap"] * 100)
    recall_deltas = (
        list(pd.to_numeric(best["delta_recall"], errors="coerce") * 100)
        if "delta_recall" in best.columns
        else [float("nan")] * len(best)
    )
    colors = ["#2E8B57" if v >= 0 else "#B22222" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4.0, top_n * 0.85)))
    bars = ax.barh(strategies[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=1)

    for bar, recall in zip(bars, recall_deltas[::-1]):
        if pd.notna(recall):
            x = bar.get_width()
            offset = 0.4 if x >= 0 else -0.4
            ha = "left" if x >= 0 else "right"
            ax.text(
                x + offset,
                bar.get_y() + bar.get_height() / 2,
                f"recall {recall:+.1f} pp",
                va="center",
                ha=ha,
                fontsize=8,
                color="#333333",
            )

    ax.set_xlabel("Fairness-Gap Improvement (percentage points vs baseline)")
    ax.set_title(
        f"Top {top_n} Binning Strategies — Fairness-Gap Improvement\n"
        f"({model_type.replace('_', ' ').title()}, best mitigation per strategy)",
        fontsize=11,
    )
    plt.tight_layout()
    save_figure(fig, output_file, dpi=300)
    plt.close(fig)
    return output_file


def save_top_n_binning_strategy_age_group_small_multiples(
    full_df: pd.DataFrame,
    per_group_df: pd.DataFrame,
    output_file,
    model_type: str = "logistic_regression",
    top_n: int = 5,
    min_recall_delta: float = -0.03,
    age_metric: str = "tpr",
):
    """Small multiples: per-strategy age-group improvement bars.

    One subplot per top-N strategy. Group labels are strategy-specific — a footnote
    warns the reader not to compare labels across subplots.
    """
    if per_group_df is None or per_group_df.empty:
        return None

    best = _select_best_per_strategy(full_df, model_type, min_recall_delta).head(top_n)
    if best.empty:
        return None

    subplot_data = []
    for _, row in best.iterrows():
        filtered = _filter_per_group_for_selected(per_group_df, row, "age_group")
        if filtered.empty:
            logger.warning(
                "save_top_n_binning_strategy_age_group_small_multiples: no per_group rows "
                "for strategy '%s'",
                row.get("binning_strategy"),
            )
            continue
        sub = filtered[filtered["metric"] == age_metric].copy()
        if sub.empty:
            continue
        if "improvement" in sub.columns:
            sub["improvement"] = pd.to_numeric(sub["improvement"], errors="coerce")
            missing = sub["improvement"].isna()
            if missing.any():
                sub.loc[missing, "improvement"] = sub[missing].apply(_group_improvement, axis=1)
        else:
            sub["improvement"] = sub.apply(_group_improvement, axis=1)
        sub = sub.dropna(subset=["improvement"])
        if sub.empty:
            continue
        sub["group_label"] = [
            pretty_group_label(attr, grp) for attr, grp in zip(sub["sensitive_attr"], sub["group"])
        ]
        subplot_data.append((str(row.get("binning_strategy", "")), sub))

    if len(subplot_data) < 2:
        logger.warning(
            "save_top_n_binning_strategy_age_group_small_multiples: fewer than 2 strategies "
            "have per_group data — skipping"
        )
        return None

    n_plots = len(subplot_data)
    if n_plots <= 3:
        n_rows, n_cols = 1, n_plots
    else:
        import math

        n_cols = math.ceil(n_plots / 2)
        n_rows = 2

    all_improvements = np.concatenate([sub["improvement"].values for _, sub in subplot_data])
    max_abs = max(float(np.nanmax(np.abs(all_improvements))), 0.02)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax_idx, (strategy_name, sub) in enumerate(subplot_data):
        ax = axes_flat[ax_idx]
        agg = sub.groupby("group_label", as_index=False)["improvement"].mean()
        colors = ["#2E8B57" if v >= 0 else "#B22222" for v in agg["improvement"]]
        ax.bar(agg["group_label"], agg["improvement"] * 100, color=colors, edgecolor="white")
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title(strategy_name, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel(f"{age_metric.upper()} improvement (pp)" if ax_idx % n_cols == 0 else "")
        ax.set_ylim(-max_abs * 115, max_abs * 115)
        ax.tick_params(axis="x", rotation=35, labelsize=8)

    for ax_idx in range(len(subplot_data), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(
        f"Age-Group {age_metric.upper()} Improvement by Binning Strategy (top {n_plots})",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "Group labels are strategy-specific — do not compare labels across subplots",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, output_file, dpi=300)
    plt.close(fig)
    return output_file
