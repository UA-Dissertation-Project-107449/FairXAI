"""Smoke tests for all viz functions: assert each creates a file without raising."""

import matplotlib

matplotlib.use("Agg")  # must be before any other matplotlib import

import numpy as np
import pandas as pd

from fairxai.comparison.metric_tables import (
    build_fairness_evidence_summary,
    build_group_metric_deltas,
)
from fairxai.viz import fairness_comparison as fairness_comparison_viz
from fairxai.viz.fairness import (
    plot_bias_amplification_waterfall,
    plot_fairness_metric_heatmap,
    plot_group_performance_gaps,
)
from fairxai.viz.fairness_comparison import (
    save_before_after_metric_radar,
    save_cross_model_baseline_radar,
    save_cross_model_best_available_radar,
    save_group_before_after_bars,
    save_group_delta_bars,
    save_group_error_consequence_bars,
    save_group_performance_gap_bars,
    save_intersectional_heatmap,
    save_mitigation_delta_matrix,
)
from fairxai.viz.transformations import (
    plot_before_after_distributions,
    plot_scaling_effects,
    plot_transformation_impact,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers  (module-level, not shared via conftest — viz-specific)
# ---------------------------------------------------------------------------

_MITIGATIONS = ["smote", "reweighing", "calibrated_eq_odds", "threshold_optimizer"]
_BINNINGS = ["equal_width_3", "equal_width_5", "jenks_3"]
_MODELS = ["logistic_regression", "random_forest", "svm", "xgboost"]


def _make_full_comparison_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        mit = _MITIGATIONS[i % len(_MITIGATIONS)]
        model = _MODELS[i % len(_MODELS)]
        binn = _BINNINGS[i % len(_BINNINGS)]
        rows.append(
            {
                "experiment_id": f"exp_{i}",
                "model_type": model,
                "dataset": "cleveland",
                "mitigation_technique": mit,
                "binning_strategy": binn,
                "training_method": "standard",
                "model_variant": "default",
                "f1_value": rng.uniform(0.6, 0.9),
                "recall_value": rng.uniform(0.5, 0.9),
                "precision_value": rng.uniform(0.5, 0.9),
                "auc_value": rng.uniform(0.7, 0.95),
                "fairness_gap": rng.uniform(0.05, 0.35),
                "dem_parity_age_group_max_diff": rng.uniform(0.05, 0.4),
                "eq_odds_age_group_tpr_diff": rng.uniform(0.05, 0.35),
                "eq_odds_age_group_fpr_diff": rng.uniform(0.02, 0.2),
                "dem_parity_sex_max_diff": rng.uniform(0.05, 0.3),
                "eq_odds_sex_tpr_diff": rng.uniform(0.03, 0.25),
                "eq_odds_sex_fpr_diff": rng.uniform(0.02, 0.15),
                "fairness_gain_pct": rng.uniform(-5, 40),
                "baseline_score": rng.uniform(0.65, 0.8),
                "score_value": rng.uniform(0.60, 0.82),
                "delta_f1": rng.uniform(-0.05, 0.06),
                "delta_recall": rng.uniform(-0.02, 0.08),
                "delta_precision": rng.uniform(-0.05, 0.05),
                "delta_auc": rng.uniform(-0.03, 0.04),
                "delta_accuracy": rng.uniform(-0.04, 0.04),
                "delta_fairness_gap": rng.uniform(-0.02, 0.12),
                "delta_dp_gap": rng.uniform(-0.02, 0.12),
                "delta_eq_tpr_gap": rng.uniform(-0.02, 0.10),
                "delta_eq_fpr_gap": rng.uniform(-0.02, 0.08),
                "performance_cost_pct": rng.uniform(0, 6),
            }
        )
    # Add a few explicit baseline rows
    for model in _MODELS:
        rows.append(
            {
                "experiment_id": f"base_{model}",
                "model_type": model,
                "dataset": "cleveland",
                "mitigation_technique": "baseline",
                "binning_strategy": "equal_width_5",
                "training_method": "standard",
                "model_variant": "default",
                "f1_value": 0.72,
                "recall_value": 0.70,
                "precision_value": 0.74,
                "auc_value": 0.80,
                "fairness_gap": 0.25,
                "dem_parity_age_group_max_diff": 0.22,
                "eq_odds_age_group_tpr_diff": 0.18,
                "eq_odds_age_group_fpr_diff": 0.10,
                "dem_parity_sex_max_diff": 0.15,
                "eq_odds_sex_tpr_diff": 0.12,
                "eq_odds_sex_fpr_diff": 0.08,
                "fairness_gain_pct": 0.0,
                "baseline_score": 0.72,
                "score_value": 0.72,
                "delta_f1": 0.0,
                "delta_recall": 0.0,
                "delta_precision": 0.0,
                "delta_auc": 0.0,
                "delta_accuracy": 0.0,
                "delta_fairness_gap": 0.0,
                "delta_dp_gap": 0.0,
                "delta_eq_tpr_gap": 0.0,
                "delta_eq_fpr_gap": 0.0,
                "performance_cost_pct": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _make_per_group_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    attrs = ["age_group_cat", "sex_cat"]
    groups_map = {
        "age_group_cat": ["<40", "40-49", "50-59", "60-69", "70+"],
        "sex_cat": ["Female", "Male"],
    }
    metrics = ["demographic_parity_rate", "tpr", "fpr"]
    rows = []
    for mit in _MITIGATIONS:
        for attr in attrs:
            for grp in groups_map[attr]:
                for metric in metrics:
                    bv = rng.uniform(0.3, 0.7)
                    ev = rng.uniform(0.3, 0.7)
                    rows.append(
                        {
                            "experiment_id": f"{mit}_{attr}",
                            "dataset": "cleveland",
                            "model_type": "logistic_regression",
                            "binning_strategy": "fixed_10yr",
                            "training_method": "single_split",
                            "mitigation_technique": mit,
                            "model_variant": "default",
                            "sensitive_attr": attr,
                            "group": grp,
                            "metric": metric,
                            "baseline_value": bv,
                            "experiment_value": ev,
                            "delta": ev - bv,
                            "baseline_overall_value": 0.50,
                            "experiment_overall_value": 0.52,
                        }
                    )
    return pd.DataFrame(rows)


def _make_summary_df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        [
            {
                "model_type": m,
                "f1_score": rng.uniform(0.6, 0.85),
                "recall": rng.uniform(0.55, 0.85),
                "precision": rng.uniform(0.6, 0.85),
                "auc_roc": rng.uniform(0.70, 0.90),
                "fairness_gap": rng.uniform(0.05, 0.30),
            }
            for m in _MODELS
        ]
    )


def _make_fairness_json(attr: str = "age_group_cat") -> dict:
    groups = ["<40", "40-49", "50-59", "60-69", "70+"] if "age" in attr else ["Female", "Male"]
    group_metrics = {
        g: {"tpr": 0.6 + i * 0.05, "fpr": 0.1 + i * 0.02, "count": 50} for i, g in enumerate(groups)
    }
    group_precision = {
        g: {"precision": 0.7 + i * 0.03, "predicted_positive_count": 30}
        for i, g in enumerate(groups)
    }
    return {
        "test_metrics": {
            "group_fairness": {
                attr: {
                    "equalized_odds": {"group_metrics": group_metrics},
                    "predictive_parity": {"group_precision": group_precision},
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# Smoke tests — one per function
# ---------------------------------------------------------------------------


class TestTransformationPlots:
    def test_plot_transformation_impact_creates_file(self, tmp_path):
        before = {"f1": 0.72, "recall": 0.70, "fairness_gap": 0.25}
        after = {"f1": 0.69, "recall": 0.74, "fairness_gap": 0.14}
        out = tmp_path / "impact.png"
        result = plot_transformation_impact(before, after, out)
        assert result is not None
        assert out.exists()

    def test_plot_transformation_impact_empty_returns_none(self, tmp_path):
        assert plot_transformation_impact({}, {"f1": 0.7}, tmp_path / "x.png") is None

    def test_plot_before_after_distributions_creates_file(self, tmp_path):
        rng = np.random.default_rng(3)
        cols = ["trestbps", "chol", "thalach"]
        before = pd.DataFrame(rng.standard_normal((50, 3)), columns=cols)
        after = pd.DataFrame(rng.standard_normal((70, 3)) + 0.5, columns=cols)
        out = tmp_path / "dist.png"
        result = plot_before_after_distributions(before, after, cols, out)
        assert result is not None
        assert out.exists()

    def test_plot_before_after_distributions_empty_cols_returns_none(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        assert plot_before_after_distributions(df, df, [], tmp_path / "x.png") is None

    def test_plot_scaling_effects_creates_file(self, tmp_path):
        rng = np.random.default_rng(4)
        cols = ["trestbps", "chol"]
        raw = pd.DataFrame(rng.uniform(100, 200, (40, 2)), columns=cols)
        scaled = pd.DataFrame(rng.standard_normal((40, 2)), columns=cols)
        out = tmp_path / "scale.png"
        result = plot_scaling_effects(raw, scaled, out)
        assert result is not None
        assert out.exists()

    def test_plot_scaling_effects_no_common_cols_returns_none(self, tmp_path):
        raw = pd.DataFrame({"a": [1, 2]})
        scaled = pd.DataFrame({"b": [1, 2]})
        assert plot_scaling_effects(raw, scaled, tmp_path / "x.png") is None


class TestFairnessPlots:
    def test_plot_fairness_metric_heatmap_creates_file(self, tmp_path):
        df = _make_full_comparison_df()
        out = tmp_path / "heatmap.png"
        result = plot_fairness_metric_heatmap(df, "age_group_cat", out)
        assert result is not None
        assert out.exists()

    def test_plot_fairness_metric_heatmap_missing_cols_returns_none(self, tmp_path):
        df = pd.DataFrame({"mitigation_technique": ["smote"], "f1_value": [0.7]})
        assert plot_fairness_metric_heatmap(df, "age_group_cat", tmp_path / "x.png") is None

    def test_plot_fairness_metric_heatmap_only_baseline_returns_none(self, tmp_path):
        df = _make_full_comparison_df()
        df = df[df["mitigation_technique"] == "baseline"]
        assert plot_fairness_metric_heatmap(df, "age_group_cat", tmp_path / "x.png") is None

    def test_plot_group_performance_gaps_creates_file(self, tmp_path):
        before = _make_fairness_json("age_group_cat")
        after = _make_fairness_json("age_group_cat")
        out = tmp_path / "gaps.png"
        result = plot_group_performance_gaps(before, after, "age_group_cat", out)
        assert result is not None
        assert out.exists()

    def test_plot_group_performance_gaps_sex_attr(self, tmp_path):
        before = _make_fairness_json("sex_cat")
        after = _make_fairness_json("sex_cat")
        out = tmp_path / "gaps_sex.png"
        result = plot_group_performance_gaps(before, after, "sex_cat", out)
        assert result is not None
        assert out.exists()

    def test_plot_bias_amplification_waterfall_creates_file(self, tmp_path):
        stages = {
            "raw_data": 0.31,
            "preprocessed": 0.28,
            "trained_baseline": 0.22,
            "mitigated": 0.14,
        }
        out = tmp_path / "waterfall.png"
        result = plot_bias_amplification_waterfall(stages, out)
        assert result is not None
        assert out.exists()

    def test_plot_bias_amplification_waterfall_single_stage_returns_none(self, tmp_path):
        assert plot_bias_amplification_waterfall({"only_stage": 0.3}, tmp_path / "x.png") is None


class TestExperimentPlots:
    def test_save_intersectional_heatmap_creates_file(self, tmp_path):
        df = _make_per_group_df()
        out = tmp_path / "intersectional.png"
        result = save_intersectional_heatmap(df, "demographic_parity_rate", out)
        assert result is not None
        assert out.exists()

    def test_save_intersectional_heatmap_unknown_metric_returns_none(self, tmp_path):
        df = _make_per_group_df()
        assert save_intersectional_heatmap(df, "nonexistent_metric", tmp_path / "x.png") is None

    def test_save_before_after_metric_radar_creates_file(self, tmp_path):
        df = _make_full_comparison_df()
        out = tmp_path / "before_after_radar.png"
        selected = df[
            (df["mitigation_technique"] != "baseline") & (df["binning_strategy"] == "equal_width_5")
        ].iloc[0]
        result = save_before_after_metric_radar(df, out, selected_row=selected)
        assert result is not None
        assert out.exists()

    def test_save_mitigation_delta_matrix_creates_file(self, tmp_path):
        df = _make_full_comparison_df()
        out = tmp_path / "delta_matrix.png"
        result = save_mitigation_delta_matrix(df, out)
        assert result is not None
        assert out.exists()

    def test_fairness_comparison_delta_matrix_does_not_need_score_value(self, tmp_path):
        df = _make_full_comparison_df().drop(
            columns=["score_value", "baseline_score", "fairness_gain_pct"], errors="ignore"
        )
        out = tmp_path / "delta_no_score.png"
        result = fairness_comparison_viz.save_mitigation_delta_matrix(df, out)
        assert result is not None
        assert out.exists()

    def test_group_before_after_and_delta_create_files(self, tmp_path):
        full_df = _make_full_comparison_df()
        per_group = _make_per_group_df()
        selected = full_df[
            (full_df["mitigation_technique"] != "baseline")
            & (full_df["binning_strategy"] == "equal_width_5")
        ].iloc[0]

        before_after = tmp_path / "group_before_after.png"
        result = save_group_before_after_bars(per_group, before_after, "age_group_cat", selected)
        assert result is not None
        assert before_after.exists()

        delta = tmp_path / "group_delta.png"
        result = save_group_delta_bars(per_group, delta, "age_group_cat", selected)
        assert result is not None
        assert delta.exists()

    def test_group_performance_gaps_from_paired_rows_create_file(self, tmp_path):
        full_df = _make_full_comparison_df()
        per_group = _make_per_group_df()
        selected = full_df[
            (full_df["mitigation_technique"] != "baseline")
            & (full_df["binning_strategy"] == "equal_width_5")
        ].iloc[0]

        out = tmp_path / "group_performance_gaps.png"
        result = save_group_performance_gap_bars(per_group, out, "age_group_cat", selected)
        assert result is not None
        assert out.exists()

    def test_cross_model_new_radars_create_files(self, tmp_path):
        df = _make_full_comparison_df()
        baseline_out = tmp_path / "baseline_model_radar.png"
        best_out = tmp_path / "best_model_radar.png"

        assert save_cross_model_baseline_radar(df, baseline_out) is not None
        assert baseline_out.exists()
        assert save_cross_model_best_available_radar(df, best_out) is not None
        assert best_out.exists()

    def test_save_group_error_consequence_bars_creates_file(self, tmp_path):
        rng = np.random.default_rng(7)
        groups_map = {
            "age_group_cat": ["<40", "40-49", "50-59", "60-69"],
            "sex_cat": ["Female", "Male"],
        }
        rows = []
        for attr, groups in groups_map.items():
            for grp in groups:
                for metric in ("fnr", "fpr"):
                    bv = rng.uniform(0.15, 0.45)
                    ev = rng.uniform(0.10, 0.40)
                    rows.append(
                        {
                            "experiment_id": "smote_exp",
                            "dataset": "cleveland",
                            "model_type": "logistic_regression",
                            "binning_strategy": "fixed_10yr",
                            "training_method": "single_split",
                            "mitigation_technique": "smote",
                            "model_variant": "default",
                            "sensitive_attr": attr,
                            "group": grp,
                            "metric": metric,
                            "baseline_value": bv,
                            "experiment_value": ev,
                            "delta": ev - bv,
                            "positive_count": rng.integers(5, 20),
                            "negative_count": rng.integers(5, 20),
                        }
                    )
        per_group = pd.DataFrame(rows)
        full_df = _make_full_comparison_df()
        selected = full_df[
            (full_df["mitigation_technique"] != "baseline")
            & (full_df["binning_strategy"] == "equal_width_5")
        ].iloc[0]

        out = tmp_path / "error_consequences.png"
        result = save_group_error_consequence_bars(per_group, out, "age_group_cat", selected)
        assert result is not None
        assert out.exists()

    def test_save_group_error_consequence_bars_empty_returns_none(self, tmp_path):
        full_df = _make_full_comparison_df()
        selected = full_df[full_df["mitigation_technique"] != "baseline"].iloc[0]
        result = save_group_error_consequence_bars(
            pd.DataFrame(), tmp_path / "x.png", "age_group_cat", selected
        )
        assert result is None

    def test_fairness_evidence_summary_creates_file(self, tmp_path):
        full_df = _make_full_comparison_df()
        group_deltas = build_group_metric_deltas(_make_per_group_df())
        summary = build_fairness_evidence_summary(
            full_df,
            group_deltas,
            {
                "selection": {
                    "primary_model_type": "logistic_regression",
                    "min_recall_delta": -0.03,
                    "top_n": 5,
                }
            },
        )
        assert not summary.empty
        assert {"groups_improved", "groups_worsened", "delta_fairness_gap"}.issubset(
            summary.columns
        )
