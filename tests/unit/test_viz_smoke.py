"""Smoke tests for all viz functions: assert each creates a file without raising."""

import matplotlib

matplotlib.use("Agg")  # must be before any other matplotlib import

import numpy as np
import pandas as pd
import pytest

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
                "model_type": model,
                "mitigation_technique": mit,
                "binning_strategy": binn,
                "training_method": "standard",
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
            }
        )
    # Add a few explicit baseline rows
    for model in _MODELS:
        rows.append(
            {
                "model_type": model,
                "mitigation_technique": "baseline",
                "binning_strategy": "equal_width_5",
                "training_method": "standard",
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
                            "mitigation_technique": mit,
                            "sensitive_attr": attr,
                            "group": grp,
                            "metric": metric,
                            "baseline_value": bv,
                            "experiment_value": ev,
                            "delta": ev - bv,
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

    def test_save_cross_model_radar_creates_file(self, tmp_path):
        df = _make_summary_df()
        out = tmp_path / "radar.png"
        result = save_cross_model_radar(df, out)
        assert result is not None
        assert out.exists()

    def test_save_cross_model_radar_missing_cols_returns_none(self, tmp_path):
        df = pd.DataFrame({"model_type": ["logistic_regression"], "f1_score": [0.7]})
        assert save_cross_model_radar(df, tmp_path / "x.png") is None

    def test_save_mitigation_effectiveness_matrix_creates_file(self, tmp_path):
        df = _make_full_comparison_df()
        out = tmp_path / "effectiveness.png"
        result = save_mitigation_effectiveness_matrix(df, out)
        assert result is not None
        assert out.exists()

    def test_save_mitigation_effectiveness_matrix_missing_col_returns_none(self, tmp_path):
        df = pd.DataFrame({"mitigation_technique": ["smote"]})
        assert save_mitigation_effectiveness_matrix(df, tmp_path / "x.png") is None

    def test_save_pareto_all_models_creates_file(self, tmp_path):
        df = _make_full_comparison_df()
        out = tmp_path / "pareto.png"
        result = save_pareto_all_models(df, out)
        assert result is not None
        assert out.exists()

    def test_save_pareto_all_models_missing_cols_returns_none(self, tmp_path):
        df = pd.DataFrame({"model_type": ["logistic_regression"], "f1_value": [0.7]})
        assert save_pareto_all_models(df, tmp_path / "x.png") is None
