"""Unit tests for run_experiment_comparison helpers."""

import sys
from pathlib import Path

import pandas as pd

# Add scripts/experiments to path so we can import the module directly.
_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "scripts" / "experiments"
sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from run_experiment_comparison import (  # noqa: E402
    _baseline_key_from_row,
    _extract_per_group_fairness,
    _load_baseline_per_group,
    _normalize_sensitive_attr,
)

from fairxai.comparison.config import load_comparison_config  # noqa: E402
from fairxai.comparison.metric_tables import (  # noqa: E402
    build_experiment_index,
    build_group_metric_deltas,
    build_metric_deltas,
    build_metric_values,
)
from fairxai.comparison.naming import figure_filename  # noqa: E402
from fairxai.comparison.plot_frames import build_metric_plot_frame  # noqa: E402


class TestExtractPerGroupFairness:
    def test_known_structure_returns_records(self, minimal_fairness_metrics_dict):
        records = _extract_per_group_fairness(minimal_fairness_metrics_dict)
        assert len(records) > 0

    def test_demographic_parity_rate_present(self, minimal_fairness_metrics_dict):
        records = _extract_per_group_fairness(minimal_fairness_metrics_dict)
        metrics = {r["metric"] for r in records}
        assert "demographic_parity_rate" in metrics

    def test_tpr_fpr_present(self, minimal_fairness_metrics_dict):
        records = _extract_per_group_fairness(minimal_fairness_metrics_dict)
        metrics = {r["metric"] for r in records}
        assert "tpr" in metrics
        assert "fpr" in metrics

    def test_groups_match_input(self, minimal_fairness_metrics_dict):
        records = _extract_per_group_fairness(minimal_fairness_metrics_dict)
        groups = {r["group"] for r in records}
        assert "40-49" in groups
        assert "50-59" in groups

    def test_empty_dict_returns_empty_list(self):
        assert _extract_per_group_fairness({}) == []

    def test_none_returns_empty_list(self):
        assert _extract_per_group_fairness(None) == []


class TestLoadBaselinePerGroup:
    def test_loads_records_when_file_exists(self, sample_baseline_fairness_json, tmp_run_root):
        records = _load_baseline_per_group(tmp_run_root, "cleveland", "logistic_regression")
        assert len(records) > 0

    def test_each_record_has_source_field(self, sample_baseline_fairness_json, tmp_run_root):
        records = _load_baseline_per_group(tmp_run_root, "cleveland", "logistic_regression")
        assert all(r.get("source") == "baseline_assess" for r in records)

    def test_returns_empty_when_file_absent(self, tmp_run_root):
        records = _load_baseline_per_group(
            tmp_run_root, "nonexistent_dataset", "logistic_regression"
        )
        assert records == []


class TestBaselineLookupKeyIncludesModelType:
    """Regression test: baseline key must include model_type to avoid cross-model confusion."""

    def test_baseline_key_includes_model_variant(self):
        """Verify the baseline key separates LR variants as well as model type."""
        # Construct a small df and simulate the lookup construction from run_comparison_analysis.
        df = pd.DataFrame(
            {
                "dataset": ["cleveland", "cleveland"],
                "model_type": ["logistic_regression", "logistic_regression"],
                "model_variant": ["c_0_5", "c_1_0"],
                "binning_strategy": ["fixed_10yr", "fixed_10yr"],
                "training_method": ["single_split", "single_split"],
                "mitigation_technique": ["baseline", "baseline"],
                "score_value": [0.75, 0.72],
                "fairness_gap": [0.1, 0.12],
                "experiment_id": ["e1", "e2"],
                "status": ["success", "success"],
            }
        )
        baseline_df = df[df["mitigation_technique"] == "baseline"]
        lookup = {}
        for _, row in baseline_df.iterrows():
            lookup[_baseline_key_from_row(row, include_variant=True)] = row

        # Two variants → two separate baseline entries (not overwriting each other)
        assert len(lookup) == 2
        assert ("cleveland", "logistic_regression", "c_0_5", "fixed_10yr", "single_split") in lookup
        assert ("cleveland", "logistic_regression", "c_1_0", "fixed_10yr", "single_split") in lookup


class TestSensitiveAttrNormalization:
    def test_cat_suffix_normalizes_for_join(self):
        assert _normalize_sensitive_attr("age_group_cat") == "age_group"
        assert _normalize_sensitive_attr("sex_cat") == "sex"
        assert _normalize_sensitive_attr("sex") == "sex"


class TestCanonicalMetricTables:
    def _full_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "experiment_id": "base",
                    "dataset": "cleveland",
                    "model_type": "logistic_regression",
                    "model_variant": "c_1_0",
                    "binning_strategy": "fixed_10yr",
                    "training_method": "single_split",
                    "mitigation_technique": "baseline",
                    "f1_value": 0.70,
                    "recall_value": 0.65,
                    "precision_value": 0.75,
                    "auc_value": 0.80,
                    "accuracy_value": 0.72,
                    "fairness_gap": 0.20,
                    "dp_max_diff": 0.18,
                    "eq_odds_age_group_tpr_diff": 0.14,
                    "eq_odds_age_group_fpr_diff": 0.10,
                },
                {
                    "experiment_id": "mit",
                    "dataset": "cleveland",
                    "model_type": "logistic_regression",
                    "model_variant": "c_1_0",
                    "binning_strategy": "fixed_10yr",
                    "training_method": "single_split",
                    "mitigation_technique": "smote",
                    "f1_value": 0.68,
                    "recall_value": 0.70,
                    "precision_value": 0.72,
                    "auc_value": 0.79,
                    "accuracy_value": 0.71,
                    "fairness_gap": 0.12,
                    "dp_max_diff": 0.11,
                    "eq_odds_age_group_tpr_diff": 0.09,
                    "eq_odds_age_group_fpr_diff": 0.07,
                },
            ]
        )

    def test_metric_values_and_deltas_are_long_and_directional(self):
        full_df = self._full_df()
        values = build_metric_values(full_df)
        deltas = build_metric_deltas(full_df)

        assert {"metric_family", "metric", "value", "higher_is_better"}.issubset(values.columns)
        assert {"baseline_value", "experiment_value", "delta", "improvement"}.issubset(
            deltas.columns
        )
        f1 = deltas[deltas["metric"] == "f1"].iloc[0]
        fairness = deltas[deltas["metric"] == "fairness_gap"].iloc[0]
        assert round(float(f1["delta"]), 6) == -0.02
        assert round(float(f1["improvement"]), 6) == -0.02
        assert round(float(fairness["delta"]), 6) == -0.08
        assert round(float(fairness["improvement"]), 6) == 0.08

    def test_plot_frame_adapter_does_not_require_score_value(self):
        full_df = self._full_df()
        index = build_experiment_index(full_df)
        values = build_metric_values(full_df)
        deltas = build_metric_deltas(full_df)

        plot_frame = build_metric_plot_frame(index, values, deltas)

        assert plot_frame is not None
        assert "score_value" not in plot_frame.columns
        assert {"f1_value", "recall_value", "delta_f1", "delta_fairness_gap"}.issubset(
            plot_frame.columns
        )

    def test_group_metric_deltas_normalize_improvement_direction(self):
        per_group = pd.DataFrame(
            [
                {
                    "experiment_id": "mit",
                    "dataset": "cleveland",
                    "model_type": "logistic_regression",
                    "model_variant": "c_1_0",
                    "binning_strategy": "fixed_10yr",
                    "training_method": "single_split",
                    "mitigation_technique": "smote",
                    "sensitive_attr": "age_group_cat",
                    "group": "60-69",
                    "metric": "fpr",
                    "baseline_value": 0.30,
                    "experiment_value": 0.20,
                },
                {
                    "experiment_id": "mit",
                    "dataset": "cleveland",
                    "model_type": "logistic_regression",
                    "model_variant": "c_1_0",
                    "binning_strategy": "fixed_10yr",
                    "training_method": "single_split",
                    "mitigation_technique": "smote",
                    "sensitive_attr": "age_group_cat",
                    "group": "60-69",
                    "metric": "demographic_parity_rate",
                    "baseline_value": 0.80,
                    "experiment_value": 0.62,
                    "baseline_overall_value": 0.50,
                    "experiment_overall_value": 0.52,
                },
            ]
        )
        deltas = build_group_metric_deltas(per_group)

        assert set(deltas["sensitive_attr"]) == {"age_group"}
        assert round(float(deltas[deltas["metric"] == "fpr"]["improvement"].iloc[0]), 6) == 0.10
        dp = deltas[deltas["metric"] == "demographic_parity_rate"].iloc[0]
        assert round(float(dp["distance_improvement"]), 6) == 0.20

    def test_comparison_config_loads_yaml_defaults(self, tmp_path):
        cfg = tmp_path / "comparison.yaml"
        cfg.write_text("selection:\n  primary_model_type: random_forest\n", encoding="utf-8")
        loaded = load_comparison_config(Path(__file__).parents[2], str(cfg))

        assert loaded["selection"]["primary_model_type"] == "random_forest"
        assert loaded["canonical_outputs"]["enabled"] is True
        assert "legacy_score" not in loaded

    def test_figure_filename_uses_dataset_scoped_template(self):
        cfg = load_comparison_config(Path(__file__).parents[2])
        filename = figure_filename(
            cfg,
            "fairness_metric_heatmap",
            dataset="kaggle_heart",
            sensitive_attr="age_group_cat",
        )

        assert filename == "kaggle_heart_age_group_fairness_metric_heatmap.png"

        performance_filename = figure_filename(
            cfg,
            "group_performance_gaps",
            dataset="kaggle_heart",
            model_label="lr",
            sensitive_attr="sex_cat",
        )
        assert performance_filename == "kaggle_heart_lr_primary_sex_performance_gaps.png"
