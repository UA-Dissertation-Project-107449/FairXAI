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
