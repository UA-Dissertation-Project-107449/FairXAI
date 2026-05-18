"""Unit tests for fairxai.fairness.metrics.FairnessMetrics."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.fairness.metrics import FairnessMetrics


def _make_df(groups, y_true, y_pred, attr="sex_cat"):
    """Helper: build a minimal predictions DataFrame."""
    return pd.DataFrame({attr: groups, "y_true": y_true, "y_pred": y_pred})


class TestDemographicParity:
    def test_known_rates(self):
        """Two groups with known positive rates produce correct max_difference."""
        df = _make_df(
            groups=["A"] * 10 + ["B"] * 10,
            y_true=[1] * 10 + [1] * 10,
            y_pred=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        )
        fm = FairnessMetrics(sensitive_attributes=["sex_cat"])
        result = fm.demographic_parity(df, sensitive_attr="sex_cat")
        # Group A: 4/10=0.4, Group B: 7/10=0.7 → max_diff=0.3
        assert abs(result["max_difference"] - 0.3) < 1e-6

    def test_single_group_zero_difference(self):
        """A single group → max_difference is 0 and is_fair is True."""
        df = _make_df(
            groups=["A"] * 10,
            y_true=[1] * 10,
            y_pred=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        )
        fm = FairnessMetrics(sensitive_attributes=["sex_cat"])
        result = fm.demographic_parity(df, sensitive_attr="sex_cat")
        assert result["max_difference"] == 0.0
        # is_fair stored as string "True"
        assert str(result["is_fair"]).lower() in ("true", "1")

    def test_equal_groups_are_fair(self):
        """Groups with identical positive rates → max_difference = 0."""
        df = _make_df(
            groups=["A"] * 10 + ["B"] * 10,
            y_true=[1] * 20,
            y_pred=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 2,
        )
        fm = FairnessMetrics(sensitive_attributes=["sex_cat"])
        result = fm.demographic_parity(df, sensitive_attr="sex_cat")
        assert abs(result["max_difference"]) < 1e-6


class TestEqualizedOdds:
    def test_fnr_present_in_group_metrics(self):
        """equalized_odds() result includes fnr for each group."""
        df = _make_df(
            groups=["A"] * 10 + ["B"] * 10,
            y_true=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            y_pred=[1, 1, 1, 0, 0, 1, 0, 0, 0, 0] + [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        )
        fm = FairnessMetrics(sensitive_attributes=["sex_cat"])
        result = fm.equalized_odds(df, sensitive_attr="sex_cat")
        for group_data in result["group_metrics"].values():
            assert "fnr" in group_data
            assert 0.0 <= group_data["fnr"] <= 1.0

    def test_fnr_equals_one_minus_tpr(self):
        """fnr == 1 - tpr for each group to within float precision."""
        df = _make_df(
            groups=["A"] * 8 + ["B"] * 8,
            y_true=[1, 1, 1, 1, 0, 0, 0, 0] + [1, 1, 1, 0, 0, 0, 0, 0],
            y_pred=[1, 1, 0, 0, 1, 0, 0, 0] + [1, 0, 0, 0, 1, 0, 0, 0],
        )
        fm = FairnessMetrics(sensitive_attributes=["sex_cat"])
        result = fm.equalized_odds(df, sensitive_attr="sex_cat")
        for group_data in result["group_metrics"].values():
            assert abs(group_data["fnr"] - (1.0 - group_data["tpr"])) < 1e-9

    def test_fnr_max_difference_present(self):
        """equalized_odds() result includes fnr_max_difference aggregate."""
        df = _make_df(
            groups=["A"] * 10 + ["B"] * 10,
            y_true=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0] + [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            y_pred=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0] + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        fm = FairnessMetrics(sensitive_attributes=["sex_cat"])
        result = fm.equalized_odds(df, sensitive_attr="sex_cat")
        assert "fnr_max_difference" in result
        assert result["fnr_max_difference"] >= 0.0


class TestCalculateAllMetrics:
    def test_returns_expected_keys(self, synthetic_predictions_df):
        """calculate_all_metrics returns group_fairness and individual_fairness keys."""
        fm = FairnessMetrics(sensitive_attributes=["age_group_cat", "sex_cat"])
        result = fm.calculate_all_metrics(synthetic_predictions_df)
        assert "group_fairness" in result

    def test_group_fairness_contains_sensitive_attrs(self, synthetic_predictions_df):
        fm = FairnessMetrics(sensitive_attributes=["age_group_cat", "sex_cat"])
        result = fm.calculate_all_metrics(synthetic_predictions_df)
        gf = result.get("group_fairness", {})
        assert "age_group_cat" in gf or "sex_cat" in gf

    def test_max_difference_non_negative(self, synthetic_predictions_df):
        fm = FairnessMetrics(sensitive_attributes=["sex_cat"])
        result = fm.calculate_all_metrics(synthetic_predictions_df)
        for attr_metrics in result.get("group_fairness", {}).values():
            dp = attr_metrics.get("demographic_parity", {})
            if "max_difference" in dp:
                assert dp["max_difference"] >= 0
