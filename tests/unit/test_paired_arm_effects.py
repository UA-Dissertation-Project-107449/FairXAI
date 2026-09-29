"""The paired bootstrap over a mitigation arm and its baseline.

Two properties carry the fix. Pairing the draw removes the cohort noise the two
arms share, so a real change is detectable where an unpaired comparison of two
wide intervals is not. And a max-gap difference, which is untestable on a single
arm because it is bounded below by zero, becomes testable once it is differenced
against a baseline that carries the same upward bias.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fairxai.fairness.uncertainty import (  # noqa: E402
    PAIRED_PERFORMANCE_METRICS,
    paired_arm_differences,
)

SENSITIVE = ["sex", "age_group"]


def _cohort(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, n),
            "sex": rng.choice(["f", "m"], n),
            "age_group": rng.choice(["<50", "50-60", ">60"], n),
        }
    )
    df["y_proba"] = rng.random(n)
    df["y_pred"] = (df["y_proba"] > 0.5).astype(int)
    return df


def _exchangeable_arm(baseline: pd.DataFrame, seed: int = 99) -> pd.DataFrame:
    """Same rows, independently drawn predictions: nothing was changed."""
    rng = np.random.default_rng(seed)
    arm = baseline.copy()
    arm["y_proba"] = rng.random(len(arm))
    arm["y_pred"] = (arm["y_proba"] > 0.5).astype(int)
    return arm


def _biased_arm(baseline: pd.DataFrame) -> pd.DataFrame:
    """An arm that predicts positive for every woman: a large, real sex effect."""
    arm = baseline.copy()
    arm.loc[arm["sex"] == "f", "y_pred"] = 1
    return arm


class TestPairedDifferences:
    def test_identical_arms_have_zero_difference_everywhere(self) -> None:
        baseline = _cohort()
        result = paired_arm_differences(baseline, baseline.copy(), SENSITIVE, n_boot=100)

        assert not result.is_empty
        assert np.allclose(result.table["difference"], 0.0)
        assert not result.table["significant"].any()

    def test_a_real_effect_is_detected_on_the_attribute_it_touches(self) -> None:
        baseline = _cohort()
        result = paired_arm_differences(baseline, _biased_arm(baseline), SENSITIVE, n_boot=200)
        gaps = result.table[
            (result.table["metric"] == "demographic_parity")
            & (result.table["quantity"] == "max_difference")
        ].set_index("attribute")

        assert gaps.loc["sex", "significant"]
        assert not gaps.loc["age_group", "significant"]

    def test_max_gap_difference_is_testable_unlike_the_gap_itself(self) -> None:
        """The interval on a gap *difference* contains zero when nothing changed.

        On a single arm the same statistic essentially never does, which is why
        the assessment marks it descriptive rather than inferential.
        """
        baseline = _cohort(n=400)
        result = paired_arm_differences(
            baseline, _exchangeable_arm(baseline), SENSITIVE, n_boot=300, random_state=3
        )
        gaps = result.table[result.table["quantity"].str.contains("difference")]

        assert len(gaps) > 0
        assert not gaps["excludes_zero"].any()

    def test_performance_change_is_reported_next_to_the_fairness_change(self) -> None:
        baseline = _cohort()
        result = paired_arm_differences(baseline, _biased_arm(baseline), SENSITIVE, n_boot=100)
        performance = result.table[result.table["scope"] == "performance"]

        assert set(performance["quantity"]) == set(PAIRED_PERFORMANCE_METRICS)
        # Predicting positive for every woman can only raise recall.
        recall = performance.set_index("quantity").loc["recall"]
        assert recall["difference"] > 0

    def test_significant_effects_is_the_adjusted_shortlist(self) -> None:
        baseline = _cohort()
        result = paired_arm_differences(baseline, _biased_arm(baseline), SENSITIVE, n_boot=200)

        shortlist = result.significant_effects()
        assert len(shortlist) == int(result.metadata["n_significant"])
        assert (shortlist["p_value_bh"] < shortlist["alpha"]).all()
        # BH can only shrink the shortlist relative to the raw p-values.
        assert len(shortlist) <= len(result.significant_effects(adjusted=False))


class TestPairedGuards:
    def test_different_row_counts_are_refused(self) -> None:
        baseline = _cohort(n=200)
        with pytest.raises(ValueError, match="same test rows"):
            paired_arm_differences(baseline, baseline.head(100), SENSITIVE, n_boot=50)

    def test_different_rows_of_the_same_length_are_refused(self) -> None:
        baseline = _cohort(n=200, seed=0)
        other = _cohort(n=200, seed=1)
        with pytest.raises(ValueError, match="not the same test rows"):
            paired_arm_differences(baseline, other, SENSITIVE, n_boot=50)

    def test_missing_sensitive_columns_yield_an_empty_result(self) -> None:
        baseline = _cohort(n=100)
        result = paired_arm_differences(baseline, baseline.copy(), ["not_a_column"], n_boot=50)
        assert result.is_empty

    @pytest.mark.parametrize("kwargs", [{"n_boot": 1}, {"alpha": 0.0}, {"alpha": 1.0}])
    def test_out_of_range_arguments_raise(self, kwargs: dict) -> None:
        baseline = _cohort(n=100)
        with pytest.raises(ValueError):
            paired_arm_differences(baseline, baseline.copy(), SENSITIVE, **kwargs)

    def test_seed_fixes_the_result(self) -> None:
        baseline = _cohort()
        arm = _biased_arm(baseline)
        first = paired_arm_differences(baseline, arm, SENSITIVE, n_boot=60, random_state=7)
        second = paired_arm_differences(baseline, arm, SENSITIVE, n_boot=60, random_state=7)

        pd.testing.assert_frame_equal(first.table, second.table)

    def test_worker_count_is_a_performance_knob_only(self) -> None:
        baseline = _cohort()
        arm = _biased_arm(baseline)
        serial = paired_arm_differences(
            baseline, arm, SENSITIVE, n_boot=40, random_state=5, n_jobs=1
        )
        parallel = paired_arm_differences(
            baseline, arm, SENSITIVE, n_boot=40, random_state=5, n_jobs=2
        )

        pd.testing.assert_frame_equal(
            serial.table.drop(columns=["method"]), parallel.table.drop(columns=["method"])
        )


def test_calibration_study_exposes_the_paired_null() -> None:
    """The exchangeable-arms scenario is what calibrates the paired test."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "studies"))
    from run_bootstrap_calibration import exchangeable_arms

    baseline, arm = exchangeable_arms(200, seed=0)
    assert baseline["y_true"].equals(arm["y_true"])
    assert not baseline["y_pred"].equals(arm["y_pred"])


def test_stage_10_writes_one_paired_row_set_per_arm(tmp_path: Path) -> None:
    """The stage reads back the arms it persisted and compares each to its baseline."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))
    from run_mitigation_comparison import _write_paired_effects

    baseline = _cohort(n=200)
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir(parents=True)
    baseline.to_csv(predictions_dir / "base.csv", index=False)
    _biased_arm(baseline).to_csv(predictions_dir / "arm.csv", index=False)

    index = [
        {
            "file": "base.csv",
            "dataset": "toy",
            "model_type": "logistic_regression",
            "technique": "baseline",
            "constraint_attr": "none",
        },
        {
            "file": "arm.csv",
            "dataset": "toy",
            "model_type": "logistic_regression",
            "technique": "reweighting",
            "constraint_attr": "sex",
        },
    ]
    _write_paired_effects(tmp_path, index, SENSITIVE, {"enabled": True, "n_boot": 60})

    effects = pd.read_csv(tmp_path / "paired_effects.csv")
    assert set(effects["technique"]) == {"reweighting"}
    assert set(effects["constraint_attr"]) == {"sex"}
    assert effects["significant"].any()


def test_stage_10_paired_effects_can_be_switched_off(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))
    from run_mitigation_comparison import _write_paired_effects

    _write_paired_effects(tmp_path, [{"file": "x.csv"}], SENSITIVE, {"enabled": False})
    assert not (tmp_path / "paired_effects.csv").exists()
