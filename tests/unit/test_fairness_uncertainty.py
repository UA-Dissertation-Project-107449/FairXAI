"""Unit tests for bootstrap confidence intervals on fairness metrics."""

import numpy as np
import pandas as pd
import pytest

from fairxai.fairness.uncertainty import (
    BOOTSTRAP_BASE_REPLICATES,
    BOOTSTRAP_LARGE_REPLICATES,
    STRATIFY_GROUP,
    STRATIFY_GROUP_OUTCOME,
    STRATIFY_NONE,
    _benjamini_hochberg,
    _bootstrap_p_value,
    _replicate_indices,
    _stratum_blocks,
    adaptive_bootstrap_replicates,
    bootstrap_fairness_metrics,
    flatten_fairness_metrics,
)

SENSITIVE = ["sex", "age_group"]


def _cohort(n: int = 300, seed: int = 0, bias: float = 0.0) -> pd.DataFrame:
    """A prediction frame. ``bias`` shifts scores down for one sex only."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, n),
            "sex": rng.choice(["f", "m"], n),
            "age_group": rng.choice(["<50", "50-60", ">60"], n),
        }
    )
    df["y_proba"] = np.clip(rng.random(n) - np.where(df["sex"] == "f", bias, 0.0), 0, 1)
    df["y_pred"] = (df["y_proba"] > 0.5).astype(int)
    return df


# --- flattening -------------------------------------------------------------


def test_flatten_keys_group_quantities_and_drops_verdicts():
    results = {
        "group_fairness": {
            "sex": {
                "demographic_parity": {
                    "metric": "demographic_parity",
                    "sensitive_attribute": "sex",
                    "group_rates": {
                        "f": {"positive_rate": 0.4, "count": 120},
                        "m": {"positive_rate": 0.6, "count": 180},
                    },
                    "overall_rate": 0.52,
                    "max_difference": 0.2,
                    "is_fair": False,
                }
            }
        },
        "calibration": {},
    }

    flat = flatten_fairness_metrics(results)

    assert flat[("group_fairness", "sex", "demographic_parity", "f", "positive_rate")] == 0.4
    assert flat[("group_fairness", "sex", "demographic_parity", "", "max_difference")] == 0.2
    # Verdicts are thresholded booleans, counts are fixed by the design.
    assert all("is_fair" != key[-1] for key in flat)
    assert all("count" != key[-1] for key in flat)
    # Strings never become statistics.
    assert all(key[-1] not in {"metric", "sensitive_attribute"} for key in flat)


def test_flatten_drops_per_bin_calibration_detail():
    results = {
        "group_fairness": {},
        "calibration": {
            "sex": {
                "group_calibration": {
                    "f": {"ece": 0.1, "count": 40, "bins": [{"bin": 0, "mean_true": 0.2}]}
                },
                "max_ece_difference": 0.05,
            }
        },
    }

    flat = flatten_fairness_metrics(results)

    assert ("calibration", "sex", "calibration", "f", "ece") in flat
    assert ("calibration", "sex", "calibration", "", "max_ece_difference") in flat
    assert not any("bin" in str(key[-1]) for key in flat)


# --- statistical primitives -------------------------------------------------


def test_bootstrap_p_value_is_floored_by_the_replicate_count():
    all_positive = np.ones(199)

    p = _bootstrap_p_value(all_positive)

    # Not zero: 199 replicates cannot resolve below 2/200.
    assert p == pytest.approx(2 / 200)


def test_bootstrap_p_value_is_one_for_a_difference_centred_on_zero():
    symmetric = np.linspace(-1, 1, 201)

    assert _bootstrap_p_value(symmetric) == pytest.approx(1.0, abs=0.02)


def test_benjamini_hochberg_is_monotone_and_never_shrinks_a_p_value():
    raw = np.array([0.001, 0.01, 0.04, 0.2, 0.9])

    adjusted = _benjamini_hochberg(raw)

    assert np.all(adjusted >= raw)
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-12)
    assert adjusted.max() <= 1.0


def test_benjamini_hochberg_matches_the_step_up_definition():
    raw = np.array([0.01, 0.02, 0.03])

    adjusted = _benjamini_hochberg(raw)

    # 0.01*3/1 = 0.03, 0.02*3/2 = 0.03, 0.03*3/3 = 0.03 → all pulled to 0.03.
    assert adjusted == pytest.approx([0.03, 0.03, 0.03])


def test_adaptive_replicates_thin_only_large_cohorts():
    assert adaptive_bootstrap_replicates(300) == BOOTSTRAP_BASE_REPLICATES
    assert adaptive_bootstrap_replicates(10_000) == BOOTSTRAP_BASE_REPLICATES
    assert adaptive_bootstrap_replicates(68_000) == BOOTSTRAP_LARGE_REPLICATES


# --- resampling -------------------------------------------------------------


def test_stratified_replicates_preserve_every_stratum_size():
    strata = pd.Series(["a"] * 10 + ["b"] * 3 + ["c"] * 7)
    blocks = _stratum_blocks(strata, len(strata))
    rng = np.random.default_rng(0)

    idx = _replicate_indices(len(strata), blocks, rng)

    assert len(idx) == len(strata)
    assert strata.iloc[idx].value_counts().to_dict() == {"a": 10, "c": 7, "b": 3}


def test_unstratified_replicates_do_not_fix_group_sizes():
    strata = pd.Series(["a"] * 10 + ["b"] * 10)
    rng = np.random.default_rng(0)

    counts = {
        tuple(
            sorted(strata.iloc[_replicate_indices(20, None, rng)].value_counts().to_dict().items())
        )
        for _ in range(20)
    }

    assert len(counts) > 1


# --- end to end -------------------------------------------------------------


def test_a_planted_disparity_is_recovered_with_the_right_sign():
    df = _cohort(n=400, bias=0.35)

    result = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=300)

    findings = result.significant_differences()
    sex_rates = findings[
        (findings["attribute"] == "sex") & (findings["quantity"] == "positive_rate")
    ]
    assert len(sex_rates) == 1
    row = sex_rates.iloc[0]
    # 'f' had its scores pushed down, so it must be the lower-selected group.
    assert (row["group_a"], row["group_b"]) == ("f", "m")
    assert row["difference"] < 0
    assert row["ci_high"] < 0


def test_an_undisturbed_cohort_yields_no_adjusted_findings():
    df = _cohort(n=400, seed=7)

    result = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=300)

    assert result.significant_differences().empty


def test_max_gap_intervals_are_reported_but_not_treated_as_tests():
    """The max-of-differences statistic is bounded below by zero.

    Its interval is descriptive only, which is exactly why the pairwise table
    exists; this test pins the asymmetry so nobody 'fixes' the gap CI into a
    significance claim later.
    """
    df = _cohort(n=400, seed=7)

    result = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=300)

    gaps = result.table[result.table["quantity"].str.contains("difference")]
    assert not gaps.empty
    assert (gaps["ci_low"] >= 0).all()
    assert result.significant_differences().empty


def test_intervals_bracket_their_point_estimates_for_per_group_quantities():
    """Holds for rates, not for the descriptive-only quantities.

    A per-group ECE is biased upward under resampling, and on a real cohort
    (Cleveland, age 40-49) its interval sat entirely above its own point
    estimate. Bracketing is a property of the quantities that are not flagged
    descriptive-only, so the assertion is scoped to those.
    """
    df = _cohort(n=400)

    result = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=200)

    per_group = result.table[(result.table["group"] != "") & (~result.table["descriptive_only"])]
    assert not per_group.empty
    assert (per_group["ci_low"] <= per_group["point"]).all()
    assert (per_group["point"] <= per_group["ci_high"]).all()


def test_results_are_reproducible_for_a_fixed_seed():
    df = _cohort(n=250)

    first = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=100, random_state=11)
    second = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=100, random_state=11)
    third = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=100, random_state=12)

    pd.testing.assert_frame_equal(first.table, second.table)
    assert not first.table["ci_low"].equals(third.table["ci_low"])


def test_stratification_degrades_when_a_stratum_cannot_be_resampled():
    df = _cohort(n=200)
    # One row carries a group of its own: group×outcome cannot resample it.
    df.loc[df.index[0], "age_group"] = "unique"

    result = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=50)

    assert result.metadata["stratify_requested"] == STRATIFY_GROUP_OUTCOME
    assert result.metadata["stratify_used"] in {STRATIFY_GROUP, STRATIFY_NONE}


def test_requested_stratification_is_kept_when_every_stratum_is_large_enough():
    result = bootstrap_fairness_metrics(_cohort(n=400), SENSITIVE, n_boot=50)

    assert result.metadata["stratify_used"] == STRATIFY_GROUP_OUTCOME


def test_individual_fairness_is_never_bootstrapped():
    """It is O(n^2), and resampling puts duplicate rows at distance zero."""
    result = bootstrap_fairness_metrics(_cohort(n=200), SENSITIVE, n_boot=50)

    assert result.metadata["individual_fairness_bootstrapped"] is False
    assert "individual_fairness" not in set(result.table["scope"])


def test_unknown_sensitive_columns_yield_an_empty_but_well_formed_result():
    result = bootstrap_fairness_metrics(_cohort(n=100), ["not_a_column"], n_boot=50)

    assert result.is_empty
    assert result.pairwise.empty
    assert "ci_low" in result.table.columns
    assert "p_value_bh" in result.pairwise.columns


def test_an_empty_frame_is_handled_without_raising():
    empty = _cohort(n=10).iloc[0:0]

    assert bootstrap_fairness_metrics(empty, SENSITIVE, n_boot=50).is_empty


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_boot": 1}, "n_boot"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"alpha": -0.1}, "alpha"),
    ],
)
def test_invalid_parameters_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_fairness_metrics(_cohort(n=100), SENSITIVE, **kwargs)


def test_metadata_records_what_actually_ran():
    result = bootstrap_fairness_metrics(_cohort(n=300), SENSITIVE, n_boot=64, random_state=5)

    meta = result.metadata
    assert meta["n_boot"] == 64
    assert meta["random_state"] == 5
    assert meta["n_rows"] == 300
    assert meta["sensitive_attributes"] == SENSITIVE
    assert meta["method"] == "percentile"
    assert meta["p_value_resolution"] == pytest.approx(2 / 65)


def test_every_pair_of_groups_is_compared_once():
    result = bootstrap_fairness_metrics(_cohort(n=300), SENSITIVE, n_boot=50)

    age = result.pairwise[
        (result.pairwise["attribute"] == "age_group")
        & (result.pairwise["quantity"] == "positive_rate")
    ]
    # Three age bands → three unordered pairs, each appearing once.
    assert len(age) == 3
    assert len(set(zip(age["group_a"], age["group_b"]))) == 3


# --- descriptive-only quantities -------------------------------------------


def _cohort_with_a_constant_group(n: int = 200, seed: int = 3) -> pd.DataFrame:
    """A cohort whose smallest group cannot vary under resampling."""
    df = _cohort(n=n, seed=seed)
    idx = df.index[:4]
    df.loc[idx, "age_group"] = "tiny"
    df.loc[idx, "y_true"] = 1
    df.loc[idx, "y_pred"] = 1
    df.loc[idx, "y_proba"] = 0.9
    return df


def test_gap_quantities_are_reported_descriptive_only():
    df = _cohort(n=400, seed=7)

    result = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=100)

    gaps = result.table[result.table["quantity"].str.contains("difference")]
    assert not gaps.empty
    assert gaps["descriptive_only"].all()


def test_per_group_calibration_error_is_reported_descriptive_only():
    """ECE is a nonnegative plug-in statistic, biased upward on small groups.

    Its interval can sit entirely above its own point estimate, so it carries
    the same warning label as a max-gap row rather than reading as an estimate
    anyone can test against zero.
    """
    df = _cohort(n=400, seed=7)

    result = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=100)

    ece = result.table[result.table["quantity"] == "ece"]
    assert not ece.empty
    assert ece["descriptive_only"].all()


def test_group_rates_are_not_descriptive_only():
    df = _cohort(n=400, seed=7)

    result = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=100)

    rates = result.table[result.table["quantity"].isin(["positive_rate", "tpr", "precision"])]
    assert not rates.empty
    assert not rates["descriptive_only"].any()


# --- degenerate replicate spread -------------------------------------------


def test_a_quantity_that_cannot_vary_under_resampling_is_flagged_degenerate():
    df = _cohort_with_a_constant_group()

    result = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=100, random_state=5)

    tiny = result.table[
        (result.table["attribute"] == "age_group")
        & (result.table["group"] == "tiny")
        & (result.table["quantity"] == "tpr")
    ]
    assert not tiny.empty
    assert tiny["degenerate"].all()


def test_ordinary_group_quantities_are_not_flagged_degenerate():
    df = _cohort(n=400, seed=7)

    result = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=100)

    rates = result.table[
        (result.table["attribute"] == "sex") & (result.table["quantity"] == "positive_rate")
    ]
    assert not rates.empty
    assert not rates["degenerate"].any()


# --- parallel replicates ----------------------------------------------------


def test_parallel_replicates_reproduce_the_serial_result():
    """Worker count is a performance knob, never a change to the numbers."""
    df = _cohort(n=250)

    serial = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=60, random_state=11, n_jobs=1)
    parallel = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=60, random_state=11, n_jobs=2)

    pd.testing.assert_frame_equal(serial.table, parallel.table)
    pd.testing.assert_frame_equal(serial.pairwise, parallel.pairwise)


def test_metadata_records_the_worker_count():
    df = _cohort(n=200)

    result = bootstrap_fairness_metrics(df, SENSITIVE, n_boot=40, n_jobs=2)

    assert result.metadata["n_jobs"] == 2
