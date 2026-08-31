"""Unit tests for subgroup-resolved SHAP attribution summaries."""

import numpy as np
import pandas as pd
import pytest

from fairxai.explainability.subgroup import (
    DEFAULT_MIN_GROUP_SIZE,
    summarise_subgroup_shap,
)

FEATURES = ["chol", "thalach", "oldpeak"]


def _split_attribution_matrix(n_per_group: int = 60, seed: int = 0) -> tuple:
    """Two groups whose attribution structure differs by construction.

    Group ``a`` is driven by ``chol``, group ``b`` by ``thalach``, with the same
    total attribution mass in both. Any correct implementation must report a
    large ``share_gap`` on those two features and none worth speaking of on
    ``oldpeak``.
    """
    rng = np.random.default_rng(seed)
    a = np.column_stack(
        [
            rng.normal(1.0, 0.01, n_per_group),
            rng.normal(0.2, 0.01, n_per_group),
            rng.normal(0.1, 0.01, n_per_group),
        ]
    )
    b = np.column_stack(
        [
            rng.normal(0.2, 0.01, n_per_group),
            rng.normal(1.0, 0.01, n_per_group),
            rng.normal(0.1, 0.01, n_per_group),
        ]
    )
    shap_abs = np.abs(np.vstack([a, b]))
    sensitive = pd.DataFrame({"sex": ["a"] * n_per_group + ["b"] * n_per_group})
    return shap_abs, sensitive


def test_detects_structural_attribution_disparity():
    shap_abs, sensitive = _split_attribution_matrix()

    summary = summarise_subgroup_shap(shap_abs, FEATURES, sensitive)

    assert summary is not None
    disparity = summary.disparity.set_index("feature")
    assert disparity.loc["chol", "share_gap"] > 0.4
    assert disparity.loc["thalach", "share_gap"] > 0.4
    assert disparity.loc["oldpeak", "share_gap"] < 0.05
    assert set(disparity.loc["chol", ["max_group", "min_group"]]) == {"a", "b"}


def test_share_separates_structure_from_confidence():
    """A uniformly scaled group differs in magnitude but not in structure."""
    rng = np.random.default_rng(1)
    base = np.abs(rng.normal(1.0, 0.05, size=(120, len(FEATURES))))
    scaled = base.copy()
    scaled[60:] *= 0.25  # group b gets a quarter of the attribution mass
    sensitive = pd.DataFrame({"sex": ["a"] * 60 + ["b"] * 60})

    summary = summarise_subgroup_shap(scaled, FEATURES, sensitive)

    assert summary is not None
    disparity = summary.disparity
    # The magnitude gap is real and reported...
    assert disparity["mean_abs_shap_gap"].max() > 0.5
    # ...but nothing about *which* features explain the group has changed.
    assert disparity["share_gap"].max() < 0.05


def test_identical_groups_agree_on_ranking():
    rng = np.random.default_rng(2)
    means = np.array([1.0, 0.5, 0.1])
    shap_abs = np.abs(rng.normal(means, 0.01, size=(120, len(FEATURES))))
    sensitive = pd.DataFrame({"sex": ["a"] * 60 + ["b"] * 60})

    summary = summarise_subgroup_shap(shap_abs, FEATURES, sensitive, top_k=3)

    assert summary is not None
    agreement = summary.agreement
    assert agreement["spearman_vs_overall"].to_numpy() == pytest.approx(1.0)
    assert agreement["top3_overlap_vs_overall"].to_numpy() == pytest.approx(1.0)
    assert (agreement["max_rank_shift_vs_overall"] == 0).all()


def test_small_groups_are_dropped_and_reported():
    shap_abs = np.abs(np.random.default_rng(3).normal(1.0, 0.1, size=(65, len(FEATURES))))
    sensitive = pd.DataFrame({"sex": ["a"] * 30 + ["b"] * 30 + ["c"] * 5})

    summary = summarise_subgroup_shap(shap_abs, FEATURES, sensitive, min_group_size=10)

    assert summary is not None
    assert set(summary.per_group["group"]) == {"a", "b"}
    assert summary.skipped["sex"] == {"c": 5}
    assert (summary.disparity["n_groups"] == 2).all()


def test_returns_none_when_no_attribute_has_two_usable_groups():
    shap_abs = np.abs(np.random.default_rng(4).normal(1.0, 0.1, size=(40, len(FEATURES))))
    sensitive = pd.DataFrame({"sex": ["a"] * 35 + ["b"] * 5})

    assert summarise_subgroup_shap(shap_abs, FEATURES, sensitive, min_group_size=10) is None


def test_missing_group_labels_are_excluded():
    shap_abs = np.abs(np.random.default_rng(5).normal(1.0, 0.1, size=(60, len(FEATURES))))
    sensitive = pd.DataFrame({"sex": ["a"] * 25 + ["b"] * 25 + [None] * 10})

    summary = summarise_subgroup_shap(shap_abs, FEATURES, sensitive, min_group_size=5)

    assert summary is not None
    assert set(summary.per_group["group"]) == {"a", "b"}
    assert summary.per_group.groupby("group")["n"].max().to_dict() == {"a": 25, "b": 25}


def test_multiple_attributes_share_one_long_table():
    shap_abs = np.abs(np.random.default_rng(6).normal(1.0, 0.1, size=(80, len(FEATURES))))
    sensitive = pd.DataFrame(
        {
            "sex": ["a"] * 40 + ["b"] * 40,
            "age_group": ["young"] * 20 + ["old"] * 60,
        }
    )

    summary = summarise_subgroup_shap(shap_abs, FEATURES, sensitive, min_group_size=10)

    assert summary is not None
    assert set(summary.per_group["attribute"]) == {"sex", "age_group"}
    assert set(summary.disparity["attribute"]) == {"sex", "age_group"}
    assert len(summary.per_group) == 4 * len(FEATURES)  # 4 groups across 2 attributes


def test_shares_sum_to_one_per_group():
    shap_abs, sensitive = _split_attribution_matrix()

    summary = summarise_subgroup_shap(shap_abs, FEATURES, sensitive)

    assert summary is not None
    totals = summary.per_group.groupby("group")["share"].sum()
    assert totals.to_numpy() == pytest.approx(1.0)


def test_ranks_are_dense_and_start_at_one():
    shap_abs, sensitive = _split_attribution_matrix()

    summary = summarise_subgroup_shap(shap_abs, FEATURES, sensitive)

    assert summary is not None
    for _, block in summary.per_group.groupby("group"):
        assert sorted(block["rank"]) == list(range(1, len(FEATURES) + 1))


def test_all_zero_group_does_not_divide_by_zero():
    shap_abs = np.vstack(
        [
            np.abs(np.random.default_rng(7).normal(1.0, 0.1, size=(40, len(FEATURES)))),
            np.zeros((40, len(FEATURES))),
        ]
    )
    sensitive = pd.DataFrame({"sex": ["a"] * 40 + ["b"] * 40})

    summary = summarise_subgroup_shap(shap_abs, FEATURES, sensitive, min_group_size=10)

    assert summary is not None
    zero_block = summary.per_group[summary.per_group["group"] == "b"]
    assert (zero_block["share"] == 0.0).all()
    assert np.isfinite(summary.disparity["share_gap"]).all()
    # A group with no attribution at all has no feature ordering to compare.
    zero_agreement = summary.agreement[summary.agreement["group"] == "b"]
    assert zero_agreement["spearman_vs_overall"].isna().all()


@pytest.mark.parametrize(
    "shap_abs, sensitive, message",
    [
        (np.zeros((10, 3)), pd.DataFrame({"sex": ["a"] * 9}), "rows"),
        (np.zeros((10, 2)), pd.DataFrame({"sex": ["a"] * 10}), "columns"),
        (np.zeros(10), pd.DataFrame({"sex": ["a"] * 10}), "2D"),
    ],
)
def test_shape_mismatches_raise(shap_abs, sensitive, message):
    with pytest.raises(ValueError, match=message):
        summarise_subgroup_shap(shap_abs, FEATURES, sensitive)


def test_default_group_floor_is_thirty():
    """The floor is a reported methodological choice, not an implementation detail."""
    assert DEFAULT_MIN_GROUP_SIZE == 30
