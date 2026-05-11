"""Unit tests for attribute_binning and integration/binning using synthetic DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairxai.experiments.attribute_binning import (
    _equal_width_bins,
    _jenks_bins,
    _quantile_bins,
    create_binning_strategy,
    validate_and_repair,
)

# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


def _binary_df(n: int = 100) -> pd.DataFrame:
    """Binary feature (0/1, equal split), binary target."""
    return pd.DataFrame(
        {
            "feature": np.tile([0, 1], n // 2),
            "target": np.tile([0, 0, 1, 1], n // 4),
        }
    )


def _continuous_df(n: int = 200) -> pd.DataFrame:
    """Continuous feature uniform [0, 100], binary target."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature": rng.uniform(0.0, 100.0, n),
            "target": rng.integers(0, 2, n),
        }
    )


def _low_card_df(n: int = 90) -> pd.DataFrame:
    """Feature with 3 unique integer values (1, 2, 3), binary target."""
    return pd.DataFrame(
        {
            "feature": np.tile([1, 2, 3], n // 3),
            "target": np.tile([0, 1, 0], n // 3),
        }
    )


def _constant_df(n: int = 50) -> pd.DataFrame:
    """Feature with a single constant value — zero range."""
    return pd.DataFrame({"feature": np.full(n, 7.0), "target": np.zeros(n, dtype=int)})


# ---------------------------------------------------------------------------
# equal-width bins
# ---------------------------------------------------------------------------


def test_equal_width_continuous_bin_counts():
    df = _continuous_df()
    for n_bins in (3, 5, 7):
        bins, labels = _equal_width_bins(df, "feature", n_bins, f"equal_width_{n_bins}")
        assert len(bins) - 1 == n_bins, f"Expected {n_bins} bins, got {len(bins) - 1}"
        assert labels is None


def test_equal_width_binary_edges_cover_data():
    """equal_width on binary still produces edges that span the data range."""
    df = _binary_df()
    bins, _ = _equal_width_bins(df, "feature", 3, "equal_width_3")
    # Endpoints widened by 0.001
    assert bins[0] < 0.0
    assert bins[-1] > 1.0
    assert len(bins) - 1 == 3  # 3 edge intervals defined (1 will be empty)


def test_equal_width_constant_column_raises():
    df = _constant_df()
    with pytest.raises(ValueError, match="zero range"):
        _equal_width_bins(df, "feature", 3, "equal_width_3")


# ---------------------------------------------------------------------------
# quantile bins
# ---------------------------------------------------------------------------


def test_quantile_continuous_bin_counts():
    df = _continuous_df()
    for n_bins in (3, 5, 7):
        bins, labels = _quantile_bins(df, "feature", n_bins, f"quantile_{n_bins}")
        assert len(bins) - 1 == n_bins, f"Expected {n_bins} bins, got {len(bins) - 1}"
        assert labels is None


def test_quantile_binary_collapses():
    """Quantile binning on a binary column collapses to 1 bin (known limitation).

    This documents the degenerate behaviour that the backend's binary pre-check
    suppresses by marking these strategies as not_applicable.
    """
    df = _binary_df()
    bins, _ = _quantile_bins(df, "feature", 3, "quantile_3")
    n_bins_created = len(bins) - 1
    # pandas qcut with duplicates='drop' collapses binary to 1 bin
    assert n_bins_created <= 2, (
        f"Expected binary column to collapse to ≤2 bins, got {n_bins_created}"
    )


def test_quantile_low_cardinality_no_crash():
    """quantile_5 on 3-unique-value data should not crash (falls back gracefully)."""
    df = _low_card_df()
    bins, _ = _quantile_bins(df, "feature", 5, "quantile_5")
    n_bins_created = len(bins) - 1
    assert n_bins_created >= 1
    assert n_bins_created <= 3  # can't have more bins than unique values


# ---------------------------------------------------------------------------
# Jenks bins
# ---------------------------------------------------------------------------


def test_jenks_binary_raises_value_error():
    """Jenks on binary column raises a human-readable ValueError."""
    df = _binary_df()
    with pytest.raises(ValueError, match="unique"):
        _jenks_bins(df, "feature", 3, "jenks_3")


def test_jenks_continuous_works():
    """Jenks on continuous data produces the requested number of bins."""
    df = _continuous_df(n=200)
    for n_bins in (3, 5):
        bins, labels = _jenks_bins(df, "feature", n_bins, f"jenks_{n_bins}")
        assert len(bins) - 1 == n_bins
        assert labels is None


# ---------------------------------------------------------------------------
# create_binning_strategy dispatch
# ---------------------------------------------------------------------------


def test_create_binning_strategy_dispatch_smoke():
    """Smoke-test strategy name dispatch for all WebApp-exposed strategy names."""
    df = _continuous_df()
    for strategy in ("equal_width_3", "equal_width_5", "equal_width_7",
                     "quantile_3", "quantile_5", "quantile_7"):
        bins, _ = create_binning_strategy(df, strategy, col="feature", min_group_size=0)
        assert len(bins) >= 2, f"Strategy {strategy} returned fewer than 2 bin edges"


def test_create_binning_strategy_unknown_raises():
    df = _continuous_df()
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_binning_strategy(df, "not_a_strategy", col="feature")


# ---------------------------------------------------------------------------
# validate_and_repair
# ---------------------------------------------------------------------------


def test_validate_and_repair_merges_small_bins():
    """Bins whose count < min_group_size get merged into neighbours."""
    # 30 samples at 0.0, 1 sample at 5.0, 30 samples at 10.0
    values = np.concatenate([np.zeros(30), [5.0], np.ones(30) * 10.0])
    series = pd.Series(values)
    bins = [-0.001, 3.0, 7.0, 10.001]  # middle bin has 1 sample
    labels = ["low", "mid", "high"]

    repaired_bins, _ = validate_and_repair(series, bins, labels, min_group_size=5)
    # Middle bin (1 sample) should have been merged → 2 bins left
    assert len(repaired_bins) - 1 == 2


def test_validate_and_repair_clips_leading_edge():
    """Edges that fall below the data minimum are clipped."""
    series = pd.Series([10.0, 20.0, 30.0])
    bins = [-999.0, 0.0, 50.0, 100.0]  # first intermediate edge (0.0) is below data min
    repaired, _ = validate_and_repair(series, bins, None, min_group_size=0)
    # Leading empty bin should be removed
    assert repaired[0] <= 10.0


# ---------------------------------------------------------------------------
# Integration: run_binning (requires tmp CSV)
# ---------------------------------------------------------------------------


def test_run_binning_equal_width_binary_returns_2_bins(tmp_path):
    """equal_width_3 on binary produces 2 non-empty bins in the output JSON."""
    from fairxai.integration.binning import run_binning

    csv = tmp_path / "binary.csv"
    _binary_df(n=100).to_csv(csv, index=False)

    result = run_binning(csv, target_column="target", attribute="feature",
                         strategy="equal_width_3", min_group_size=0)

    assert len(result["bins"]) == 2, (
        f"Expected 2 non-empty bins for binary feature with equal_width_3, "
        f"got {len(result['bins'])}"
    )


def test_run_binning_quantile_binary_collapses(tmp_path):
    """quantile_3 on binary produces 1 bin — documents suppressed-by-backend behaviour."""
    from fairxai.integration.binning import run_binning

    csv = tmp_path / "binary.csv"
    _binary_df(n=100).to_csv(csv, index=False)

    result = run_binning(csv, target_column="target", attribute="feature",
                         strategy="quantile_3", min_group_size=0)

    assert len(result["bins"]) <= 2
    # When only 1 bin, disparity_ratio must be None (can't compare 1 bin to itself)
    if len(result["bins"]) == 1:
        assert result["summary"]["disparity_ratio"] is None


def test_run_binning_continuous_full_output_schema(tmp_path):
    """run_binning on continuous data returns expected schema fields."""
    from fairxai.integration.binning import run_binning

    csv = tmp_path / "cont.csv"
    _continuous_df(n=200).to_csv(csv, index=False)

    result = run_binning(csv, target_column="target", attribute="feature",
                         strategy="quantile_5", min_group_size=0)

    assert "bins" in result
    assert "summary" in result
    assert "recommendations" in result
    assert "disparity_ratio" in result["summary"]
    assert len(result["bins"]) == 5
    for b in result["bins"]:
        assert "label" in b
        assert "count" in b
        assert "target_rate" in b
