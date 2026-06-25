"""Unit tests for the synthetic dataset generator package."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairxai.data.synthetic import (
    SyntheticConfig,
    build_grid,
    build_smoke_grid,
    generate,
    inject_missingness,
    write_dataset,
)


def test_build_grid_has_unique_dataset_ids():
    grid = build_grid()
    ids = [cfg.dataset_id() for cfg in grid]
    assert len(grid) == 36
    assert len(set(ids)) == len(ids)


def test_generate_is_deterministic():
    cfg = build_grid()[0]
    df_a, _ = generate(cfg)
    df_b, _ = generate(cfg)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_lowcard_column_is_categorical_ground_truth():
    cfg = SyntheticConfig(tier="abstract", seed=1, n_samples=500, lowcard_levels=10)
    _, ground_truth = generate(cfg)
    lowcard = next(c for c in ground_truth if c.name == "lowcard_0")
    assert lowcard.expected_semantic_type == "categorical"
    assert lowcard.n_distinct_design == 10


def test_missingness_writes_real_nans_and_labels_ground_truth():
    cfg = SyntheticConfig(
        tier="abstract",
        seed=2,
        n_samples=1000,
        missing_mechanism="mcar",
        missing_pct=0.3,
        n_missing_features=3,
    )
    df, ground_truth = generate(cfg)
    assert df.isna().sum().sum() > 0
    affected = [c for c in ground_truth if c.missing_mechanism == "mcar"]
    assert len(affected) == 3
    # Target and sensitive columns are never nulled.
    assert df["target"].isna().sum() == 0
    assert df["sex"].isna().sum() == 0


def test_inject_missingness_none_is_noop():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    out = inject_missingness(df, ["a"], "none", 0.5, np.random.default_rng(0))
    assert out.isna().sum().sum() == 0


def test_mar_missingness_varies_with_conditioning_column():
    n = 4000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "feat": rng.normal(size=n),
            "age_group": rng.choice(["<40", "70+"], size=n),
        }
    )
    out = inject_missingness(
        df, ["feat"], "mar", 0.3, np.random.default_rng(1), conditioning_column="age_group"
    )
    rate_a = out.loc[df["age_group"] == "<40", "feat"].isna().mean()
    rate_b = out.loc[df["age_group"] == "70+", "feat"].isna().mean()
    overall = out["feat"].isna().mean()
    # MAR: missingness depends on the conditioning column, so per-group rates
    # diverge while the overall rate stays near the designed fraction.
    assert abs(rate_a - rate_b) > 0.05
    assert abs(overall - 0.3) < 0.05


def test_write_dataset_roundtrips_nans(tmp_path):
    cfg = build_smoke_grid()[1]  # the missingness smoke config
    df, ground_truth = generate(cfg)
    csv_path, meta_path = write_dataset(df, cfg, ground_truth, tmp_path)
    assert csv_path.exists() and meta_path.exists()
    reloaded = pd.read_csv(csv_path)
    assert reloaded.isna().sum().sum() > 0
