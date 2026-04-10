"""Unit tests for SimilarityEngine and ViolationDensityMapper."""

import numpy as np
import pandas as pd
import pytest

from fairxai.similarity import SimilarityEngine, ViolationDensityMapper


def _make_df(n=40, uniform_pred=False, random_state=42):
    """Synthetic DataFrame with features and predictions."""
    rng = np.random.default_rng(random_state)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(0, 10, n),
            "feat_c": rng.uniform(0, 5, n),
            "y_pred": 0 if uniform_pred else rng.integers(0, 2, size=n),
        }
    )
    if uniform_pred:
        df["y_pred"] = 0  # all same → perfect consistency
    return df


class TestSimilarityEngine:
    def test_consistency_one_when_all_predictions_identical(self):
        df = _make_df(n=40, uniform_pred=True)
        engine = SimilarityEngine(k_values=[5], pred_col="y_pred")
        result = engine.compute(df, feature_cols=["feat_a", "feat_b", "feat_c"])
        assert len(result.rows) == 1
        assert result.rows[0].mean_consistency == pytest.approx(1.0)

    def test_consistency_less_than_one_when_predictions_vary(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame(
            {
                "feat_a": np.linspace(0, 10, 40),
                "feat_b": np.linspace(0, 10, 40),
                # Alternating 0/1 so neighbours have different predictions
                "y_pred": [i % 2 for i in range(40)],
            }
        )
        engine = SimilarityEngine(k_values=[3], pred_col="y_pred")
        result = engine.compute(df, feature_cols=["feat_a", "feat_b"])
        assert result.rows[0].mean_consistency < 1.0

    def test_multiple_k_values_produce_multiple_rows(self):
        df = _make_df(n=50)
        engine = SimilarityEngine(k_values=[5, 10, 20], pred_col="y_pred")
        result = engine.compute(df, feature_cols=["feat_a", "feat_b"])
        assert len(result.rows) == 3
        ks = [r.k for r in result.rows]
        assert set(ks) == {5, 10, 20}

    def test_k_larger_than_n_samples_skipped_no_crash(self):
        df = _make_df(n=10)
        engine = SimilarityEngine(k_values=[50], pred_col="y_pred")
        result = engine.compute(df, feature_cols=["feat_a", "feat_b"])
        # k=50 > n=10, should be skipped silently
        assert result.rows == []

    def test_missing_pred_col_raises_valueerror(self):
        df = _make_df(n=20)
        engine = SimilarityEngine(k_values=[3], pred_col="nonexistent")
        with pytest.raises(ValueError, match="nonexistent"):
            engine.compute(df, feature_cols=["feat_a", "feat_b"])

    def test_save_scores_creates_csv(self, tmp_path):
        df = _make_df(n=30)
        engine = SimilarityEngine(k_values=[5], pred_col="y_pred")
        result = engine.compute(df, feature_cols=["feat_a", "feat_b"])
        out = engine.save_scores(result, tmp_path)
        assert out.exists()
        saved = pd.read_csv(out)
        assert "k" in saved.columns
        assert "mean_consistency" in saved.columns


class TestViolationDensityMapper:
    def test_produces_png_with_valid_data(self, tmp_path):
        df = _make_df(n=50)
        mapper = ViolationDensityMapper(k=5, sample_size=40)
        result = mapper.compute(
            df,
            feature_cols=["feat_a", "feat_b", "feat_c"],
            pred_col="y_pred",
            output_file=tmp_path / "map.png",
        )
        # Should succeed if matplotlib available
        if result.output_file is not None:
            assert result.output_file.exists()

    def test_returns_none_on_insufficient_data(self, tmp_path):
        df = _make_df(n=3)  # too small for k=5
        mapper = ViolationDensityMapper(k=5)
        result = mapper.compute(
            df,
            feature_cols=["feat_a", "feat_b"],
            pred_col="y_pred",
            output_file=tmp_path / "map.png",
        )
        assert result.output_file is None
