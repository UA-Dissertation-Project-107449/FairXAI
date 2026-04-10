"""Unit tests for ClusteringEngine."""

import numpy as np
import pandas as pd
import pytest

from fairxai.clustering import ClusteringEngine, ClusteringError


def _make_df(n=60, n_clusters=3, random_state=42):
    """Synthetic DataFrame with clear cluster structure."""
    rng = np.random.default_rng(random_state)
    centres = np.linspace(0, 10, n_clusters)
    labels = rng.integers(0, n_clusters, size=n)
    feat_a = centres[labels] + rng.normal(0, 0.5, n)
    feat_b = centres[labels] * 0.8 + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "feat_a": feat_a,
            "feat_b": feat_b,
            "heart_disease": rng.integers(0, 2, size=n),
            "age_group": rng.choice(["<40", "40-49", "50+"], size=n),
        }
    )


class TestClusteringEngineKMeans:
    def test_returns_valid_integer_assignments(self):
        df = _make_df()
        cfg = {"kmeans": {"parameters": {"n_clusters": [3], "n_init": 5, "random_state": 42}}}
        engine = ClusteringEngine(config=cfg)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])

        assert result.group_cluster.dtype in [np.int32, np.int64, int]
        assert result.group_cluster.isna().sum() == 0
        assert len(result.group_cluster) == len(df)
        assert result.group_cluster.min() == 0

    def test_no_nan_labels(self):
        df = _make_df(n=80)
        cfg = {"kmeans": {"parameters": {"n_clusters": [2, 3], "n_init": 5, "random_state": 0}}}
        engine = ClusteringEngine(config=cfg)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])
        assert not result.group_cluster.isna().any()

    def test_silhouette_selects_k_from_grid(self):
        """With clearly separated clusters, best k should be selected from grid."""
        df = _make_df(n_clusters=3)
        cfg = {"kmeans": {"parameters": {"n_clusters": [2, 3, 4], "n_init": 5, "random_state": 42}}}
        engine = ClusteringEngine(config=cfg)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])
        # Best k is somewhere in the grid (not hardcoded)
        assert result.n_clusters in [2, 3, 4]
        assert result.silhouette > 0

    def test_feature_cols_respected(self):
        df = _make_df()
        engine = ClusteringEngine(config={"kmeans": {"parameters": {"n_clusters": [3], "n_init": 5, "random_state": 42}}})
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])
        assert set(result.feature_cols) == {"feat_a", "feat_b"}


class TestClusteringEngineErrors:
    def test_n_clusters_gt_n_samples_raises_clean_error(self):
        """When all k values exceed n_samples, ClusteringError is raised."""
        df = _make_df(n=5)
        cfg = {"kmeans": {"parameters": {"n_clusters": [10, 20], "n_init": 5}}}
        engine = ClusteringEngine(config=cfg)
        with pytest.raises(ClusteringError):
            engine.fit(df, feature_cols=["feat_a", "feat_b"])

    def test_no_usable_features_raises_clean_error(self):
        df = pd.DataFrame({"heart_disease": [0, 1, 0], "age_group": ["<40", "50+", "<40"]})
        engine = ClusteringEngine(config={"kmeans": {"parameters": {"n_clusters": [2]}}})
        with pytest.raises(ClusteringError):
            engine.fit(df)  # All numeric excluded or non-numeric


class TestClusteringEngineDiagnostics:
    def test_diagnostics_saved_to_csv(self, tmp_path):
        df = _make_df()
        cfg = {"kmeans": {"parameters": {"n_clusters": [2, 3], "n_init": 5, "random_state": 0}}}
        engine = ClusteringEngine(config=cfg)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])
        out = engine.save_diagnostics(result, tmp_path)
        assert out.exists()
        saved = pd.read_csv(out)
        assert "method" in saved.columns
        assert "silhouette" in saved.columns
        assert len(saved) >= 2  # one row per k
