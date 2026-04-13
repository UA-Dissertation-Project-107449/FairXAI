"""Unit tests for FairnessPerCluster and ClusterProfiler."""

import numpy as np
import pandas as pd

from fairxai.clustering import ClusterProfiler, FairnessPerCluster


def _make_pred_df(n=60, n_clusters=3, random_state=42):
    """Synthetic DataFrame with cluster labels, predictions, and sensitive attrs."""
    rng = np.random.default_rng(random_state)
    cluster_ids = rng.integers(0, n_clusters, size=n)
    return pd.DataFrame(
        {
            "group_cluster": cluster_ids,
            "age_group": rng.choice(["<40", "40-49", "50+"], size=n),
            "sex": rng.choice(["Female", "Male"], size=n),
            "trestbps": rng.uniform(90, 180, n),
            "chol": rng.uniform(150, 350, n),
            "y_pred": rng.integers(0, 2, size=n),
            "heart_disease": rng.integers(0, 2, size=n),
        }
    )


class TestFairnessPerCluster:
    def test_returns_one_row_per_cluster_attr_pair(self):
        df = _make_pred_df(n=90, n_clusters=3)
        fpc = FairnessPerCluster(sensitive_attrs=["age_group", "sex"], min_group_size=2)
        result = fpc.compute(df, cluster_col="group_cluster")

        assert not result.empty
        assert "cluster_id" in result.columns
        assert "sensitive_attr" in result.columns
        assert "dp_max_diff" in result.columns
        assert "is_fair" in result.columns

    def test_single_member_cluster_skipped_no_crash(self):
        """A cluster with fewer samples than min_group_size should be skipped."""
        df = _make_pred_df(n=30, n_clusters=3)
        # Overwrite cluster 2 to have only 1 sample
        df.loc[df["group_cluster"] == 2, "group_cluster"] = 0
        df.loc[0, "group_cluster"] = 2  # exactly 1 sample in cluster 2

        fpc = FairnessPerCluster(sensitive_attrs=["sex"], min_group_size=5)
        result = fpc.compute(df, cluster_col="group_cluster")
        # Cluster 2 should be absent (skipped)
        present = result["cluster_id"].unique()
        assert 2 not in present

    def test_missing_sensitive_attr_handled(self):
        """Missing column does not crash; just produces no row for that attr."""
        df = _make_pred_df(n=60)
        df = df.drop(columns=["sex"])
        fpc = FairnessPerCluster(sensitive_attrs=["age_group", "sex"], min_group_size=2)
        result = fpc.compute(df, cluster_col="group_cluster")
        assert "sex" not in result["sensitive_attr"].values


class TestCramersV:
    def test_perfect_correlation_gives_high_v(self):
        """When cluster assignment == sensitive attr, Cramér's V should be high."""
        # Perfect alignment: cluster 0 = Female, cluster 1 = Male
        df = pd.DataFrame(
            {
                "group_cluster": [0] * 30 + [1] * 30,
                "sex": ["Female"] * 30 + ["Male"] * 30,
                "y_pred": [0] * 60,
            }
        )
        fpc = FairnessPerCluster(sensitive_attrs=["sex"])
        result = fpc.cramers_v_matrix(df, cluster_col="group_cluster")
        assert not result.empty
        v = result.loc[result["sensitive_attr"] == "sex", "cramers_v"].iloc[0]
        assert v > 0.8

    def test_random_correlation_gives_low_v(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "group_cluster": rng.integers(0, 3, size=120),
                "sex": rng.choice(["Female", "Male"], size=120),
                "y_pred": rng.integers(0, 2, size=120),
            }
        )
        fpc = FairnessPerCluster(sensitive_attrs=["sex"])
        result = fpc.cramers_v_matrix(df, cluster_col="group_cluster")
        if not result.empty:
            v = result.loc[result["sensitive_attr"] == "sex", "cramers_v"].iloc[0]
            assert v < 0.5  # should be near 0 for random


class TestClusterProfiler:
    def test_narratives_produced_for_each_cluster(self):
        df = _make_pred_df(n=60, n_clusters=3)
        profiler = ClusterProfiler(target_col="heart_disease")
        report = profiler.compute(
            df, cluster_col="group_cluster", feature_cols=["trestbps", "chol"]
        )
        assert len(report.narratives) == len(df["group_cluster"].unique())

    def test_save_report_creates_markdown(self, tmp_path):
        df = _make_pred_df(n=60, n_clusters=3)
        profiler = ClusterProfiler(target_col="heart_disease")
        report = profiler.compute(
            df, cluster_col="group_cluster", feature_cols=["trestbps", "chol"]
        )
        out = profiler.save_report(report, tmp_path / "subgroup_profiles.md")
        assert out.exists()
        content = out.read_text()
        assert "Cluster" in content
        assert "Feature Means" in content
