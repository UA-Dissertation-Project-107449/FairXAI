"""Unit tests for ClusteringEngine."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

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
        engine = ClusteringEngine(
            config={"kmeans": {"parameters": {"n_clusters": [3], "n_init": 5, "random_state": 42}}}
        )
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])
        assert set(result.feature_cols) == {"feat_a", "feat_b"}


class TestClusteringEngineValidityGate:
    """min_cluster_size / min_clusters disqualify degenerate solutions.

    Defaults must be a no-op so the WebApp adapter is byte-for-byte unchanged.
    """

    def test_default_params_do_not_filter(self):
        df = _make_df(n=60)
        cfg = {"kmeans": {"parameters": {"n_clusters": [3], "n_init": 5, "random_state": 42}}}
        engine = ClusteringEngine(config=cfg)  # defaults: abs=1, frac=0.0
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])
        assert result.n_clusters >= 2

    def test_min_cluster_size_too_large_raises(self):
        """No solution can have clusters larger than n → everything disqualified."""
        df = _make_df(n=60)
        cfg = {"kmeans": {"parameters": {"n_clusters": [3], "n_init": 5, "random_state": 42}}}
        engine = ClusteringEngine(config=cfg, min_cluster_size_abs=10_000)
        with pytest.raises(ClusteringError):
            engine.fit(df, feature_cols=["feat_a", "feat_b"])

    def test_effective_threshold_uses_fraction(self):
        """max(abs, frac*n): frac=0.9 on 60 rows → 54 > any 3-way split → raises."""
        df = _make_df(n=60)
        cfg = {"kmeans": {"parameters": {"n_clusters": [3], "n_init": 5, "random_state": 42}}}
        engine = ClusteringEngine(config=cfg, min_cluster_size_abs=1, min_cluster_size_frac=0.9)
        with pytest.raises(ClusteringError):
            engine.fit(df, feature_cols=["feat_a", "feat_b"])

    def test_modest_threshold_still_passes_balanced(self):
        """A balanced 3-way split of 90 rows survives a min size of 20."""
        df = _make_df(n=90, n_clusters=3)
        cfg = {"kmeans": {"parameters": {"n_clusters": [3], "n_init": 5, "random_state": 42}}}
        engine = ClusteringEngine(config=cfg, min_cluster_size_abs=20)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])
        import numpy as np

        _, counts = np.unique(result.group_cluster.to_numpy(), return_counts=True)
        assert counts.min() >= 20

    def test_min_silhouette_floor_rejects_below_threshold(self):
        """An opt-in silhouette floor disqualifies the winner when it's too weak."""
        df = _make_df(n=60)
        cfg = {"kmeans": {"parameters": {"n_clusters": [3], "n_init": 5, "random_state": 42}}}
        engine = ClusteringEngine(config=cfg, min_silhouette=0.99)  # impossibly high
        with pytest.raises(ClusteringError, match="stable enough"):
            engine.fit(df, feature_cols=["feat_a", "feat_b"])

    def test_min_silhouette_none_disables_floor(self):
        """Default (None) keeps the floor off → WebApp byte-identical."""
        df = _make_df(n=60)
        cfg = {"kmeans": {"parameters": {"n_clusters": [3], "n_init": 5, "random_state": 42}}}
        engine = ClusteringEngine(config=cfg)  # min_silhouette defaults to None
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])
        assert result.n_clusters >= 2


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

    def test_dbscan_high_noise_candidate_is_rejected(self):
        rng = np.random.default_rng(7)
        cluster_a = rng.normal(0, 0.02, size=(6, 2))
        cluster_b = rng.normal(3, 0.02, size=(6, 2))
        noise = rng.uniform(-20, 20, size=(40, 2))
        X = np.vstack([cluster_a, cluster_b, noise])
        df = pd.DataFrame(X, columns=["feat_a", "feat_b"])
        cfg = {
            "dbscan": {
                "parameters": {
                    "eps": [0.25],
                    "min_samples": [3],
                    "max_noise_fraction": 0.30,
                }
            }
        }
        engine = ClusteringEngine(config=cfg)

        with pytest.raises(ClusteringError) as exc:
            engine.fit(df, feature_cols=["feat_a", "feat_b"])

        notes = [d.note or "" for d in exc.value.diagnostics]
        assert any("rejected: noise_fraction" in note for note in notes)

    def test_dbscan_keeps_accepted_noise_as_separate_cluster(self):
        rng = np.random.default_rng(9)
        cluster_a = rng.normal(0, 0.03, size=(12, 2))
        cluster_b = rng.normal(3, 0.03, size=(12, 2))
        noise = np.array([[12.0, 12.0], [-12.0, -12.0]])
        X = np.vstack([cluster_a, cluster_b, noise])
        df = pd.DataFrame(X, columns=["feat_a", "feat_b"])
        cfg = {
            "dbscan": {
                "parameters": {
                    "eps": [0.35],
                    "min_samples": [3],
                    "max_noise_fraction": 0.30,
                }
            }
        }
        engine = ClusteringEngine(config=cfg)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])

        counts = result.group_cluster.value_counts()
        assert result.method == "dbscan"
        assert counts.min() == 2
        assert result.n_clusters == 3


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


class TestClusteringEngineHierarchicalRowCeiling:
    """Ward linkage is O(n^2) in memory, so it is capped by row count.

    Every other method in the engine is memory-bounded, so exceeding the ceiling
    must drop hierarchical alone and leave the rest of the study intact.
    """

    def test_hierarchical_runs_below_the_ceiling(self):
        df = _make_df(n=60)
        cfg = {"hierarchical": {"parameters": {"n_clusters": [3], "max_samples": 100}}}
        engine = ClusteringEngine(config=cfg)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])

        assert result.method == "hierarchical"
        assert len(result.group_cluster) == len(df)

    def test_hierarchical_is_skipped_above_the_ceiling(self):
        df = _make_df(n=60)
        cfg = {
            "hierarchical": {"parameters": {"n_clusters": [3], "max_samples": 10}},
            "kmeans": {"parameters": {"n_clusters": [3], "n_init": 5, "random_state": 42}},
        }
        engine = ClusteringEngine(config=cfg)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])

        # KMeans still produces a solution; hierarchical contributed no candidate.
        assert result.method == "kmeans"
        notes = [d.note for d in result.diagnostics if d.method == "hierarchical"]
        assert notes and "skipped" in notes[0]
        assert "max_samples=10" in notes[0]

    def test_skipped_hierarchical_is_visible_in_diagnostics_csv(self, tmp_path):
        df = _make_df(n=60)
        cfg = {
            "hierarchical": {"parameters": {"n_clusters": [3], "max_samples": 10}},
            "kmeans": {"parameters": {"n_clusters": [3], "n_init": 5, "random_state": 42}},
        }
        engine = ClusteringEngine(config=cfg)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])
        saved = pd.read_csv(engine.save_diagnostics(result, tmp_path))

        skipped = saved[saved["method"] == "hierarchical"]
        assert len(skipped) == 1
        assert "skipped" in str(skipped.iloc[0]["note"])

    def test_ceiling_can_be_disabled(self):
        df = _make_df(n=60)
        cfg = {"hierarchical": {"parameters": {"n_clusters": [3], "max_samples": 0}}}
        engine = ClusteringEngine(config=cfg)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])

        assert result.method == "hierarchical"

    def test_shipped_config_ceiling_splits_the_cardio70k_variants(self):
        """10k subsample must cluster hierarchically; the full 70k run must not."""
        import yaml

        root = Path(__file__).resolve().parents[2]
        cfg = yaml.safe_load(open(root / "configs" / "experiments" / "clustering.yaml"))
        ceiling = cfg["clustering_methods"]["hierarchical"]["parameters"]["max_samples"]

        assert 9822 <= ceiling, "the cardio70k 10k subsample must stay under the ceiling"
        assert ceiling < 68749, "the full cardio70k train+test row set must exceed it"


class TestClusteringEngineDbscanMemoryBudget:
    """DBSCAN's neighbour lists are O(n^2) when eps is large relative to density.

    The cost depends on eps, not on row count alone, so the guard is per eps: a
    cheap eps must still run at full row count while an expensive one is dropped.
    """

    @staticmethod
    def _dbscan_config(max_neighbor_gib, eps_grid):
        return {
            "dbscan": {
                "parameters": {
                    "eps": eps_grid,
                    "min_samples": [5],
                    "max_neighbor_gib": max_neighbor_gib,
                }
            }
        }

    def test_estimate_grows_quadratically_in_rows(self):
        """The estimate is what the budget is checked against, so it must scale."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 13))

        small = ClusteringEngine._estimate_dbscan_neighbor_gib(X, 5.0, 10_000)
        large = ClusteringEngine._estimate_dbscan_neighbor_gib(X, 5.0, 20_000)

        assert large == pytest.approx(small * 4, rel=0.01)

    def test_estimate_grows_with_eps(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 13))

        tight = ClusteringEngine._estimate_dbscan_neighbor_gib(X, 0.5, 68_749)
        wide = ClusteringEngine._estimate_dbscan_neighbor_gib(X, 5.0, 68_749)

        assert wide > tight * 100

    def test_expensive_eps_is_skipped_and_recorded(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(300, 2)), columns=["feat_a", "feat_b"])
        engine = ClusteringEngine(config=self._dbscan_config(1e-9, [5.0]))

        with pytest.raises(ClusteringError) as excinfo:
            engine.fit(df, feature_cols=["feat_a", "feat_b"])

        notes = [d.note or "" for d in excinfo.value.diagnostics if d.method == "dbscan"]
        assert notes, "the skipped candidate must still appear in diagnostics"
        assert all("max_neighbor_gib" in n for n in notes)

    def test_cheap_eps_still_runs_under_the_same_budget(self):
        """A budget that rejects a wide eps must not reject a tight one."""
        rng = np.random.default_rng(0)
        blobs = np.vstack(
            [rng.normal(loc, 0.15, size=(100, 2)) for loc in ([0, 0], [6, 6], [0, 6])]
        )
        df = pd.DataFrame(blobs, columns=["feat_a", "feat_b"])
        engine = ClusteringEngine(config=self._dbscan_config(0.001, [0.5, 5.0]))

        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])

        by_eps = {
            d.params["eps"]: (d.note or "") for d in result.diagnostics if d.method == "dbscan"
        }
        assert "max_neighbor_gib" in by_eps[5.0], "wide eps must exceed this budget"
        assert "max_neighbor_gib" not in by_eps[0.5], "tight eps must stay affordable"
        assert result.method == "dbscan"

    def test_budget_can_be_disabled(self):
        rng = np.random.default_rng(0)
        blobs = np.vstack(
            [rng.normal(loc, 0.15, size=(100, 2)) for loc in ([0, 0], [6, 6], [0, 6])]
        )
        df = pd.DataFrame(blobs, columns=["feat_a", "feat_b"])
        engine = ClusteringEngine(config=self._dbscan_config(0, [0.5]))

        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])

        notes = [d.note or "" for d in result.diagnostics if d.method == "dbscan"]
        assert all("max_neighbor_gib" not in n for n in notes)

    def test_shipped_config_budget_splits_a_13d_eps_grid(self):
        """The shipped budget must keep the cheap end of the eps grid and drop the wide end.

        The fixture is synthetic 13-D gaussian noise, which is *sparser* than the
        real cardio70k feature space: on the real data the study skipped eps=3.0
        (10.2 GiB) and eps=5.0 (42.3 GiB), where this fixture only skips 5.0. So
        this test pins the mechanism and the budget's order of magnitude, not the
        exact cut point for any one dataset -- the guard measures density at run
        time precisely because that cut point is dataset-dependent.
        """
        import yaml

        root = Path(__file__).resolve().parents[2]
        cfg = yaml.safe_load(open(root / "configs" / "experiments" / "clustering.yaml"))
        params = cfg["clustering_methods"]["dbscan"]["parameters"]
        budget = params["max_neighbor_gib"]

        rng = np.random.default_rng(0)
        X = rng.normal(size=(2000, 13))  # 13-D standardised, as the study clusters
        over = {
            eps: ClusteringEngine._estimate_dbscan_neighbor_gib(X, eps, 68_749) > budget
            for eps in params["eps"]
        }

        assert over[5.0], "the widest eps must be skipped even on sparse data"
        assert not any(
            over[e] for e in params["eps"] if e <= 2.0
        ), "the cheap end of the grid must survive the budget at the full row count"

    def test_estimate_ignores_non_finite_rows(self):
        """pairwise_distances rejects NaN, so the estimate must exclude those rows."""
        rng = np.random.default_rng(0)
        clean = rng.normal(size=(500, 13))
        dirty = np.vstack([clean, np.full((10, 13), np.nan)])

        assert ClusteringEngine._estimate_dbscan_neighbor_gib(dirty, 5.0, 68_749) == pytest.approx(
            ClusteringEngine._estimate_dbscan_neighbor_gib(clean, 5.0, 68_749), rel=0.15
        )

    def test_estimate_returns_none_when_nothing_is_finite(self):
        allnan = np.full((50, 13), np.nan)

        assert ClusteringEngine._estimate_dbscan_neighbor_gib(allnan, 5.0, 68_749) is None

    def test_nan_data_fails_as_a_contained_clustering_error(self):
        """NaN must surface as the engine's own error, not the budget check's.

        Every sklearn method here rejects NaN, so NaN data legitimately produces
        no solution -- that is pre-existing behaviour and four_site_uci has always
        ended this way. What matters is *which* error escapes. The budget check
        runs outside the per-candidate try, so before this was contained it raised
        ValueError("Input contains NaN") straight out of pairwise_distances,
        replacing the engine's own ClusteringError and reporting a memory guard as
        the cause of a data-quality failure.
        """
        rng = np.random.default_rng(0)
        blobs = np.vstack(
            [rng.normal(loc, 0.15, size=(100, 2)) for loc in ([0, 0], [6, 6], [0, 6])]
        )
        df = pd.DataFrame(blobs, columns=["feat_a", "feat_b"])
        df.loc[0, "feat_a"] = np.nan
        engine = ClusteringEngine(config=self._dbscan_config(4.0, [0.5]))

        with pytest.raises(ClusteringError) as excinfo:
            engine.fit(df, feature_cols=["feat_a", "feat_b"])

        assert "No clustering method produced a valid solution" in str(excinfo.value)


def _blobs_with_planted_noise(n_noise=9, seed=11):
    """Two tight blobs plus scattered points no blob can claim.

    The noise points are the whole question: DBSCAN sets them aside, KMeans has
    to absorb them. Scored on the kept points only, DBSCAN was paid for the
    discarding; scored on all of them it is not.
    """
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(0.0, 0.05, size=(21, 2))
    cluster_b = rng.normal(4.0, 0.05, size=(20, 2))
    noise = rng.uniform(-8.0, 12.0, size=(n_noise, 2))
    X = np.vstack([cluster_a, cluster_b, noise])
    return pd.DataFrame(X, columns=["feat_a", "feat_b"])


class TestDbscanNoiseIsScored:
    """DBSCAN is scored on every point, with noise as the label it really is."""

    _DBSCAN_CFG = {
        "dbscan": {"parameters": {"eps": [0.4], "min_samples": [3], "max_noise_fraction": 0.30}}
    }

    def test_silhouette_is_the_all_points_score(self):
        from sklearn.preprocessing import StandardScaler

        df = _blobs_with_planted_noise()
        engine = ClusteringEngine(config=self._DBSCAN_CFG)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])

        assert result.method == "dbscan"
        assert result.n_noise > 0

        X = StandardScaler().fit_transform(df[["feat_a", "feat_b"]].values)
        labels = result.group_cluster.to_numpy()
        noise_label = labels.max()
        kept = labels != noise_label

        all_points = silhouette_score(X, labels)
        kept_only = silhouette_score(X[kept], labels[kept])

        assert result.silhouette == pytest.approx(all_points)
        # The old rule; kept here so the test fails if the score drifts back to it.
        assert kept_only > all_points

    def test_a_partition_of_every_point_wins_over_one_that_discards(self):
        df = _blobs_with_planted_noise()
        cfg = {
            "kmeans": {"parameters": {"n_clusters": [2, 3], "n_init": 10, "random_state": 0}},
            **self._DBSCAN_CFG,
        }
        engine = ClusteringEngine(config=cfg)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])

        assert result.method == "kmeans"
        assert result.n_noise == 0

    def test_noise_count_is_reported(self, tmp_path):
        df = _blobs_with_planted_noise()
        engine = ClusteringEngine(config=self._DBSCAN_CFG)
        result = engine.fit(df, feature_cols=["feat_a", "feat_b"])

        assert result.noise_fraction == pytest.approx(result.n_noise / len(df))
        saved = pd.read_csv(engine.save_diagnostics(result, tmp_path))
        assert "n_noise" in saved.columns
        assert saved["n_noise"].max() == result.n_noise
