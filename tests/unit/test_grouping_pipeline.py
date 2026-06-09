"""Unit tests for the leakage-safe pre-train clustering module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairxai.clustering.grouping_pipeline import (
    GROUP_CLUSTER_COL,
    assign_clusters_nearest_centroid,
    cluster_and_persist,
)


def _two_blob_df(n_per: int = 40, seed: int = 0) -> pd.DataFrame:
    """Two well-separated Gaussian blobs in 2D + a binary target."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[-5.0, -5.0], scale=0.5, size=(n_per, 2))
    b = rng.normal(loc=[5.0, 5.0], scale=0.5, size=(n_per, 2))
    x = np.vstack([a, b])
    return pd.DataFrame(
        {
            "f0": x[:, 0],
            "f1": x[:, 1],
            "heart_disease": np.tile([0, 1], n_per),
        }
    )


def test_assign_nearest_centroid_matches_obvious_blobs():
    """Test rows near a train blob get that blob's label."""
    train = _two_blob_df(n_per=30, seed=1)
    labels = np.array([0] * 30 + [1] * 30)
    # Two test points: one near blob 0 (-5,-5), one near blob 1 (5,5).
    test = pd.DataFrame({"f0": [-4.8, 4.9], "f1": [-5.1, 5.2], "heart_disease": [0, 1]})

    assigned = assign_clusters_nearest_centroid(train, test, ["f0", "f1"], labels)

    assert assigned[0] == labels[0]  # near blob whose label is labels[0..29]
    assert assigned[1] == labels[30]  # near the other blob
    assert assigned[0] != assigned[1]


def _write_splits(tmp_path, dataset, binning, train_df, test_df):
    ds_dir = tmp_path / f"{dataset}_{binning}"
    ds_dir.mkdir(parents=True)
    train_df.to_csv(ds_dir / f"{dataset}_train.csv", index=False)
    test_df.to_csv(ds_dir / f"{dataset}_test.csv", index=False)
    return ds_dir


def test_cluster_and_persist_writes_labels_and_is_leakage_safe(tmp_path):
    """group_cluster lands in both splits; test labels are a subset of train labels."""
    train_df = _two_blob_df(n_per=40, seed=2)
    test_df = _two_blob_df(n_per=15, seed=3)
    ds_dir = _write_splits(tmp_path, "toy", "fixed_10yr", train_df, test_df)

    result = cluster_and_persist(
        dataset="toy",
        processed_dir=tmp_path,
        binning="fixed_10yr",
        method_cfg={"kmeans": {}},
        feature_exclude=["heart_disease", GROUP_CLUSTER_COL],
        out_dir=tmp_path / "out",
    )

    assert result is not None
    tr = pd.read_csv(ds_dir / "toy_train.csv")
    te = pd.read_csv(ds_dir / "toy_test.csv")
    assert GROUP_CLUSTER_COL in tr.columns
    assert GROUP_CLUSTER_COL in te.columns
    # No leakage: test labels can only be ones the train fit produced.
    assert set(te[GROUP_CLUSTER_COL].unique()) <= set(tr[GROUP_CLUSTER_COL].unique())
    # Diagnostics were emitted.
    assert (tmp_path / "out" / "cluster_diagnostics.csv").exists()


def test_cluster_and_persist_is_idempotent(tmp_path):
    """A second call skips when group_cluster is already present in both splits."""
    train_df = _two_blob_df(n_per=40, seed=4)
    test_df = _two_blob_df(n_per=15, seed=5)
    _write_splits(tmp_path, "toy", "fixed_10yr", train_df, test_df)

    kwargs = dict(
        dataset="toy",
        processed_dir=tmp_path,
        binning="fixed_10yr",
        method_cfg={"kmeans": {}},
        feature_exclude=["heart_disease", GROUP_CLUSTER_COL],
        out_dir=tmp_path / "out",
    )
    first = cluster_and_persist(**kwargs)
    second = cluster_and_persist(**kwargs)

    assert first is not None
    assert second is None  # idempotent skip


def test_cluster_and_persist_writes_scaled_variant(tmp_path):
    """Labels must land in the *scaled* split — the file the trainer reads."""
    train_df = _two_blob_df(n_per=40, seed=6)
    test_df = _two_blob_df(n_per=15, seed=7)
    ds_dir = tmp_path / "toy_fixed_10yr"
    ds_dir.mkdir(parents=True)
    # Both plain and scaled variants exist (as preprocess produces them).
    for suffix in ("_train.csv", "_train_scaled.csv"):
        train_df.to_csv(ds_dir / f"toy{suffix}", index=False)
    for suffix in ("_test.csv", "_test_scaled.csv"):
        test_df.to_csv(ds_dir / f"toy{suffix}", index=False)

    cluster_and_persist(
        dataset="toy",
        processed_dir=tmp_path,
        binning="fixed_10yr",
        method_cfg={"kmeans": {}},
        feature_exclude=["heart_disease", GROUP_CLUSTER_COL],
        out_dir=tmp_path / "out",
    )

    for fname in ("toy_train_scaled.csv", "toy_test_scaled.csv", "toy_train.csv", "toy_test.csv"):
        cols = pd.read_csv(ds_dir / fname).columns
        assert GROUP_CLUSTER_COL in cols, f"{fname} missing {GROUP_CLUSTER_COL}"


def test_cluster_and_persist_skips_when_validity_gate_fails(tmp_path):
    """No valid clustering (min size impossibly large) → skip, no group_cluster."""
    train_df = _two_blob_df(n_per=40, seed=8)
    test_df = _two_blob_df(n_per=15, seed=9)
    ds_dir = _write_splits(tmp_path, "toy", "fixed_10yr", train_df, test_df)

    result = cluster_and_persist(
        dataset="toy",
        processed_dir=tmp_path,
        binning="fixed_10yr",
        method_cfg={"kmeans": {}},
        feature_exclude=["heart_disease", GROUP_CLUSTER_COL],
        out_dir=tmp_path / "out",
        min_cluster_size_abs=10_000,  # no solution can satisfy this
    )

    assert result is None
    # group_cluster must NOT have been written to the splits.
    assert GROUP_CLUSTER_COL not in pd.read_csv(ds_dir / "toy_train.csv").columns
    assert GROUP_CLUSTER_COL not in pd.read_csv(ds_dir / "toy_test.csv").columns


def test_cluster_and_persist_missing_splits_returns_none(tmp_path):
    """No split files → graceful None (no crash)."""
    result = cluster_and_persist(
        dataset="absent",
        processed_dir=tmp_path,
        binning="fixed_10yr",
        method_cfg={"kmeans": {}},
        feature_exclude=None,
        out_dir=tmp_path / "out",
    )
    assert result is None
