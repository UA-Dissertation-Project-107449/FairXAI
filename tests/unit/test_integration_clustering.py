from types import SimpleNamespace

import pandas as pd
import pytest

from fairxai.cli import main as cli_main
from fairxai.integration import clustering as clustering_module


def _write_csv(tmp_path):
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "feat_a": [0.0, 0.1, 4.9, 5.0],
            "feat_b": [0.0, 0.2, 5.1, 5.0],
            "target": [0, 0, 1, 1],
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def test_run_clustering_uses_selected_method_config(tmp_path, monkeypatch):
    captured = {}

    class FakeEngine:
        def __init__(self, config=None, feature_exclude=None, min_silhouette=None, **kwargs):
            captured["config"] = config
            captured["feature_exclude"] = feature_exclude
            captured["min_silhouette"] = min_silhouette

        def fit(self, df, feature_cols=None):
            return SimpleNamespace(
                group_cluster=pd.Series([0, 0, 1, 1], index=df.index),
                method="kmeans",
                n_clusters=2,
                silhouette=0.82,
                feature_cols=["feat_a", "feat_b"],
            )

    class FakeProfiler:
        def __init__(self, target_col):
            self.target_col = target_col

        def compute(self, df, cluster_col, feature_cols):
            return SimpleNamespace(
                narratives={0: "Low values", 1: "High values"},
                global_means=pd.Series({"feat_a": 2.5, "feat_b": 2.575}),
                feature_means=pd.DataFrame(
                    {"feat_a": [0.05, 4.95], "feat_b": [0.1, 5.05]},
                    index=[0, 1],
                ),
            )

    monkeypatch.setattr(clustering_module, "ClusteringEngine", FakeEngine)
    monkeypatch.setattr(clustering_module, "ClusterProfiler", FakeProfiler)

    result = clustering_module.run_clustering(_write_csv(tmp_path), "target", method="kmeans")

    assert captured["config"] == {"kmeans": {}}
    assert captured["feature_exclude"] == ["target"]
    # WebApp adapter turns the stability floor on by default.
    assert captured["min_silhouette"] == 0.05
    assert result["requested_method"] == "kmeans"
    assert result["method"] == "kmeans"
    assert result["n_clusters"] == 2


def test_run_clustering_rejects_invalid_method(tmp_path):
    with pytest.raises(ValueError, match="Unsupported clustering method"):
        clustering_module.run_clustering(_write_csv(tmp_path), "target", method="spectral")


def test_cli_forwards_clustering_method(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_run_clustering(csv_path, target_column, pca2d=None, method="auto"):
        captured.update(
            {
                "csv_path": str(csv_path),
                "target_column": target_column,
                "pca2d": pca2d,
                "method": method,
            }
        )
        return {"requested_method": method, "method": method, "clusters": []}

    monkeypatch.setattr(clustering_module, "run_clustering", fake_run_clustering)
    csv_path = _write_csv(tmp_path)

    exit_code = cli_main.main(
        [
            "clustering",
            "--filename",
            str(csv_path),
            "--target-column",
            "target",
            "--method",
            "dbscan",
        ]
    )

    assert exit_code == 0
    assert captured["method"] == "dbscan"
    assert captured["target_column"] == "target"
    assert '"requested_method": "dbscan"' in capsys.readouterr().out
