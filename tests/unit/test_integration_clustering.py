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
            captured["feature_cols"] = feature_cols
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
    # Explicit columns, not feature_exclude: an exclude list is added to the
    # engine's hardcoded defaults, an explicit list replaces them.
    assert captured["feature_exclude"] is None
    assert captured["feature_cols"] == ["feat_a", "feat_b"]
    # WebApp adapter turns the stability floor on by default.
    assert captured["min_silhouette"] == 0.05
    assert result["requested_method"] == "kmeans"
    assert result["method"] == "kmeans"
    assert result["n_clusters"] == 2
    assert result["feature_columns"] == ["feat_a", "feat_b"]


def test_run_clustering_rejects_invalid_method(tmp_path):
    with pytest.raises(ValueError, match="Unsupported clustering method"):
        clustering_module.run_clustering(_write_csv(tmp_path), "target", method="spectral")


def test_cli_forwards_clustering_method(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_run_clustering(csv_path, target_column, method="auto", **kwargs):
        captured.update(
            {
                "csv_path": str(csv_path),
                "target_column": target_column,
                "method": method,
                **kwargs,
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
            "--index-column",
            "patient_id",
            "--sensitive-columns",
            # One argument with a space in it: proof the roles survive the trip
            # from the WebApp through argv without being re-split.
            "race group",
        ]
    )

    assert exit_code == 0
    assert captured["method"] == "dbscan"
    assert captured["target_column"] == "target"
    assert captured["index_column"] == "patient_id"
    assert captured["sensitive_columns"] == ["race group"]
    assert '"requested_method": "dbscan"' in capsys.readouterr().out


def test_cli_reads_the_column_list_off_a_stored_projection(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_run_clustering(csv_path, target_column, method="auto", **kwargs):
        captured.update(kwargs)
        return {"requested_method": method, "method": method, "clusters": []}

    monkeypatch.setattr(clustering_module, "run_clustering", fake_run_clustering)
    projection = tmp_path / "pca2d.json"
    projection.write_text('{"points": [[1.0, 2.0, 0]], "feature_columns": ["feat_a", "feat_b"]}')

    exit_code = cli_main.main(
        [
            "clustering",
            "--filename",
            str(_write_csv(tmp_path)),
            "--target-column",
            "target",
            "--pca2d-file",
            str(projection),
        ]
    )

    assert exit_code == 0
    assert captured["pca2d"] == [[1.0, 2.0, 0]]
    assert captured["pca2d_feature_columns"] == ["feat_a", "feat_b"]
    capsys.readouterr()


def test_cli_accepts_a_bare_coordinate_list_as_columns_unknown(tmp_path, monkeypatch, capsys):
    # Jobs profiled before the column list was published still send bare coords.
    # Unknown columns must read as unknown, not as "matches whatever we clustered".
    captured = {}

    def fake_run_clustering(csv_path, target_column, method="auto", **kwargs):
        captured.update(kwargs)
        return {"requested_method": method, "method": method, "clusters": []}

    monkeypatch.setattr(clustering_module, "run_clustering", fake_run_clustering)

    exit_code = cli_main.main(
        [
            "clustering",
            "--filename",
            str(_write_csv(tmp_path)),
            "--target-column",
            "target",
            "--pca2d-json",
            "[[1.0, 2.0, 0]]",
        ]
    )

    assert exit_code == 0
    assert captured["pca2d"] == [[1.0, 2.0, 0]]
    assert captured["pca2d_feature_columns"] is None
    capsys.readouterr()


# --- projection alignment ----------------------------------------------------
# The stored projection may cover different columns than the clustering did:
# characterization drops the index column, we drop the sensitive ones. Reusing
# it then places every point using features the clustering never saw, which is
# what makes clean clusters look interleaved on screen. Only the projection's
# own published column list can settle whether the two match.


def _result_over(feature_cols, index):
    return SimpleNamespace(
        group_cluster=pd.Series([0, 0, 1, 1], index=index),
        method="kmeans",
        n_clusters=2,
        silhouette=0.9,
        feature_cols=feature_cols,
    )


def _frame():
    return pd.DataFrame(
        {
            "feat_a": [0.0, 0.1, 4.9, 5.0],
            "feat_b": [0.0, 0.2, 5.1, 5.0],
            # Named to match the engine's hard-coded default exclude list, which
            # is exactly how the two column sets drift apart in practice.
            "sex": [0, 1, 0, 1],
            "target": [0, 0, 1, 1],
            "group_cluster": [0, 0, 1, 1],
        }
    )


_STORED = [[1.0, 2.0, 0], [3.0, 4.0, 0], [5.0, 6.0, 1], [7.0, 8.0, 1]]


def test_pca_clusters_reuse_stored_projection_when_columns_match():
    df = _frame()

    points, explained = clustering_module._build_pca_clusters(
        df,
        _result_over(["feat_a", "feat_b", "sex"], df.index),
        _STORED,
        ["feat_a", "feat_b", "sex"],
    )

    # Coordinates come through untouched; only the label becomes the cluster id.
    assert [p[:2] for p in points] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    assert [p[2] for p in points] == [0, 0, 1, 1]
    # Unknown here on purpose: the figure belongs to the run that built the coords.
    assert explained is None


def test_pca_clusters_recompute_when_the_column_sets_differ():
    df = _frame()

    # We held "sex" out as sensitive; the stored projection includes it.
    points, explained = clustering_module._build_pca_clusters(
        df, _result_over(["feat_a", "feat_b"], df.index), _STORED, ["feat_a", "feat_b", "sex"]
    )

    assert [p[:2] for p in points] != [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    assert explained is not None
    assert 0.0 < explained <= 1.0


def test_pca_clusters_recompute_when_the_projection_does_not_name_its_columns():
    # The regression this guards: inferring the column list from the CSV instead
    # of reading it off the projection. That inference cannot see that
    # characterization dropped the index column, so it declares a match between a
    # 2-feature projection and a 3-feature clustering and reuses the wrong coords.
    df = _frame()

    points, explained = clustering_module._build_pca_clusters(
        df, _result_over(["feat_a", "feat_b", "sex"], df.index), _STORED, None
    )

    assert [p[:2] for p in points] != [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    assert explained is not None


# --- feature-set selection ---------------------------------------------------


def test_feature_columns_exclude_target_index_and_sensitive():
    df = _frame().assign(patient_id=[1, 2, 3, 4])

    cols = clustering_module.resolve_feature_columns(
        df, target_column="target", index_column="patient_id", sensitive_columns=["sex"]
    )

    assert cols == ["feat_a", "feat_b"]


def test_feature_columns_keep_a_column_named_sex_when_it_was_not_nominated():
    # The engine's own default list drops "sex" by name. On an arbitrary upload
    # that is the wrong call: the WebApp asks the user which columns are
    # sensitive, and a column nobody nominated is just a feature.
    cols = clustering_module.resolve_feature_columns(_frame(), target_column="target")

    assert cols == ["feat_a", "feat_b", "sex"]
