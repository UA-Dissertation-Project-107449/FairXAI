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


def test_feature_columns_keep_sensitive_columns_and_report_the_rest():
    # Sensitive attributes are features now. Excluding them used to leave narrow
    # datasets with a single column, and clustering one column is not clustering.
    df = _frame().assign(patient_id=[1, 2, 3, 4], notes=["a", "b", "c", "d"])

    cols, excluded = clustering_module.resolve_feature_columns(
        df, target_column="target", index_column="patient_id", sensitive_columns=["sex"]
    )

    assert cols == ["feat_a", "feat_b", "sex"]
    assert excluded == [
        {"column": "target", "reason": "target"},
        {"column": "group_cluster", "reason": "engine_output"},
        {"column": "patient_id", "reason": "index"},
        {"column": "notes", "reason": "non_numeric"},
    ]


def test_feature_columns_keep_a_column_named_sex_when_it_was_not_nominated():
    # The engine's own default list drops "sex" by name. On an arbitrary upload
    # that is the wrong call: the WebApp asks the user which columns are
    # sensitive, and a column nobody nominated is just a feature.
    cols, _ = clustering_module.resolve_feature_columns(_frame(), target_column="target")

    assert cols == ["feat_a", "feat_b", "sex"]


def test_feature_columns_drop_an_undeclared_integer_identifier():
    df = _frame().assign(record_no=[10, 11, 12, 13])

    cols, excluded = clustering_module.resolve_feature_columns(df, target_column="target")

    assert "record_no" not in cols
    assert {"column": "record_no", "reason": "identifier"} in excluded


def test_feature_columns_keep_an_all_distinct_float_measurement():
    # The trap this guards: `nunique == len(df)` alone. A continuous measurement
    # over few rows is routinely all-distinct, and dropping it would be a worse
    # and quieter bug than the one the identifier rule exists to prevent.
    df = _frame().assign(cholesterol=[210.4, 233.1, 188.7, 265.2])

    cols, excluded = clustering_module.resolve_feature_columns(df, target_column="target")

    assert "cholesterol" in cols
    assert all(item["column"] != "cholesterol" for item in excluded)


def test_run_clustering_refuses_a_single_feature_column(tmp_path):
    # dataset26: three features, one declared as index, one as sensitive. The
    # engine used to run and report a silhouette of 1.0 — what partitioning a
    # line always gives — and only the PCA overlay noticed, by returning nothing.
    csv_path = tmp_path / "narrow.csv"
    pd.DataFrame(
        {
            "feature_0": [1, 2, 3, 4],
            "feature_1": [0.0, 0.1, 4.9, 5.0],
            "target_variable": [0, 0, 1, 1],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="at least 2 numeric feature columns"):
        clustering_module.run_clustering(
            csv_path, "target_variable", index_column="feature_0", sensitive_columns=["feature_1"]
        )


def test_dbscan_eps_grid_scales_with_the_column_count():
    # Thirteen standardised columns put typical neighbour distances well past
    # the engine's 1.0 default ceiling, at which every point is noise.
    rng = pd.DataFrame(
        {f"col_{index}": [(row * 7 + index * 3) % 11 for row in range(60)] for index in range(13)}
    )

    grid = clustering_module._dbscan_eps_grid(rng, list(rng.columns))

    assert len(grid) >= 1
    assert min(grid) > 1.0


def test_dbscan_eps_grid_measures_boolean_columns_too():
    # The engine scales and clusters booleans like any other column, so a grid
    # built without them is sized for fewer dimensions than DBSCAN is handed.
    numeric = pd.DataFrame(
        {f"col_{index}": [(row * 5 + index) % 7 for row in range(40)] for index in range(3)}
    )
    with_flags = numeric.assign(
        **{f"flag_{index}": [(row + index) % 2 == 0 for row in range(40)] for index in range(6)}
    )

    numeric_grid = clustering_module._dbscan_eps_grid(numeric, list(numeric.columns))
    flag_grid = clustering_module._dbscan_eps_grid(with_flags, list(with_flags.columns))

    assert max(flag_grid) > max(numeric_grid)


def test_dbscan_eps_grid_falls_back_when_there_is_nothing_to_measure():
    frame = pd.DataFrame({"only_row": [1.0]})

    assert clustering_module._dbscan_eps_grid(frame, ["only_row"]) == list(
        clustering_module._DBSCAN_DEFAULT_EPS_GRID
    )


def test_engine_config_carries_the_eps_grid_for_dbscan_only():
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 9.0], "b": [1.0, 0.0, 3.0, 8.0]})

    dbscan_config = clustering_module._engine_config_for_method("dbscan", frame, ["a", "b"])
    kmeans_config = clustering_module._engine_config_for_method("kmeans", frame, ["a", "b"])

    assert dbscan_config["dbscan"]["parameters"]["eps"]
    assert kmeans_config == {"kmeans": {}}
    assert clustering_module._engine_config_for_method("auto", frame, ["a", "b"]) is None


def test_clustering_failure_reports_why_the_attempts_were_rejected():
    # "No valid solution" names no cause, so the UI can only say "failed".
    from fairxai.clustering.engine import ClusteringError
    from fairxai.clustering.models import ClusterDiagnostics

    error = ClusteringError(
        "No clustering method produced a valid solution",
        diagnostics=[
            ClusterDiagnostics(
                method="dbscan",
                params={"eps": 2.7, "min_samples": 5},
                n_clusters=3,
                silhouette=0.19,
                note="rejected: noise_fraction=56.0% > max_noise_fraction=30.0%",
            ),
            ClusterDiagnostics(
                method="dbscan",
                params={"eps": 1.0, "min_samples": 5},
                n_clusters=0,
                silhouette=None,
                note="only 0 cluster(s) + 297 noise",
            ),
        ],
    )

    message = clustering_module._describe_clustering_failure(error, "dbscan")

    assert "eps=2.7" in message
    assert "noise_fraction=56.0%" in message


def test_run_clustering_keeps_the_diagnostics_when_it_rewrites_the_message(tmp_path, monkeypatch):
    # The message is rewritten for the UI; the structured attempts behind it are
    # the only machine-readable record of why, so they must survive the rewrite.
    from fairxai.clustering.engine import ClusteringError
    from fairxai.clustering.models import ClusterDiagnostics

    diagnostics = [
        ClusterDiagnostics(
            method="dbscan",
            params={"eps": 2.7, "min_samples": 5},
            n_clusters=3,
            silhouette=0.19,
            note="rejected: noise_fraction=56.0% > max_noise_fraction=30.0%",
        )
    ]

    class FailingEngine:
        def __init__(self, **kwargs):
            pass

        def fit(self, df, feature_cols=None):
            raise ClusteringError(
                "No clustering method produced a valid solution", diagnostics=diagnostics
            )

    monkeypatch.setattr(clustering_module, "ClusteringEngine", FailingEngine)

    with pytest.raises(ClusteringError) as excinfo:
        clustering_module.run_clustering(
            _write_csv(tmp_path), target_column="target", method="dbscan"
        )

    assert excinfo.value.diagnostics == diagnostics
    assert "noise_fraction=56.0%" in str(excinfo.value)


def test_clustering_failure_keeps_the_original_message_without_diagnostics():
    from fairxai.clustering.engine import ClusteringError

    error = ClusteringError("No clustering method produced a valid solution")

    assert clustering_module._describe_clustering_failure(error, "dbscan") == str(error)
