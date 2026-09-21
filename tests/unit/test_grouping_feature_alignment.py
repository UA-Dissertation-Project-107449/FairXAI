"""Unit tests for the grouping study's training-feature alignment.

The grouping study reads the *unscaled* splits so cluster profiles stay in
clinical units, but those splits carry columns training dropped, the encoded
forms of the sensitive attributes, and missing values training imputed. These
tests pin the three helpers that close that gap.
"""

import sys
from pathlib import Path

import pandas as pd

_STUDIES_DIR = Path(__file__).parent.parent.parent / "scripts" / "studies"
sys.path.insert(0, str(_STUDIES_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import run_grouping_analysis as rga  # noqa: E402


def _write_scaled(tmp_path, dataset, binning, columns):
    """Write the training artifact the alignment reads its column list from."""
    ds_dir = tmp_path / f"{dataset}_{binning}"
    ds_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({c: [0.0] for c in columns}).to_csv(
        ds_dir / f"{dataset}_train_scaled.csv", index=False
    )
    return ds_dir


class TestTrainingFeatureSpace:
    def test_columns_absent_from_training_are_dropped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rga, "_PROCESSED_DIR", tmp_path)
        _write_scaled(tmp_path, "ds", "clinical", ["cp", "trestbps", "heart_disease"])
        df = pd.DataFrame({"cp": [1.0], "trestbps": [2.0], "ca": [3.0], "heart_disease": [0]})

        cols, meta = rga._training_feature_space("ds", "clinical", df, ["heart_disease"])

        assert cols == ["cp", "trestbps"]
        assert meta["dropped"] == ["ca"]
        assert meta["aligned"] is True

    def test_excluded_columns_stay_excluded_even_when_training_kept_them(
        self, tmp_path, monkeypatch
    ):
        """Alignment narrows the feature set; it must never widen it."""
        monkeypatch.setattr(rga, "_PROCESSED_DIR", tmp_path)
        _write_scaled(tmp_path, "ds", "clinical", ["cp", "heart_disease"])
        df = pd.DataFrame({"cp": [1.0], "heart_disease": [0]})

        cols, _ = rga._training_feature_space("ds", "clinical", df, ["heart_disease"])

        assert cols == ["cp"]

    def test_missing_scaled_split_falls_back_to_engine_defaults(self, tmp_path, monkeypatch):
        """No training artifact must not mean no clustering."""
        monkeypatch.setattr(rga, "_PROCESSED_DIR", tmp_path)
        df = pd.DataFrame({"cp": [1.0], "heart_disease": [0]})

        cols, meta = rga._training_feature_space("ds", "clinical", df, ["heart_disease"])

        assert cols == []
        assert meta["aligned"] is False

    def test_zero_overlap_falls_back_rather_than_clustering_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rga, "_PROCESSED_DIR", tmp_path)
        _write_scaled(tmp_path, "ds", "clinical", ["totally", "different"])
        df = pd.DataFrame({"cp": [1.0], "heart_disease": [0]})

        cols, meta = rga._training_feature_space("ds", "clinical", df, ["heart_disease"])

        assert cols == []
        assert meta["aligned"] is False


class TestExpandExclusions:
    def test_encoded_forms_of_an_excluded_attribute_are_added(self):
        df = pd.DataFrame(
            {"sex": [0], "sex_bin": [0], "sex_extended": [0], "age_group": [0], "cp": [0]}
        )

        expanded = rga._expand_exclusions(df, ["sex", "age_group"])

        assert expanded == ["age_group", "sex", "sex_bin", "sex_extended"]

    def test_unrelated_column_sharing_a_prefix_survives(self):
        """Only ``<attr>`` and ``<attr>_<encoding>`` are encodings of ``<attr>``."""
        df = pd.DataFrame({"sex": [0], "sextant": [0]})

        assert rga._expand_exclusions(df, ["sex"]) == ["sex"]

    def test_names_absent_from_the_frame_are_kept(self):
        """The list is also passed to the engine, which sees other frames."""
        df = pd.DataFrame({"cp": [0]})

        assert rga._expand_exclusions(df, ["ethnicity"]) == ["ethnicity"]


class TestImputeLikeTraining:
    def test_residual_nan_is_median_filled_and_counted(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, None], "b": [1.0, None, 3.0, 4.0]})

        filled, n_rows = rga._impute_like_training(df, ["a", "b"])

        assert n_rows == 2  # two distinct rows carried a NaN
        assert not filled[["a", "b"]].isna().any().any()
        assert filled.loc[3, "a"] == 2.0  # median of 1, 2, 3

    def test_the_callers_frame_is_never_modified(self):
        """The caller writes its frame back to the splits every other stage reads."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, None]})
        before = df.copy()

        rga._impute_like_training(df, ["a"])

        pd.testing.assert_frame_equal(df, before)

    def test_row_count_is_preserved(self):
        """Predictions are merged by position, so rows must never be dropped."""
        df = pd.DataFrame({"a": [1.0, None, 3.0]})

        filled, _ = rga._impute_like_training(df, ["a"])

        assert len(filled) == 3

    def test_complete_data_is_untouched(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})

        filled, n_rows = rga._impute_like_training(df, ["a"])

        assert n_rows == 0
        pd.testing.assert_frame_equal(filled, df)

    def test_columns_outside_the_feature_set_keep_their_nan(self):
        df = pd.DataFrame({"a": [1.0, None], "untouched": [None, None]})

        filled, _ = rga._impute_like_training(df, ["a"])

        assert filled["untouched"].isna().all()
