"""Regression tests for non-dropping cardiac stratification fallbacks."""

import numpy as np
import pandas as pd
import pytest

from fairxai.data.preprocessors import CardiacPreprocessor


def test_rare_detailed_strata_fall_back_without_dropping_rows() -> None:
    rows = []
    row_id = 0
    for target in (0, 1):
        for age_group in ("40-49", "50-59"):
            for sex in ("Male", "Male", "Male", "Female"):
                rows.append(
                    {
                        "row_id": row_id,
                        "heart_disease": target,
                        "age_group": age_group,
                        "sex": sex,
                    }
                )
                row_id += 1
    df = pd.DataFrame(rows)
    original = df.copy(deep=True)
    preprocessor = CardiacPreprocessor(sensitive_attrs=["age_group", "sex"])

    train, test = preprocessor.stratified_split(
        df,
        target="heart_disease",
        test_size=0.25,
        random_state=42,
    )

    assert len(train) + len(test) == len(df)
    assert set(train["row_id"]) | set(test["row_id"]) == set(df["row_id"])
    assert set(train["row_id"]).isdisjoint(test["row_id"])
    assert test.groupby(["heart_disease", "age_group"]).size().eq(1).all()
    pd.testing.assert_frame_equal(df, original)


def test_keep_strategy_retains_rows_while_drop_rows_removes_missing() -> None:
    df = pd.DataFrame(
        {
            "heart_disease": [0, 1, 0, 1],
            "trestbps": [120.0, np.nan, 140.0, 130.0],
        }
    )
    pre = CardiacPreprocessor(sensitive_attrs=[])

    kept, _ = pre.handle_missing_values(df, strategy="keep")
    dropped, _ = pre.handle_missing_values(df, strategy="drop_rows")

    assert len(kept) == 4  # pooled cohorts keep the missing row for imputation
    assert len(dropped) == 3  # clean cohorts drop it (complete-case)


def test_imputation_is_fitted_on_train_and_reused_on_test() -> None:
    # Train median of [10, 20, 30] is 20; the test fold's own median (100) must
    # never be used — the missing test value must be filled with the train median.
    train = pd.DataFrame({"heart_disease": [0, 1, 0], "bp": [10.0, 20.0, 30.0]})
    test = pd.DataFrame({"heart_disease": [1, 0], "bp": [100.0, np.nan]})
    pre = CardiacPreprocessor(sensitive_attrs=[])

    pre.prepare_features(train, target="heart_disease", fit=True)
    x_test, _, _ = pre.prepare_features(test, target="heart_disease", fit=False)

    assert x_test["bp"].iloc[1] == 20.0  # train median, not test median (100)


def test_all_missing_nullable_int_column_is_filled() -> None:
    # An all-missing pandas nullable Int64 column: its median is pd.NA (not a
    # float), so the fallback must still fill it with a numeric constant rather
    # than leave <NA> or inject the "unknown" string.
    df = pd.DataFrame(
        {
            "heart_disease": [0, 1],
            "ca": pd.array([pd.NA, pd.NA], dtype="Int64"),
            "trestbps": [120.0, 130.0],
        }
    )
    pre = CardiacPreprocessor(sensitive_attrs=[])
    x, _, _ = pre.prepare_features(df, target="heart_disease", fit=True)

    assert not x["ca"].isnull().any()
    assert (x["ca"] == 0).all()


def test_unknown_missing_strategy_raises() -> None:
    df = pd.DataFrame({"heart_disease": [0, 1], "bp": [np.nan, 130.0]})
    pre = CardiacPreprocessor(sensitive_attrs=[])
    with pytest.raises(ValueError, match="Unknown missing-value strategy"):
        pre.handle_missing_values(df, strategy="median")


def test_extra_exclude_drops_model_only_columns() -> None:
    df = pd.DataFrame(
        {
            "heart_disease": [0, 1],
            "trestbps": [120.0, 130.0],
            "source_site": ["cleveland", "hungarian"],
            "chol": [np.nan, np.nan],
        }
    )
    pre = CardiacPreprocessor(sensitive_attrs=[])
    _, _, feature_names = pre.prepare_features(
        df, target="heart_disease", extra_exclude=["source_site", "chol"], fit=True
    )
    assert "source_site" not in feature_names
    assert "chol" not in feature_names
    assert "trestbps" in feature_names
