"""Regression tests for non-dropping cardiac stratification fallbacks."""

import pandas as pd

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
