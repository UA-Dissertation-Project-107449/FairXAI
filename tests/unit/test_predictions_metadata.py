"""Unit tests for prediction metadata carrying (sensitive + extra meta)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairxai.models.baseline import (
    BaselineLogisticRegression,
    generate_predictions_with_metadata,
)


def _fit_tiny():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=40), "f2": rng.normal(size=40)})
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
    model = BaselineLogisticRegression()
    model.train(X, y)
    return model, X, y


def test_extra_meta_columns_are_carried():
    """`extra_meta` (e.g. continuous age_raw) rides along with predictions for
    post-hoc analysis, without being a model feature or a sensitive attr."""
    model, X, y = _fit_tiny()
    sensitive = pd.DataFrame({"sex": ["M", "F"] * 20})
    extra = pd.DataFrame({"age_raw": np.arange(40, dtype=float)})

    preds = generate_predictions_with_metadata(model, X, y, sensitive, extra_meta=extra)

    assert "age_raw" in preds.columns
    assert list(preds["age_raw"]) == list(np.arange(40, dtype=float))
    assert "sex" in preds.columns


def test_extra_meta_defaults_to_no_change():
    model, X, y = _fit_tiny()
    sensitive = pd.DataFrame({"sex": ["M", "F"] * 20})

    preds = generate_predictions_with_metadata(model, X, y, sensitive)

    assert "age_raw" not in preds.columns
    assert "sex" in preds.columns
