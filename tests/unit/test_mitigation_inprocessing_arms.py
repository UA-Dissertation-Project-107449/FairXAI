"""How the in-processing arms are predicted with, scored and explained.

``ExponentiatedGradient`` and ``GridSearch`` are both reductions, but they predict
differently: GridSearch picks one grid member, EG draws a member per row from
``weights_``. Every claim about an in-processing arm has to follow from that.
"""

import numpy as np
import pandas as pd
import pytest

from fairxai.fairness.mitigation import MitigationEngine


@pytest.fixture
def biased_split():
    """200 rows where the label tracks the sensitive attribute.

    A tight demographic-parity constraint only binds when the unconstrained model
    is unfair, and EG only mixes predictors when the constraint binds. On easy data
    it collapses onto a single member and says nothing about the mixed case.
    """
    rng = np.random.default_rng(3)
    n = 200
    sex = rng.integers(0, 2, size=n)
    f1 = rng.normal(size=n) + 1.5 * sex
    X = pd.DataFrame({"f1": f1, "f2": rng.normal(size=n), "sex_feature": sex.astype(float)})
    y = pd.Series(((f1 + rng.normal(scale=0.5, size=n)) > 0.7).astype(int), name="target")
    sensitive = pd.DataFrame({"sex": sex})
    return X, y, sensitive


def _arm(technique, split, **kwargs):
    X, y, sensitive = split
    engine = MitigationEngine()
    return engine.apply_technique(
        technique_name=technique,
        stage="in-processing",
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        sensitive_train=sensitive,
        sensitive_test=sensitive,
        sensitive_attr="sex",
        **kwargs,
    )


def test_exponentiated_gradient_labels_are_reproducible(biased_split):
    """Unseeded, EG draws a different predictor per row on every call."""
    first = _arm("exponentiated_gradient", biased_split, eps=0.01)
    second = _arm("exponentiated_gradient", biased_split, eps=0.01)

    np.testing.assert_array_equal(first["predictions"]["y_pred"], second["predictions"]["y_pred"])
    assert first["test_metrics"]["accuracy"] == second["test_metrics"]["accuracy"]


def test_exponentiated_gradient_is_scored_by_its_own_ensemble(biased_split):
    """The AUC has to score the mixture, not one member of it."""
    X, _, _ = biased_split
    result = _arm("exponentiated_gradient", biased_split, eps=0.01)
    model = result["model"]

    assert (model.weights_ > 0).sum() > 1, "fixture no longer exercises a mixed EG"
    np.testing.assert_allclose(
        result["predictions"]["y_proba"], model._pmf_predict(X)[:, 1].astype(float)
    )


def test_grid_search_is_scored_by_its_chosen_member(biased_split):
    """GridSearch predicts with predictors_[best_idx_]; predictors_[0] is a different model."""
    X, _, _ = biased_split
    result = _arm("grid_search", biased_split)
    model = result["model"]

    chosen = model.predictors_[model.best_idx_].predict_proba(X)[:, 1]
    np.testing.assert_allclose(result["predictions"]["y_proba"], chosen)
    if model.best_idx_ != 0:
        first = model.predictors_[0].predict_proba(X)[:, 1]
        assert not np.allclose(chosen, first)
