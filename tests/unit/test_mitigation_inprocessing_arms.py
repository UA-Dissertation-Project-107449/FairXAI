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
