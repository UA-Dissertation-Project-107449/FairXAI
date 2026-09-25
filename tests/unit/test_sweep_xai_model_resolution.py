"""Which model stage 12 (sweep) explains for each mitigation arm.

The sweep resolved every fairlearn arm to the first member of ``predictors_``, so
its SHAP and LIME output described a model neither reduction predicts with. These
tests pin the resolution itself: ``GridSearch`` is explained through its chosen
member, ``ExponentiatedGradient`` through the weighted mixture of its members, and
the wrapper and post-processing paths are untouched.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["chol", "thalach", "oldpeak"]


@pytest.fixture(scope="module")
def sweep():
    spec = importlib.util.spec_from_file_location(
        "run_combinatorial_experiments",
        ROOT / "scripts" / "experiments" / "run_combinatorial_experiments.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fitted(n: int = 80, seed: int = 0):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = (X["chol"] + rng.normal(0, 0.3, n) > 0).astype(int)
    return LogisticRegression(max_iter=200).fit(X, y), X


class _Wrapped:
    """A fairxai model wrapper: the sklearn estimator hangs off ``.model``."""

    def __init__(self, estimator):
        self.model = estimator


class _GridSearch:
    """``GridSearch``: fits a grid, then predicts with ``predictors_[best_idx_]``."""

    def __init__(self, predictors, best_idx, base_estimator=None):
        self.predictors_ = predictors
        self.best_idx_ = best_idx
        # fairlearn keeps the unfitted template here; it must never be explained.
        self.estimator = base_estimator


class _Randomised:
    """``ExponentiatedGradient``: draws a member of ``predictors_`` per row."""

    def __init__(self, predictors, weights):
        self.predictors_ = pd.Series(list(predictors))
        self.weights_ = pd.Series(list(weights))

    def _pmf_predict(self, X):
        positive = sum(
            weight * np.asarray(member.predict(X))
            for weight, member in zip(self.weights_, self.predictors_)
        )
        return np.column_stack([1.0 - positive, positive])


class _PostProcessor:
    """``ThresholdOptimizer``: wraps a base estimator it only re-thresholds."""

    def __init__(self, estimator):
        self.estimator_ = estimator


def test_grid_search_resolves_to_its_chosen_member(sweep):
    other, _ = _fitted(seed=1)
    chosen, _ = _fitted(seed=2)
    unfitted = type(chosen)()

    resolved = sweep._unwrap_for_xai(_GridSearch([other, chosen], 1, base_estimator=unfitted))

    assert resolved is chosen


def test_randomised_reduction_resolves_to_a_weighted_mixture(sweep):
    first, X = _fitted(seed=1)
    second, _ = _fitted(seed=2)

    resolved = sweep._unwrap_for_xai(_Randomised([first, second], weights=[0.25, 0.75]))

    assert isinstance(resolved, sweep._MixtureProba)
    expected = 0.25 * first.predict_proba(X)[:, 1] + 0.75 * second.predict_proba(X)[:, 1]
    np.testing.assert_allclose(resolved.predict_proba(X)[:, 1], expected)
    # Neither member alone is the arm, which is the defect this replaces.
    assert not np.allclose(resolved.predict_proba(X)[:, 1], first.predict_proba(X)[:, 1])


def test_zero_weight_members_are_left_out_of_the_mixture(sweep):
    kept, X = _fitted(seed=1)
    never_drawn, _ = _fitted(seed=2)

    resolved = sweep._unwrap_for_xai(_Randomised([kept, never_drawn], weights=[1.0, 0.0]))

    assert [member for _, member in resolved.mixture] == [kept]
    np.testing.assert_allclose(resolved.predict_proba(X)[:, 1], kept.predict_proba(X)[:, 1])


def test_mixture_is_explainable_by_shap(sweep):
    """SHAP needs a callable; LIME needs predict_proba. The mixture answers both."""
    pytest.importorskip("shap")
    from fairxai.explainability.tabular import shap_explain_tabular

    first, X = _fitted(n=40, seed=1)
    second, _ = _fitted(n=40, seed=2)
    mixture = sweep._unwrap_for_xai(_Randomised([first, second], weights=[0.5, 0.5]))

    explanation = shap_explain_tabular(mixture, X, max_samples=20)

    assert explanation.feature_names == FEATURES
    assert explanation.shap_values.shape[0] == 20
    probabilities = mixture.predict_proba(X)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(mixture.predict(X), (probabilities[:, 1] >= 0.5) * 1)


def test_wrapper_and_post_processor_paths_are_unchanged(sweep):
    estimator, _ = _fitted()

    assert sweep._unwrap_for_xai(_Wrapped(estimator)) is estimator
    assert sweep._unwrap_for_xai(_PostProcessor(estimator)) is estimator
    assert sweep._unwrap_for_xai(estimator) is estimator
    assert sweep._unwrap_for_xai(object()) is None


def test_fairlearn_reductions_resolve_the_same_way(sweep):
    """The fakes above stand in for these two; this is the real thing."""
    pytest.importorskip("fairlearn")
    from fairlearn.reductions import (
        DemographicParity,
        ExponentiatedGradient,
        GridSearch,
    )
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    n = 200
    sex = rng.integers(0, 2, size=n)
    chol = rng.normal(size=n) + 1.5 * sex
    X = pd.DataFrame({"chol": chol, "thalach": rng.normal(size=n), "oldpeak": sex.astype(float)})
    y = pd.Series(((chol + rng.normal(scale=0.5, size=n)) > 0.7).astype(int))

    # A tight constraint on unfair data is what makes EG mix several predictors.
    eg = ExponentiatedGradient(
        LogisticRegression(max_iter=200), constraints=DemographicParity(), eps=0.01
    )
    eg.fit(X, y, sensitive_features=sex)
    mixture = sweep._unwrap_for_xai(eg)
    assert isinstance(mixture, sweep._MixtureProba)
    assert len(mixture.mixture) == int((eg.weights_ > 0).sum())
    assert mixture.predict_proba(X).shape == (n, 2)

    gs = GridSearch(LogisticRegression(max_iter=200), constraints=DemographicParity())
    gs.fit(X, y, sensitive_features=sex)
    assert sweep._unwrap_for_xai(gs) is gs.predictors_[gs.best_idx_]
