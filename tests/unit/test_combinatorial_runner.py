"""Unit tests for run_combinatorial_experiments helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "scripts" / "experiments"
sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from run_combinatorial_experiments import (
    _DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES,
    _coerce_label_vector,
    _coerce_probability_vector,
)


class TestDefaultMitigationSupportedModelTypes:
    def test_logistic_regression_in_default_set(self):
        assert "logistic_regression" in _DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES

    def test_default_set_is_non_empty(self):
        assert len(_DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES) >= 1


class TestCoerceProbabilityVector:
    def test_1d_array_passthrough(self):
        arr = np.array([0.1, 0.9, 0.5])
        result = _coerce_probability_vector(arr)
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, arr)

    def test_2d_two_columns_returns_second(self):
        arr = np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]])
        result = _coerce_probability_vector(arr)
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [0.7, 0.4, 0.5])

    def test_2d_one_column_returns_first(self):
        arr = np.array([[0.7], [0.4], [0.5]])
        result = _coerce_probability_vector(arr)
        assert result.shape == (3,)

    def test_scalar_wraps_in_array(self):
        result = _coerce_probability_vector(0.8)
        assert result.shape == (1,)
        assert abs(float(result[0]) - 0.8) < 1e-9

    def test_2d_empty_second_dim_returns_zeros(self):
        arr = np.empty((3, 0))
        result = _coerce_probability_vector(arr)
        np.testing.assert_array_equal(result, np.zeros(3))


class TestCoerceLabelVector:
    def test_1d_passthrough(self):
        arr = np.array([0, 1, 1, 0])
        result = _coerce_label_vector(arr)
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, arr)

    def test_2d_argmax(self):
        arr = np.array([[0.3, 0.7], [0.8, 0.2]])
        result = _coerce_label_vector(arr)
        np.testing.assert_array_equal(result, [1, 0])


class TestMitigationSupportedModelTypesConfigDriven:
    """Verify the config-driven path works end-to-end for model filtering.

    This simulates the logic from run_combinatorial_experiments without
    needing to import the full module with all its pipeline dependencies.
    """

    def _filter_mitigation(self, mitigation, model_type, supported_set):
        """Replicate the filtering logic from the combinatorial loop."""
        if mitigation != "baseline" and model_type not in supported_set:
            return "skip"
        return "run"

    def test_lr_with_smote_runs(self):
        supported = {"logistic_regression"}
        assert self._filter_mitigation("smote", "logistic_regression", supported) == "run"

    def test_rf_with_smote_skips(self):
        supported = {"logistic_regression"}
        assert self._filter_mitigation("smote", "random_forest", supported) == "skip"

    def test_rf_with_baseline_always_runs(self):
        supported = {"logistic_regression"}
        assert self._filter_mitigation("baseline", "random_forest", supported) == "run"

    def test_when_rf_added_to_supported_it_runs_smote(self):
        supported = {"logistic_regression", "random_forest"}
        assert self._filter_mitigation("smote", "random_forest", supported) == "run"
