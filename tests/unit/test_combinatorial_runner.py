"""Unit tests for run_combinatorial_experiments helpers."""

import sys
from pathlib import Path

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "scripts" / "experiments"
sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from run_combinatorial_experiments import (  # noqa: E402
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


def test_postprocessing_base_model_follows_model_type():
    """Post-processing must post-process the configured family, not always an LR."""
    import importlib.util
    from pathlib import Path

    from fairxai.models import RandomForestModel

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "combi_runner", root / "scripts" / "experiments" / "run_combinatorial_experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    built = module._build_postprocessing_base_model(
        model_type="random_forest",
        model_params={"n_estimators": 10, "max_depth": 3},
    )
    assert isinstance(built, RandomForestModel)


def _load_runner_module():
    """Load the runner as a standalone module so its globals can be patched."""
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "combi_runner", root / "scripts" / "experiments" / "run_combinatorial_experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sweep_engine_uses_the_row_model_family(monkeypatch):
    """Stage 11 rows must mitigate the family the row actually trains."""
    module = _load_runner_module()

    seen = {}

    class _SpyEngine:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(module, "MitigationEngine", _SpyEngine)
    module._build_mitigation_engine(
        {"model_type": "xgboost", "model_params": {"max_depth": 3}, "random_seed": 42}
    )
    assert seen["model_type"] == "xgboost"
    assert seen["model_params"] == {"max_depth": 3}
    assert seen["random_state"] == 42


def test_sweep_engine_defaults_to_logistic_regression(monkeypatch):
    """A row without a family keeps the historical logistic-regression engine."""
    module = _load_runner_module()

    seen = {}

    class _SpyEngine:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(module, "MitigationEngine", _SpyEngine)
    module._build_mitigation_engine({})
    assert seen["model_type"] == "logistic_regression"
    assert seen["model_params"] == {}
    assert seen["random_state"] == 42


def test_combo_experiments_are_planned_per_supported_family():
    """Combos default to the mitigation families, but can be narrowed on cost."""
    module = _load_runner_module()

    families = module._resolve_combo_model_types(
        {"mitigation_supported_model_types": ["logistic_regression", "random_forest"]}
    )
    assert families == ["logistic_regression", "random_forest"]

    families = module._resolve_combo_model_types(
        {
            "mitigation_supported_model_types": ["logistic_regression", "random_forest"],
            "mitigation_combo_model_types": ["logistic_regression"],
        }
    )
    assert families == ["logistic_regression"]


def test_combo_model_types_fall_back_to_logistic_regression():
    """An empty config must not silently plan zero combo experiments."""
    module = _load_runner_module()

    assert module._resolve_combo_model_types({}) == ["logistic_regression"]
    assert module._resolve_combo_model_types(
        {"mitigation_supported_model_types": [], "mitigation_combo_model_types": []}
    ) == ["logistic_regression"]


def test_combinatorial_config_lists_combo_families():
    """The shipped config must declare both mitigation and combo family lists."""
    import yaml

    root = Path(__file__).resolve().parents[2]
    with open(root / "configs" / "experiments" / "combinatorial.yaml") as f:
        config = yaml.safe_load(f)

    supported = [str(m).strip().lower() for m in config["mitigation_supported_model_types"]]
    combos = [str(m).strip().lower() for m in config["mitigation_combo_model_types"]]

    assert "random_forest" in supported
    # All four families are mitigated. The O(n^2) SVM cost that kept it out is a
    # cardio70k concern; the shipped datasets are cleveland_uci (303 rows) and
    # four_site_uci (918), where an RBF fit costs a fraction of a second.
    assert {"logistic_regression", "random_forest", "svm", "xgboost"} == set(supported)
    assert set(combos).issubset(set(supported))
    assert {"logistic_regression", "random_forest", "svm", "xgboost"} == set(combos)


def _tiny_splits(seed=11, n=120):
    """Synthetic single split with one binary sensitive column."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    y = pd.Series((X["f1"] + rng.normal(scale=0.5, size=n) > 0).astype(int), name="heart_disease")
    sensitive = pd.DataFrame({"sex": rng.integers(0, 2, size=n)})
    half = n // 2
    return {
        "X_train": X.iloc[:half].reset_index(drop=True),
        "y_train": y.iloc[:half].reset_index(drop=True),
        "X_test": X.iloc[half:].reset_index(drop=True),
        "y_test": y.iloc[half:].reset_index(drop=True),
        "X_train_raw": X.iloc[:half].reset_index(drop=True),
        "sensitive_train": sensitive.iloc[:half].reset_index(drop=True),
        "sensitive_test": sensitive.iloc[half:].reset_index(drop=True),
        "sensitive_cols": ["sex"],
    }


def test_single_split_result_carries_mitigation_metadata(tmp_path):
    """Whether reweighting actually weighted anything must survive to disk.

    The cuML forest silently ignores sample weights, so a row can be labelled
    "reweighting" while being a plain baseline; the log line that says so is
    gone by the time anyone reads the results.
    """
    import logging

    from fairxai.experiments.versioning import ExperimentVersioning

    module = _load_runner_module()
    versioning = ExperimentVersioning(base_results_dir=tmp_path)

    result = module.run_single_split_experiment(
        "exp_test",
        {
            "dataset": "synthetic",
            "binning_strategy": "fixed_10yr",
            "mitigation_technique": "reweighting",
            "training_method": "single_split",
            "random_seed": 42,
            "model_type": "logistic_regression",
            "model_params": {},
            "sensitive_attributes": ["sex"],
            "xai": {"enabled": False, "mode": "disabled"},
        },
        _tiny_splits(),
        versioning,
        logging.getLogger("test"),
    )

    assert result["mitigation_metadata"]["sample_weight_applied"] is True
    assert result["mitigation_metadata"]["model_type"] == "logistic_regression"


class TestResolveXgbDevice:
    """XGBoost only accepts 'cpu' or 'cuda'; every other accelerator must clamp.

    Passing 'rocm' through raises XGBoostError("Invalid argument for `device`")
    at fit time, which would kill every XGBoost arm of the sweep on an AMD box.
    """

    def _resolve(self, monkeypatch, detected):
        module = _load_runner_module()
        monkeypatch.setattr(module, "detect_accelerator", lambda requested: detected)
        return module._resolve_xgb_device({"accelerator": "auto"})

    def test_cuda_is_passed_through(self, monkeypatch):
        assert self._resolve(monkeypatch, "cuda") == "cuda"

    def test_cpu_is_passed_through(self, monkeypatch):
        assert self._resolve(monkeypatch, "cpu") == "cpu"

    def test_rocm_falls_back_to_cpu(self, monkeypatch):
        assert self._resolve(monkeypatch, "rocm") == "cpu"

    def test_unknown_accelerator_falls_back_to_cpu(self, monkeypatch):
        assert self._resolve(monkeypatch, "mps") == "cpu"

    def test_rocm_fallback_is_logged(self, monkeypatch, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            self._resolve(monkeypatch, "rocm")

        assert any("rocm" in record.message.lower() for record in caplog.records)
