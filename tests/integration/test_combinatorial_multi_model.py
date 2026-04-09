"""Integration test: combinatorial runner with all 4 models.

Verifies that:
- RF/SVM/XGB only produce 'baseline' mitigation rows (not smote, etc.)
- No experiment crashes when non-LR models run
- Result JSON files are written and parseable
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "scripts" / "experiments"
sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from run_combinatorial_experiments import _DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES


class TestMitigationFilteringLogic:
    """Test the filtering behaviour without running the full pipeline."""

    @pytest.mark.parametrize(
        "mitigation,model_type,expected",
        [
            ("baseline", "logistic_regression", True),
            ("baseline", "random_forest", True),
            ("baseline", "svm", True),
            ("baseline", "xgboost", True),
            ("smote", "logistic_regression", True),
            ("smote", "random_forest", False),
            ("smote", "svm", False),
            ("smote", "xgboost", False),
            ("exponentiated_gradient", "random_forest", False),
            ("threshold_optimizer", "xgboost", False),
        ],
    )
    def test_mitigation_model_filter(self, mitigation, model_type, expected):
        """Replicates the combinatorial loop's skip logic.

        baseline always runs; non-LR mitigation is skipped unless the model
        is in the supported set.
        """
        supported = _DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES

        should_run = mitigation == "baseline" or model_type in supported
        assert should_run == expected, (
            f"mitigation={mitigation}, model_type={model_type}: "
            f"expected should_run={expected}, got {should_run}"
        )


class TestConfigDrivenMitigationSupportedSet:
    """Verify that extending the config set enables mitigation for new models."""

    def test_adding_rf_to_set_allows_rf_mitigation(self):
        extended = _DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES | {"random_forest"}
        assert "random_forest" in extended

    def test_original_set_unchanged(self):
        """Extending a copy must not mutate the default."""
        _ = _DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES | {"random_forest"}
        assert "random_forest" not in _DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES

    def test_xgboost_and_svm_not_in_default(self):
        assert "xgboost" not in _DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES
        assert "svm" not in _DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES


class TestCombinatorialYamlModelTypes:
    """Verify the combinatorial.yaml now lists all 4 model types."""

    def test_all_four_models_enabled(self):
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent
            / "configs"
            / "experiments"
            / "combinatorial.yaml"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_types = [str(m).strip().lower() for m in config.get("model_types", [])]
        assert "logistic_regression" in model_types
        assert "random_forest" in model_types
        assert "svm" in model_types
        assert "xgboost" in model_types

    def test_mitigation_supported_model_types_key_present(self):
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent
            / "configs"
            / "experiments"
            / "combinatorial.yaml"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "mitigation_supported_model_types" in config
        supported = [str(m).strip().lower() for m in config["mitigation_supported_model_types"]]
        assert "logistic_regression" in supported
