"""Tests for the confidence-interval files the assessment stage writes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_FAIRXAI_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_FAIRXAI_ROOT / "src"))
sys.path.insert(0, str(_FAIRXAI_ROOT / "scripts" / "common"))

from assess_predictions import _resolve_n_bootstrap, _write_uncertainty  # noqa: E402

from fairxai.fairness.uncertainty import (  # noqa: E402
    BOOTSTRAP_BASE_REPLICATES,
    BOOTSTRAP_LARGE_REPLICATES,
)

SENSITIVE = ["sex", "age_group"]


def _predictions(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, n),
            "sex": rng.choice(["f", "m"], n),
            "age_group": rng.choice(["<50", "50-60", ">60"], n),
        }
    )
    df["y_proba"] = rng.random(n)
    df["y_pred"] = (df["y_proba"] > 0.5).astype(int)
    return df


def test_auto_scales_replicates_to_the_cohort():
    assert _resolve_n_bootstrap("auto", 300) == BOOTSTRAP_BASE_REPLICATES
    assert _resolve_n_bootstrap(None, 68_000) == BOOTSTRAP_LARGE_REPLICATES
    assert _resolve_n_bootstrap(50, 68_000) == 50


def test_writes_both_tables_and_returns_metadata(tmp_path):
    metadata = _write_uncertainty(
        _predictions(), SENSITIVE, tmp_path, "cleveland_logistic_regression", {"n_bootstrap": 60}
    )

    ci = pd.read_csv(tmp_path / "cleveland_logistic_regression_ci.csv")
    pairwise = pd.read_csv(tmp_path / "cleveland_logistic_regression_pairwise_ci.csv")

    assert {"ci_low", "ci_high", "point", "includes_zero"} <= set(ci.columns)
    assert {"group_a", "group_b", "difference", "p_value_bh", "significant"} <= set(
        pairwise.columns
    )
    assert metadata["n_boot"] == 60
    assert metadata["n_rows"] == 300
    assert "n_significant_differences" in metadata


def test_disabled_by_configuration_writes_nothing(tmp_path):
    metadata = _write_uncertainty(
        _predictions(), SENSITIVE, tmp_path, "cleveland_logistic_regression", {"enabled": False}
    )

    assert metadata == {}
    assert list(tmp_path.iterdir()) == []


def test_no_resolved_sensitive_attributes_writes_nothing(tmp_path):
    metadata = _write_uncertainty(_predictions(), [], tmp_path, "cleveland", {"n_bootstrap": 20})

    assert metadata == {}
    assert list(tmp_path.iterdir()) == []


def test_a_bootstrap_failure_never_fails_the_assessment(tmp_path):
    """Intervals are additive; losing them must not cost the point estimates."""
    broken = _predictions().drop(columns=["y_pred"])

    metadata = _write_uncertainty(broken, SENSITIVE, tmp_path, "cleveland", {"n_bootstrap": 20})

    assert metadata == {}
    assert list(tmp_path.iterdir()) == []


def test_config_alpha_reaches_the_written_table(tmp_path):
    _write_uncertainty(
        _predictions(), SENSITIVE, tmp_path, "cleveland", {"n_bootstrap": 60, "alpha": 0.10}
    )

    ci = pd.read_csv(tmp_path / "cleveland_ci.csv")
    assert (ci["alpha"] == 0.10).all()
