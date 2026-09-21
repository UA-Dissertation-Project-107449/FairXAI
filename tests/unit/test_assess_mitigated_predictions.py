"""Tests for the bootstrap intervals written per mitigation arm."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_FAIRXAI_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_FAIRXAI_ROOT / "src"))
sys.path.insert(0, str(_FAIRXAI_ROOT / "scripts" / "common"))

from assess_mitigated_predictions import (  # noqa: E402
    _ARM_KEYS,
    _load_index,
    _prefix_arm,
    _select_arms,
    assess_arm,
)

_UNCERTAINTY_CFG = {"enabled": True, "n_bootstrap": 40, "alpha": 0.05, "n_jobs": 1}


def _arm_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, n),
            "sex": rng.choice(["female", "male"], n),
            "age_group": rng.choice(["<40", "40-49", "50-59"], n),
            "feature_a": rng.normal(size=n),
        }
    )
    df["y_proba"] = rng.random(n)
    df["y_pred"] = (df["y_proba"] > 0.5).astype(int)
    return df


def _write_arms(tmp_path: Path) -> Path:
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir(parents=True)
    entries = [
        {
            "file": "cleveland_uci_logistic_regression_baseline_none.csv",
            "dataset": "cleveland_uci",
            "model_type": "logistic_regression",
            "technique": "baseline",
            "constraint_attr": "none",
        },
        {
            "file": "cleveland_uci_logistic_regression_reweighing_sex.csv",
            "dataset": "cleveland_uci",
            "model_type": "logistic_regression",
            "technique": "reweighing",
            "constraint_attr": "sex",
        },
    ]
    for seed, entry in enumerate(entries):
        _arm_frame(seed=seed).to_csv(predictions_dir / entry["file"], index=False)
    (predictions_dir / "index.json").write_text(json.dumps(entries))
    return predictions_dir


def test_load_index_returns_entries(tmp_path):
    predictions_dir = _write_arms(tmp_path)
    entries = _load_index(predictions_dir)
    assert len(entries) == 2
    assert {e["technique"] for e in entries} == {"baseline", "reweighing"}


def test_load_index_missing_is_empty_not_fatal(tmp_path):
    assert _load_index(tmp_path) == []


def test_select_arms_filters_on_every_key(tmp_path):
    entries = _load_index(_write_arms(tmp_path))
    assert len(_select_arms(entries, ["cleveland_uci"], None, None)) == 2
    assert len(_select_arms(entries, None, None, ["baseline"])) == 1
    assert _select_arms(entries, ["four_site_uci"], None, None) == []


def test_prefix_arm_puts_identity_first():
    frame = pd.DataFrame({"metric": ["demographic_parity"], "point": [0.1]})
    entry = {
        "dataset": "cleveland_uci",
        "model_type": "svm",
        "technique": "reweighing",
        "constraint_attr": "sex",
    }
    tagged = _prefix_arm(frame, entry)
    assert list(tagged.columns[: len(_ARM_KEYS)]) == _ARM_KEYS
    assert tagged.loc[0, "technique"] == "reweighing"
    # The original frame keeps its own columns.
    assert list(frame.columns) == ["metric", "point"]


def test_assess_arm_writes_both_tables(tmp_path):
    predictions_dir = _write_arms(tmp_path)
    entries = _load_index(predictions_dir)
    output_dir = tmp_path / "prediction_fairness"
    output_dir.mkdir()

    assessed = assess_arm(
        entries[0], predictions_dir, output_dir, ["age_group", "sex"], _UNCERTAINTY_CFG
    )

    stem = Path(entries[0]["file"]).stem
    assert (output_dir / f"{stem}_ci.csv").exists()
    assert (output_dir / f"{stem}_pairwise_ci.csv").exists()
    assert not assessed["table"].empty
    assert {"metric", "quantity", "ci_low", "ci_high"} <= set(assessed["table"].columns)
    assert {"group_a", "group_b", "p_value_bh", "tested"} <= set(assessed["pairwise"].columns)


def test_assess_arm_skips_a_missing_file(tmp_path):
    predictions_dir = _write_arms(tmp_path)
    output_dir = tmp_path / "prediction_fairness"
    output_dir.mkdir()
    entry = {
        "file": "not_written.csv",
        "dataset": "cleveland_uci",
        "model_type": "svm",
        "technique": "reweighing",
        "constraint_attr": "sex",
    }
    assert assess_arm(entry, predictions_dir, output_dir, ["sex"], _UNCERTAINTY_CFG) == {}


def test_assess_arm_skips_when_no_sensitive_attribute_resolves(tmp_path):
    predictions_dir = _write_arms(tmp_path)
    output_dir = tmp_path / "prediction_fairness"
    output_dir.mkdir()
    entries = _load_index(predictions_dir)
    assert (
        assess_arm(entries[0], predictions_dir, output_dir, ["not_a_column"], _UNCERTAINTY_CFG)
        == {}
    )


@pytest.mark.parametrize("technique", ["baseline", "reweighing"])
def test_every_arm_is_assessable_including_the_baseline(tmp_path, technique):
    """The unmitigated arm goes through the same pass as the mitigated ones."""
    predictions_dir = _write_arms(tmp_path)
    output_dir = tmp_path / "prediction_fairness"
    output_dir.mkdir()
    entry = next(e for e in _load_index(predictions_dir) if e["technique"] == technique)

    assessed = assess_arm(
        entry, predictions_dir, output_dir, ["age_group", "sex"], _UNCERTAINTY_CFG
    )

    assert assessed["metadata"]
    assert not assessed["pairwise"].empty
