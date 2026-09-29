"""The default binning folder keeps the age bands each cohort's schema declares.

Preprocess re-cuts ``age_group`` once per binning strategy, and training reads
the ``{dataset}_{default_binning}`` folder. It used to cut that folder with the
built-in ``fixed_10yr`` bands for every cohort, so cleveland_uci's schema merge
of 70+ into 60+ never reached a training split.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_FAIRXAI_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_FAIRXAI_ROOT / "src"))
sys.path.insert(0, str(_FAIRXAI_ROOT / "scripts" / "common"))

from preprocess_data import _resolve_age_bins  # noqa: E402

from fairxai.experiments.attribute_binning import apply_binning  # noqa: E402
from fairxai.experiments.data_io import resolve_default_binning  # noqa: E402

_CONFIGS = _FAIRXAI_ROOT / "configs"


def _schema(dataset: str) -> dict:
    with open(_CONFIGS / "schema" / "cardiac.json") as handle:
        return json.load(handle)["datasets"][dataset]


def _default_binning() -> str:
    with open(_CONFIGS / "pipelines" / "cardiac.yaml") as handle:
        return resolve_default_binning(yaml.safe_load(handle))


def _ages(low: int = 29, high: int = 77) -> pd.DataFrame:
    return pd.DataFrame({"age_raw": np.arange(low, high + 1, dtype=float)})


def test_cleveland_default_split_has_no_70_plus_band() -> None:
    default = _default_binning()
    df = _ages()

    bins, labels, source = _resolve_age_bins(df, _schema("cleveland_uci"), default, default)
    binned = apply_binning(df, bins, labels, col="age_raw", output_col="age_group")

    assert source == "schema"
    assert "70+" not in set(binned["age_group"].astype(str))
    assert set(binned["age_group"].astype(str)) == {"<40", "40-49", "50-59", "60+"}


def test_four_site_default_split_keeps_its_70_plus_band() -> None:
    default = _default_binning()
    df = _ages()

    bins, labels, _ = _resolve_age_bins(df, _schema("four_site_uci"), default, default)
    binned = apply_binning(df, bins, labels, col="age_raw", output_col="age_group")

    assert "70+" in set(binned["age_group"].astype(str))


def test_alternative_strategies_ignore_the_schema_bands() -> None:
    bins, labels, source = _resolve_age_bins(
        _ages(), _schema("cleveland_uci"), "clinical", _default_binning()
    )

    assert source == "strategy"
    assert labels == ["<45", "45-54", "55-64", "65+"]


def test_schema_bins_in_days_are_converted_to_years() -> None:
    default = _default_binning()

    bins, _, source = _resolve_age_bins(
        _ages(), _schema("cardio70k"), default, default, age_unit="days"
    )

    assert source == "schema"
    assert bins == [0.0, 40.0, 50.0, 60.0, 70.0, 120.0]
