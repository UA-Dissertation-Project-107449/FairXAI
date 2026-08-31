"""Tests for sensitive-attribute decoding in the assessment stage.

The bug these pin: band labels were assigned by zipping the observed distinct
values, *sorted as text*, against a fixed canonical list. ``"<40"`` sorts after
``"70+"`` in ASCII, so every age band in the fairness report was named after a
different band's data.
"""

import sys
from pathlib import Path

import pandas as pd

_FAIRXAI_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_FAIRXAI_ROOT / "src"))
sys.path.insert(0, str(_FAIRXAI_ROOT / "scripts" / "common"))

from assess_predictions import (  # noqa: E402
    _age_band_sort_key,
    decode_sensitive_attributes,
)

FIXED_10YR = ["<40", "40-49", "50-59", "60-69", "70+"]
CLINICAL = ["<45", "45-54", "55-64", "65+"]


def test_labelled_age_column_keeps_its_own_labels():
    """The regression. Each band must keep the rows it actually describes."""
    df = pd.DataFrame({"age_group": ["<40"] * 3 + ["40-49"] * 7 + ["70+"] * 5})

    out = decode_sensitive_attributes(df)

    counts = out["age_group_cat"].value_counts().to_dict()
    assert counts == {"<40": 3, "40-49": 7, "70+": 5}


def test_a_four_band_binning_is_not_renamed_to_the_five_band_list():
    """The clinical binning has no ``<40``; positional zipping invented one."""
    df = pd.DataFrame({"age_group": CLINICAL * 4})

    out = decode_sensitive_attributes(df)

    assert set(out["age_group_cat"].unique()) == set(CLINICAL)


def test_age_categories_are_ordered_by_age_not_by_ascii():
    df = pd.DataFrame({"age_group": list(reversed(FIXED_10YR))})

    out = decode_sensitive_attributes(df)

    assert list(out["age_group_cat"].cat.categories) == FIXED_10YR
    assert out["age_group_cat"].cat.ordered


def test_sort_key_puts_the_open_lower_band_first():
    """``"<40"`` opens the scale even though ``<`` is 0x3C and ``7`` is 0x37."""
    assert sorted(FIXED_10YR, key=_age_band_sort_key) == FIXED_10YR
    assert sorted(CLINICAL, key=_age_band_sort_key) == CLINICAL
    assert _age_band_sort_key("<40") < _age_band_sort_key("40-49")


def test_a_full_numeric_encoding_decodes_to_the_canonical_bands():
    """Ascending codes are ascending ages, so a complete set can be named."""
    df = pd.DataFrame({"age_group": [-1.4, -0.7, 0.0, 0.7, 1.4]})

    out = decode_sensitive_attributes(df)

    assert list(out["age_group_cat"].astype(str)) == FIXED_10YR


def test_a_partial_numeric_encoding_is_not_given_guessed_band_names():
    """Fewer codes than canonical bands: naming them would name them wrong."""
    df = pd.DataFrame({"age_group": [0.0, 1.0, 2.0]})

    out = decode_sensitive_attributes(df)

    labels = set(out["age_group_cat"].astype(str))
    assert labels == {"age_band_0", "age_band_1", "age_band_2"}
    assert not labels & set(FIXED_10YR)


def test_labelled_sex_column_passes_through_rather_than_relying_on_sort_order():
    """``"F"``/``"M"`` must not be renamed by sort position."""
    df = pd.DataFrame({"sex": ["M", "F", "M", "F"]})

    out = decode_sensitive_attributes(df)

    assert set(out["sex_cat"]) == {"M", "F"}


def test_numeric_sex_decodes_zero_to_female():
    df = pd.DataFrame({"sex": [0, 1, 1, 0]})

    out = decode_sensitive_attributes(df)

    assert list(out["sex_cat"]) == ["Female", "Male", "Male", "Female"]
