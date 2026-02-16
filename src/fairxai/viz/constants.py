"""Shared constants and normalization helpers for notebook visualizations."""

from __future__ import annotations

import pandas as pd

CARDIAC_CATEGORY_VALUE_LABEL_MAPPING: dict[str, dict[str, str]] = {
    "cp": {
        "0": "TA",
        "1": "ATA",
        "2": "NAP",
        "3": "ASY",
        "TA": "TA",
        "ATA": "ATA",
        "NAP": "NAP",
        "ASY": "ASY",
    },
    "restecg": {
        "0": "Normal",
        "1": "ST",
        "2": "LVH",
        "Normal": "Normal",
        "ST": "ST",
        "LVH": "LVH",
    },
    "slope": {
        "0": "Down",
        "1": "Flat",
        "2": "Up",
        "Down": "Down",
        "Flat": "Flat",
        "Up": "Up",
    },
    "exang": {
        "0": "N",
        "1": "Y",
        "N": "N",
        "Y": "Y",
    },
    "fbs": {
        "0": "No",
        "1": "Yes",
        "No": "No",
        "Yes": "Yes",
    },
}

CARDIAC_CATEGORY_DISPLAY_ORDER: dict[str, list[str]] = {
    "cp": ["TA", "ATA", "NAP", "ASY"],
    "restecg": ["Normal", "ST", "LVH"],
    "slope": ["Up", "Flat", "Down"],
    "exang": ["N", "Y"],
    "fbs": ["No", "Yes"],
}


def normalize_cardiac_category_series(feature_name: str, series: pd.Series) -> pd.Series:
    mapping = CARDIAC_CATEGORY_VALUE_LABEL_MAPPING.get(feature_name)
    if mapping:
        normalized = series.astype(str).map(lambda value: mapping.get(str(value), str(value)))
    else:
        normalized = series.astype(str)

    order = CARDIAC_CATEGORY_DISPLAY_ORDER.get(feature_name)
    if order:
        normalized = pd.Categorical(normalized, categories=order, ordered=True)
    return normalized
