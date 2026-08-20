"""Visualization style constants shared by FairXAI plotting APIs."""

from __future__ import annotations

# Keyed on the dataset ids the pipeline actually emits (configs/pipelines/cardiac.yaml,
# configs/experiments/combinatorial.yaml). Seaborn raises ValueError on a dict palette
# that is missing a hue level rather than falling back to a default colour, so a stale
# key here is a crash at plot time, not a cosmetic difference.
_CLEVELAND = "#0072B2"
_FOUR_SITE = "#009E73"
_CARDIO70K = "#D55E00"

PALETTE_DATASET = {
    "cleveland_uci": _CLEVELAND,
    "four_site_uci": _FOUR_SITE,
    "cardio70k": _CARDIO70K,
    # Pre-rename ids, kept so already-committed notebooks keep their colours.
    "cleveland": _CLEVELAND,
    "kaggle_heart": _FOUR_SITE,
}

PALETTE_SEX = {
    "Female": "#CC79A7",
    "Male": "#56B4E9",
    "Other": "#9E9E9E",
}

PALETTE_TARGET = {
    0: "#2E8B57",
    1: "#B22222",
}

UNITS = {
    "trestbps": "mm Hg",
    "chol": "mg/dl",
    "thalach": "bpm",
    "oldpeak": "ST depression",
    "ap_hi": "mm Hg",
    "ap_lo": "mm Hg",
    "height": "cm",
    "weight": "kg",
    "bmi": "kg/m^2",
}
