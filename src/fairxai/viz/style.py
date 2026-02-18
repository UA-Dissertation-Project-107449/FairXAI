"""Visualization style constants shared by FairXAI plotting APIs."""

from __future__ import annotations

PALETTE_DATASET = {
    "cleveland": "#0072B2",
    "kaggle_heart": "#009E73",
    "cardio70k": "#D55E00",
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
