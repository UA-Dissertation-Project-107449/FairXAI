"""Missingness injection for synthetic datasets.

Writes real NaNs into selected feature columns under either MCAR (missing
completely at random) or MAR (missing at random, conditioned on another column).
Target, sensitive and index columns are never nulled so the study can still
score class balance and group structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _row_missing_probabilities(
    conditioning: pd.Series, missing_pct: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-row missing probability for MAR, averaging ``missing_pct``.

    Numeric conditioning columns use rank position; categorical columns assign a
    deterministic per-category weight. Weights are rescaled so the mean row
    probability equals ``missing_pct`` and clipped to ``[0, 1]``.
    """
    n = len(conditioning)
    if n == 0:
        return np.zeros(0)

    if pd.api.types.is_numeric_dtype(conditioning):
        ranks = conditioning.rank(method="average", na_option="bottom").to_numpy()
        weights = ranks / ranks.max() if ranks.max() > 0 else np.ones(n)
    else:
        categories = list(pd.unique(conditioning.astype("object")))
        # Stable per-category weights in (0, 1], higher for later categories.
        weight_map = {cat: (idx + 1) / len(categories) for idx, cat in enumerate(categories)}
        weights = conditioning.astype("object").map(weight_map).to_numpy(dtype=float)

    mean_weight = float(np.nanmean(weights)) if np.isfinite(weights).any() else 1.0
    if mean_weight <= 0:
        return np.full(n, missing_pct)
    probs = np.clip(weights * (missing_pct / mean_weight), 0.0, 1.0)
    return probs


def inject_missingness(
    df: pd.DataFrame,
    target_columns: list[str],
    mechanism: str,
    missing_pct: float,
    rng: np.random.Generator,
    conditioning_column: str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with real NaNs written into ``target_columns``.

    Parameters
    ----------
    df : DataFrame
        Source frame (not mutated).
    target_columns : list[str]
        Columns eligible to receive NaNs.
    mechanism : str
        ``"mcar"``, ``"mar"`` or ``"none"``.
    missing_pct : float
        Target missing fraction per affected column (0..1).
    rng : numpy Generator
        Seeded RNG for determinism.
    conditioning_column : str | None
        For MAR, the column the missingness probability depends on. Falls back
        to MCAR when absent.
    """
    if mechanism == "none" or missing_pct <= 0 or not target_columns:
        return df.copy()

    out = df.copy()
    n = len(out)

    if mechanism == "mar" and conditioning_column and conditioning_column in out.columns:
        probs = _row_missing_probabilities(out[conditioning_column], missing_pct, rng)
    else:  # mcar (or mar without a usable conditioning column)
        probs = np.full(n, missing_pct)

    for column in target_columns:
        if column not in out.columns:
            continue
        mask = rng.random(n) < probs
        out.loc[mask, column] = np.nan

    return out
