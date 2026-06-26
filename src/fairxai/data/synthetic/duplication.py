"""Duplicate-row injection for synthetic datasets.

Real datasets often carry duplicated records (re-submitted forms, ETL fan-out,
joins gone wrong). This module copies a controlled fraction of rows verbatim so
the study can observe how profiling responds to duplication. Copies are exact
(including any injected NaNs) and the result is shuffled deterministically so the
duplicates are not all clustered at the tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_duplicates(
    df: pd.DataFrame,
    duplicate_pct: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``duplicate_pct`` of its rows duplicated.

    Parameters
    ----------
    df : DataFrame
        Source frame (not mutated).
    duplicate_pct : float
        Fraction of the original row count to copy verbatim (0..1). ``0.20`` on a
        100-row frame appends 20 duplicate rows (120 total).
    rng : numpy Generator
        Seeded RNG for deterministic row selection and shuffling.
    """
    n = len(df)
    if duplicate_pct <= 0 or n == 0:
        return df.copy()

    n_dup = int(round(n * duplicate_pct))
    if n_dup <= 0:
        return df.copy()

    replace = n_dup > n
    picks = rng.choice(n, size=n_dup, replace=replace)
    duplicated = pd.concat([df, df.iloc[picks]], ignore_index=True)
    order = rng.permutation(len(duplicated))
    return duplicated.iloc[order].reset_index(drop=True)
