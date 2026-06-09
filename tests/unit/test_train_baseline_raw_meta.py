"""Unit tests for sourcing `*_raw` prediction metadata from the raw split.

The scaled split drops `*_raw` columns; `_load_raw_meta` recovers them from the
raw sibling split so `age_raw` rides into the baseline prediction CSVs (needed by
the age-binning fairness sweep).
"""

import sys
from pathlib import Path

import pandas as pd

_FAIRXAI_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_FAIRXAI_ROOT / "src"))
sys.path.insert(0, str(_FAIRXAI_ROOT / "scripts" / "common"))

from train_baseline import _load_raw_meta  # noqa: E402


def test_recovers_age_raw_from_raw_sibling(tmp_path):
    scaled = tmp_path / "cleveland_train_scaled.csv"
    raw = tmp_path / "cleveland_train.csv"
    # Scaled frame has no `*_raw`; raw sibling carries age_raw (same rows/order).
    scaled_df = pd.DataFrame({"chol": [0.1, -0.2, 0.3], "age_group": [0, 1, 2]})
    scaled_df.to_csv(scaled, index=False)
    pd.DataFrame({"chol": [240, 200, 260], "age_raw": [54.0, 61.0, 49.0]}).to_csv(raw, index=False)

    meta = _load_raw_meta(scaled, scaled_df)

    assert meta is not None
    assert list(meta.columns) == ["age_raw"]
    assert list(meta["age_raw"]) == [54.0, 61.0, 49.0]


def test_returns_none_when_no_raw_anywhere(tmp_path):
    scaled = tmp_path / "cleveland_train_scaled.csv"
    raw = tmp_path / "cleveland_train.csv"
    scaled_df = pd.DataFrame({"chol": [0.1, 0.2], "age_group": [0, 1]})
    scaled_df.to_csv(scaled, index=False)
    pd.DataFrame({"chol": [240, 200]}).to_csv(raw, index=False)

    assert _load_raw_meta(scaled, scaled_df) is None


def test_falls_back_to_scaled_when_raw_sibling_missing(tmp_path):
    scaled = tmp_path / "cleveland_train_scaled.csv"
    # No raw sibling on disk; scaled frame itself happens to carry age_raw.
    scaled_df = pd.DataFrame({"chol": [0.1, 0.2], "age_raw": [54.0, 61.0]})
    scaled_df.to_csv(scaled, index=False)

    meta = _load_raw_meta(scaled, scaled_df)

    assert meta is not None
    assert list(meta["age_raw"]) == [54.0, 61.0]


def test_returns_none_on_row_count_mismatch(tmp_path):
    scaled = tmp_path / "cleveland_train_scaled.csv"
    raw = tmp_path / "cleveland_train.csv"
    scaled_df = pd.DataFrame({"chol": [0.1, 0.2, 0.3], "age_group": [0, 1, 2]})
    scaled_df.to_csv(scaled, index=False)
    # Raw has a different row count → can't safely align → no carry.
    pd.DataFrame({"age_raw": [54.0, 61.0]}).to_csv(raw, index=False)

    assert _load_raw_meta(scaled, scaled_df) is None
