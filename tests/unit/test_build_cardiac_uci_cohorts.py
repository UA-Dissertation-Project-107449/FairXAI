"""Unit tests for the raw-UCI cardiac cohort builder.

Synthetic tests cover the dedup + encoding logic (CI-safe). A ``local_data`` test
asserts the acceptance counts against the gitignored raw UCI files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "utils"))

from build_cardiac_uci_cohorts import (  # noqa: E402
    UCI_DIR,
    UCI_RAW_COLUMNS,
    _dedup_four_site,
    _finalize,
    build,
)


def _row(**overrides):
    base = {c: 1.0 for c in UCI_RAW_COLUMNS}
    base["source_site"] = "cleveland"
    base.update(overrides)
    return base


class TestFinalize:
    def test_cp_and_slope_shift_to_zero_based(self):
        df = pd.DataFrame([_row(cp=1.0, slope=1.0), _row(cp=4.0, slope=3.0)])
        out = _finalize(df)
        assert list(out["cp"]) == [0.0, 3.0]
        assert list(out["slope"]) == [0.0, 2.0]

    def test_chol_zero_becomes_missing(self):
        df = pd.DataFrame([_row(chol=0.0), _row(chol=240.0)])
        out = _finalize(df)
        assert pd.isna(out["chol"].iloc[0])
        assert out["chol"].iloc[1] == 240.0

    def test_trestbps_zero_becomes_missing(self):
        # 0 mmHg resting BP is a physiological impossibility (sentinel), not a
        # measurement; fold into missing so it is imputed, not row-dropped.
        df = pd.DataFrame([_row(trestbps=0.0), _row(trestbps=130.0)])
        out = _finalize(df)
        assert pd.isna(out["trestbps"].iloc[0])
        assert out["trestbps"].iloc[1] == 130.0

    def test_num_and_sex_are_integer_typed(self):
        df = pd.DataFrame([_row(num=2.0, sex=0.0)])
        out = _finalize(df)
        assert str(out["num"].dtype) == "Int64"
        assert str(out["sex"].dtype) == "Int64"
        # target_mapping keys are "0".."4"; integer stringifies cleanly.
        assert str(out["num"].iloc[0]) == "2"

    def test_thal_remapped_to_canonical(self):
        # 3=normal, 6=fixed defect, 7=reversible defect -> contiguous 0/1/2.
        df = pd.DataFrame([_row(thal=3.0), _row(thal=6.0), _row(thal=7.0)])
        out = _finalize(df)
        assert list(out["thal"]) == [0, 1, 2]


class TestDedup:
    def test_exact_canonical_duplicate_removed(self):
        a = _row(age=63.0, trestbps=145.0, chol=233.0, thalach=150.0, oldpeak=2.3, num=0.0)
        dup = dict(a, source_site="hungarian")  # same canonical key, different site
        other = _row(age=41.0, trestbps=130.0, chol=204.0, thalach=172.0, oldpeak=1.4, num=1.0)
        df = pd.DataFrame([a, dup, other])
        deduped, removed = _dedup_four_site(df)
        assert removed == 1
        assert len(deduped) == 2
        # keep="first" retains the cleveland occurrence
        assert deduped.iloc[0]["source_site"] == "cleveland"

    def test_missing_values_in_key_compared_as_equal(self):
        # chol is part of the canonical dedup key; two rows sharing a missing chol
        # (and all other key fields) must be treated as duplicates.
        a = _row(chol=float("nan"))
        b = _row(chol=float("nan"))
        df = pd.DataFrame([a, b])
        _, removed = _dedup_four_site(df)
        assert removed == 1


@pytest.mark.local_data
class TestFullBuild:
    def test_acceptance_counts(self, tmp_path):
        if not (UCI_DIR / "processed.cleveland.data").exists():
            pytest.skip("raw UCI files not present")
        m = build(out_dir=tmp_path)  # never overwrite the real gitignored cohorts
        assert m["cleveland_rows"] == 303
        assert m["four_site_source_rows"] == 920
        assert m["four_site_final_rows"] == 918
        assert m["four_site_dedup_removed"] == 2
        assert m["four_site_rows_per_site"] == {
            "cleveland": 303,
            "hungarian": 293,
            "va": 199,
            "switzerland": 123,
        }
        # chol==0 folded into missing -> 201 unavailable (29 '?' + 172 zeros).
        assert m["four_site_missing"]["chol"]["missing"] == 201
        assert m["four_site_missing"]["ca"]["missing"] == 609
