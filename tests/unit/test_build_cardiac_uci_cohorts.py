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
    _site_confounding_report,
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


class TestSiteConfounding:
    """The pooled cohort's age gradient may partly be a site gradient.

    The four hospitals differ in both prevalence and age mix, so an age band's
    positive rate is read from a different mix of sites than the band next to it.
    The report exists to make that visible, which needs the per-site rates *and*
    the same rates split by band within each site.
    """

    @staticmethod
    def _cohort(rows):
        return pd.DataFrame(
            [{"source_site": site, "age": age, "num": num} for site, age, num in rows]
        )

    def test_bands_come_from_the_schema(self):
        report = _site_confounding_report(self._cohort([("cleveland", 55, 1)]))
        assert report["age_bands"] == ["<40", "40-49", "50-59", "60-69", "70+"]

    def test_reports_positive_rate_and_age_mix_per_site(self):
        cohort = self._cohort(
            [("cleveland", 35, 0), ("cleveland", 55, 1), ("va", 65, 1), ("va", 66, 3)]
        )

        per_site = _site_confounding_report(cohort)["per_site"]

        assert per_site["cleveland"]["n"] == 2
        assert per_site["cleveland"]["positive_rate_pct"] == 50.0
        assert per_site["cleveland"]["age_mix_pct"]["<40"] == 50.0
        # num is the 0-4 severity code: anything above 0 is disease.
        assert per_site["va"]["positive_rate_pct"] == 100.0
        assert per_site["va"]["age_mix_pct"]["60-69"] == 100.0

    def test_band_tables_follow_the_schema_order(self):
        cohort = self._cohort([("va", 65, 1), ("va", 35, 0), ("va", 45, 1)])

        report = _site_confounding_report(cohort)

        assert list(report["per_age_band"]) == ["<40", "40-49", "60-69"]
        assert list(report["per_site_and_age_band"]["va"]) == ["<40", "40-49", "60-69"]

    def test_pooled_gradient_can_disappear_within_site(self):
        """The Simpson case the check is for: age only tracks which site it is.

        Neither site has any age gradient at all, but the young patients are
        nearly all from the low-prevalence site and the old ones from the
        high-prevalence site, so pooling manufactures one.
        """
        rows = [("hungarian", 35, i % 4 == 0) for i in range(20)]  # 25% positive, all young
        rows += [("switzerland", 65, i % 4 != 0) for i in range(20)]  # 75% positive, all old

        report = _site_confounding_report(self._cohort(rows))

        pooled = report["per_age_band"]
        assert pooled["<40"]["positive_rate_pct"] == 25.0
        assert pooled["60-69"]["positive_rate_pct"] == 75.0
        # Within each site there is one band only, holding that site's own rate:
        # the 50-point pooled gap is entirely between-site.
        within = report["per_site_and_age_band"]
        assert list(within["hungarian"]) == ["<40"]
        assert list(within["switzerland"]) == ["60-69"]
        assert within["hungarian"]["<40"]["positive_rate_pct"] == 25.0
        assert within["switzerland"]["60-69"]["positive_rate_pct"] == 75.0


@pytest.mark.local_data
def test_four_site_age_gradient_is_partly_a_site_gradient(tmp_path):
    """The real cohort: does "age" partly mean "Hungary/Switzerland"?

    Partly, yes. Site prevalence runs from 36% (Hungary) to 94% (Switzerland),
    and the age mixes differ sharply (20% of Hungary's patients are under 40
    against 3% of VA's), so the pooled 34% -> 74% gradient overstates the
    within-site one. It does not vanish: three of the four sites keep a positive
    gradient of their own, so age is a real signal *and* partly a site proxy.
    These numbers are what the dissertation cites, so they are pinned here.
    """
    if not (UCI_DIR / "processed.cleveland.data").exists():
        pytest.skip("raw UCI files not present")

    report = build(out_dir=tmp_path)["four_site_site_confounding"]
    per_site = report["per_site"]
    pooled = report["per_age_band"]
    within = report["per_site_and_age_band"]

    assert per_site["hungarian"]["positive_rate_pct"] == 36.2
    assert per_site["switzerland"]["positive_rate_pct"] == 93.5
    assert per_site["hungarian"]["age_mix_pct"]["<40"] == 20.1
    assert per_site["va"]["age_mix_pct"]["<40"] == 2.5

    pooled_gap = pooled["60-69"]["positive_rate_pct"] - pooled["<40"]["positive_rate_pct"]
    assert round(pooled_gap, 1) == 39.2

    # Same gap inside each site, where it cannot be carried by the site mix.
    within_gaps = {
        site: bands["60-69"]["positive_rate_pct"] - bands["<40"]["positive_rate_pct"]
        for site, bands in within.items()
    }
    assert all(gap > 0 for gap in within_gaps.values()), within_gaps
    assert max(within_gaps.values()) < pooled_gap, within_gaps
