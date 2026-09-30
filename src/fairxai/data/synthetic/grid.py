"""Default study grid for the profiling-sensitivity study.

Rather than a full Cartesian product (which explodes), each knob family is swept
one at a time off a per-tier baseline ("temperature study" style).

Seeding is paired: every config of one grid shares the same seed, so a condition
differs from its baseline only by the knob (common random numbers). The generators
draw missingness and duplication from their own derived streams, so those knobs
act on exactly the baseline frame. Replicates are separate grids built with
different seeds (see ``scripts/studies/run_profiling_sensitivity_study.py``).
"""

from __future__ import annotations

from dataclasses import replace

from .config import SyntheticConfig

TIERS = ("abstract", "healthcare")


def _baseline(tier: str, seed: int) -> SyntheticConfig:
    return SyntheticConfig(
        tier=tier,
        seed=seed,
        n_samples=2000,
        n_features=12,
        minority_ratio=0.5,
        # Baseline carries real signal (not near-random) so downstream clustering
        # and complexity metrics have structure to find; the separability sweep
        # still probes the hard, low-signal regime.
        class_sep=1.8,
        missing_mechanism="none",
        missing_pct=0.0,
        n_missing_features=0,
        n_binary=1,
        n_lowcard=2,
        lowcard_levels=10,
        n_highcard=1,
        duplicate_pct=0.0,
        label="base",
    )


def build_grid(seed: int = 20260625) -> list[SyntheticConfig]:
    """Return the default 24-dataset grid (12 per tier), all sharing ``seed``.

    One knob family is swept at a time off a per-tier baseline. Sweeps are kept
    deliberately compact (the study reads conditions, not a full grid search):
    missingness is well represented (it is the headline knob), cardinality and
    duplication each get the cases that matter, and the secondary knobs
    (imbalance / separability / size) carry the contrast points only.
    """
    configs: list[SyntheticConfig] = []

    for tier in TIERS:
        base = _baseline(tier, seed)
        configs.append(base)

        # Missingness sweep: {mcar, mar} x {0.20, 0.40} (real NaNs; the headline).
        for mechanism in ("mcar", "mar"):
            for pct in (0.20, 0.40):
                configs.append(
                    replace(
                        base,
                        missing_mechanism=mechanism,
                        missing_pct=pct,
                        n_missing_features=3,
                        label="missingness",
                    )
                )

        # Class-imbalance sweep (strong imbalance; balanced baseline is the ref).
        for minority in (0.10, 0.02):
            configs.append(replace(base, minority_ratio=minority, label="imbalance"))

        # Separability sweep: the hard, low-signal contrast to the baseline.
        configs.append(replace(base, class_sep=0.5, label="separability"))

        # Size sweep: n=120 triggers the low-card type boundary (baseline is large).
        configs.append(replace(base, n_samples=120, label="size"))

        # Cardinality / type-mix sweep (15-level low-card + extra high-card cols).
        configs.append(replace(base, lowcard_levels=15, n_highcard=3, label="cardinality"))

        # Duplicate-rows sweep: a fraction of rows copied verbatim.
        for dup in (0.05, 0.20):
            configs.append(replace(base, duplicate_pct=dup, label="duplicates"))

    return configs


def build_smoke_grid(seed: int = 20260625) -> list[SyntheticConfig]:
    """A 5-dataset abstract-tier subset for fast smoke tests, sharing ``seed``."""
    base = _baseline("abstract", seed)
    return [
        base,
        replace(
            base,
            missing_mechanism="mar",
            missing_pct=0.20,
            n_missing_features=3,
            label="missingness",
        ),
        replace(base, minority_ratio=0.10, label="imbalance"),
        replace(base, n_samples=120, label="size"),
        replace(base, duplicate_pct=0.20, label="duplicates"),
    ]
