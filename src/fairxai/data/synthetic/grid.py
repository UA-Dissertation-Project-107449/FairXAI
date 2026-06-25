"""Default study grid for the profiling-sensitivity study.

Rather than a full Cartesian product (which explodes), each knob family is swept
one at a time off a per-tier baseline ("temperature study" style). Every config
gets a distinct derived seed so the whole grid is reproducible.
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
        class_sep=1.0,
        missing_mechanism="none",
        missing_pct=0.0,
        n_missing_features=0,
        n_binary=1,
        n_lowcard=2,
        lowcard_levels=10,
        n_highcard=1,
        label="base",
    )


def build_grid(base_seed: int = 20260625) -> list[SyntheticConfig]:
    """Return the default ~36-dataset grid."""
    configs: list[SyntheticConfig] = []

    def _add(cfg: SyntheticConfig) -> None:
        # Distinct seed per dataset guarantees unique ids and independent draws.
        configs.append(replace(cfg, seed=base_seed + len(configs)))

    for tier in TIERS:
        base = _baseline(tier, base_seed)
        _add(base)

        # Missingness sweep: {mcar, mar} x {0.05, 0.20, 0.40}
        for mechanism in ("mcar", "mar"):
            for pct in (0.05, 0.20, 0.40):
                _add(
                    replace(
                        base,
                        missing_mechanism=mechanism,
                        missing_pct=pct,
                        n_missing_features=3,
                        label="missingness",
                    )
                )

        # Class-imbalance sweep
        for minority in (0.40, 0.30, 0.10, 0.02):
            _add(replace(base, minority_ratio=minority, label="imbalance"))

        # Separability sweep
        for sep in (0.5, 2.0):
            _add(replace(base, class_sep=sep, label="separability"))

        # Size / difficulty sweep (n=120 triggers the low-card type boundary)
        for n_samples in (120, 500, 5000):
            _add(replace(base, n_samples=n_samples, label="size"))

        # Cardinality / type-mix sweep (explicit 10- and 15-level cases)
        for levels in (10, 15):
            _add(replace(base, lowcard_levels=levels, n_highcard=3, label="cardinality"))

    return configs


def build_smoke_grid(base_seed: int = 20260625) -> list[SyntheticConfig]:
    """A 4-dataset abstract-tier subset for fast smoke tests."""
    base = _baseline("abstract", base_seed)
    raw = [
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
    ]
    return [replace(cfg, seed=base_seed + idx) for idx, cfg in enumerate(raw)]
