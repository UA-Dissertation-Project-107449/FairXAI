"""Configuration and ground-truth dataclasses for synthetic dataset generation.

The generators in this package produce datasets with *known* design intent. Each
column carries a :class:`GroundTruthColumn` describing the semantic type it was
designed to have, so a study can compare the live profiler output against truth
(for example to validate the categorical-vs-continuous type-inference fix).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GroundTruthColumn:
    """Designed-in description of a single generated column."""

    name: str
    role: str  # "feature" | "target" | "sensitive" | "index"
    expected_semantic_type: str  # "binary"|"categorical"|"continuous"|"identifier"
    expected_inferred_type: str  # "binary"|"categorical"|"numerical"|"text"
    n_distinct_design: int  # distinct values designed in (pre-missingness)
    missing_mechanism: str = "none"  # "mcar"|"mar"|"none"
    missing_pct_design: float = 0.0  # target missing fraction injected


@dataclass(frozen=True)
class SyntheticConfig:
    """Knobs for one synthetic dataset.

    A study varies one knob family at a time off a per-tier baseline.
    """

    tier: str  # "abstract" | "healthcare"
    seed: int
    n_samples: int = 2000
    n_features: int = 12  # informative+noise (abstract) / extra clinical (healthcare)
    minority_ratio: float = 0.5  # fraction of the positive (minority) class
    class_sep: float = 1.0  # separability / signal strength
    # missingness
    missing_mechanism: str = "none"  # "mcar" | "mar" | "none"
    missing_pct: float = 0.0  # per-targeted-feature missing fraction
    n_missing_features: int = 0  # how many feature columns receive NaNs
    # cardinality / type-mix
    n_binary: int = 1  # injected binary numeric columns
    n_lowcard: int = 2  # low-cardinality categorical-numeric columns
    lowcard_levels: int = 10  # distinct values per low-card column (the "10-value" case)
    n_highcard: int = 1  # high-cardinality continuous columns
    # duplicated rows
    duplicate_pct: float = 0.0  # fraction of rows copied verbatim (real duplicate records)
    label: str = ""  # human label for the sweep this config belongs to

    def dataset_id(self) -> str:
        """Stable, descriptive filename slug encoding the condition."""
        prefix = f"{self.label}__" if self.label else ""
        miss = (
            f"{self.missing_mechanism}{self.missing_pct:.2f}x{self.n_missing_features}"
            if self.missing_mechanism != "none" and self.missing_pct > 0
            else "nomiss"
        )
        dup = f"_dup{self.duplicate_pct:.2f}" if self.duplicate_pct > 0 else ""
        return (
            f"{prefix}{self.tier}"
            f"_n{self.n_samples}_f{self.n_features}"
            f"_min{self.minority_ratio:.2f}_sep{self.class_sep:.1f}"
            f"_{miss}_lc{self.lowcard_levels}x{self.n_lowcard}{dup}"
            f"_seed{self.seed}"
        )
