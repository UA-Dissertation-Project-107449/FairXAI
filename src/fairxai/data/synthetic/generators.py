"""Two-tier synthetic dataset generators.

* ``generate_abstract`` — sklearn ``make_classification`` core (continuous
  informative features) plus injected sensitive attributes and a controlled mix
  of binary / low-cardinality / high-cardinality columns.
* ``generate_healthcare`` — cardiac-flavoured frame mirroring the column shape of
  ``tests/conftest.py`` with a logistic diagnosis target.

Both return ``(DataFrame, list[GroundTruthColumn])`` and apply missingness via
:func:`fairxai.data.synthetic.missingness.inject_missingness`. Expected semantic
types are derived from design intent, never by running the profiler, so the study
can detect divergence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from .config import GroundTruthColumn, SyntheticConfig
from .missingness import inject_missingness

_AGE_GROUPS = ["<40", "40-49", "50-59", "60-69", "70+"]
_RACES = ["White", "Black", "Asian", "Hispanic"]
_SEXES = ["Female", "Male"]


def _expected_lowcard_type(n_distinct: int) -> str:
    """Designed semantic type for a low-cardinality numeric *code* column.

    These columns are categorical by construction (small integer code sets such
    as 0-9 scales). Two levels is binary; anything else is categorical. The
    profiler reaches the same verdict via either the cardinality cap (<=12) or
    the low distinct-ratio rule for larger code sets in big frames.
    """
    return "binary" if n_distinct == 2 else "categorical"


def _missing_feature_names(feature_names: list[str], n_missing: int) -> set[str]:
    """First ``n_missing`` feature columns receive NaNs (deterministic)."""
    if n_missing <= 0:
        return set()
    return set(feature_names[:n_missing])


def _apply_and_label_missingness(
    df: pd.DataFrame,
    cfg: SyntheticConfig,
    ground_truth: list[GroundTruthColumn],
    eligible_features: list[str],
    conditioning_column: str | None,
) -> pd.DataFrame:
    """Inject missingness and record the mechanism/pct on each affected column."""
    targets = sorted(_missing_feature_names(eligible_features, cfg.n_missing_features))
    rng = np.random.default_rng(cfg.seed + 7919)
    df = inject_missingness(
        df,
        target_columns=targets,
        mechanism=cfg.missing_mechanism,
        missing_pct=cfg.missing_pct,
        rng=rng,
        conditioning_column=conditioning_column,
    )
    target_set = set(targets) if cfg.missing_mechanism != "none" else set()
    for col in ground_truth:
        if col.name in target_set:
            col.missing_mechanism = cfg.missing_mechanism
            col.missing_pct_design = cfg.missing_pct
    return df


def generate_abstract(
    cfg: SyntheticConfig,
) -> tuple[pd.DataFrame, list[GroundTruthColumn]]:
    """Abstract make_classification dataset with injected structure."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_samples

    n_informative = max(2, cfg.n_features // 2)
    X, y = make_classification(
        n_samples=n,
        n_features=cfg.n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        weights=[1.0 - cfg.minority_ratio, cfg.minority_ratio],
        class_sep=cfg.class_sep,
        flip_y=0.01,
        random_state=cfg.seed,
    )

    data: dict[str, np.ndarray] = {}
    ground_truth: list[GroundTruthColumn] = []
    feature_names: list[str] = []

    for i in range(cfg.n_features):
        name = f"feat_{i}"
        data[name] = X[:, i]
        feature_names.append(name)
        ground_truth.append(GroundTruthColumn(name, "feature", "continuous", "numerical", n))

    # Sensitive attributes (string-typed → categorical/binary).
    data["sex"] = rng.choice(_SEXES, size=n)
    ground_truth.append(GroundTruthColumn("sex", "sensitive", "binary", "binary", 2))
    data["age_group"] = rng.choice(_AGE_GROUPS, size=n)
    ground_truth.append(
        GroundTruthColumn("age_group", "sensitive", "categorical", "categorical", len(_AGE_GROUPS))
    )
    data["race"] = rng.choice(_RACES, size=n)
    ground_truth.append(
        GroundTruthColumn("race", "sensitive", "categorical", "categorical", len(_RACES))
    )

    _add_type_mix(data, ground_truth, feature_names, cfg, rng, n)

    data["target"] = y.astype(int)
    ground_truth.append(GroundTruthColumn("target", "target", "binary", "binary", 2))

    df = pd.DataFrame(data)
    df = _apply_and_label_missingness(
        df, cfg, ground_truth, feature_names, conditioning_column="age_group"
    )
    return df, ground_truth


def generate_healthcare(
    cfg: SyntheticConfig,
) -> tuple[pd.DataFrame, list[GroundTruthColumn]]:
    """Cardiac-flavoured dataset with a logistic diagnosis target."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_samples

    age = rng.integers(29, 80, size=n)
    age_group = pd.cut(
        age,
        bins=[0, 39, 49, 59, 69, 200],
        labels=_AGE_GROUPS,
    ).astype("object")
    sex = rng.choice(_SEXES, size=n)
    race = rng.choice(_RACES, size=n)
    trestbps = rng.normal(130, 17, size=n).clip(90, 200)
    chol = rng.normal(245, 52, size=n).clip(120, 560)
    thalach = rng.normal(150, 23, size=n).clip(70, 210)
    oldpeak = rng.gamma(1.2, 0.9, size=n).clip(0, 6.5)
    ca = rng.integers(0, 4, size=n)  # 4 distinct → categorical

    # Diagnosis driven by a standardized linear score; class_sep sharpens signal.
    def _z(arr: np.ndarray) -> np.ndarray:
        return (arr - arr.mean()) / (arr.std() + 1e-9)

    score = (
        0.9 * _z(age.astype(float))
        + 0.5 * _z(trestbps)
        + 0.4 * _z(chol)
        + 0.7 * _z(oldpeak)
        - 0.6 * _z(thalach)
    )
    noise = rng.normal(0, 1.0 / max(cfg.class_sep, 1e-3), size=n)
    latent = score + noise
    threshold = np.quantile(latent, 1.0 - cfg.minority_ratio)
    heart_disease = (latent >= threshold).astype(int)

    data: dict[str, np.ndarray] = {
        "sex": sex,
        "age_group": (
            age_group.to_numpy() if hasattr(age_group, "to_numpy") else np.asarray(age_group)
        ),
        "race": race,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "ca": ca,
    }
    ground_truth: list[GroundTruthColumn] = [
        GroundTruthColumn("sex", "sensitive", "binary", "binary", 2),
        GroundTruthColumn("age_group", "sensitive", "categorical", "categorical", len(_AGE_GROUPS)),
        GroundTruthColumn("race", "sensitive", "categorical", "categorical", len(_RACES)),
        GroundTruthColumn("trestbps", "feature", "continuous", "numerical", n),
        GroundTruthColumn("chol", "feature", "continuous", "numerical", n),
        GroundTruthColumn("thalach", "feature", "continuous", "numerical", n),
        GroundTruthColumn("oldpeak", "feature", "continuous", "numerical", n),
        GroundTruthColumn("ca", "feature", "categorical", "numerical", 4),
    ]
    feature_names = ["trestbps", "chol", "thalach", "oldpeak", "ca"]

    _add_type_mix(data, ground_truth, feature_names, cfg, rng, n)

    data["heart_disease"] = heart_disease
    ground_truth.append(GroundTruthColumn("heart_disease", "target", "binary", "binary", 2))

    df = pd.DataFrame(data)
    df = _apply_and_label_missingness(
        df, cfg, ground_truth, feature_names, conditioning_column="age_group"
    )
    return df, ground_truth


def _add_type_mix(
    data: dict[str, np.ndarray],
    ground_truth: list[GroundTruthColumn],
    feature_names: list[str],
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    n: int,
) -> None:
    """Append the configured binary / low-card / high-card numeric columns."""
    for i in range(cfg.n_binary):
        name = f"bin_{i}"
        data[name] = rng.integers(0, 2, size=n)
        feature_names.append(name)
        ground_truth.append(GroundTruthColumn(name, "feature", "binary", "binary", 2))

    for i in range(cfg.n_lowcard):
        name = f"lowcard_{i}"
        levels = max(3, cfg.lowcard_levels)
        data[name] = rng.integers(0, levels, size=n)
        feature_names.append(name)
        ground_truth.append(
            GroundTruthColumn(name, "feature", _expected_lowcard_type(levels), "numerical", levels)
        )

    for i in range(cfg.n_highcard):
        name = f"highcard_{i}"
        data[name] = rng.normal(0, 1, size=n)
        feature_names.append(name)
        ground_truth.append(GroundTruthColumn(name, "feature", "continuous", "numerical", n))
