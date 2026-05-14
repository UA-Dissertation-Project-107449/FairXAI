"""Shared display labels for dissertation plots."""

from __future__ import annotations


MITIGATION_LABELS = {
    "baseline": "Baseline",
    "adasyn": "ADASYN",
    "adasyn+exponentiated_gradient": "ADASYN + EG",
    "adasyn+exponentiated_gradient+threshold_optimizer": "ADASYN + EG + TO",
    "exponentiated_gradient": "Exp. Gradient",
    "grid_search": "Fairlearn Grid",
    "reweighting": "Reweighting",
    "reweighting+threshold_optimizer": "Reweighting + TO",
    "reweighing": "Reweighting",
    "smote": "SMOTE",
    "smote+exponentiated_gradient": "SMOTE + EG",
    "smote+exponentiated_gradient+threshold_optimizer": "SMOTE + EG + TO",
    "smote+threshold_optimizer": "SMOTE + TO",
    "threshold_optimizer": "Threshold Opt.",
}


def display_mitigation(value: object) -> str:
    """Return a compact label for mitigation names."""
    value_str = str(value)
    if value_str in MITIGATION_LABELS:
        return MITIGATION_LABELS[value_str]
    return value_str.replace("_", " ").replace("+", " + ").title()


def normalize_sensitive_attr(attr: object) -> str:
    """Normalize attrs like age_group_cat to age_group."""
    attr_str = str(attr)
    if attr_str.endswith("_cat"):
        return attr_str[: -len("_cat")]
    return attr_str


def pretty_group_label(attr: object, group: object) -> str:
    """Return readable subgroup labels for plots."""
    group_str = str(group)
    if normalize_sensitive_attr(attr) == "sex":
        return {"0": "Female", "1": "Male"}.get(group_str, group_str)
    return group_str
