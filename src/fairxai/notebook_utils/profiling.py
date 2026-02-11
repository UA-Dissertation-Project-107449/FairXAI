"""Helpers for profiling notebooks and profile JSONs."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from . import PALETTE_DATASET, PALETTE_SEX


def find_profile_files(results_dir: Path, pattern: str, datasets: list[str]) -> dict[str, Path]:
    files = list(results_dir.rglob(pattern)) if results_dir.exists() else []
    mapped: dict[str, Path] = {}
    for path in files:
        stem = path.stem
        name = stem.replace("_data_profile", "") if stem.endswith("_data_profile") else stem
        if name in datasets and name not in mapped:
            mapped[name] = path
    return mapped


def load_profiles(files: dict[str, Path]) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for name, path in files.items():
        try:
            with open(path, "r") as f:
                profiles[name] = json.load(f)
        except Exception as exc:
            print(f"Failed to load {name}: {exc}")
    return profiles


def dataset_overview_rows(profiles: dict[str, dict], datasets: list[str]) -> pd.DataFrame:
    rows = []
    for name in datasets:
        profile = profiles.get(name, {})
        basic = profile.get("basic_stats", {})
        rows.append({
            "dataset": name,
            "samples": basic.get("n_samples"),
            "features": basic.get("n_features"),
            "target_prevalence": basic.get("target_prevalence"),
        })
    return pd.DataFrame(rows)


def sensitive_distribution_rows(
    profiles: dict[str, dict],
    datasets: list[str],
    attributes: list[str],
) -> pd.DataFrame:
    rows = []
    for name in datasets:
        profile = profiles.get(name, {})
        for attr in attributes:
            dist = profile.get("sensitive_attr_distribution", {}).get(attr, {})
            counts = dist.get("counts", {})
            proportions = dist.get("proportions", {})
            total = sum(counts.values()) if counts else 0
            for group, count in counts.items():
                pct = proportions.get(group)
                if pct is None and total:
                    pct = count / total
                rows.append({
                    "dataset": name,
                    "attribute": attr,
                    "group": group,
                    "count": count,
                    "pct": pct,
                    "underrepresented": pct is not None and pct < 0.10,
                })
    return pd.DataFrame(rows)


def plot_sensitive_proportions(df: pd.DataFrame, attribute: str, datasets: list[str]) -> None:
    subset = df[df["attribute"] == attribute].copy()
    if subset.empty:
        print(f"No data for {attribute}")
        return
    pivot = subset.pivot_table(index="dataset", columns="group", values="pct", aggfunc="sum").fillna(0)
    pivot = pivot.reindex(datasets)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(8, 4))
    ax.set_title(f"{attribute} proportions")
    ax.set_ylabel("proportion")
    ax.set_ylim(0, 1.0)
    ax.legend(title=attribute, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()


def representation_balance_rows(
    profiles: dict[str, dict],
    datasets: list[str],
    attributes: list[str],
) -> pd.DataFrame:
    rows = []
    for name in datasets:
        profile = profiles.get(name, {})
        for attr in attributes:
            balance = profile.get("representation_balance", {}).get(attr, {})
            rows.append({
                "dataset": name,
                "attribute": attr,
                "cv": balance.get("coefficient_of_variation"),
                "min_group": balance.get("min_group_size"),
                "max_group": balance.get("max_group_size"),
                "size_ratio": balance.get("size_ratio"),
            })
    return pd.DataFrame(rows)


def plot_balance_cv(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["cv"]).copy()
    if plot_df.empty:
        print("No representation balance data available.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=plot_df, x="dataset", y="cv", hue="attribute", ax=ax)
    ax.axhline(0.3, color="#999999", linestyle="--", linewidth=1)
    ax.axhline(0.7, color="#999999", linestyle=":", linewidth=1)
    ax.set_title("Representation balance (CV)")
    ax.set_ylabel("CV")
    ax.set_ylim(0, max(0.8, plot_df["cv"].max() * 1.2))
    ax.legend(title="attribute", loc="upper right")
    plt.tight_layout()


def plot_size_ratio_heatmap(df: pd.DataFrame, datasets: list[str]) -> None:
    ratio_df = df.dropna(subset=["size_ratio"]).copy()
    if ratio_df.empty:
        print("No size ratio data available.")
        return
    heat = ratio_df.pivot(index="dataset", columns="attribute", values="size_ratio").reindex(datasets)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="Reds", ax=ax)
    ax.set_title("Group size ratio (max/min)")
    plt.tight_layout()


def group_statistics_rows(
    profiles: dict[str, dict],
    datasets: list[str],
    attributes: list[str],
) -> pd.DataFrame:
    rows = []
    for name in datasets:
        profile = profiles.get(name, {})
        for attr in attributes:
            groups = profile.get("group_statistics", {}).get(attr, {})
            for group, stats in groups.items():
                rows.append({
                    "dataset": name,
                    "attribute": attr,
                    "group": group,
                    "n": stats.get("n_samples"),
                    "prevalence": stats.get("target_prevalence"),
                })
    return pd.DataFrame(rows)


def plot_prevalence_heatmap_by_age(df: pd.DataFrame, age_order: list[str]) -> None:
    age_df = df[df["attribute"] == "age_group"].copy()
    if age_df.empty:
        print("No age-group prevalence data available.")
        return
    age_df["group"] = pd.Categorical(age_df["group"], categories=age_order, ordered=True)
    heat = age_df.pivot(index="group", columns="dataset", values="prevalence").reindex(age_order)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1, ax=ax)
    ax.set_title("Prevalence by age group")
    ax.set_xlabel("dataset")
    ax.set_ylabel("age_group")
    plt.tight_layout()


def plot_prevalence_by_sex(df: pd.DataFrame) -> None:
    sex_df = df[df["attribute"] == "sex"].copy()
    if sex_df.empty:
        print("No sex prevalence data available.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=sex_df, x="dataset", y="prevalence", hue="group", ax=ax, palette=PALETTE_SEX)
    ax.set_title("Prevalence by sex")
    ax.set_ylabel("prevalence")
    ax.set_ylim(0, 1.0)
    ax.legend(title="sex", loc="upper right")
    plt.tight_layout()


def spd_rows(
    profiles: dict[str, dict],
    datasets: list[str],
    attributes: list[str],
) -> pd.DataFrame:
    rows = []
    for name in datasets:
        profile = profiles.get(name, {})
        for attr in attributes:
            spd = profile.get("label_imbalance_by_group", {}).get(attr, {}).get("statistical_parity_difference", {})
            rows.append({
                "dataset": name,
                "attribute": attr,
                "max_spd": spd.get("max_difference"),
                "max_ratio": spd.get("max_ratio"),
            })
    return pd.DataFrame(rows)


def plot_spd_bars(df: pd.DataFrame) -> None:
    plot_spd = df.dropna(subset=["max_spd"]).copy()
    if plot_spd.empty:
        print("No SPD data available.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(data=plot_spd, x="dataset", y="max_spd", hue="attribute", ax=axes[0])
    axes[0].axhline(0.1, color="#999999", linestyle="--", linewidth=1)
    axes[0].set_title("Max SPD")
    axes[0].set_ylim(0, max(0.2, plot_spd["max_spd"].max() * 1.2))
    if plot_spd["max_ratio"].notna().any():
        sns.barplot(data=plot_spd.dropna(subset=["max_ratio"]), x="dataset", y="max_ratio", hue="attribute", ax=axes[1])
        axes[1].set_title("Max ratio")
    else:
        axes[1].axis("off")
    for ax in axes:
        ax.legend(title="attribute", loc="upper right")
    plt.tight_layout()


def positive_rate_rows(
    profiles: dict[str, dict],
    datasets: list[str],
    attribute: str,
) -> pd.DataFrame:
    rows = []
    for name in datasets:
        rates = profiles.get(name, {}).get("label_imbalance_by_group", {}).get(attribute, {}).get("positive_rates", {})
        for group, value in rates.items():
            rows.append({"dataset": name, "age_group": group, "prevalence": value})
    return pd.DataFrame(rows)


def plot_positive_rates_by_age(df: pd.DataFrame, age_order: list[str]) -> None:
    if df.empty:
        print("No positive-rate series available.")
        return
    df["age_group"] = pd.Categorical(df["age_group"], categories=age_order, ordered=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=df, x="age_group", y="prevalence", hue="dataset", marker="o", ax=ax, palette=PALETTE_DATASET)
    ax.set_title("Positive rate by age group")
    ax.set_ylabel("prevalence")
    ax.set_ylim(0, 1.0)
    ax.legend(title="dataset")
    plt.tight_layout()
