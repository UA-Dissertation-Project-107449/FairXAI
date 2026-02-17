"""Comparison visualizations for cross-dataset analysis.

This module is intentionally scaffolded for incremental implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_correlation_heatmap_grid(
	corr_targets: list[tuple[str, pd.DataFrame, list[str]]],
	categorical_feature_names: set[str] | None = None,
	categorical_series_normalizer: Callable[[str, pd.Series], pd.Series] | None = None,
	figsize: tuple[float, float] = (24, 6),
	cmap: str = "coolwarm",
	annot: bool = True,
	annot_size: int = 8,
	save_path: Path | None = None,
	show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
	if not corr_targets:
		raise ValueError("`corr_targets` must contain at least one item.")

	def numeric_for_corr(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
		cols: dict[str, pd.Series] = {}
		for feature in features:
			if feature not in df.columns:
				continue
			series = df[feature]
			if (
				categorical_feature_names
				and feature in categorical_feature_names
				and categorical_series_normalizer is not None
			):
				mapped = categorical_series_normalizer(feature, series)
				codes = pd.Series(mapped).cat.codes.replace(-1, np.nan)
				cols[feature] = codes
			else:
				cols[feature] = pd.to_numeric(series, errors="coerce")

		numeric = pd.DataFrame(cols)
		numeric = numeric.dropna(axis=1, how="all")
		numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
		return numeric

	fig, axes = plt.subplots(1, len(corr_targets), figsize=figsize)
	axes = np.atleast_1d(axes).ravel()

	for idx, (name, df, features) in enumerate(corr_targets):
		numeric = numeric_for_corr(df, features) if not df.empty else pd.DataFrame()
		if numeric.shape[1] < 2:
			axes[idx].set_title(f"{name} correlations (insufficient numeric features)")
			axes[idx].axis("off")
			continue

		corr = numeric.corr(numeric_only=True)
		sns.heatmap(
			corr,
			ax=axes[idx],
			cmap=cmap,
			center=0,
			vmin=-1,
			vmax=1,
			annot=annot,
			fmt=".2f",
			annot_kws={"size": annot_size},
			cbar=True,
		)
		axes[idx].set_title(f"{name} correlations (n={corr.shape[0]})")

	plt.tight_layout()
	if save_path:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=300, bbox_inches="tight")
	if show:
		plt.show()
	return fig, axes


def plot_pca_kmeans_scatter_grid(
	datasets: dict[str, pd.DataFrame],
	target_col: str | None = None,
	n_clusters: int = 3,
	sample_size: int = 1500,
	random_state: int = 42,
	figsize: tuple[float, float] = (15, 4),
	save_path: Path | None = None,
	show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
	if not datasets:
		raise ValueError("`datasets` must contain at least one dataframe.")

	fig, axes = plt.subplots(1, len(datasets), figsize=figsize)
	axes = np.atleast_1d(axes).ravel()

	for idx, (name, df) in enumerate(datasets.items()):
		numeric = df.select_dtypes(include=[np.number])
		if target_col:
			numeric = numeric.drop(columns=[target_col], errors="ignore")
		numeric = numeric.dropna(axis=0)

		if numeric.empty:
			axes[idx].set_title(f"{name} clustering (no data)")
			axes[idx].axis("off")
			continue

		sampled = numeric.sample(n=min(sample_size, len(numeric)), random_state=random_state)
		scaled = StandardScaler().fit_transform(sampled)
		embedding = PCA(n_components=2, random_state=random_state).fit_transform(scaled)
		clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(embedding)

		axes[idx].scatter(embedding[:, 0], embedding[:, 1], c=clusters, s=8, cmap="tab10")
		axes[idx].set_title(f"{name} clustering (PCA)")
		axes[idx].set_xlabel("PC1")
		axes[idx].set_ylabel("PC2")

	plt.tight_layout()
	if save_path:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=300, bbox_inches="tight")
	if show:
		plt.show()
	return fig, axes


def plot_two_dataset_feature_distributions(
	dataset_a: pd.DataFrame,
	dataset_b: pd.DataFrame,
	shared_features: list[str],
	dataset_a_name: str,
	dataset_b_name: str,
	dataset_palette: dict[str, str],
	units: dict[str, str] | None = None,
	categorical_feature_names: set[str] | None = None,
	categorical_series_normalizer: Callable[[str, pd.Series], pd.Series] | None = None,
	categorical_order_map: dict[str, list[str]] | None = None,
	save_path_prefix: Path | None = None,
	show: bool = False,
) -> dict[str, list[str]]:
	units = units or {}
	categorical_feature_names = categorical_feature_names or set()
	categorical_order_map = categorical_order_map or {}

	numeric_features = [feature for feature in shared_features if feature not in categorical_feature_names]
	categorical_features = [feature for feature in shared_features if feature in categorical_feature_names]

	def numeric_series(df: pd.DataFrame, feature: str) -> pd.Series:
		return pd.to_numeric(df[feature], errors="coerce")

	if numeric_features:
		fig, axes = plt.subplots(len(numeric_features), 1, figsize=(10, max(4, len(numeric_features) * 2.2)))
		axes = np.atleast_1d(axes).ravel()

		for idx, feature in enumerate(numeric_features):
			sns.kdeplot(
				numeric_series(dataset_a, feature).dropna(),
				ax=axes[idx],
				label=dataset_a_name,
				color=dataset_palette[dataset_a_name],
				fill=True,
				alpha=0.3,
			)
			sns.kdeplot(
				numeric_series(dataset_b, feature).dropna(),
				ax=axes[idx],
				label=dataset_b_name,
				color=dataset_palette[dataset_b_name],
				fill=True,
				alpha=0.3,
			)
			unit = units.get(feature)
			title = f"{feature} distribution ({dataset_a_name} vs {dataset_b_name})" + (f" ({unit})" if unit else "")
			axes[idx].set_title(title)
			if unit:
				axes[idx].set_xlabel(f"{feature} ({unit})")
			axes[idx].legend(loc="best")

		plt.tight_layout()
		if save_path_prefix:
			numeric_path = save_path_prefix.parent / f"{save_path_prefix.name}_numeric.png"
			numeric_path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(numeric_path, dpi=300, bbox_inches="tight")
		if show:
			plt.show()

	if categorical_features:
		rows = len(categorical_features)
		fig, axes = plt.subplots(rows, 1, figsize=(10, max(4, rows * 3.2)))
		axes = np.atleast_1d(axes).reshape(-1)

		for idx, feature in enumerate(categorical_features):
			if categorical_series_normalizer is not None:
				dataset_a_values = categorical_series_normalizer(feature, dataset_a[feature])
				dataset_b_values = categorical_series_normalizer(feature, dataset_b[feature])
			else:
				dataset_a_values = dataset_a[feature].astype(str)
				dataset_b_values = dataset_b[feature].astype(str)

			order = categorical_order_map.get(feature)
			dataset_a_counts = pd.Series(dataset_a_values).value_counts()
			dataset_b_counts = pd.Series(dataset_b_values).value_counts()
			if order:
				dataset_a_counts = dataset_a_counts.reindex(order, fill_value=0)
				dataset_b_counts = dataset_b_counts.reindex(order, fill_value=0)

			counts_df = pd.DataFrame({dataset_a_name: dataset_a_counts, dataset_b_name: dataset_b_counts}).fillna(0)
			pct_df = counts_df.div(counts_df.sum(axis=0), axis=1)
			pct_df.plot(
				kind="bar",
				ax=axes[idx],
				color=[dataset_palette[dataset_a_name], dataset_palette[dataset_b_name]],
			)

			for container_index, container in enumerate(axes[idx].containers):
				count_series = counts_df.iloc[:, container_index]
				labels = [f"{int(count)} ({value:.0%})" for value, count in zip(container.datavalues, count_series)]
				axes[idx].bar_label(container, labels=labels, fontsize=9)

			axes[idx].set_title(f"{feature} proportions ({dataset_a_name} vs {dataset_b_name})")
			axes[idx].set_xlabel(feature)
			axes[idx].set_ylabel("proportion")
			axes[idx].set_ylim(0, 1.08)

		plt.tight_layout()
		if save_path_prefix:
			categorical_path = save_path_prefix.parent / f"{save_path_prefix.name}_categorical.png"
			categorical_path.parent.mkdir(parents=True, exist_ok=True)
			fig.savefig(categorical_path, dpi=300, bbox_inches="tight")
		if show:
			plt.show()

	return {
		"numeric_features": numeric_features,
		"categorical_features": categorical_features,
	}


def summarize_ks_test_between_datasets(
	dataset_a: pd.DataFrame,
	dataset_b: pd.DataFrame,
	features: list[str],
	dataset_a_name: str = "dataset_a",
	dataset_b_name: str = "dataset_b",
	alpha: float = 0.05,
	min_unique_values: int = 2,
) -> pd.DataFrame:
	if not features:
		return pd.DataFrame(
			columns=[
				"feature",
				"ks_stat",
				"p_value",
				"n_a",
				"n_b",
				"distributions_differ",
			]
		)

	rows: list[dict[str, object]] = []
	for feature in features:
		if feature not in dataset_a.columns or feature not in dataset_b.columns:
			continue

		a_vals = pd.to_numeric(dataset_a[feature], errors="coerce").dropna()
		b_vals = pd.to_numeric(dataset_b[feature], errors="coerce").dropna()

		if a_vals.nunique() < min_unique_values or b_vals.nunique() < min_unique_values:
			continue

		stat = ks_2samp(a_vals, b_vals)
		rows.append(
			{
				"feature": feature,
				"ks_stat": float(stat.statistic),
				"p_value": float(stat.pvalue),
				"n_a": int(a_vals.shape[0]),
				"n_b": int(b_vals.shape[0]),
				"distributions_differ": "Yes" if stat.pvalue < alpha else "No",
			}
		)

	result = pd.DataFrame(rows)
	if result.empty:
		return result

	result = result.sort_values(by=["p_value", "ks_stat"], ascending=[True, False]).reset_index(drop=True)
	result = result.rename(columns={"n_a": f"n_{dataset_a_name}", "n_b": f"n_{dataset_b_name}"})
	return result


def plot_feature_drift_matrix(*args, **kwargs):
	raise NotImplementedError("plot_feature_drift_matrix is planned but not implemented yet.")


def plot_dataset_similarity_radar(*args, **kwargs):
	raise NotImplementedError("plot_dataset_similarity_radar is planned but not implemented yet.")


def plot_group_representation_bars(*args, **kwargs):
	raise NotImplementedError("plot_group_representation_bars is planned but not implemented yet.")

