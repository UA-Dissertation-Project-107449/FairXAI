"""ClusteringEngine: fit multiple unsupervised methods and select the best."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .models import ClusterDiagnostics, ClusterResult

logger = logging.getLogger(__name__)

_DEFAULT_EXCLUDE = ["heart_disease", "age_group", "sex", "ethnicity", "group_cluster"]
_DEFAULT_MIN_SILHOUETTE = 0.05
_DEFAULT_MAX_DBSCAN_NOISE_FRACTION = 0.30


class ClusteringError(ValueError):
    """Raised when clustering cannot produce a valid solution."""

    def __init__(
        self, message: str, diagnostics: Optional[List[ClusterDiagnostics]] = None
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or []


class ClusteringEngine:
    """Fit K-Means, Hierarchical, DBSCAN, and GMM; select the best assignment.

    Best is defined as the method + hyperparameter combination with the highest
    silhouette score.  GMM uses BIC for internal candidate selection but is
    still compared to others via silhouette.

    Args:
        config: Dict from ``clustering.yaml["clustering_methods"]``.  Only
            methods present as keys in this dict are fitted.  Pass ``None``
            to use sensible defaults for all four methods.
        feature_exclude: Extra columns to strip before clustering (in addition
            to the YAML ``data.feature_selection.exclude`` list).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        feature_exclude: Optional[List[str]] = None,
    ) -> None:
        self._config = config or {}
        self._extra_exclude = set(feature_exclude or [])
        self._scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> ClusterResult:
        """Fit all configured methods and return the best cluster assignment.

        Args:
            df: Input DataFrame (already preprocessed / scaled values OK too).
            feature_cols: Explicit list of columns to cluster on.  When
                ``None``, all numeric columns not in the exclude list are used.

        Returns:
            :class:`ClusterResult` with ``group_cluster`` Series (int 0..k-1).

        Raises:
            ClusteringError: When no method produces a valid solution.
        """
        cols = self._resolve_feature_cols(df, feature_cols)
        if not cols:
            raise ClusteringError("No usable feature columns found after exclusion.")

        X = self._scaler.fit_transform(df[cols].values)
        n_samples = len(X)

        all_diagnostics: List[ClusterDiagnostics] = []
        candidates: List[tuple] = []  # (silhouette, labels, diagnostics)

        # -- K-Means -------------------------------------------------------
        if not self._config or "kmeans" in self._config:
            km_cfg = (self._config or {}).get("kmeans", {})
            result = self._fit_kmeans(X, km_cfg, n_samples)
            all_diagnostics.extend(result["diagnostics"])
            if result["best"] is not None:
                candidates.append(result["best"])

        # -- Hierarchical --------------------------------------------------
        if not self._config or "hierarchical" in self._config:
            hier_cfg = (self._config or {}).get("hierarchical", {})
            result = self._fit_hierarchical(X, hier_cfg, n_samples)
            all_diagnostics.extend(result["diagnostics"])
            if result["best"] is not None:
                candidates.append(result["best"])

        # -- DBSCAN --------------------------------------------------------
        if not self._config or "dbscan" in self._config:
            db_cfg = (self._config or {}).get("dbscan", {})
            result = self._fit_dbscan(X, db_cfg, n_samples)
            all_diagnostics.extend(result["diagnostics"])
            if result["best"] is not None:
                candidates.append(result["best"])

        # -- Gaussian Mixture ----------------------------------------------
        if not self._config or "gaussian_mixture" in self._config:
            gmm_cfg = (self._config or {}).get("gaussian_mixture", {})
            result = self._fit_gmm(X, gmm_cfg, n_samples)
            all_diagnostics.extend(result["diagnostics"])
            if result["best"] is not None:
                candidates.append(result["best"])

        if not candidates:
            raise ClusteringError(
                "No clustering method produced a valid solution "
                "(check data size, DBSCAN eps, or n_clusters grid).",
                diagnostics=all_diagnostics,
            )

        # Pick overall winner: highest silhouette
        best_sil, best_labels, best_diag = max(candidates, key=lambda t: t[0])
        if best_sil < _DEFAULT_MIN_SILHOUETTE:
            raise ClusteringError(
                "No clustering method produced a stable enough solution "
                f"(best silhouette={best_sil:.4f}, minimum={_DEFAULT_MIN_SILHOUETTE:.2f}).",
                diagnostics=all_diagnostics,
            )

        group_cluster = pd.Series(best_labels, index=df.index, name="group_cluster", dtype=int)

        logger.info(
            "[SUCCESS] Best clustering: method=%s n_clusters=%d silhouette=%.4f",
            best_diag.method,
            best_diag.n_clusters,
            best_sil,
        )

        return ClusterResult(
            group_cluster=group_cluster,
            method=best_diag.method,
            n_clusters=best_diag.n_clusters,
            silhouette=best_sil,
            feature_cols=cols,
            diagnostics=all_diagnostics,
        )

    def save_diagnostics(self, result: ClusterResult, output_dir: Path) -> Path:
        """Write cluster_diagnostics.csv to *output_dir*."""
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [d.to_dict() for d in result.diagnostics]
        df = pd.DataFrame(rows)
        out = output_dir / "cluster_diagnostics.csv"
        df.to_csv(out, index=False)
        logger.info("[SUCCESS] cluster_diagnostics saved to %s", out)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_feature_cols(self, df: pd.DataFrame, explicit: Optional[List[str]]) -> List[str]:
        if explicit is not None:
            return [c for c in explicit if c in df.columns]
        exclude_from_yaml = set(
            (self._config or {})
            .get("data", {})
            .get("feature_selection", {})
            .get("exclude", _DEFAULT_EXCLUDE)
        )
        exclude = exclude_from_yaml | self._extra_exclude
        return [c for c in df.select_dtypes(include="number").columns if c not in exclude]

    # -- K-Means -----------------------------------------------------------

    def _fit_kmeans(self, X: np.ndarray, cfg: Dict, n_samples: int) -> Dict:
        params = cfg.get("parameters", {})
        k_grid = params.get("n_clusters", [3, 4, 5, 6])
        init = params.get("init", "k-means++")
        n_init = params.get("n_init", 10)
        max_iter = params.get("max_iter", 300)
        random_state = params.get("random_state", 42)

        diagnostics = []
        best: Optional[tuple] = None

        for k in k_grid:
            if k >= n_samples:
                logger.debug("kmeans: k=%d >= n_samples=%d, skipping", k, n_samples)
                continue
            try:
                km = KMeans(
                    n_clusters=k,
                    init=init,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=random_state,
                )
                labels = km.fit_predict(X)
                sil = silhouette_score(X, labels)
                diag = ClusterDiagnostics(
                    method="kmeans",
                    params={"n_clusters": k, "init": init},
                    n_clusters=k,
                    silhouette=sil,
                )
                diagnostics.append(diag)
                if best is None or sil > best[0]:
                    best = (sil, labels, diag)
            except Exception as exc:
                logger.debug("kmeans k=%d failed: %s", k, exc)

        return {"diagnostics": diagnostics, "best": best}

    # -- Hierarchical ------------------------------------------------------

    def _fit_hierarchical(self, X: np.ndarray, cfg: Dict, n_samples: int) -> Dict:
        params = cfg.get("parameters", {})
        k_grid = params.get("n_clusters", [3, 4, 5, 6])
        linkage = params.get("linkage", "ward")

        diagnostics = []
        best: Optional[tuple] = None

        for k in k_grid:
            if k >= n_samples:
                continue
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels = agg.fit_predict(X)
                sil = silhouette_score(X, labels)
                diag = ClusterDiagnostics(
                    method="hierarchical",
                    params={"n_clusters": k, "linkage": linkage},
                    n_clusters=k,
                    silhouette=sil,
                )
                diagnostics.append(diag)
                if best is None or sil > best[0]:
                    best = (sil, labels, diag)
            except Exception as exc:
                logger.debug("hierarchical k=%d failed: %s", k, exc)

        return {"diagnostics": diagnostics, "best": best}

    # -- DBSCAN ------------------------------------------------------------

    def _fit_dbscan(self, X: np.ndarray, cfg: Dict, n_samples: int) -> Dict:
        params = cfg.get("parameters", {})
        eps_grid = params.get("eps", [0.3, 0.5, 0.7, 1.0])
        min_samples_grid = params.get("min_samples", [5, 10, 15])
        max_noise_fraction = float(
            params.get("max_noise_fraction", _DEFAULT_MAX_DBSCAN_NOISE_FRACTION)
        )

        diagnostics = []
        best: Optional[tuple] = None

        for eps in eps_grid:
            for min_s in min_samples_grid:
                try:
                    db = DBSCAN(eps=eps, min_samples=min_s, metric="euclidean")
                    labels_raw = db.fit_predict(X)
                    # Filter noise points (label == -1) for silhouette
                    mask = labels_raw != -1
                    n_noise = (~mask).sum()
                    noise_fraction = float(n_noise / n_samples) if n_samples else 1.0
                    unique_labels = np.unique(labels_raw[mask])
                    n_clusters = len(unique_labels)

                    if n_clusters < 2:
                        diag = ClusterDiagnostics(
                            method="dbscan",
                            params={"eps": eps, "min_samples": min_s},
                            n_clusters=n_clusters,
                            silhouette=None,
                            note=f"only {n_clusters} cluster(s) + {n_noise} noise",
                        )
                        diagnostics.append(diag)
                        continue

                    # Re-label so cluster ids are 0..k-1 (noise stays excluded)
                    labels_clean = labels_raw[mask]
                    sil = silhouette_score(X[mask], labels_clean)
                    if noise_fraction > max_noise_fraction:
                        diag = ClusterDiagnostics(
                            method="dbscan",
                            params={"eps": eps, "min_samples": min_s},
                            n_clusters=n_clusters,
                            silhouette=sil,
                            note=(
                                f"rejected: noise_fraction={noise_fraction:.1%} "
                                f"> max_noise_fraction={max_noise_fraction:.1%}"
                            ),
                        )
                        diagnostics.append(diag)
                        continue

                    diag = ClusterDiagnostics(
                        method="dbscan",
                        params={"eps": eps, "min_samples": min_s},
                        n_clusters=n_clusters + int(n_noise > 0),
                        silhouette=sil,
                        note=(
                            f"{n_noise} noise points kept as noise cluster"
                            if n_noise > 0
                            else "no noise"
                        ),
                    )
                    diagnostics.append(diag)

                    # Keep DBSCAN noise separate instead of folding it into the largest cluster.
                    relabeled = labels_raw.copy()
                    if n_noise > 0:
                        relabeled[~mask] = n_clusters

                    if best is None or sil > best[0]:
                        best = (sil, relabeled, diag)
                except Exception as exc:
                    logger.debug("dbscan eps=%.2f min_s=%d failed: %s", eps, min_s, exc)

        return {"diagnostics": diagnostics, "best": best}

    # -- Gaussian Mixture --------------------------------------------------

    def _fit_gmm(self, X: np.ndarray, cfg: Dict, n_samples: int) -> Dict:
        params = cfg.get("parameters", {})
        k_grid = params.get("n_components", [3, 4, 5, 6])
        cov_type = params.get("covariance_type", "full")
        max_iter = params.get("max_iter", 100)
        n_init = params.get("n_init", 10)
        random_state = params.get("random_state", 42)

        diagnostics = []
        # First select best GMM candidate via BIC (lower = better)
        best_bic_candidate: Optional[tuple] = None  # (bic, labels, diag)
        best: Optional[tuple] = None

        for k in k_grid:
            if k >= n_samples:
                continue
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    max_iter=max_iter,
                    n_init=n_init,
                    random_state=random_state,
                )
                gmm.fit(X)
                labels = gmm.predict(X)
                bic = gmm.bic(X)

                n_unique = len(np.unique(labels))
                sil = silhouette_score(X, labels) if n_unique >= 2 else None

                diag = ClusterDiagnostics(
                    method="gaussian_mixture",
                    params={"n_components": k, "covariance_type": cov_type},
                    n_clusters=n_unique,
                    silhouette=sil,
                    bic=bic,
                )
                diagnostics.append(diag)

                if best_bic_candidate is None or bic < best_bic_candidate[0]:
                    best_bic_candidate = (bic, labels, diag)
            except Exception as exc:
                logger.debug("gmm k=%d failed: %s", k, exc)

        # Convert GMM winner to silhouette-comparable entry
        if best_bic_candidate is not None:
            _, labels, diag = best_bic_candidate
            n_unique = len(np.unique(labels))
            if n_unique >= 2:
                sil = silhouette_score(X, labels)
                best = (sil, labels, diag)

        return {"diagnostics": diagnostics, "best": best}
