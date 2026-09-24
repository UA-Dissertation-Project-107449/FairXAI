"""ClusteringEngine: fit multiple unsupervised methods and select the best."""

from __future__ import annotations

import logging
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .models import ClusterDiagnostics, ClusterResult

logger = logging.getLogger(__name__)

_DEFAULT_EXCLUDE = ["heart_disease", "age_group", "sex", "ethnicity", "group_cluster"]
_DEFAULT_MIN_SILHOUETTE = 0.05
_DEFAULT_MAX_DBSCAN_NOISE_FRACTION = 0.30

# Row ceiling for agglomerative clustering. Ward linkage without a connectivity
# graph goes through scipy's hierarchy.ward, which materialises the full
# condensed distance matrix: n(n-1)/2 float64 entries, rebuilt once per k in the
# grid. That is 0.37 GiB at 10k rows but 17.6 GiB at 68.7k, which is what killed
# the cardio70k grouping study (OOM at 28.1 GiB anon-rss on a 30 GB machine).
# Every other method here is memory-bounded, so the ceiling drops hierarchical
# alone rather than the whole study. 10_000 is set so the cardio70k 10k
# subsample still clusters hierarchically and the full 70k run does not.
_DEFAULT_MAX_HIERARCHICAL_SAMPLES = 10_000

# Memory budget for a single DBSCAN candidate's neighbour lists. DBSCAN holds,
# for every row, the indices of every row within ``eps``; that is n * k * 8 bytes
# with k the average neighbourhood size, so cost is driven by ``eps`` and density,
# not by rows alone. The shipped grid reaches eps=5.0, and in 13-D standardised
# space the expected pairwise distance is only ~5.1, so at that eps roughly half
# the dataset is a neighbour of every row: ~31 GiB at 68.7k rows, a second OOM in
# the same study that the hierarchical ceiling above does not cover. A row ceiling
# would be the wrong instrument here because eps=0.5 stays cheap at any n, so the
# budget is checked per eps from a measured density estimate instead.
_DEFAULT_MAX_DBSCAN_NEIGHBOR_GIB = 4.0

# The estimate below counts neighbour indices only. Measured peak RSS ran
# 1.66-1.74x that across n=5k/10k/20k at eps=5.0 (tree structures, the boolean
# core-sample mask and temporary copies), so the estimate is scaled by 2.0 to
# stay conservative at the top of that range.
_DBSCAN_NEIGHBOR_SAFETY_FACTOR = 2.0

# Rows sampled to estimate neighbourhood density. The estimate needs the fraction
# of pairs within eps, which a uniform subsample gives to well within the accuracy
# a power-of-two budget check needs; 2000 rows is one 4M-entry distance matrix.
_DBSCAN_DENSITY_SAMPLE_ROWS = 2000


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
        min_clusters: int = 2,
        min_cluster_size_abs: int = 1,
        min_cluster_size_frac: float = 0.0,
        min_silhouette: Optional[float] = None,
    ) -> None:
        self._config = config or {}
        self._extra_exclude = set(feature_exclude or [])
        self._scaler = StandardScaler()
        # Validity gate (defaults are a no-op → WebApp byte-identical). A solution
        # is valid only if it has >= min_clusters and every cluster holds at least
        # max(min_cluster_size_abs, ceil(min_cluster_size_frac * n)) samples.
        self.min_clusters = min_clusters
        self.min_cluster_size_abs = min_cluster_size_abs
        self.min_cluster_size_frac = min_cluster_size_frac
        # Optional silhouette floor; None = disabled (no floor). Use
        # _DEFAULT_MIN_SILHOUETTE (0.05) to opt into evidence_cleanup's stability gate.
        self.min_silhouette = min_silhouette

    def _effective_min_cluster_size(self, n_samples: int) -> int:
        return max(self.min_cluster_size_abs, ceil(self.min_cluster_size_frac * n_samples))

    def _is_valid_solution(self, labels: "np.ndarray", min_size: int) -> bool:
        uniq, counts = np.unique(labels, return_counts=True)
        return len(uniq) >= self.min_clusters and counts.min() >= min_size

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

        # Validity gate: drop degenerate solutions (single effective cluster, or
        # any cluster below the min size). Disqualifies the whole solution so a
        # lopsided DBSCAN (e.g. 88% in one cluster + tiny rest) can't win on
        # silhouette over a balanced KMeans. Defaults make this a no-op.
        min_size = self._effective_min_cluster_size(n_samples)
        valid = [c for c in candidates if self._is_valid_solution(np.asarray(c[1]), min_size)]
        if not valid:
            raise ClusteringError(
                f"No clustering solution met the validity gate "
                f"(min_clusters={self.min_clusters}, min_cluster_size={min_size} "
                f"= max({self.min_cluster_size_abs}, ceil({self.min_cluster_size_frac}*{n_samples}))). "
                "All candidates were degenerate (single dominant cluster or sub-threshold groups).",
                diagnostics=all_diagnostics,
            )

        # Pick overall winner among valid solutions: highest silhouette.
        best_sil, best_labels, best_diag = max(valid, key=lambda t: t[0])

        # Optional silhouette floor (opt-in; default None = no floor → WebApp byte-identical).
        # evidence_cleanup wanted an unconditional 0.05 floor for dissertation evidence
        # quality; expose it as a knob (default _DEFAULT_MIN_SILHOUETTE) rather than always-on.
        if self.min_silhouette is not None and best_sil < self.min_silhouette:
            raise ClusteringError(
                "No clustering method produced a stable enough solution "
                f"(best silhouette={best_sil:.4f}, minimum={self.min_silhouette:.2f}).",
                diagnostics=all_diagnostics,
            )

        group_cluster = pd.Series(best_labels, index=df.index, name="group_cluster", dtype=int)
        n_noise = int(best_diag.n_noise or 0)
        noise_fraction = n_noise / n_samples if n_samples else 0.0

        logger.info(
            "[SUCCESS] Best clustering: method=%s n_clusters=%d silhouette=%.4f noise=%d (%.1f%%)",
            best_diag.method,
            best_diag.n_clusters,
            best_sil,
            n_noise,
            noise_fraction * 100,
        )

        return ClusterResult(
            group_cluster=group_cluster,
            method=best_diag.method,
            n_clusters=best_diag.n_clusters,
            silhouette=best_sil,
            feature_cols=cols,
            diagnostics=all_diagnostics,
            n_noise=n_noise,
            noise_fraction=noise_fraction,
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
        max_samples = int(params.get("max_samples", _DEFAULT_MAX_HIERARCHICAL_SAMPLES))

        diagnostics = []
        best: Optional[tuple] = None

        if max_samples > 0 and n_samples > max_samples:
            # Recorded rather than silently dropped: the diagnostics CSV is the
            # audit trail for which candidates a run actually considered, so a
            # skipped method has to be visible there.
            logger.warning(
                "hierarchical SKIPPED: %d rows exceeds max_samples=%d. Ward linkage "
                "needs an n(n-1)/2 distance matrix (%.1f GiB here), which does not fit. "
                "Raise clustering_methods.hierarchical.parameters.max_samples to force it.",
                n_samples,
                max_samples,
                n_samples * (n_samples - 1) / 2 * 8 / 2**30,
            )
            diagnostics.append(
                ClusterDiagnostics(
                    method="hierarchical",
                    params={"linkage": linkage, "max_samples": max_samples},
                    n_clusters=0,
                    silhouette=None,
                    note=f"skipped: n_samples={n_samples} exceeds max_samples={max_samples}",
                )
            )
            return {"diagnostics": diagnostics, "best": best}

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

    @staticmethod
    def _estimate_dbscan_neighbor_gib(X: np.ndarray, eps: float, n_samples: int) -> Optional[float]:
        """Estimated peak GiB of DBSCAN's neighbour lists for one ``eps``.

        Takes a uniform subsample, measures the fraction of pairs within ``eps``
        (the diagonal counts: a point is its own neighbour in the radius query),
        and scales that density up to the full row count. Returns the index bytes
        times :data:`_DBSCAN_NEIGHBOR_SAFETY_FACTOR`.

        Non-finite rows are dropped before measuring, because ``pairwise_distances``
        rejects NaN outright. Returns ``None`` when no estimate is possible, which
        the caller reads as "unknown, do not block": a budget check that cannot run
        must never be what stops a dataset being clustered.
        """
        if n_samples <= 0:
            return 0.0
        finite = X[np.isfinite(X).all(axis=1)]
        if len(finite) < 2:
            return None
        m = min(_DBSCAN_DENSITY_SAMPLE_ROWS, len(finite))
        rng = np.random.default_rng(0)  # fixed: the budget check must be reproducible
        sample = (
            finite[rng.choice(len(finite), size=m, replace=False)] if len(finite) > m else finite
        )
        within = pairwise_distances(sample, sample) <= eps
        fraction = float(within.mean())
        index_bytes = n_samples * fraction * n_samples * 8
        return index_bytes / 2**30 * _DBSCAN_NEIGHBOR_SAFETY_FACTOR

    def _fit_dbscan(self, X: np.ndarray, cfg: Dict, n_samples: int) -> Dict:
        params = cfg.get("parameters", {})
        eps_grid = params.get("eps", [0.3, 0.5, 0.7, 1.0])
        min_samples_grid = params.get("min_samples", [5, 10, 15])
        max_noise_fraction = float(
            params.get("max_noise_fraction", _DEFAULT_MAX_DBSCAN_NOISE_FRACTION)
        )
        max_neighbor_gib = float(params.get("max_neighbor_gib", _DEFAULT_MAX_DBSCAN_NEIGHBOR_GIB))

        diagnostics = []
        best: Optional[tuple] = None

        for eps in eps_grid:
            if max_neighbor_gib > 0:
                try:
                    projected = self._estimate_dbscan_neighbor_gib(X, eps, n_samples)
                except Exception as exc:
                    # The estimate is an optimisation, not a correctness check, and
                    # it runs outside the per-candidate try below. Letting it raise
                    # would abort the whole dataset over a guard that is only meant
                    # to skip one candidate, so an unusable estimate falls through
                    # to the fit and whatever that does.
                    logger.debug("dbscan eps=%.2f budget estimate failed: %s", eps, exc)
                    projected = None
                if projected is not None and projected > max_neighbor_gib:
                    # Recorded per eps rather than dropped silently: the
                    # diagnostics CSV is the audit trail for which candidates a
                    # run actually considered.
                    logger.warning(
                        "dbscan eps=%.2f SKIPPED at %d rows: neighbour lists need "
                        "~%.1f GiB, over max_neighbor_gib=%.1f. Raise "
                        "clustering_methods.dbscan.parameters.max_neighbor_gib to "
                        "force it.",
                        eps,
                        n_samples,
                        projected,
                        max_neighbor_gib,
                    )
                    for min_s in min_samples_grid:
                        diagnostics.append(
                            ClusterDiagnostics(
                                method="dbscan",
                                params={"eps": eps, "min_samples": min_s},
                                n_clusters=0,
                                silhouette=None,
                                note=(
                                    f"skipped: estimated {projected:.1f} GiB of "
                                    f"neighbours at n_samples={n_samples} exceeds "
                                    f"max_neighbor_gib={max_neighbor_gib}"
                                ),
                            )
                        )
                    continue
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
                            n_noise=int(n_noise),
                            note=f"only {n_clusters} cluster(s) + {n_noise} noise",
                        )
                        diagnostics.append(diag)
                        continue

                    # Keep DBSCAN noise separate instead of folding it into the
                    # largest cluster; the noise label is one of the groups
                    # downstream, so it is one of the groups the score is about.
                    relabeled = labels_raw.copy()
                    if n_noise > 0:
                        relabeled[~mask] = n_clusters

                    # Score every point, noise included. Scoring the kept points
                    # alone paid DBSCAN for discarding the rows it fitted worst,
                    # so a solution that called 18% of the cohort noise beat a
                    # KMeans partition of all of it.
                    sil = silhouette_score(X, relabeled)
                    if noise_fraction > max_noise_fraction:
                        diag = ClusterDiagnostics(
                            method="dbscan",
                            params={"eps": eps, "min_samples": min_s},
                            n_clusters=n_clusters,
                            silhouette=sil,
                            n_noise=int(n_noise),
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
                        n_noise=int(n_noise),
                        note=(
                            f"{n_noise} noise points ({noise_fraction:.1%}) kept "
                            f"as noise cluster and scored"
                            if n_noise > 0
                            else "no noise"
                        ),
                    )
                    diagnostics.append(diag)

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
