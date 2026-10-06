"""Build the results-chapter numbers ledger from the FairXAI runs named in runs.yaml.

Every number quoted in the dissertation's Chapters 6 and 7 should be traceable to a row
this script prints. It reads small CSV/JSON summaries and saved per-row predictions, never
a model, and writes one Markdown file.

    python3 scripts/thesis/build_ledger.py [--out output/thesis/chapter6_ledger.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from thesis_runs import (
    C70_MODELS,
    CALIBRATION,
    CARD,
    CARDIAC_COHORTS,
    DEFAULT_OUT,
    DERM,
    FOUR_SITE_DATA,
    REDUNDANT,
    RUN_C70,
    RUN_C70_SUB,
    RUN_CARDIAC,
    RUN_DERM_AUG,
    RUN_DERM_CACHED,
    SYNTHETIC,
    UCI_MODELS,
    USAB,
    rel,
    study_dir,
)

lines: list[str] = []
# Where performance_intervals.py wrote its CSVs; --intervals overrides.
intervals_dir = DEFAULT_OUT


def emit(text: str = "") -> None:
    lines.append(text)


def short(run: str) -> str:
    """``run_20260927_182904_...`` -> ``20260927_182904``."""
    return "_".join(run.split("_")[1:3])


def table(df: pd.DataFrame, floatfmt: str = ".3f") -> None:
    emit(df.to_markdown(index=False, floatfmt=floatfmt))
    emit()


# ---------------------------------------------------------------------------
def section_calibration() -> None:
    emit("## L1. Null calibration, B sweep (one row per stratification and B)")
    emit()
    for name in (
        "calibration_summary",
        "calibration_summary_paired",
        "calibration_summary_by_metric",
    ):
        f = CALIBRATION / f"{name}.csv"
        emit(f"Source: `{rel(f)}`")
        emit()
        table(pd.read_csv(f).sort_values("n_boot", kind="stable"), ".4f")


def section_resampling() -> None:
    emit("## L2. Resampling scheme actually in force (closes L22)")
    emit()
    rows = []
    for run in (RUN_CARDIAC, RUN_C70):
        pf = CARD / "runs" / run / "baseline" / "prediction_fairness"
        for f in sorted(pf.glob("*pairwise_ci.csv")):
            d = pd.read_csv(f, usecols=["stratify", "n_boot"])
            split = "cv" if "_cv_" in f.name else "test"
            rows.append(
                {
                    "run": run,
                    "file": f.name.replace("_pairwise_ci.csv", ""),
                    "split": split,
                    "stratify_used": ",".join(sorted(d.stratify.unique())),
                    "n_boot": int(d.n_boot.iloc[0]),
                }
            )
    table(pd.DataFrame(rows))


def section_triage() -> None:
    emit("## L3. Readiness verdicts and flagged categories")
    emit()
    cases = [
        (CARD / "runs" / RUN_CARDIAC / "recommendations" / "cleveland_uci", "cleveland_uci"),
        (CARD / "runs" / RUN_CARDIAC / "recommendations" / "four_site_uci", "four_site_uci"),
        (CARD / "runs" / RUN_C70 / "recommendations" / "cardio70k", "cardio70k"),
        (DERM / "runs" / RUN_DERM_AUG / "recommendations" / "pad_ufes_20", "pad_ufes_20"),
    ]
    rows, detail = [], []
    for path, name in cases:
        f = path / "triage.json"
        if not f.exists():
            rows.append({"cohort": name, "verdict": "MISSING", "source": rel(path)})
            continue
        d = json.load(open(f))
        d = d.get("triage", d)
        recs = d["recommendations"]
        prio = pd.Series([r.get("priority") for r in recs]).value_counts().to_dict()
        cats = sorted({r.get("category") for r in recs if r.get("category")})
        rows.append(
            {
                "cohort": name,
                "verdict": d["readiness_status"],
                "priorities": ", ".join(f"{k}:{v}" for k, v in sorted(prio.items())),
                "categories": "".join(cats),
                "n_recs": len(recs),
                "source": rel(f),
            }
        )
        for r in recs:
            detail.append(
                {
                    "cohort": name,
                    "priority": r.get("priority"),
                    "cat": r.get("category"),
                    "title": r.get("title"),
                }
            )
    table(pd.DataFrame(rows))
    emit("Detail:")
    emit()
    table(pd.DataFrame(detail))


def section_baseline() -> None:
    emit("## L4. Baseline predictive performance (held-out test split)")
    emit()
    emit("Stage-7 baseline, from `baseline/results/training_results.json` -- QUOTE THESE.")
    emit()
    rows = []
    for run in (RUN_CARDIAC, RUN_C70_SUB, RUN_C70):
        f = CARD / "runs" / run / "baseline" / "results" / "training_results.json"
        d = json.load(open(f))
        for cohort, models in d.items():
            for model, r in models.items():
                te, tr = r.get("test_metrics", {}), r.get("train_metrics", {})
                cvm = (r.get("cv_results") or {}).get("metrics", {})
                rows.append(
                    {
                        "run": short(run),
                        "cohort": cohort,
                        "model": model,
                        "n_train": r.get("n_train"),
                        "n_test": r.get("n_test"),
                        "n_features": r.get("n_features"),
                        "test_f1": te.get("f1_score"),
                        "test_auc": te.get("auc_roc"),
                        "test_recall": te.get("recall"),
                        "test_precision": te.get("precision"),
                        "test_acc": te.get("accuracy"),
                        "train_f1": tr.get("f1_score"),
                        "cv_f1": cvm.get("f1_score", {}).get("mean"),
                        "cv_f1_sd": cvm.get("f1_score", {}).get("std"),
                        "cv_auc": cvm.get("auc_roc", {}).get("mean"),
                        "cv_auc_sd": cvm.get("auc_roc", {}).get("std"),
                    }
                )
        emit(f"Source: `{rel(f)}`")
    emit()
    table(pd.DataFrame(rows))
    emit(
        "The dissertation_figures `baseline_model_comparison.csv` below is the BEST-F1 baseline "
        "cell per family out of the sweep's 70 baseline cells (generate_dissertation_plots.py "
        "~line 711): a maximum selected on the test split. DO NOT QUOTE as the baseline."
    )
    emit()
    frames = []
    for run in (RUN_CARDIAC, RUN_C70_SUB, RUN_C70):
        f = (
            CARD
            / "studies"
            / "dissertation_figures"
            / run
            / "model_stability"
            / "baseline_model_comparison.csv"
        )
        d = pd.read_csv(f)
        d.insert(0, "run", run)
        frames.append(d)
        emit(f"Source: `{rel(f)}`")
    emit()
    table(pd.concat(frames))

    emit("### Dermatology backbones, augmented and cached no-augmentation arms")
    emit()
    frames = []
    for run in (RUN_DERM_AUG, RUN_DERM_CACHED):
        f = DERM / "runs" / run / "baseline" / "comparison" / "model_comparison.csv"
        d = pd.read_csv(f)
        d.insert(0, "run", run)
        frames.append(d)
    derm = pd.concat(frames)
    cols = [
        "run",
        "model",
        "feature_cache",
        "n_test",
        "epochs_run",
        "best_epoch",
        "train_time_seconds",
    ]
    cols += ["accuracy", "precision", "recall", "f1", "auc"]
    table(derm[cols])
    fair = [c for c in derm.columns if c.endswith("_dp_delta") or c.endswith("_eo_delta")]
    table(derm[["run", "model"] + fair])

    aug = derm[derm.run == RUN_DERM_AUG].set_index("model")
    noaug = derm[derm.run == RUN_DERM_CACHED].set_index("model")
    perf = ["accuracy", "f1", "auc", "recall"]
    diff = (aug[perf] - noaug[perf]).reset_index()
    emit(
        "Augmented minus cached no-aug arm. NOT a matched pair: the arms differ in "
        "augmentation AND feature caching."
    )
    emit()
    table(diff, "+.3f")


def section_pairwise() -> None:
    emit("## L5. Pairwise evidence (the only inferential object)")
    emit()
    emit(
        "`distinct` drops equalized_odds/tpr (= equal_opportunity/tpr) and equalized_odds/fnr "
        "(= 1 - tpr). The BH family in the files includes those rows, which makes the "
        "adjustment slightly more conservative than necessary."
    )
    emit()
    rows, sig_rows = [], []
    for run in (RUN_CARDIAC, RUN_C70_SUB, RUN_C70):
        pf = CARD / "runs" / run / "baseline" / "prediction_fairness"
        for f in sorted(pf.glob("*pairwise_ci.csv")):
            d = pd.read_csv(f)
            split = "cv" if "_cv_" in f.name else "test"
            stem = f.name.replace("_cv_pairwise_ci.csv", "").replace("_pairwise_ci.csv", "")
            cohort = (
                "cleveland_uci"
                if stem.startswith("cleveland")
                else ("four_site_uci" if stem.startswith("four_site") else "cardio70k")
            )
            model = stem.replace(cohort + "_", "")
            key = list(zip(d.metric, d.quantity))
            dd = d[[k not in REDUNDANT for k in key]]
            for attr, g in [("all", dd)] + list(dd.groupby("attribute")):
                rows.append(
                    {
                        "run": short(run),
                        "cohort": cohort,
                        "model": model,
                        "split": split,
                        "attr": attr,
                        "rows_raw": len(d) if attr == "all" else np.nan,
                        "sig_raw": int(d.significant.sum()) if attr == "all" else np.nan,
                        "distinct": len(g),
                        "tested": int(g.tested.sum()),
                        "untested": int((~g.tested).sum()),
                        "sig": int(g.significant.sum()),
                    }
                )
            if split == "test":
                s = dd[dd.significant]
                for _, r in s.iterrows():
                    sig_rows.append(
                        {
                            "run": short(run),
                            "cohort": cohort,
                            "model": model,
                            "attr": r.attribute,
                            "quantity": r.quantity,
                            "a": r.group_a,
                            "b": r.group_b,
                            "diff": r.difference,
                            "lo": r.ci_low,
                            "hi": r.ci_high,
                            "p_bh": r.p_value_bh,
                            "n_a": r.n_group_a,
                            "n_b": r.n_group_b,
                        }
                    )
            untested = d[~d.tested]
            if len(untested) and split == "test":
                reasons = untested.untested_reason.value_counts().to_dict()
                rows[-1]["untested_reasons"] = str(reasons)
    table(pd.DataFrame(rows), ".0f")
    emit("### Significant distinct comparisons, test split")
    emit()
    table(pd.DataFrame(sig_rows), ".3f")


def section_gaps() -> None:
    emit("## L6. Descriptive parity gaps (max-min), test split -- NOT tests")
    emit()
    rows = []
    for run in (RUN_CARDIAC, RUN_C70):
        pf = CARD / "runs" / run / "baseline" / "prediction_fairness"
        for f in sorted(pf.glob("*_ci.csv")):
            if "pairwise" in f.name or "_cv_" in f.name:
                continue
            d = pd.read_csv(f)
            d = d[d.group.isna()]
            for _, r in d.iterrows():
                rows.append(
                    {
                        "file": f.name.replace("_ci.csv", ""),
                        "attr": r.attribute,
                        "metric": r.metric,
                        "quantity": r.quantity,
                        "point": r.point,
                        "lo": r.ci_low,
                        "hi": r.ci_high,
                        "descriptive_only": r.descriptive_only,
                    }
                )
    table(pd.DataFrame(rows))


def section_clusters() -> None:
    emit("## L7. Discovered subgroups (clusters fit on the train split, stage 7)")
    emit()
    for run, cohorts in CARDIAC_COHORTS:
        for c in cohorts:
            base = CARD / "runs" / run / "grouping_pretrain" / c
            emit(f"### {c}  (`{rel(base)}`)")
            emit()
            a = pd.read_csv(base / "cluster_assignments.csv", index_col=0)
            first = a.iloc[0]
            emit(
                f"selected: {first.method_used}, k={int(first.n_clusters)}, "
                f"silhouette {first.silhouette:.3f}; train rows {len(a)}; "
                f"sizes {a.group_cluster.value_counts().sort_index().to_dict()}"
            )
            emit()
            # The profile table's header row lists the clustering features.
            prof = (base / "subgroup_profiles.md").read_text().splitlines()
            head = next((ln for ln in prof if ln.startswith("|") and "group_cluster" in ln), "")
            feats = [x.strip() for x in head.strip("|").split("|")][1:]
            emit(f"clustering features ({len(feats)}): {', '.join(feats)}")
            emit()
            diag = pd.read_csv(base / "cluster_diagnostics.csv")
            table(diag.dropna(axis=1, how="all"), ".4f")
            emit(f"Profiles: `{rel(base / 'subgroup_profiles.md')}`")
            emit()
            # In-sample table from the grouping study: listed for traceability, not quoted.
            f = study_dir("grouping", run) / c / "fairness_by_cluster.csv"
            emit(f"In-sample cluster fairness (DO NOT QUOTE): `{rel(f)}`")
            emit()
            table(pd.read_csv(f), ".3f")


def section_binning() -> None:
    emit("## L8. Binning sensitivity (stage 9, `experiments/attribute_binning/comparison.csv`)")
    emit()
    for run in (RUN_CARDIAC, RUN_C70):
        f = CARD / "runs" / run / "experiments" / "attribute_binning" / "comparison.csv"
        d = pd.read_csv(f)
        emit(f"Source: `{rel(f)}`")
        emit()
        table(d)
        summ = d.groupby("dataset").agg(
            schemes=("strategy", "nunique"),
            sp_min=("max_sp_difference", "min"),
            sp_max=("max_sp_difference", "max"),
            groups_min=("n_groups", "min"),
            groups_max=("n_groups", "max"),
            smallest_group=("min_group_size", "min"),
        )
        table(summ.reset_index())


def section_binning_predictions() -> None:
    emit(
        "## L8b. Binning sensitivity on predictions (pooled CV predictions, all 26 schemes, per model)"
    )
    emit()
    emit(
        "Source: `baseline/age_binning_sensitivity/<cohort>/<model>/age_binning_sensitivity.csv`, "
        "regime = baseline. `label_sp` is the stage-9 label-level gap for the same scheme."
    )
    emit()
    rows = []
    for run, cohorts in (
        (RUN_CARDIAC, ["cleveland_uci", "four_site_uci"]),
        (RUN_C70, ["cardio70k"]),
    ):
        lab = pd.read_csv(
            CARD / "runs" / run / "experiments" / "attribute_binning" / "comparison.csv"
        )
        for c in cohorts:
            label = lab[lab.dataset == c].set_index("strategy").max_sp_difference
            for f in sorted(
                (CARD / "runs" / run / "baseline" / "age_binning_sensitivity" / c).glob(
                    "*/age_binning_sensitivity.csv"
                )
            ):
                d = pd.read_csv(f)
                d = d[d.regime == "baseline"]
                g = d.groupby("strategy").agg(
                    dp=("dp_gap", "first"),
                    eo=("eo_tpr_gap", "first"),
                    nmin=("n", "min"),
                    n=("n", "sum"),
                )
                g["label_sp"] = label
                ok = g[g.nmin >= 30]
                rows.append(
                    {
                        "cohort": c,
                        "model": f.parent.name,
                        "N": int(g.n.iloc[0]),
                        "dp_min": g.dp.min(),
                        "dp_max": g.dp.max(),
                        "schemes_nmin30": len(ok),
                        "dp_min_nmin30": ok.dp.min(),
                        "dp_max_nmin30": ok.dp.max(),
                        "dp_clinical": g.dp.get("clinical"),
                        "r_label_pred": g.dp.corr(g.label_sp),
                        "eo_tpr_min": g.eo.min(),
                        "eo_tpr_max": g.eo.max(),
                    }
                )
    table(pd.DataFrame(rows))


def section_mitigation() -> None:
    emit("## L9. Single-technique mitigation, stage 10 (each arm carries its own pairwise table)")
    emit()
    for run in (RUN_CARDIAC, RUN_C70):
        base = CARD / "runs" / run / "experiments" / "mitigation"
        arms = pd.read_csv(base / "prediction_fairness" / "arms_summary.csv")
        perf = pd.read_csv(base / "summary.csv")
        emit(
            f"Sources: `{rel(base / 'prediction_fairness' / 'arms_summary.csv')}`, `{rel(base / 'summary.csv')}`"
        )
        emit()
        keep = ["dataset", "model_type", "technique", "constraint_attr", "stratify_used", "n_rows"]
        keep += ["n_pairwise_comparisons", "n_pairwise_tested", "n_significant_differences"]
        a = arms[keep].copy()
        perf = perf.rename(columns={"constraint_attr": "constraint_attr"})
        perf["constraint_attr"] = perf["constraint_attr"].fillna("none")
        pcols = ["dataset", "model_type", "technique", "constraint_attr", "f1_score", "auc_roc"]
        pcols += ["dp_sex_max_diff", "dp_age_group_max_diff", "eq_odds_max_diff"]
        m = a.merge(
            perf[pcols].drop_duplicates(["dataset", "model_type", "technique", "constraint_attr"]),
            on=["dataset", "model_type", "technique", "constraint_attr"],
            how="left",
        )
        table(m)


def _nondominated(df: pd.DataFrame, perf: str, gap: str) -> pd.Series:
    """True where no other row has perf >= and gap <= with one strict."""
    p = df[perf].to_numpy()
    g = df[gap].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        dom = (p >= p[i]) & (g <= g[i]) & ((p > p[i]) | (g < g[i]))
        keep[i] = not dom.any()
    return pd.Series(keep, index=df.index)


def section_sweep() -> None:
    emit("## L10. Combinatorial sweep, stage 11 -- point estimates only, no intervals")
    emit()
    emit(
        "Non-dominated sets computed here per Ch5 sec:method:evaluation:tradeoffs: one fairness "
        "criterion at a time, F1 as the predictive axis, single_split and kfold_cv kept apart "
        "(different evaluation protocols). The pareto_*.csv files shipped by stage 12 are NOT "
        "used: they rank against the aggregate fairness_gap, which Ch5 rules out."
    )
    emit()
    crit = [
        "dem_parity_age_group_max_diff",
        "dem_parity_sex_max_diff",
        "eq_odds_age_group_tpr_diff",
        "eq_odds_sex_tpr_diff",
    ]
    for run, cohorts in (
        (RUN_CARDIAC, ["cleveland_uci", "four_site_uci"]),
        (RUN_C70, ["cardio70k"]),
    ):
        for c in cohorts:
            f = CARD / "runs" / run / "experiments" / "comparisons" / "data" / f"tradeoff_{c}.csv"
            t = pd.read_csv(f)
            t = t[t.status == "success"]
            emit(f"### {c}: {len(t)} cells  (`{rel(f)}`)")
            emit()
            emit(f"cells by model: {t.model_type.value_counts().to_dict()}")
            emit()
            base = t[t.mitigation_technique == "baseline"]
            emit(
                f"baseline cells: {len(base)}; F1 range {base.f1_value.min():.3f}-{base.f1_value.max():.3f}; "
                f"all-cell F1 range {t.f1_value.min():.3f}-{t.f1_value.max():.3f}"
            )
            emit()
            for tm in ("single_split", "kfold_cv"):
                sub = t[t.training_method == tm]
                for cr in crit:
                    s = sub.dropna(subset=[cr, "f1_value"])
                    nd = s[_nondominated(s, "f1_value", cr)].sort_values(
                        "f1_value", ascending=False
                    )
                    cols = [
                        "binning_strategy",
                        "mitigation_technique",
                        "model_type",
                        "model_variant",
                        "f1_value",
                        cr,
                    ]
                    if tm == "kfold_cv":
                        cols.append("f1_score_std")
                    emit(f"**{tm} / {cr}**: {len(nd)} non-dominated of {len(s)}")
                    emit()
                    table(nd[cols])

    # svm on cardio70k: completed cells that the comparison excludes
    res = CARD / "runs" / RUN_C70 / "experiments" / "results" / "cardio70k"
    counts: dict = {}
    for sp in ("holdout", "cv"):
        for f in res.glob(f"{sp}/*.json"):
            cf = json.load(open(f))["configuration"]
            if cf.get("model_type") == "svm":
                k = (sp, cf.get("mitigation_technique"))
                counts[k] = counts.get(k, 0) + 1
    emit("### svm on cardio70k (excluded from the comparison)")
    emit()
    emit(f"Completed cells by (split, technique): {counts}; total {sum(counts.values())}.")
    emit("Planned per family on cardiac: 260 (from tradeoff model counts), so 54/260 completed.")
    emit()


def section_ablation() -> None:
    emit("## L11. Sensitive-attribute ablation (feature_selection study, point estimates)")
    emit()
    rows = []
    for run in (RUN_CARDIAC, RUN_C70_SUB, RUN_C70):
        study = study_dir("feature_selection", run)
        for r in sorted((study / "runs").glob("fs_*__*")):
            mode, model = r.name[3:].split("__")
            pred = r / "baseline" / "results" / "predictions"
            for f in sorted(pred.glob("*_test.csv")):
                cohort = f.name.replace(f"_{model}_test.csv", "")
                d = pd.read_csv(f)
                row = {
                    "run": short(run),
                    "cohort": cohort,
                    "mode": mode,
                    "model": model,
                    "n": len(d),
                    "f1": f1_score(d.y_true, d.y_pred),
                    "auc": roc_auc_score(d.y_true, d.y_proba),
                }
                for a in ("sex", "age_group"):
                    if a in d:
                        rates = d.groupby(a).y_pred.mean()
                        row[f"dp_{a}"] = rates.max() - rates.min()
                rows.append(row)
    table(pd.DataFrame(rows))


def section_shap() -> None:
    emit("## L12. Attribution disparity (subgroup SHAP, holdout)")
    emit()
    rows, glob_rows = [], []
    for run in (RUN_CARDIAC, RUN_C70):
        xai = CARD / "runs" / run / "baseline" / "results" / "xai"
        for m in sorted(xai.iterdir()):
            sh = m / "holdout" / "shap"
            if not (sh / "summary.csv").exists():
                rows.append({"model": m.name, "note": "no SHAP (excluded)"})
                continue
            s = pd.read_csv(sh / "summary.csv")
            s["share"] = s.mean_abs_shap / s.mean_abs_shap.sum()
            glob_rows.append(
                {
                    "model": m.name,
                    "top1": s.feature.iloc[0],
                    "top1_share": s.share.iloc[0],
                    "top3": ", ".join(s.feature.iloc[:3]),
                    "n_features": len(s),
                }
            )
            disp = pd.read_csv(sh / "subgroup_disparity.csv")
            agr = pd.read_csv(sh / "subgroup_agreement.csv")
            for attr, g in disp.groupby("attribute"):
                top = g.sort_values("share_gap", ascending=False).iloc[0]
                ag = agr[agr.attribute == attr]
                rows.append(
                    {
                        "model": m.name,
                        "attr": attr,
                        "groups": int(top.n_groups),
                        "max_share_gap": top.share_gap,
                        "feature": top.feature,
                        "max_group": top.max_group,
                        "min_group": top.min_group,
                        "max_mag_gap": g.mean_abs_shap_gap.max(),
                        "min_spearman": ag.spearman_vs_overall.min(),
                        "min_top5": ag.top5_overlap_vs_overall.min(),
                        "worst_group": (
                            ag.sort_values("spearman_vs_overall").group.iloc[0] if len(ag) else None
                        ),
                        "worst_n": (
                            ag.sort_values("spearman_vs_overall").n.iloc[0] if len(ag) else None
                        ),
                    }
                )
    emit("Cohort-wide ranking:")
    emit()
    table(pd.DataFrame(glob_rows))
    emit("Per-attribute disparity and rank agreement:")
    emit()
    table(pd.DataFrame(rows))


def split_of(cohort: str) -> str:
    """The prediction split the chapter quotes: held-out for Cardio70k, pooled CV otherwise."""
    return "test" if cohort == "cardio70k" else "cv"


def models_of(cohort: str) -> list[str]:
    return C70_MODELS if cohort == "cardio70k" else UCI_MODELS


def section_derm_groups() -> None:
    emit("## L4b. Dermatology recall and AUC by Fitzpatrick group (point estimates, no intervals)")
    emit()
    rows = []
    for arm, run in (("augmented", RUN_DERM_AUG), ("cached_no_aug", RUN_DERM_CACHED)):
        path = DERM / "runs" / run / "baseline" / "prediction_fairness" / "fairness_groups.csv"
        emit(f"Source ({arm}): `{rel(path)}`")
        g = pd.read_csv(path)
        g = g[g.sensitive_attribute == "fitzpatrick_group"]
        for key, d in g.groupby("run_key"):
            d = d.set_index("group")
            light, mid = d.loc["I-II"], d.loc["III-IV"]
            rows.append(
                {
                    "arm": arm,
                    "backbone": key.removeprefix("pad_ufes_20_"),
                    "recall_I-II": light.recall,
                    "recall_III-IV": mid.recall,
                    "recall_diff": mid.recall - light.recall,
                    "auc_I-II": light.auc,
                    "auc_III-IV": mid.auc,
                }
            )
    emit()
    groups = g.drop_duplicates("group")[["group", "n", "prevalence", "degenerate"]]
    emit("Groups (last arm; identical test split in both):")
    emit()
    table(groups)
    table(pd.DataFrame(rows))


def section_four_site_bands() -> None:
    emit("## L6b. Four-site outcome rate by site and age band (raw data, descriptive only)")
    emit()
    emit(f"Source: `{rel(FOUR_SITE_DATA)}`, the `age_group` bands as stored (right-closed).")
    emit()
    d = pd.read_csv(FOUR_SITE_DATA, usecols=["source_site", "age_group", "heart_disease"])
    bands = sorted(d.age_group.unique(), key=lambda b: (not b.startswith("<"), b))
    site = d.groupby("source_site").heart_disease.agg(n="size", prevalence="mean")
    table(site.sort_values("prevalence").reset_index())
    band = d.groupby("age_group").heart_disease.agg(n="size", rate="mean")
    share = pd.crosstab(d.age_group, d.source_site, normalize="index").add_prefix("share_")
    emit("Pooled rate per band, and each site's share of the band:")
    emit()
    table(band.join(share).loc[bands].reset_index())
    rate = d.pivot_table(index="source_site", columns="age_group", values="heart_disease")
    size = d.pivot_table(
        index="source_site", columns="age_group", values="heart_disease", aggfunc="size"
    )
    cells = pd.DataFrame(index=rate.index)
    for b in bands:
        cells[b] = [f"{r:.3f} ({int(n)})" if pd.notna(n) else "-" for r, n in zip(rate[b], size[b])]
    young, old = bands[0], "60-69"
    if old in rate:
        cells[f"{old} minus {young}"] = (rate[old] - rate[young]).round(3)
    emit("Within-site rate (n) per band:")
    emit()
    emit(cells.reset_index().to_markdown(index=False))
    emit()


def cluster_gaps(attribute: str) -> pd.DataFrame:
    """Within-cluster parity and outcome-rate gaps over one attribute, groups of n >= 30."""
    rows = []
    for run, cohorts in CARDIAC_COHORTS:
        pred = CARD / "runs" / run / "baseline" / "results" / "predictions"
        for cohort in cohorts:
            for model in models_of(cohort):
                path = pred / f"{cohort}_{model}_{split_of(cohort)}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path, usecols=["y_true", "y_pred", "group_cluster", attribute])
                for cl, c in df.groupby("group_cluster"):
                    s = c.groupby(attribute).agg(
                        n=("y_true", "size"), pos=("y_pred", "mean"), out=("y_true", "mean")
                    )
                    s = s[s.n >= 30]
                    if len(s) < 2:
                        continue
                    rows.append(
                        {
                            "cohort": cohort,
                            "split": split_of(cohort),
                            "cluster": cl,
                            "n": len(c),
                            "groups": len(s),
                            "outcome_rate": c.y_true.mean(),
                            "outcome_gap": s.out.max() - s.out.min(),
                            "model": model,
                            "parity_gap": s.pos.max() - s.pos.min(),
                        }
                    )
    return pd.DataFrame(rows)


def section_cluster_age() -> None:
    emit("## L7b. Age and sex gaps within discovered clusters (saved predictions, groups n >= 30)")
    emit()
    emit(
        "Point estimates, max - min over the groups of one cluster. No interval: a bootstrap "
        "interval on a max-min gap excludes zero on null data (see L1); only pairwise "
        "differences are testable, and the pairwise layer does not cross cluster with age or sex."
    )
    emit()
    keys = ["cohort", "split", "cluster", "n", "groups", "outcome_rate", "outcome_gap"]
    for attribute in ("age_group", "sex"):
        emit(f"By {attribute}:")
        emit()
        df = cluster_gaps(attribute)
        table(df.pivot_table(index=keys, columns="model", values="parity_gap").reset_index())


# Techniques that act on the group, in table order; the two within-group resamplers pool.
ACTING = [
    "exponentiated_gradient",
    "threshold_optimizer",
    "grid_search",
    "reweighting",
    "group_aware_resampling",
]
GROUP_AWARE = {
    "smote_group": "group_aware_resampling",
    "uniform_sampling": "group_aware_resampling",
}


def section_mitigation_effects() -> None:
    emit("## L9b. Mitigation: change in the targeted parity gap and in F1, mean over families")
    emit()
    emit(
        "Targeted gap = demographic_parity max_difference on the constraint attribute; F1 from "
        "scope performance. Group-aware resampling pools smote_group and uniform_sampling (mean "
        "and counts over both). sig_down / sig_up: arms whose gap change is significant (BH "
        "within the arm), by direction. UCI cohorts: pooled CV; Cardio70k: held-out split."
    )
    emit()
    rows, sig = [], []
    for run, cohorts in CARDIAC_COHORTS:
        base = CARD / "runs" / run / "experiments" / "mitigation"
        for cohort in cohorts:
            path = base / (
                "paired_effects.csv" if split_of(cohort) == "test" else "paired_effects_cv.csv"
            )
            emit(f"Source ({cohort}): `{rel(path)}`")
            pe = pd.read_csv(path)
            pe = pe[pe.dataset == cohort].copy()
            pe["significant"] = pe.significant.fillna(False).astype(bool)
            pe["group"] = pe.technique.replace(GROUP_AWARE)
            gap = pe[
                (pe.scope == "group_fairness")
                & (pe.metric == "demographic_parity")
                & (pe.quantity == "max_difference")
                & (pe.attribute == pe.constraint_attr)
            ].copy()
            gap["down"] = gap.significant & (gap.difference < 0)
            gap["up"] = gap.significant & (gap.difference > 0)
            f1 = pe[(pe.scope == "performance") & (pe.quantity == "f1")]
            keys = ["constraint_attr", "group"]
            agg = gap.groupby(keys).agg(
                baseline_gap=("baseline", "mean"),
                d_gap=("difference", "mean"),
                arms=("difference", "size"),
                sig_down=("down", "sum"),
                sig_up=("up", "sum"),
            )
            agg = agg.join(f1.groupby(keys).difference.mean().rename("d_f1")).reset_index()
            agg.insert(0, "cohort", cohort)
            rows.append(agg)
            arm_keys = ["dataset", "constraint_attr", "technique", "model_type"]
            hit = gap[gap.significant][arm_keys + ["baseline", "arm", "difference"]]
            f1_ci = f1[arm_keys].assign(
                d_f1=[
                    ci(d, lo, hi, s)
                    for d, lo, hi, s in zip(f1.difference, f1.ci_low, f1.ci_high, f1.significant)
                ]
            )
            sig.append(hit.merge(f1_ci, on=arm_keys, how="left"))
    emit()
    df = pd.concat(rows)
    order = {t: i for i, t in enumerate(ACTING)}
    df = df.sort_values(
        ["cohort", "constraint_attr", "group"],
        key=lambda s: s.map(order).fillna(99) if s.name == "group" else s,
    )
    table(df.rename(columns={"group": "technique"}))
    emit("Every arm with a significant targeted-gap change, and its F1 change [95% interval]:")
    emit()
    table(pd.concat(sig))


def section_coef_shap() -> None:
    emit("## L12b. Logistic regression |coefficient| vs mean |SHAP| (holdout), and top-two shares")
    emit()
    rows, tops = [], []
    for run, cohorts in CARDIAC_COHORTS:
        res = CARD / "runs" / run / "baseline" / "results"
        for cohort in cohorts:
            imp = pd.read_csv(res / "predictions" / f"{cohort}_logistic_regression_importance.csv")
            lr = res / "xai" / f"{cohort}__logistic_regression" / "holdout" / "shap" / "summary.csv"
            m = imp.merge(pd.read_csv(lr), on="feature")
            rows.append(
                {
                    "cohort": cohort,
                    "features": len(m),
                    "spearman": spearmanr(m.abs_coefficient, m.mean_abs_shap).statistic,
                    "top_coef": imp.sort_values("abs_coefficient").feature.iloc[-1],
                }
            )
            for model in models_of(cohort):
                p = res / "xai" / f"{cohort}__{model}" / "holdout" / "shap" / "summary.csv"
                if not p.exists():
                    continue
                s = pd.read_csv(p).sort_values("mean_abs_shap", ascending=False)
                share = (s.mean_abs_shap / s.mean_abs_shap.sum()).to_numpy()
                tops.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "top1": s.feature.iloc[0],
                        "share1": share[0],
                        "top2": s.feature.iloc[1],
                        "share2": share[1],
                        "top4": ", ".join(sorted(s.feature.iloc[:4])),
                    }
                )
    table(pd.DataFrame(rows))
    table(pd.DataFrame(tops))


def _arm_parts(name: str, cohort: str) -> tuple[str, str, str]:
    """``<cohort>_<family>_<technique>_<constraint>`` -> (family, technique, constraint)."""
    rest = name.removeprefix(cohort + "_")
    family = next(f for f in UCI_MODELS if rest.startswith(f + "_"))
    rest = rest.removeprefix(family + "_")
    constraint = next(c for c in ("group_cluster", "age_group", "sex", "none") if rest.endswith(c))
    return family, rest.removesuffix("_" + constraint), constraint


def section_shap_mitigation() -> None:
    emit("## L12c. Subgroup SHAP before and after mitigation (stage 10, held-out rows)")
    emit()
    emit(
        "baseline_none = the unmitigated model on the same held-out rows; compare mitigated arms "
        "with it, not with L12 (training rows). gap = max share_gap over features; rho = min "
        "Spearman of a group's ranking vs the overall one."
    )
    emit()
    arms = []
    for run, cohorts in CARDIAC_COHORTS:
        root = CARD / "runs" / run / "experiments" / "mitigation" / "subgroup_shap"
        emit(f"Source: `{rel(root)}`")
        for cohort in cohorts:
            found = [
                d
                for d in sorted(root.glob(f"{cohort}_*"))
                if (d / "subgroup_disparity.csv").exists()
            ]
            if not found:
                emit(
                    f"{cohort}: no subgroup SHAP (no attribute keeps two groups of 30 held-out rows)."
                )
            for d in found:
                family, technique, constraint = _arm_parts(d.name, cohort)
                row = {
                    "cohort": cohort,
                    "model": family,
                    "technique": technique,
                    "constraint": constraint,
                }
                for attr, g in pd.read_csv(d / "subgroup_disparity.csv").groupby("attribute"):
                    row[f"{attr}_gap"] = g.share_gap.max()
                for attr, g in pd.read_csv(d / "subgroup_agreement.csv").groupby("attribute"):
                    row[f"{attr}_rho"] = g.spearman_vs_overall.min()
                arms.append(row)
    emit()
    df = pd.DataFrame(arms)
    df["arm"] = df.model + "/" + df.technique + "/" + df.constraint
    summary = []
    for (cohort, is_base), g in df.groupby(["cohort", df.technique == "baseline"]):
        for attr in ("age_group", "sex", "group_cluster"):
            if f"{attr}_gap" not in g or g[f"{attr}_gap"].isna().all():
                continue
            hi = g.loc[g[f"{attr}_gap"].idxmax()]
            lo = g.loc[g[f"{attr}_rho"].idxmin()]
            summary.append(
                {
                    "cohort": cohort,
                    "arms": "baseline_none" if is_base else f"mitigated ({len(g)})",
                    "attribute": attr,
                    "max_gap": hi[f"{attr}_gap"],
                    "max_gap_arm": hi.arm,
                    "min_rho": lo[f"{attr}_rho"],
                    "min_rho_arm": lo.arm,
                }
            )
    table(pd.DataFrame(summary))
    emit("Per arm:")
    emit()
    table(df.drop(columns="arm"))


def ci(point: float, low: float, high: float, star: bool = False) -> str:
    """``+0.024 [-0.007, +0.052]``, starred when significant after BH."""
    return f"{point:+.3f} [{low:+.3f}, {high:+.3f}]{'*' if star else ''}"


def _intervals(name: str) -> pd.DataFrame | None:
    path = intervals_dir / f"{name}.csv"
    if not path.exists():
        emit(f"MISSING: `{rel(path)}`; run `scripts/thesis/performance_intervals.py`.")
        emit()
        return None
    emit(f"Source: `{rel(path)}`")
    emit()
    return pd.read_csv(path)


def section_baseline_intervals() -> None:
    emit("## L4c. Baseline performance with 95% percentile intervals (held-out test split)")
    emit()
    df = _intervals("baseline_performance")
    if df is None:
        return
    df["cell"] = [
        f"{p:.3f} [{lo:.3f}, {hi:.3f}]" for p, lo, hi in zip(df.point, df.ci_low, df.ci_high)
    ]
    keys = ["run", "cohort", "model", "n_boot", "stratify"]
    wide = df.pivot_table(index=keys, columns="quantity", values="cell", aggfunc="first")
    wide = wide.reset_index()
    wide["run"] = wide.run.map(short)
    table(wide[keys + ["f1", "auc", "accuracy", "precision", "recall"]])


def section_ablation_intervals() -> None:
    emit("## L11b. Ablation: paired change against exclude_sensitive (shared test rows)")
    emit()
    emit(
        "Difference [95% interval], * = significant after BH within the comparison. Separate "
        "fits scored on the same rows: the interval covers test-sample noise, not refit noise. "
        "dp_* = demographic_parity max_difference (descriptive gap, see L7b)."
    )
    emit()
    df = _intervals("ablation_paired")
    if df is None:
        return
    keys = ["run", "cohort", "model", "mode"]
    perf = df[df.scope == "performance"].assign(col="d_" + df.quantity)
    dp = df[(df.metric == "demographic_parity") & (df.quantity == "max_difference")]
    dp = dp.assign(col="d_dp_" + dp.attribute.astype(str))
    both = pd.concat([perf, dp])
    both["cell"] = [
        ci(d, lo, hi, s)
        for d, lo, hi, s in zip(both.difference, both.ci_low, both.ci_high, both.significant)
    ]
    wide = both.pivot_table(index=keys, columns="col", values="cell", aggfunc="first")
    wide = wide.reset_index()
    wide["run"] = wide.run.map(short)
    cols = [c for c in ("d_f1", "d_auc", "d_dp_age_group", "d_dp_sex") if c in wide]
    table(wide[keys + cols])


def section_synthetic() -> None:
    emit("## L14. Synthetic profiling-sensitivity study (Appendix E)")
    emit()
    if not SYNTHETIC.is_dir():
        emit(f"MISSING: `{rel(SYNTHETIC)}` (runs.yaml `studies.synthetic`).")
        emit()
        return
    emit(f"Source: `{rel(SYNTHETIC)}`")
    emit()
    for key, value in json.loads((SYNTHETIC / "study_summary.json").read_text()).items():
        emit(f"- {key}: {value}")
    emit()
    k = pd.read_csv(SYNTHETIC / "knob_response_summary.csv")
    k = k[k.knob != "base"]
    # Manipulation checks, not profiling responses: they record what the knob did.
    checks = {"semantic_type_accuracy", "top_missing_pct", "class_balance_delta"}
    checks.add("duplicate_pct_observed")
    r = k[~k.metric.isin(checks)]
    excl = (r.delta_ci95_low > 0) | (r.delta_ci95_high < 0)
    emit(
        f"Paired-delta intervals on the profiling responses: {len(r)} ({r.condition.nunique()} "
        f"conditions x {r.metric.nunique()} responses x {r.tier.nunique()} tiers); excluding "
        f"zero: {int(excl.sum())}. Manipulation checks left out: {', '.join(sorted(checks))}."
    )
    emit()
    emit("Difficulty response (ebmDifficulty, paired delta from the same seed's baseline):")
    emit()
    cols = ["tier", "knob", "condition", "knob_value", "n_replicates", "mean"]
    cols += ["delta_mean", "delta_ci95_low", "delta_ci95_high"]
    table(k[k.metric == "ebmDifficulty"][cols])
    emit("Semantic-type confusion (`type_confusion.csv`):")
    emit()
    table(pd.read_csv(SYNTHETIC / "type_confusion.csv"))


def section_usability() -> None:
    emit("## L13. Usability study (n = 5; raw record, WebApp repo)")
    emit()
    if not (USAB / "task_results.csv").exists() or not (USAB / "sus_results.csv").exists():
        emit(f"MISSING: no usability results under `{USAB}` (sibling WebApp repo).")
        emit()
        return
    t = pd.read_csv(USAB / "task_results.csv")
    s = pd.read_csv(USAB / "sus_results.csv")
    t["participant"] = t.participant.str.replace("participant_", "P")
    s["participant"] = s.participant.str.replace("participant_", "P")

    def secs(v: str) -> int:
        v = v.strip()
        mins = int(v.split("min")[0]) if "min" in v else 0
        rest = v.split("min")[1] if "min" in v else v
        sec = int(rest.rstrip("s")) if rest.rstrip("s") else 0
        return mins * 60 + sec

    t["seconds"] = t.time_on_task.map(secs)
    per = t.pivot(index="task", columns="participant", values="success")
    emit("Outcome per task and participant (raw values):")
    emit()
    emit(per.reset_index().to_markdown(index=False))
    emit()
    agg = t.groupby("task").agg(
        unaided=("success", lambda x: int((x == "yes").sum())),
        with_help=("success", lambda x: int((x == "yes-with-help").sum())),
        failed=("success", lambda x: int((x == "no").sum())),
        seq_mean=("seq", "mean"),
        seq_min=("seq", "min"),
        time_median_s=("seconds", "median"),
    )
    table(agg.reset_index(), ".1f")
    emit(f"SUS per participant: {dict(zip(s.participant, s.sus_score))}")
    emit(
        f"SUS mean {s.sus_score.mean():.1f}, sd {s.sus_score.std(ddof=1):.1f}, "
        f"min {s.sus_score.min():.1f}, max {s.sus_score.max():.1f}"
    )
    emit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "chapter6_ledger.md")
    parser.add_argument(
        "--intervals", type=Path, default=DEFAULT_OUT, help="performance_intervals.py output"
    )
    args = parser.parse_args()
    global intervals_dir
    intervals_dir = args.intervals

    emit("# Chapter 6 numbers ledger")
    emit()
    emit("Generated by FairXAI `scripts/thesis/build_ledger.py` from the runs in `runs.yaml`.")
    emit("Do not edit by hand; re-run the script.")
    emit("Every number quoted in Chapters 6 and 7 should appear in a row below.")
    emit()
    runs = (RUN_CARDIAC, RUN_C70_SUB, RUN_C70, RUN_DERM_AUG, RUN_DERM_CACHED)
    emit("Runs: " + ", ".join(f"`{r}`" for r in runs))
    emit()
    emit(f"Synthetic study: `{rel(SYNTHETIC)}`")
    emit()
    for fn in (
        section_calibration,
        section_resampling,
        section_triage,
        section_baseline,
        section_baseline_intervals,
        section_derm_groups,
        section_pairwise,
        section_gaps,
        section_four_site_bands,
        section_clusters,
        section_cluster_age,
        section_binning,
        section_binning_predictions,
        section_mitigation,
        section_mitigation_effects,
        section_sweep,
        section_ablation,
        section_ablation_intervals,
        section_shap,
        section_coef_shap,
        section_shap_mitigation,
        section_usability,
        section_synthetic,
    ):
        try:
            fn()
        except Exception as exc:  # keep going; a missing source is itself a ledger entry
            emit(f"**{fn.__name__} FAILED:** `{type(exc).__name__}: {exc}`")
            emit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
