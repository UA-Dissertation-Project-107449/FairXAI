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
from sklearn.metrics import f1_score, roc_auc_score
from thesis_runs import (
    CALIBRATION,
    CARD,
    CARDIAC_COHORTS,
    DEFAULT_OUT,
    DERM,
    REDUNDANT,
    RUN_C70,
    RUN_C70_SUB,
    RUN_CARDIAC,
    RUN_DERM_AUG,
    RUN_DERM_CACHED,
    USAB,
    rel,
    study_dir,
)

lines: list[str] = []


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
    args = parser.parse_args()

    emit("# Chapter 6 numbers ledger")
    emit()
    emit("Generated by FairXAI `scripts/thesis/build_ledger.py` from the runs in `runs.yaml`.")
    emit("Do not edit by hand; re-run the script.")
    emit("Every number quoted in Chapters 6 and 7 should appear in a row below.")
    emit()
    runs = (RUN_CARDIAC, RUN_C70_SUB, RUN_C70, RUN_DERM_AUG, RUN_DERM_CACHED)
    emit("Runs: " + ", ".join(f"`{r}`" for r in runs))
    emit()
    for fn in (
        section_calibration,
        section_resampling,
        section_triage,
        section_baseline,
        section_pairwise,
        section_gaps,
        section_clusters,
        section_binning,
        section_binning_predictions,
        section_mitigation,
        section_sweep,
        section_ablation,
        section_shap,
        section_usability,
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
