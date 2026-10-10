"""Build the dissertation's Chapter 6 figures from the runs named in runs.yaml.

Writes a vector PDF and a 300-dpi PNG per figure (the Grad-CAM panel embeds the
archived PNGs). Every figure reads the same sources as build_ledger.py, so a
number visible in a figure is a number in the ledger.

    python3 scripts/thesis/make_figures.py [--out output/thesis/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from build_ledger import _nondominated, cluster_gaps  # noqa: E402
from matplotlib.image import imread  # noqa: E402
from thesis_runs import (  # noqa: E402
    C70_MODELS,
    CARD,
    DEFAULT_OUT,
    DERM,
    REDUNDANT,
    RUN_C70,
    RUN_CARDIAC,
    RUN_DERM_AUG,
    RUN_DERM_CACHED,
    UCI_MODELS,
    split_of,
)

OUT = DEFAULT_OUT / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["TeX Gyre Pagella", "P052", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)
WIDTH = 6.0  # inches; included at \textwidth

FAMILY = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "svm": "SVM",
    "xgboost": "XGB",
}
COHORT = {"cleveland_uci": "Cleveland", "four_site_uci": "Four-site", "cardio70k": "Cardio70k"}
FAM_COLOR = {"LR": "#1b6ca8", "RF": "#3a9d5d", "SVM": "#c0762c", "XGB": "#8e4585"}
AGE_RANK = {"<40": 0, "40-49": 1, "50-59": 2, "60-69": 3, "70+": 4}
COHORTS = [(RUN_CARDIAC, "cleveland_uci"), (RUN_CARDIAC, "four_site_uci"), (RUN_C70, "cardio70k")]
# The age-binning catalogue by scheme family, in plotting order.
SCHEME_FAMILIES = {
    "fixed": ["fixed_2", "fixed_3", "fixed_5yr", "fixed_10yr"],
    "clinical": ["clinical"],
    "quantile": [f"quantile_{k}" for k in range(3, 11)]
    + [f"adaptive_quantile_{k}" for k in (5, 8, 10)],
    "equal width": [f"equal_width_{k}" for k in (3, 4, 5, 6, 8, 10)],
    "Jenks": [f"jenks_{k}" for k in (3, 4, 5)],
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


def _pairwise(run: str, cohort: str, model: str, cv: bool = False) -> pd.DataFrame:
    tag = "_cv" if cv else ""
    pf = CARD / "runs" / run / "baseline" / "prediction_fairness"
    d = pd.read_csv(pf / f"{cohort}_{model}{tag}_pairwise_ci.csv")
    return d[[(m, q) not in REDUNDANT for m, q in zip(d.metric, d.quantity)]]


# ---------------------------------------------------------------------------
def fig_evidence() -> None:
    """CV bars (untestable / tested / significant) with the held-out significant count marked."""
    specs = [
        (RUN_CARDIAC, "cleveland_uci", UCI_MODELS, "Cleveland (n=297)"),
        (RUN_CARDIAC, "four_site_uci", UCI_MODELS, "Four-site (n=918)"),
        (RUN_C70, "cardio70k", C70_MODELS, "Cardio70k (n=68,749)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.9))
    for ax, (run, cohort, models, title) in zip(axes, specs):
        total, tested, sig, sig_test = [], [], [], []
        for m in models:
            c = _pairwise(run, cohort, m, cv=True)
            d = _pairwise(run, cohort, m)
            total.append(len(c))
            tested.append(int(c.tested.sum()))
            sig.append(int(c.significant.sum()))
            sig_test.append(int(d.significant.sum()))
        y = np.arange(len(models))
        ax.barh(y, total, color="#eeeeee", edgecolor="#bbbbbb", lw=0.5, label="untestable")
        ax.barh(y, tested, color="#c9d9e8", label="tested")
        ax.barh(y, sig, color="#1b6ca8", label="significant")
        ax.scatter(
            sig_test,
            y,
            marker="|",
            s=90,
            color="#b8433f",
            lw=1.4,
            zorder=3,
            label="significant (held-out)",
        )
        ax.set_yticks(y, [FAMILY[m] for m in models])
        ax.invert_yaxis()
        ax.set_xlim(0, max(total) * 1.02)
        ax.set_title(title, loc="left")
        ax.set_xlabel("distinct comparisons")
        print(f"  evidence {cohort}: total {total} tested {tested} sig {sig} held-out {sig_test}")
    h, lab = axes[0].get_legend_handles_labels()
    h, lab = h[1:] + h[:1], lab[1:] + lab[:1]  # bars first, the held-out mark last
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "evidence_by_cohort")


def fig_foursite_age() -> None:
    """Forest plot on the four-site pooled CV predictions, all ten age-group pairs."""
    groups = list(AGE_RANK)
    order = [f"{a} vs {b}" for i, a in reversed(list(enumerate(groups))) for b in groups[:i]]
    quantities = [
        ("positive_rate", "Positive rate"),
        ("fpr", "False-positive rate"),
        ("tpr", "True-positive rate"),
    ]
    frames = []
    for m in UCI_MODELS:
        d = _pairwise(RUN_CARDIAC, "four_site_uci", m, cv=True)
        frames.append(d[(d.attribute == "age_group_cat") & d.tested].assign(model=FAMILY[m]))
    d = pd.concat(frames)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.2), sharey=True)
    offs = {"LR": -0.27, "RF": -0.09, "SVM": 0.09, "XGB": 0.27}
    for ax, (q, title) in zip(axes, quantities):
        for _, r in d[d.quantity == q].iterrows():
            a, b = r.group_a, r.group_b
            older = AGE_RANK[a] > AGE_RANK[b]
            lab, sign = (f"{a} vs {b}", 1.0) if older else (f"{b} vs {a}", -1.0)
            yv = order.index(lab) + offs[r.model]
            lo, hi = sorted((sign * r.ci_low, sign * r.ci_high))
            col = FAM_COLOR[r.model]
            ax.plot([lo, hi], [yv, yv], color=col, lw=0.9, alpha=0.9 if r.significant else 0.45)
            ax.plot(
                sign * r.difference,
                yv,
                "o",
                ms=2.8,
                color=col,
                mfc=col if r.significant else "white",
                mew=0.7,
            )
        ax.axvline(0, color="#555555", lw=0.6)
        for k in range(1, len(order)):
            ax.axhline(k - 0.5, color="#eeeeee", lw=0.4, zorder=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("older minus younger")
    axes[0].set_yticks(range(len(order)), [o.replace("-", "–") for o in order])
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    handles = [
        plt.Line2D([], [], color=FAM_COLOR[k], marker="o", ms=3, lw=1.0, label=k) for k in offs
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            color="#777777",
            marker="o",
            mfc="white",
            ms=3,
            lw=1.0,
            alpha=0.5,
            label="not significant after BH",
        )
    )
    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.07)
    )
    fig.tight_layout()
    save(fig, "foursite_age_forest")


def fig_binning() -> None:
    """Prediction-level age parity gap against the label base-rate gap, per binning scheme."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.0))
    for ax, (run, cohort) in zip(axes, COHORTS):
        rd = CARD / "runs" / run
        lab = pd.read_csv(rd / "experiments" / "attribute_binning" / "comparison.csv")
        label = lab[lab.dataset == cohort].set_index("strategy").max_sp_difference
        lo, hi = 1.0, 0.0
        sens = rd / "baseline" / "age_binning_sensitivity" / cohort
        for f in sorted(sens.glob("*/age_binning_sensitivity.csv")):
            d = pd.read_csv(f)
            d = d[(d.regime == "baseline") & (d.strategy != "clinical_adaptive")]
            g = d.groupby("strategy").agg(dp=("dp_gap", "first"), nmin=("n", "min"))
            g["label"] = label
            fam = FAMILY[f.parent.name]
            col = FAM_COLOR[fam]
            big = g[g.nmin >= 30]
            small = g[g.nmin < 30]
            ax.scatter(big.label, big.dp, s=11, color=col, lw=0, alpha=0.85, label=fam)
            ax.scatter(
                small.label, small.dp, s=11, facecolor="none", edgecolor=col, lw=0.6, alpha=0.85
            )
            if "clinical" in g.index:
                ax.scatter(
                    g.loc["clinical", "label"],
                    g.loc["clinical", "dp"],
                    s=42,
                    facecolor="none",
                    edgecolor="#222222",
                    lw=0.7,
                )
            lo, hi = min(lo, g.label.min(), g.dp.min()), max(hi, g.label.max(), g.dp.max())
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.6, ls="--")
        ax.set_title(COHORT[cohort], loc="left")
        ax.set_xlabel("label base-rate gap")
    axes[0].set_ylabel("prediction parity gap")
    h, _ = axes[1].get_legend_handles_labels()
    h += [
        plt.Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=4,
            mfc="none",
            mec="#777777",
            label="smallest group < 30",
        ),
        plt.Line2D(
            [], [], ls="", marker="o", ms=6.5, mfc="none", mec="#222222", label="clinical scheme"
        ),
    ]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "binning_spread")


def fig_binning_grid() -> None:
    """Appendix C: age parity gap under every binning scheme, per cohort and model family."""
    schemes = [s for fam in SCHEME_FAMILIES.values() for s in fam]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 4.6), sharey=True)
    offs = {"LR": -0.24, "RF": -0.08, "SVM": 0.08, "XGB": 0.24}
    for ax, (run, cohort) in zip(axes, COHORTS):
        rd = CARD / "runs" / run
        lab = pd.read_csv(rd / "experiments" / "attribute_binning" / "comparison.csv")
        label = lab[lab.dataset == cohort].set_index("strategy").max_sp_difference
        sens = rd / "baseline" / "age_binning_sensitivity" / cohort
        for f in sorted(sens.glob("*/age_binning_sensitivity.csv")):
            d = pd.read_csv(f)
            d = d[(d.regime == "baseline") & d.strategy.isin(schemes)]
            g = d.groupby("strategy").agg(dp=("dp_gap", "first"), nmin=("n", "min"))
            fam = FAMILY[f.parent.name]
            col = FAM_COLOR[fam]
            y = np.array([schemes.index(s) for s in g.index]) + offs[fam]
            big = (g.nmin >= 30).values
            ax.scatter(g.dp[big], y[big], s=9, color=col, lw=0, label=fam)
            ax.scatter(g.dp[~big], y[~big], s=9, facecolor="none", edgecolor=col, lw=0.6)
        y = [schemes.index(s) for s in label.index if s in schemes]
        x = [label[s] for s in label.index if s in schemes]
        ax.scatter(x, y, marker="|", s=40, color="#222222", lw=1.0, label="outcome-rate gap")
        edge = 0
        for k, fam in enumerate(SCHEME_FAMILIES.values()):
            if k % 2:
                ax.axhspan(edge - 0.5, edge + len(fam) - 0.5, color="#f2f2f2", lw=0, zorder=0)
            edge += len(fam)
        ax.set_title(COHORT[cohort], loc="left")
        ax.set_xlabel("age parity gap")
        ax.set_xlim(0, None)
    names = [s.replace("_", " ").replace("adaptive quantile", "adaptive q.") for s in schemes]
    axes[0].set_yticks(range(len(schemes)), names)
    axes[0].set_ylim(len(schemes) - 0.5, -0.5)
    h, _ = axes[0].get_legend_handles_labels()
    h += [
        plt.Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=3.5,
            mfc="none",
            mec="#777777",
            label="smallest group < 30",
        )
    ]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save(fig, "binning_grid")


def fig_shap_share() -> None:
    """Per-group attribution share by age group, four-site and Cardio70k logistic regression."""
    specs = [
        (RUN_CARDIAC, "four_site_uci__logistic_regression", "Four-site LR"),
        (RUN_C70, "cardio70k__logistic_regression", "Cardio70k LR"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.0), gridspec_kw={"wspace": 0.42})
    for ax, (run, key, title) in zip(axes, specs):
        xai = CARD / "runs" / run / "baseline" / "results" / "xai" / key
        s = pd.read_csv(xai / "holdout" / "shap" / "subgroup_summary.csv")
        s = s[s.attribute == "age_group"]
        piv = s.pivot(index="feature", columns="group", values="share")
        cols = [c for c in AGE_RANK if c in piv.columns]
        piv = piv[cols]
        piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
        ns = s.drop_duplicates("group").set_index("group").n
        vmax = max(0.6, float(piv.values.max()))
        im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ticks = [f"{c.replace('-', chr(8211))}\n(n={int(ns[c])})" for c in cols]
        ax.set_xticks(range(len(cols)), ticks)
        ax.set_yticks(range(len(piv)), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if v > 0.35 else "#222222",
                )
        ax.set_title(title, loc="left")
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="attribution share")
    save(fig, "shap_share_age")


def fig_gradcam() -> None:
    """Four stratified Grad-CAM cases, augmented arm beside the no-augmentation arm.

    Both arms explain the same stratified image set, so each row is one image and
    one backbone under the two training regimes; the outcome per arm comes from
    the explanation manifest, not from the caption.
    """
    arms = (("augmented", RUN_DERM_AUG), ("no augmentation", RUN_DERM_CACHED))
    cases = [
        ("resnet18", "014_PAT_207_1280_15", "ResNet-18, cancer, III–IV"),
        ("densenet121", "007_PAT_926_1758_714", "DenseNet-121, cancer, III–IV"),
        ("densenet121", "017_PAT_770_1451_136", "DenseNet-121, cancer, V–VI"),
        ("resnet18", "011_PAT_898_1706_496", "ResNet-18, benign, III–IV"),
    ]
    words = {"TP": "detected", "FN": "missed", "FP": "false alarm", "TN": "cleared"}
    fig, axes = plt.subplots(len(cases), 2, figsize=(WIDTH, 5.6))
    for row, (backbone, img, lab) in zip(axes, cases):
        for ax, (arm, run) in zip(row, arms):
            base = DERM / "runs" / run / "baseline" / "explanations" / f"pad_ufes_20_{backbone}"
            manifest = pd.read_csv(base / "manifest.csv")
            hit = manifest[(manifest.method == "gradcam") & manifest.png_path.str.contains(img)]
            ax.imshow(imread(base / "gradcam" / f"{img}.png"))
            ax.set_axis_off()
            ax.set_title(f"{lab}: {words[hit.outcome.iloc[0]]} ({arm})", loc="left", fontsize=8)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    save(fig, "gradcam_cases")


def fig_frontier() -> None:
    """F1 against sex parity, every CV sweep cell coloured by constraint attribute."""
    crit = "dem_parity_sex_max_diff"
    style = {
        "none": ("#777777", "baseline (no constraint)"),
        "sex": ("#9fb7d0", "constrained on sex"),
        "age_group": ("#d9b38c", "constrained on age"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2))
    for ax, (run, cohort) in zip(axes, COHORTS):
        data = CARD / "runs" / run / "experiments" / "comparisons" / "data"
        t = pd.read_csv(data / f"tradeoff_{cohort}.csv")
        t = t[(t.status == "success") & (t.training_method == "kfold_cv")]
        t = t.dropna(subset=[crit, "f1_value"])
        t["con"] = t.constraint_attribute.fillna("none")
        for k in ("sex", "age_group", "none"):
            s = t[t.con == k]
            size = 6 if k == "none" else 4
            ax.scatter(s[crit], s.f1_value, s=size, color=style[k][0], lw=0, label=style[k][1])
        nd = t[_nondominated(t, "f1_value", crit)].sort_values(crit)
        ax.plot(nd[crit], nd.f1_value, "-", color="#b8433f", lw=0.8)
        ax.scatter(
            nd[crit], nd.f1_value, s=14, color="#b8433f", lw=0, zorder=3, label="non-dominated"
        )
        # interval width: the F1 fold spread of each non-dominated configuration
        ax.errorbar(
            nd[crit],
            nd.f1_value,
            yerr=nd.f1_score_std,
            fmt="none",
            ecolor="#b8433f",
            elinewidth=0.5,
            alpha=0.6,
        )
        ax.set_title(f"{COHORT[cohort]} ({len(t)} cells)", loc="left")
        ax.set_xlabel("sex parity gap (5-fold)")
        distinct = len(nd.drop_duplicates(["f1_value", crit]))
        print(f"  frontier {cohort}: {len(t)} cells, {len(nd)} non-dominated, {distinct} distinct")
    axes[0].set_ylabel("F1 (5-fold mean)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "frontier_sex")


# Single techniques in table order; the plain resamplers act on no group and are left out.
TECHNIQUES = {
    "exponentiated_gradient": "Exp. gradient",
    "threshold_optimizer": "Threshold rule",
    "grid_search": "Grid search",
    "reweighting": "Reweighting",
    "smote_group": "Group SMOTE",
    "uniform_sampling": "Uniform sampl.",
}


def fig_mitigation_deltas() -> None:
    """Change in the targeted parity gap per arm, one panel per constraint attribute."""
    frames = []
    for run, cohort in COHORTS:
        base = CARD / "runs" / run / "experiments" / "mitigation"
        name = "paired_effects.csv" if split_of(cohort) == "test" else "paired_effects_cv.csv"
        pe = pd.read_csv(base / name)
        frames.append(
            pe[
                (pe.dataset == cohort)
                & (pe.scope == "group_fairness")
                & (pe.metric == "demographic_parity")
                & (pe.quantity == "max_difference")
                & (pe.attribute == pe.constraint_attr)
                & pe.technique.isin(list(TECHNIQUES))
            ]
        )
    d = pd.concat(frames)
    d["significant"] = d.significant.fillna(False).astype(bool)
    rows = [(t, c) for t in TECHNIQUES for _, c in COHORTS]
    offs = {"LR": -0.27, "RF": -0.09, "SVM": 0.09, "XGB": 0.27}
    panels = [("sex", "Sex"), ("age_group", "Age group"), ("group_cluster", "Cluster")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 4.4), sharey=True)
    for ax, (attr, title) in zip(axes, panels):
        s = d[d.constraint_attr == attr]
        for _, r in s.iterrows():
            fam = FAMILY[r.model_type]
            col = FAM_COLOR[fam]
            yv = rows.index((r.technique, r.dataset)) + offs[fam]
            alpha = 0.9 if r.significant else 0.45
            ax.plot([r.ci_low, r.ci_high], [yv, yv], color=col, lw=0.8, alpha=alpha)
            mfc = col if r.significant else "white"
            ax.plot(r.difference, yv, "o", ms=2.6, color=col, mfc=mfc, mew=0.6)
        ax.axvline(0, color="#555555", lw=0.6)
        for k in range(len(COHORTS), len(rows), len(COHORTS)):
            ax.axhline(k - 0.5, color="#cccccc", lw=0.5, zorder=0)
        ax.set_title(f"Constrained on {title.lower()}", loc="left")
        ax.set_xlabel("change in parity gap")
        print(f"  mitigation {attr}: {len(s)} arms, {int(s.significant.sum())} significant")
    labels = [f"{TECHNIQUES[t]}, {COHORT[c]}" for t, c in rows]
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    handles = [
        plt.Line2D([], [], color=FAM_COLOR[k], marker="o", ms=3, lw=1.0, label=k) for k in offs
    ]
    handles.append(
        plt.Line2D(
            [], [], color="#777777", marker="o", mfc="white", ms=3, lw=1.0, alpha=0.5,
            label="not significant after BH",
        )  # fmt: skip
    )
    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05)
    )
    fig.tight_layout()
    save(fig, "mitigation_deltas")


def fig_cluster_gaps() -> None:
    """Cardio70k held-out age and sex parity gaps inside each cluster, against the outcome gap."""
    panels = [("age_group", "Age parity gap"), ("sex", "Sex parity gap")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.2), sharey=True)
    offs = {"LR": -0.18, "RF": 0.0, "XGB": 0.18}
    for ax, (attr, title) in zip(axes, panels):
        g = cluster_gaps(attr)
        g = g[g.cohort == "cardio70k"].sort_values("outcome_rate")
        order = list(dict.fromkeys(g.cluster))
        for model, s in g.groupby("model"):
            fam = FAMILY[model]
            x = np.array([order.index(c) for c in s.cluster]) + offs[fam]
            ax.scatter(x, s.parity_gap, s=12, color=FAM_COLOR[fam], lw=0, label=fam, zorder=3)
        out = g.drop_duplicates("cluster")
        ax.scatter(
            range(len(order)),
            out.outcome_gap,
            marker="_",
            s=90,
            color="#222222",
            lw=1.2,
            label="outcome-rate gap",
        )
        ticks = [f"{c}\n{r:.0%}\n{n:,}" for c, r, n in zip(out.cluster, out.outcome_rate, out.n)]
        ax.set_xticks(range(len(order)), ticks, fontsize=6.5)
        ax.set_title(title, loc="left")
        ax.set_xlabel("cluster, outcome rate, held-out n")
    axes[0].set_ylabel("max – min over groups")
    axes[0].set_ylim(0, None)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.11))
    fig.tight_layout()
    save(fig, "cardio70k_cluster_gaps")


FIGURES = (
    fig_evidence,
    fig_foursite_age,
    fig_binning,
    fig_binning_grid,
    fig_shap_share,
    fig_gradcam,
    fig_frontier,
    fig_mitigation_deltas,
    fig_cluster_gaps,
)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="figure directory")
    OUT = parser.parse_args().out
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn()


if __name__ == "__main__":
    main()
