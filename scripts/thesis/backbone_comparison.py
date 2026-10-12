"""Paired bootstrap of the dermatology backbones' AUC, from saved test predictions.

Post hoc check behind Chapter 6's EfficientNet paragraph (advisor ask, meeting 18):
is the leading backbone's margin larger than test-sample noise, and does
augmentation change any backbone's AUC? Both runs in runs.yaml predict the same
held-out images in the same order, so every backbone and arm is scored on one
resample per replicate, and all comparisons are paired.

Patients, not images, are resampled: PAD-UFES-20 has several images per patient,
and resampling images would treat them as independent. Two BH families, as the
chapter reads them: the leader against each other backbone within each arm, and
augmented against cached for each backbone.

    python3 scripts/thesis/backbone_comparison.py [--out output/thesis] [--n-boot 4000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from thesis_runs import DEFAULT_OUT, DERM, RUN_DERM_AUG, RUN_DERM_CACHED

from fairxai.fairness.uncertainty import _benjamini_hochberg, _bootstrap_p_value

ARMS = {"augmented": RUN_DERM_AUG, "cached": RUN_DERM_CACHED}
BACKBONES = ["efficientnet_b0", "densenet121", "mobilenet_v3_large", "resnet18"]
LEADER = BACKBONES[0]


def load_predictions() -> pd.DataFrame:
    """One row per image; a y_proba column per (arm, backbone)."""
    frame = None
    for arm, run in ARMS.items():
        pred = DERM / "runs" / run / "baseline" / "results" / "predictions"
        for backbone in BACKBONES:
            df = pd.read_csv(pred / f"pad_ufes_20_{backbone}_test.csv")
            if frame is None:
                frame = df[["image_path", "patient_id", "y_true"]].copy()
            elif not (df["image_path"].values == frame["image_path"].values).all():
                raise ValueError(f"{arm} {backbone}: test images differ in order or content")
            frame[(arm, backbone)] = df["y_proba"].values
    return frame


def replicate_aucs(frame: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """AUC of every (arm, backbone) on each patient-level resample."""
    y = frame["y_true"].values
    cols = [c for c in frame.columns if isinstance(c, tuple)]
    rows = {p: idx for p, idx in frame.groupby("patient_id").indices.items()}
    patients = np.array(list(rows))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(patients, patients.size, replace=True)
        ii = np.concatenate([rows[p] for p in pick])
        if np.unique(y[ii]).size < 2:
            continue
        out.append({c: roc_auc_score(y[ii], frame[c].values[ii]) for c in cols})
    return pd.DataFrame(out)


def comparisons(frame: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    """Single AUCs, then the two families of paired differences, BH within each."""
    y = frame["y_true"].values

    def point(col):
        return roc_auc_score(y, frame[col].values)

    def row(kind, arm, a, b, est, reps):
        lo, hi = np.percentile(reps, [2.5, 97.5])
        p = _bootstrap_p_value(reps.values) if b else np.nan
        return dict(kind=kind, arm=arm, a=a, b=b, estimate=est, lo=lo, hi=hi, p=p)

    rows = []
    for arm in ARMS:
        for backbone in BACKBONES:
            col = (arm, backbone)
            rows.append(row("auc", arm, backbone, "", point(col), boot[col]))
        for other in BACKBONES[1:]:
            a, b = (arm, LEADER), (arm, other)
            rows.append(row("leader", arm, LEADER, other, point(a) - point(b), boot[a] - boot[b]))
    for backbone in BACKBONES:
        a, b = ("augmented", backbone), ("cached", backbone)
        rows.append(
            row(
                "arm",
                "augmented-cached",
                backbone,
                backbone,
                point(a) - point(b),
                boot[a] - boot[b],
            )
        )
    table = pd.DataFrame(rows)
    for kind in ("leader", "arm"):
        mask = table["kind"] == kind
        table.loc[mask, "p_adj"] = _benjamini_hochberg(table.loc[mask, "p"].values)
    for arm in ARMS:
        aucs = np.column_stack([boot[(arm, b)].values for b in BACKBONES])
        share = float(np.mean(aucs.argmax(axis=1) == 0))
        table.loc[len(table)] = dict(kind="leader_first", arm=arm, a=LEADER, estimate=share)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = load_predictions()
    print(
        f"n={len(frame)} patients={frame['patient_id'].nunique()} "
        f"prevalence={frame['y_true'].mean():.3f}"
    )
    boot = replicate_aucs(frame, args.n_boot, args.seed)
    table = comparisons(frame, boot)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "backbone_comparison.csv"
    table.to_csv(path, index=False)
    with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))
    print(f"wrote {path} (B={len(boot)})")


if __name__ == "__main__":
    main()
