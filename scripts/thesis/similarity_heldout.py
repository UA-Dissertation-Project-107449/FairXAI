"""Held-out k-NN prediction consistency per cohort and model, from the runs in runs.yaml.

Mean consistency at k = 5, 10, 20; per-group means at k = 5 by age group and sex,
with a percentile bootstrap on every pairwise group difference (groups of at
least 30, intervals unadjusted). The UCI cohorts read the pooled CV predictions,
Cardio70k its held-out test split.

    python3 scripts/thesis/similarity_heldout.py [--out output/thesis/similarity_heldout.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from thesis_runs import CARD, DEFAULT_OUT, RUN_C70, RUN_CARDIAC, UCI_MODELS

from fairxai.similarity.engine import SimilarityEngine

SOURCES = {
    "cleveland_uci": (RUN_CARDIAC, "cv"),
    "four_site_uci": (RUN_CARDIAC, "cv"),
    "cardio70k": (RUN_C70, "test"),
}
META = {
    "fold", "sample_idx", "y_true", "y_pred", "y_proba", "threshold", "confidence",
    "near_threshold", "age_group", "sex", "group_cluster", "age_raw",
}  # fmt: skip
N_BOOT, FLOOR = 4000, 30


def pairwise(groups: dict, rng: np.random.Generator) -> dict:
    """Bootstrap every pair of groups with at least FLOOR rows."""
    sig, pairs, max_diff = [], 0, 0.0
    for a, b in itertools.combinations([v for v in groups if len(groups[v]) >= FLOOR], 2):
        xa, xb = groups[a], groups[b]
        d = xa.mean() - xb.mean()
        boot = np.array(
            [rng.choice(xa, len(xa)).mean() - rng.choice(xb, len(xb)).mean() for _ in range(N_BOOT)]
        )
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pairs += 1
        max_diff = max(max_diff, abs(d))
        if lo > 0 or hi < 0:
            sig.append(f"{a} vs {b}: {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    return {"tested": pairs, "max_abs_diff": round(max_diff, 3), "sig": sig}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "similarity_heldout.json")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    out = {}
    for cohort, (run, split) in SOURCES.items():
        pred = CARD / "runs" / run / "baseline" / "results" / "predictions"
        for model in UCI_MODELS:
            path = pred / f"{cohort}_{model}_{split}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            feats = [c for c in df.select_dtypes("number").columns if c not in META]
            eng = SimilarityEngine()
            rec = {"run": run, "split": split, "n": len(df), "features": feats}
            for k in (5, 10, 20):
                rec[f"k{k}"] = float(eng.per_sample_consistency(df, feats, k).mean())
            s = eng.per_sample_consistency(df, feats, 5)
            for attr in ("age_group", "sex"):
                values = sorted(df[attr].unique(), key=str)
                groups = {v: s[(df[attr] == v).values] for v in values}
                rec[attr] = {str(v): (round(float(x.mean()), 3), len(x)) for v, x in groups.items()}
                rec[attr + "_pairs"] = pairwise(groups, rng)
            out[f"{cohort}/{model}"] = rec
            ks = " ".join(f"k{k}={rec[f'k{k}']:.3f}" for k in (5, 10, 20))
            print(cohort, model, len(df), ks)
            for attr in ("age_group", "sex"):
                print("   ", attr, rec[attr], rec[attr + "_pairs"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
