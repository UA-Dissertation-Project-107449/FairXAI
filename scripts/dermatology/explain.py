#!/usr/bin/env python3
"""Dermatology phase runner: post-hoc explainability (stage 10).

Reloads each trained baseline model for the current run and writes SHAP, LIME and
Grad-CAM saliency overlays for a small set of test images stratified by sensitive
group and outcome. No retraining. Methods and sample size come from the ``xai``
section of ``configs/pipelines/dermatology.yaml``; CLI flags override.

Invoked by the pipeline with ``RUN_ID`` exported; can also be run standalone:

    RUN_ID=<run_id> python3 scripts/dermatology/explain.py --methods gradcam
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from fairxai.explainability.image import METHODS, explain_image_model  # noqa: E402

PIPELINE = "dermatology"


def _resolve_run_id() -> str:
    run_id = os.getenv("RUN_ID")
    if run_id:
        return run_id
    base = ROOT_DIR / "output" / PIPELINE
    latest_txt = base / "latest_run.txt"
    if latest_txt.exists():
        return latest_txt.read_text().strip()
    link = base / "latest_run"
    if link.is_symlink():
        return link.resolve().name
    raise SystemExit("RUN_ID not set and no latest dermatology run found.")


def _discover_models(run_root: Path, datasets, model_types) -> list[tuple[str, dict]]:
    """Return ``(run_key, metrics_json)`` for successful models with a checkpoint."""
    results_dir = run_root / "baseline" / "results"
    dataset_filter = set(datasets) if datasets else None
    model_filter = set(model_types) if model_types else None
    out = []
    for path in sorted(results_dir.glob("*_metrics.json")):
        run_key = path.name[: -len("_metrics.json")]
        data = json.loads(path.read_text())
        if data.get("status") not in (None, "success") or not data.get("model_file"):
            continue
        model_type = data.get("model_type", "")
        dataset = run_key[: -(len(model_type) + 1)] if model_type else run_key
        if dataset_filter and dataset not in dataset_filter:
            continue
        if model_filter and model_type not in model_filter:
            continue
        out.append((run_key, data))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", help="Restrict to these datasets.")
    parser.add_argument("--model-types", nargs="*", help="Restrict to these model types.")
    parser.add_argument("--methods", nargs="*", choices=METHODS, help="Override enabled methods.")
    parser.add_argument("--n-samples", type=int, default=None, help="Images to explain per model.")
    parser.add_argument("-v", action="store_const", const=1, dest="verbose", default=0)
    parser.add_argument("-vv", action="store_const", const=2, dest="verbose")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cfg_path = ROOT_DIR / "configs" / "pipelines" / f"{PIPELINE}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    xai_cfg = cfg.get("xai", {})
    fairness_cfg = cfg.get("fairness", {})
    image_cfg = cfg.get("training", {}).get("image", {})

    if not xai_cfg.get("enabled", False) and not args.methods:
        print("[PHASE 10] xai.enabled is false and no --methods given; skipping.")
        return

    methods = args.methods or [m for m in METHODS if xai_cfg.get(m, True)]
    n_samples = args.n_samples or xai_cfg.get("n_samples", 12)
    per_cell = xai_cfg.get("per_cell", 1)
    num_samples_lime = xai_cfg.get("lime_num_samples", 1000)
    sensitive_attrs = fairness_cfg.get("sensitive_attributes", ["sex", "fitzpatrick_group"])
    image_col = image_cfg.get("image_column", "image_path")
    device = image_cfg.get("device", "cpu")
    if device == "auto":
        device = "cpu"  # explanation pass is light; pin to CPU for determinism

    run_id = _resolve_run_id()
    run_root = ROOT_DIR / "output" / PIPELINE / "runs" / run_id

    import pandas as pd

    print(f"[PHASE 10] Explaining baseline models for run {run_id} (methods: {', '.join(methods)})")
    total = 0
    for run_key, data in _discover_models(run_root, args.datasets, args.model_types):
        preds = pd.read_csv(data["test_predictions"])
        manifest = explain_image_model(
            run_root,
            run_key,
            Path(data["model_file"]),
            preds,
            image_col=image_col,
            sensitive_attrs=sensitive_attrs,
            methods=methods,
            n_samples=n_samples,
            per_cell=per_cell,
            num_samples_lime=num_samples_lime,
            device=device,
        )
        total += len(manifest)
        print(f"  {run_key}: {len(manifest)} explanation(s)")

    if total == 0:
        print("  No explanations produced.")
        return
    print(f"  Output: {run_root / 'baseline' / 'explanations'}")


if __name__ == "__main__":
    main()
