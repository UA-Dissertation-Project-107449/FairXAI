# Dermatology PAD Baseline

Scope: PAD-UFES-20, stages 1–4 and 7–11 (load through mitigate). There are no
stages 5 and 6.

Entry point:

```bash
bash scripts/dermatology/dermatology_pipeline.sh                    # all stages
bash scripts/dermatology/dermatology_pipeline.sh --go-until train   # stop after training
```

Useful overrides follow the project precedence rule: CLI flags > pipeline YAML > code defaults.

```bash
bash scripts/dermatology/dermatology_pipeline.sh \
  --datasets pad_ufes_20 \
  --model-types resnet18 \
  --device cuda \
  --epochs 5 \
  --batch-size 32
```

Device order for `--device auto`: CUDA, then ROCm, then CPU. PyTorch ROCm exposes AMD GPUs through the `cuda` runtime internally; FairXAI reports the resolved accelerator as `rocm` when `torch.version.hip` is present.

Install PyTorch with the official selector for your platform first, then install the project vision extra:

```bash
pip install -e ".[vision]"
```

Use `--no-pretrained` if ImageNet weights are not cached and the environment has no network access.

## Data augmentation (train-only)

Stage 7 supports smartphone-robustness augmentation, on by default in
`configs/pipelines/dermatology.yaml` (`training.image.use_augmentation: true`).
It is applied to the **train** split only; the eval/test transform stays
deterministic (`Resize(256) → CenterCrop(224)`), so reported metrics are
reproducible.

Transforms: `RandomResizedCrop`, horizontal + vertical flips, `RandomRotation`,
optional `GaussianBlur`, and `ColorJitter` limited to **brightness/contrast**.
Hue and saturation are deliberately excluded so skin tone is never shifted —
a fairness guard for Fitzpatrick subgroups.

```yaml
training:
  image:
    use_augmentation: true
    augmentation:
      crop_scale_min: 0.7    # RandomResizedCrop lower bound (also fixes square-squish)
      rotation_degrees: 20   # phone orientation jitter
      blur_prob: 0.2         # p of mild GaussianBlur (out-of-focus phone shots)
      brightness: 0.2        # exposure variance
      contrast: 0.2          # exposure variance
```

Override per run with `--augmentation` / `--no-augmentation`.

Key behavior when augmentation is on:

- **Feature cache is forced off.** Cached frozen features are extracted once, so a
  random crop would be frozen with no diversity. The trainer logs a warning, keeps
  the backbone frozen, and re-runs pixels → features every epoch (slower, but the
  augmentation is real). The checkpoint and metrics JSON record `feature_cache: false`.
- **No leakage into evaluation.** Early stopping validates on a deterministic
  (un-augmented) train slice, and the exported train-prediction CSV consumed by
  mitigation and XAI also uses the deterministic transform.
- **Reproducible.** Per-loader seeded generators plus `worker_init_fn` (numpy/random)
  make the augmentation stream repeat exactly across reruns for a fixed `random_state`.

See `src/fairxai/training/README.md` for the trainer-level detail.

## Post-prediction stages (8–11)

Stages 8–11 run after training without retraining the CNN. They run by default; use `--go-until`
or `RESUME_FROM=`/`GO_UNTIL=` to select them. Stage 10 follows `xai.enabled`; override it with
`--explain` / `--no-explain`.

- **8 assess** — subgroup fairness from test predictions, with post-hoc group views (binnings) recomputed
  on the same CSV. Views include `age_coarse`, `sex`, `fitzpatrick_group`, and the intersectional
  `sex_x_fitzpatrick` and `age_coarse_x_fitzpatrick` (gated by `intersection_min_group_samples`).
- **9 compare** — canonical CSV/Markdown + figures across models.
- **10 explain** — SHAP / LIME / Grad-CAM overlays for a small stratified sample.
- **11 mitigate** — two parts.
  - *Post-processing.* Group-wise decision thresholds via fairlearn `ThresholdOptimizer`, fit on the
    train predictions and applied to the test predictions, per sensitive attribute in isolation, for
    every configured constraint side-by-side (`demographic_parity`, `equalized_odds`,
    `true_positive_rate_parity`, `false_positive_rate_parity`). Output: `baseline/mitigation/`
    (before/after JSON, Markdown, per-attr×constraint CSV).
  - *Feature space* (`mitigation.feature_space`). Rebuilds each model's frozen-backbone features with
    one forward pass and runs cardiac's pre/in-processing catalog on the linear head. Output:
    `baseline/mitigation/feature_space/`. Skip it with `--no-feature-space` on
    `scripts/dermatology/mitigate.py`.

```bash
RUN_ID=<run_id> GO_UNTIL=mitigate RESUME_FROM=mitigate \
  bash scripts/dermatology/dermatology_pipeline.sh
```

Neither part retrains the CNN backbone. The feature-space arms act on the head only, so bias in the
frozen features survives. See the rationale and limitation in
[architecture/decisions.md](architecture/decisions.md#image-fairness-is-post-prediction-only-no-retrain).
