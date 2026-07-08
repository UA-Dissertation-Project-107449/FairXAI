# Dermatology PAD Baseline

Scope: PAD-UFES-20 through baseline image training.

Entry point:

```bash
bash scripts/dermatology/dermatology_pipeline.sh --go-until train
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

Stages 8–11 run on saved prediction CSVs — no retraining, no model reload — and are opt-in
(`--go-until` or `RESUME_FROM=`/`GO_UNTIL=`):

- **8 assess** — subgroup fairness from test predictions, with post-hoc group views (binnings) recomputed
  on the same CSV. Views include `age_coarse`, `sex`, `fitzpatrick_group`, and the intersectional
  `sex_x_fitzpatrick` and `age_coarse_x_fitzpatrick` (gated by `intersection_min_group_samples`).
- **9 compare** — canonical CSV/Markdown + figures across models.
- **10 explain** — SHAP / LIME / Grad-CAM overlays for a small stratified sample.
- **11 mitigate** — **post-processing only.** Group-wise decision thresholds via fairlearn
  `ThresholdOptimizer`, fit on the train predictions and applied to the test predictions, per sensitive
  attribute in isolation, for every configured constraint side-by-side
  (`demographic_parity`, `equalized_odds`, `true_positive_rate_parity`, `false_positive_rate_parity`).
  Output: `baseline/mitigation/` (before/after JSON, Markdown, per-attr×constraint CSV).

```bash
RUN_ID=<run_id> GO_UNTIL=mitigate RESUME_FROM=mitigate \
  bash scripts/dermatology/dermatology_pipeline.sh
```

Mitigation is post-processing only by deliberate scope (pre/in-processing would require retraining the
CNN). See the rationale and limitation in
[architecture/decisions.md](architecture/decisions.md#image-fairness-is-post-prediction-only-no-retrain).
