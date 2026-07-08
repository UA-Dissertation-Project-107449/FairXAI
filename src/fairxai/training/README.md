# Training Module

Hyperparameter optimization helpers (tabular) and PyTorch image baseline
training (dermatology CNNs).

## Files

| File | Purpose |
|------|---------|
| `grid_search.py` | GridSearchCV/RandomizedSearchCV wrappers, save/load HPO params |
| `vision.py` | PyTorch image baseline trainer: transfer-learning CNNs, frozen-feature caching, early stopping, train-only augmentation |
| `__init__.py` | Public exports |

## Public API

- `run_hpo`
- `train_image_baseline`

`grid_search.py` also contains `save_hpo_results` and `load_hpo_params` for
script-level use.

## Image baseline training (`vision.py`)

Trains one torchvision backbone per call (resnet18, mobilenet_v3_large,
efficientnet_b0, densenet121) with a swapped 2-class head. Two training paths:

- **Frozen-feature cache** (default when `freeze_backbone` + `cache_frozen_features`):
  the backbone runs once in eval mode, pooled features are cached, and only the
  linear head trains over them — near-independent of epoch count.
- **Per-epoch** (backbone unfrozen, or augmentation on): pixels → features every
  epoch. Slower, but required for augmentation to add real per-epoch diversity.

### Train-only augmentation

`use_augmentation=True` applies smartphone-robustness transforms to the **train**
split only (RandomResizedCrop, H/V flips, RandomRotation, optional GaussianBlur,
ColorJitter **brightness/contrast only** — no hue/saturation, to protect
skin-tone fairness). The eval/test transform stays deterministic
(`Resize(256) → CenterCrop(224)`), so reported metrics are reproducible.

Turning augmentation on **forces the frozen-feature cache off** (a cached feature
would freeze a single random crop, killing diversity); the trainer logs a warning
and the checkpoint/metrics record `feature_cache: false` for provenance. Early
stopping validates on a deterministic (un-augmented) train slice, and the exported
train-prediction CSV also uses the deterministic transform, so augmentation never
leaks into best-epoch selection or downstream mitigation/XAI. DataLoaders use
per-loader seeded generators plus `worker_init_fn` for a reproducible augmentation
stream.

## Config And Artifacts

- HPO config: `configs/experiments/hpo.yaml`
- Model defaults: `configs/models/*.yaml`
- HPO outputs: `output/cardiac/studies/hpo/best_params_<dataset>_<model>.json`
- Image training config: `configs/pipelines/dermatology.yaml` (`training.image` block)

HPO runs before baseline/combinatorial stages in the current cardiac pipeline.
Downstream scripts reload best params and then re-apply runtime hardware/job
settings. `train_image_baseline` is driven by `scripts/common/train_baseline.py`
(stage 7 of the dermatology pipeline).

## Usage

```python
from fairxai.training import run_hpo

results = run_hpo(
    model_type="logistic_regression",
    X_train=X_train,
    y_train=y_train,
    param_grid={"C": [0.1, 1.0]},
)
```

## Related

- Models: [../models/README.md](../models/README.md)
- Scripts: [../../../scripts/README.md](../../../scripts/README.md)
