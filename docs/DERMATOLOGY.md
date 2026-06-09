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
