# Explainability Module

Post-hoc explainability for FairXAI models. Two surfaces:

- **Tabular** (`tabular.py`): SHAP and LIME wrappers for the cardiac/tabular
  models. Counterfactual support remains an explicit placeholder.
- **Image** (`image.py`): SHAP, LIME, and Grad-CAM heatmaps for the dermatology
  CNN baselines, with a driver that stratifies overlays by group × outcome.

## Files

| File | Purpose |
|------|---------|
| `tabular.py` | SHAP/LIME dataclasses and helper functions (tabular models) |
| `image.py` | Image heatmaps (SHAP/LIME/Grad-CAM) + dermatology XAI driver |
| `__init__.py` | Public exports (tabular surface) |

## Public API (tabular)

- `ShapExplanation`
- `LimeExplanation`
- `shap_explain_tabular`
- `lime_explain_instance`
- `counterfactual_stub`

## Image API

Two layers, kept deliberately separate (see
[../../../docs/architecture/decisions.md](../../../docs/architecture/decisions.md),
"Image XAI Two-Layer Design"):

- **Pure heatmap functions** — no I/O, return a `[0,1]` saliency array:
  - `gradcam_heatmap` — Grad-CAM on the last conv layer
  - `lime_heatmap` — LIME image segmentation importance
  - `shap_heatmap` — SHAP pixel attribution
- **Driver** — orchestration around a saved checkpoint:
  - `select_images` — stratified sampling by group × outcome (TP/FP/TN/FN)
  - `explain_image_model` — loads a checkpoint, runs the heatmap fns, writes
    overlay PNGs

The image functions are not re-exported from `__init__.py`; import from
`fairxai.explainability.image` directly (script-facing surface).

## Config And Artifacts

XAI is enabled/configured by caller-level YAML:

- `configs/pipelines/cardiac.yaml`
- `configs/experiments/combinatorial.yaml`

Typical tabular baseline output:

```text
output/cardiac/runs/<run_id>/baseline/xai/<dataset>/
├── holdout/
└── cv/
```

Typical dermatology image-XAI output. Overlays are nested per run key
(dataset × model) then method; the group × outcome stratification lives in the
sampling and in `manifest.csv` columns, not in the directory tree:

```text
output/dermatology/runs/<run_id>/baseline/explanations/<run_key>/
├── gradcam/<NNN>_<image_id>.png
├── lime/<NNN>_<image_id>.png
├── shap/<NNN>_<image_id>.png
└── manifest.csv   # run_key, image_id, method, y_true, y_pred, outcome, <attrs>
```

## Usage

```python
from fairxai.explainability import shap_explain_tabular

explanation = shap_explain_tabular(
    model=model,
    data=X_test,
    feature_names=list(X_test.columns),
)
```

## Current Limit

`counterfactual_stub` is intentionally present and unimplemented. The
counterfactual workstream is deferred rather than silently absent.

## Related

- Plots: [../../../docs/reference/plots.md](../../../docs/reference/plots.md)
- Roadmap: [../../../docs/planning/roadmap.md](../../../docs/planning/roadmap.md)
