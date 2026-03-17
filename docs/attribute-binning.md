# Attribute Binning — Design & Usage

> **Module**: `src/fairxai/experiments/attribute_binning.py`
> **Runner**: `scripts/experiments/run_attribute_binning_analysis.py`
> **Config**: `configs/experiments/age_binning.yaml`

---

## 1  Purpose

The attribute-binning experiment answers a practical question for every
continuous or high-cardinality sensitive attribute:

> *Which discretisation strategy produces the fairest, most balanced groups
> while retaining enough samples per group for reliable analysis?*

Currently the implementation targets **age** (the most common continuous
sensitive attribute in healthcare datasets), but the architecture is designed
to generalise to any attribute that needs binning (e.g., income bands, BMI
categories, geographic regions).

---

## 2  How It Works

### 2.1  Strategy registry (config-driven)

Every binning strategy is declared in the YAML config.  The code resolves
strategies in the following priority order:

1. **Explicit config dict** passed to `create_binning_strategy()`.
2. **`BUILTIN_STRATEGIES`** — a Python dict that mirrors the YAML defaults so
   the module works even without a config file.
3. **Name-based inference** — legacy helper that infers `quantile_N` /
   `equal_width_N` patterns from the strategy name.

```yaml
# configs/experiments/age_binning.yaml  (excerpt)
binning_strategies:
  clinical:
    method: fixed
    bins: [0, 40, 55, 65, 75, 100]
    labels: ["<40", "40-54", "55-64", "65-74", "75+"]

  quantile_4:
    method: quantile
    n_bins: 4

  equal_width_5:
    method: equal_width
    n_bins: 5
```

Supported methods:

| Method        | Description                                    |
|---------------|------------------------------------------------|
| `fixed`       | User-supplied bin edges and labels             |
| `quantile`    | Bins with (approximately) equal sample counts  |
| `equal_width` | Bins with equal-width intervals over the range |
| `jenks`       | Natural breaks minimising within-group variance |
| `adaptive_quantile` | Quantile + auto-merge of under-populated bins |

### 2.2  Cross-attribute impact analysis

When multiple sensitive attributes are configured (e.g., `sex`, `race`),
the experiment computes a **cross-attribute impact matrix** for each strategy
and dataset:

| Metric                    | Meaning                                               |
|---------------------------|-------------------------------------------------------|
| `global_sp`               | Statistical parity diff for the attribute overall      |
| `max_within_bin_sp`       | Worst SP diff found in any single age bin              |
| `delta`                   | How much SP *worsens* inside bins vs the global value  |

A positive `delta` means the binning amplifies unfairness for that attribute;
a negative delta means it attenuates it.

### 2.3  Scoring

Each strategy receives a composite score (0–1) combining three weighted
dimensions:

- **Sample-size** — penalises strategies where the smallest bin has too few
  observations.
- **Balance** — penalises unequal group sizes (CV of bin counts).
- **Fairness sensitivity** — penalises strategies with high statistical parity
  differences.

Weights are configurable in the YAML under `scoring:`.

---

## 3  Recommendation Integration (Category C)

The triage recommendation engine (category **C — Representation**) includes a
**binning sensitivity sub-check**.  For any sensitive attribute with more than
two groups and a max/min size ratio exceeding `binning_size_ratio_warning`
(default **5.0×**), the engine emits a P2 recommendation suggesting the user
try alternative binning strategies.

The threshold lives in `configs/recommendations/thresholds.yaml`:

```yaml
representation:
  binning_size_ratio_warning: 5.0
```

---

## 4  Running the Experiment

```bash
# Minimal (uses defaults from YAML)
python3 scripts/experiments/run_attribute_binning_analysis.py

# Specific datasets / strategies
python3 scripts/experiments/run_attribute_binning_analysis.py \
    --datasets cleveland hungarian \
    --strategies clinical quantile_4

# With verbosity
python3 scripts/experiments/run_attribute_binning_analysis.py -vv
```

Or via the pipeline:

```bash
bash scripts/cardiac/cardiac_pipeline.sh --resume-from attribute_binning --go-until attribute_binning
```

### Output artefacts

| File                               | Content                          |
|------------------------------------|----------------------------------|
| `attribute_binning_comparison_*.csv`     | One row per strategy × dataset   |
| `attribute_binning_analysis_*.json`      | Full analysis dict (all metrics) |
| `attribute_binning_report_*.md`          | Human-readable summary + tables  |

---

## 5  Current State

The module has been renamed from `age_binning` to `attribute_binning`.
A backward-compatibility shim (`age_binning.py`) re-exports everything
from the canonical module.  The architecture already supports any
continuous or ordinal column via the `col` parameter, not just `age_raw`.

Future extensions may include:

- Decision-tree-based optimal binning.
- Domain-driven cutoffs loaded from external references.
- Binning sensitivity as part of the standard profiling stage.
