# Architecture And Design Decisions

This note captures decisions that affect how the repository is organized and
how generated artifacts should be interpreted.

## Repository Shape

| Path | Decision |
|------|----------|
| `src/fairxai/` | Reusable package code. Scripts should call into this layer instead of duplicating logic. |
| `scripts/` | Operational entry points for pipeline stages, studies, and experiments. |
| `flows/` | Prefect orchestration around the same scripts used by the bash pipelines: one flow per domain, each mirroring its bash counterpart stage for stage. |
| `configs/` | Declarative runtime settings for pipeline, domain, model, profiling, recommendation, and experiment behavior. |
| `docs/` | Architecture, guides, references, research notes, and planning docs. |
| `data/` | External/raw/processed datasets. Generated data is not package source. |
| `output/` | Run-scoped and study-scoped generated artifacts. |
| `logs/` | Run logs, warning logs, error logs, and run summaries. |
| `notebooks/` | Exploration and presentation support. Durable logic belongs in `src/` or `scripts/`. |

## Workflow

1. Scripts are the primary execution unit.
2. Bash and Prefect orchestrators coordinate scripts and checkpoint stages.
3. Notebooks consume generated artifacts and should not be the only home for research logic.
4. Docs follow code/config truth when drift appears.

## Processed Data Layout

Processed train/test splits live in per-dataset subdirectories:

```text
data/processed/<pipeline>/<dataset>_<binning>/<dataset>_train.csv
data/processed/<pipeline>/<dataset>_<binning>/<dataset>_test.csv
data/processed/<pipeline>/<dataset>_<binning>/<dataset>_train_scaled.csv
data/processed/<pipeline>/<dataset>_<binning>/<dataset>_test_scaled.csv
```

Decision points:

- `runtime.default_binning` in `configs/pipelines/cardiac.yaml` is the canonical default.
- HPO, training, mitigation, grouping, and dissertation plots should use shared dataset resolvers in `fairxai.experiments.data_io`.
- Flat processed files directly under `data/processed/cardiac/` are legacy and should not be used for current pipeline behavior.
- Combinatorial experiments deliberately address explicit `<dataset>_<binning>/` directories because they sweep binning strategies.

## Run And Study Outputs

- Run-scoped artifacts live under `output/cardiac/runs/<run_id>/`.
- Study artifacts live under `output/cardiac/studies/<study_type>/`.
- Latest-run pointers live at `output/cardiac/latest_run` and `output/cardiac/latest_run.txt`.
- Logs mirror run IDs under `logs/cardiac/runs/<run_id>/`.

## Known Artifacts And Limits

### Logistic Regression Perfect Training Scores On Balanced Cleveland

Some mitigation configurations on the small Cleveland dataset can yield perfect
training metrics after resampling. This is treated as an overfit/stability
artifact, not as primary evidence. Dissertation framing should prefer selected
mitigations and test-set tradeoffs over these sensitivity rows.

### DBSCAN Sensitivity On Cleveland

Cleveland is small and moderately high-dimensional for density clustering.
DBSCAN requires a wider `eps` search than the initial grid. If DBSCAN remains
weak, the selected clustering solution should be described as exploratory
subgroup evidence rather than strong natural phenotype discovery.

### Dermatology Scope

Dermatology has scaffolding, but the end-to-end implemented research pipeline
is cardiac. Docs should avoid implying equivalent pipeline maturity.

## Dermatology Design Notes

### Image XAI Two-Layer Design

Image explainability (`explainability/image.py`) is split into two layers on
purpose:

- **Pure heatmap functions** (`gradcam_heatmap`, `lime_heatmap`, `shap_heatmap`)
  take a model + tensor and return a normalized `[0,1]` saliency array. No file
  I/O, no checkpoint loading, no sampling — trivial to unit-test and to reuse
  outside the pipeline.
- **Driver** (`select_images`, `explain_image_model`) owns the side effects:
  checkpoint loading, stratified group × outcome sampling, overlay rendering, and
  the `manifest.csv`.

Rationale: the heatmap math is the defensible methods-chapter contribution and
must be testable without GPU/checkpoints; the orchestration is pipeline glue.
Keeping them separate also lets the assessment/figures stages consume the
heatmap fns without inheriting the driver's I/O assumptions.

### Frozen-Head Training Uses Early Stopping, Not A Fixed Epoch Count

Image baselines train a linear head over once-cached frozen-backbone features
(`training/vision.py`), so epochs are cheap but a fixed count is the wrong knob — a
linear head converges in a few epochs. `epochs` is therefore a **cap** (config 50)
and training stops via patience-based early stopping on validation AUC (fallback
`-val_loss` when a slice is single-class), restoring the best-AUC weights. The
validation slice is carved **from train**, row-stratified (not patient-grouped); it
drives stopping only and never feeds the reported **test** metric, which keeps the
upstream patient-grouped split. The same logic mirrors into the non-cached
full-model path. `epochs_run`/`best_epoch`/`early_stopped` and per-epoch
`val_loss`/`val_auc` are recorded in the metrics JSON and surfaced as
`model_comparison.csv` columns + a stage-9 learning-curve figure, so "trained to
convergence" is a checkable claim, not an assertion. This is a deliberate override
of the review doc's fixed `epochs: 5`; everything else (AdamW, lr, image_size 224,
frozen backbone, one split, one seed) stays fixed so it remains a single-axis study.

### Image Fairness Is Post-Prediction Only (No Retrain)

Dermatology fairness (`fairness/image_assessment.py`) scores from a saved
predictions CSV, never from model weights. Post-hoc **group views** (alternate
subgroup definitions, including two intersectional views `sex_x_fitzpatrick` and
`age_coarse_x_fitzpatrick`) are a CSV `groupby`, not a retraining multiplier —
"5 binnings" cost 5× a groupby, not 5× training. Support gates
(`min_group_samples=50`, intersectional `=30`) drop undersized groups from metrics
while reporting them as skipped, so small subgroups never silently inflate a
fairness delta.

**Mitigation for images is post-processing only** (stage 11,
`fairness/image_mitigation.py`). Group-wise decision thresholds via fairlearn
`ThresholdOptimizer` are fit on the saved **train** predictions and applied to the
**test** predictions (never fit and evaluated on the same rows), per sensitive
attribute *in isolation*, for every configured constraint side-by-side
(`demographic_parity`, `equalized_odds`, `true_positive_rate_parity`,
`false_positive_rate_parity`). It reuses
`PostProcessingMitigation.apply_threshold_optimizer` through a precomputed-score
estimator wrapper, so no model is loaded. Thresholds are fit only on **eligible**
train groups (>= `min_group_samples` and both classes present); fairlearn rejects
degenerate single-class groups (PAD `<20` age, `Unknown` sex/Fitzpatrick), so
those are excluded from the fit and their test rows keep the baseline prediction.
An attribute with fewer than two eligible groups is reported with a `note` and no
threshold tuning.

**Pre/in-processing runs in feature space** (stage 11 part 2,
`fairness/image_feature_mitigation.py`). It was previously out of scope on the
grounds that reweighting/SMOTE/ADASYN and the fairlearn reductions are
tabular-first and would require retraining a CNN. `freeze_backbone: true` makes
that false: the network is a fixed feature extractor and only the linear head is
learned, so the learning problem *is* tabular — an `n_rows x n_channels` matrix
and a binary label, exactly what `MitigationEngine` consumes. The matrix is
rebuilt per model with one eval-mode, no-grad forward pass over the saved
checkpoint (training does not persist its cached features, and the augmentation
path never caches them), standardised with a train-fit `StandardScaler`, then
run through the **same engine and the same technique catalog as cardiac**
(`reweighting`, `smote`, `adasyn`, `exponentiated_gradient`, `grid_search`;
`ros`/`rus` excluded in both domains). Running identical implementations, rather
than image-specific lookalikes, is what makes the cross-domain comparison a
comparison.

Two limits are recorded in every report rather than left to be rediscovered.
First, **the intervention is on the head, not the representation**: bias encoded
into the frozen features by ImageNet pre-training survives every technique here.
Learning a fair representation needs the backbone unfrozen — adversarial
debiasing is inert under a frozen backbone, since the adversary sees fixed
features and the encoder has no gradient path — and stays out of scope as a
different, far more expensive experiment. Second, the delta reference is an
**unmitigated linear head over the same standardised features**, never the CNN's
own softmax head: those are different classifiers, so measuring against the CNN
would report a head swap as a mitigation effect. The CNN's metrics travel in the
report as context only. Unlike the post-processing path, undersized groups are
not excluded from the *fit* — these techniques train a classifier that scores any
row, rather than a per-group threshold that cannot exist for a group never fit —
but they are still dropped from the fairness *metrics*, with
`group_support_train` recording what each technique actually saw.

### SCIN Is Profiling-Only With An Approximate Target

SCIN (`configs/schema/dermatology.json` → `scin`, loaded by
`DermatologyDataLoader._standardize_scin`) is a **schema-generalization +
fairness-profiling** dataset, not a second skin-cancer training set (only 25
malignant top-1 cases out of 3061 loaded locally). It proves the config-driven pipeline absorbs a
different schema — two CSVs joined on `case_id`, pre-binned native `age_group`,
`FST1..FST6` Fitzpatrick, multi-image cases, and a weighted-condition-dict label
— through the same `load → profile → recommend` stages with no pipeline rewrite.

Key decisions:
- **Scope**: profiling-only, **case-level** (one row per `case_id`, no image
  explode, no split, no training). Opt-in via `--datasets scin`; not in the
  default `runtime.datasets`, so default PAD runs are unchanged. Gradability
  training (review Step 6) stays deferred.
- **Approximate target**: `skin_cancer` = top-1 (highest-weight) condition from
  `weighted_skin_condition_label` mapped against a curated cancer-like substring
  set (`positive_label_substrings` in the schema). This is an approximate label
  for complexity/triage profiling only and must not be presented as a trained
  diagnostic. The substring set was finalized against the real 211-name label
  vocabulary — `melanoma`, `carcinoma`, `basal cell`, `squamous cell`, `scc`,
  `lymphoma`, `sarcoma`, `metastasis` — yielding 25 malignant positives (SCC/SCCIS
  8, BCC 6, cutaneous lymphoma 7, Kaposi's sarcoma 2, metastasis 1, melanoma 1).
  Premalignant Actinic Keratosis is treated as negative, consistent with PAD's
  `ACK` → negative mapping. The set stays config-driven for future tuning.
- **Recommend stage honors `--datasets`**: `scripts/common/generate_recommendations.py`
  globs every `*_standardized.csv` on disk, so a `--datasets scin` run previously
  also emitted a stale `pad_ufes_20` triage. It now accepts a `--datasets` filter
  (passed through from the pipeline shell) so the recommend phase is scoped like
  every other phase; default (no flag) still processes all standardized datasets.
- **`case_id` excluded from complexity**: it is a large signed integer key; left
  in, it would dominate distance-based complexity metrics. Added to
  `_COMPLEXITY_EXCLUDE_COLS` alongside `year`/`release`.

## Fairness Evidence Decisions

### Subgroup SHAP Is Derived From The Global Matrix, Not A Second Pass

`save_xai_outputs()` computed a full `|SHAP|` matrix and then collapsed it to one
cohort-wide `summary.csv`, which cannot answer whether the model explains its
decisions the same way for every group — the question a fairness audit needs.
`src/fairxai/explainability/subgroup.py` groups that same matrix by each
sensitive column and writes `subgroup_summary.csv`, `subgroup_disparity.csv`,
and `subgroup_agreement.csv` beside it.

- **No extra SHAP compute.** The matrix is already in memory; this is a
  group-by. A second explanation pass per group would have cost as much as the
  original and produced the same numbers.
- **Alignment is by index label, not position.** SHAP subsamples above
  `xai.global_max_samples`, so the sensitive frame is reindexed onto the rows
  SHAP actually explained. A positional join would silently mislabel groups.
  This also means the sensitive columns need not be model features — under
  `exclude_sensitive` they are not.
- **Two magnitudes, deliberately.** `mean_abs_shap` moves with model confidence,
  so a uniformly less-confident group looks "less explained" on every feature at
  once. `share` normalises each group's vector to sum to one, isolating *which*
  features carry the explanation. A disparity surviving in `share` is structural;
  one visible only in `mean_abs_shap` is a confidence difference.
- **Groups below `xai.subgroup_min_size` (30) are dropped and logged.**
  Per-feature percentiles over a handful of rows are noise, and a disparity
  driven by a five-person group is not a finding.

Explanation *quality* metrics (fidelity, cross-method agreement, stability)
remain out of scope; this closes attribution *disparity* only.

### Fairness Intervals: Max-Gap CIs Are Descriptive, Pairwise Differences Are The Test

`src/fairxai/fairness/uncertainty.py` resamples the prediction frame and re-runs
`calculate_all_metrics` per replicate, so every scalar the assessment reports
gains an interval and any metric added later inherits one. Written by the assess
stage as `<dataset>_ci.csv` and `<dataset>_pairwise_ci.csv`.

The load-bearing decision is that **the two tables are not interchangeable**:

- Most parity metrics report `max(rate) - min(rate)` over groups. That statistic
  is bounded below by zero and biased upward under resampling, so its interval
  almost never contains zero **even when no disparity exists**. Those intervals
  are descriptive spread only; a test built on them would fire on every cohort.
- The `pairwise` table holds the signed difference between two named groups,
  which is centred at zero under the null. Differences are taken *within* a
  replicate so the two groups' correlated estimates cancel; differencing two
  independently summarised intervals would overstate uncertainty and hide real
  disparities.

Supporting choices:

- **Benjamini-Hochberg, not Bonferroni.** Fairness metrics on one cohort are
  strongly dependent (`fnr = 1 - tpr`; `tpr` is reported by both equalized odds
  and equal opportunity), so family-wise control over dozens of near-duplicate
  comparisons removes real findings with the spurious ones. Bonferroni was also
  degenerate for a binary attribute — one pair per metric means no correction at
  all — while the real multiplicity is across metrics and attributes.
- **p-values rather than adjusted interval endpoints.** An adjusted percentile
  endpoint is a single extreme order statistic and needs replicate counts nobody
  can afford at 68k rows; a p-value uses the whole tail. The value is floored at
  `2/(B+1)` and the run warns when replicates cannot resolve the adjusted
  threshold.
- **Stratified by group x outcome by default**, degrading to group and then to
  i.i.d. when a stratum is too small to resample. Group sizes and per-group
  prevalence are design facts of the cohort, not quantities being estimated. The
  scheme that actually ran is recorded in the run metadata. *Measured caveat:*
  on `cleveland_uci` (n=297) the default never survives — a five-level
  `age_group` crossed with sex, `group_cluster` and outcome leaves single-row
  cells, so every split degraded `group_outcome -> group -> none`. The
  applicable row of the calibration table below is therefore `none`, not the
  default row. The conditioning argument still decides the *requested* scheme;
  it simply does not bind on the small cardiac cohorts.
- **`fairness.uncertainty.n_bootstrap`: 4000 for cardiac, not `auto`.** `auto`
  is 1000 replicates (thinned to 200 above 10k rows, mirroring
  `adaptive_shap_sample_cap`), and a bootstrap p-value cannot resolve below
  `2/(B+1)`. Three sensitive attributes give a 98-comparison family, so BH needs
  roughly p=0.0005 while 1000 replicates floor at p=0.0020: the adjusted column
  was structurally unreadable. The run warns when this holds; 4000 clears it.
- **Replicates run in parallel (`n_jobs`).** Each replicate carries its own
  spawned seed rather than drawing from one shared generator, so the worker
  count is a pure performance knob and the numbers are identical at any
  `n_jobs`. Measured on `cleveland_uci` at B=1000: 32.5s serial, 9.9s on 8
  cores.
- **Individual fairness is never bootstrapped.** Its k-NN consistency is O(n^2),
  and a resampled cohort contains duplicate rows at distance zero from each
  other, which would inflate consistency by construction.
- **Never fatal.** A bootstrap failure is logged and swallowed; intervals are an
  addition to the assessment and must not cost the point estimates downstream
  stages consume.
- **`descriptive_only` is a column, not just prose.** The max-gap argument
  applies verbatim to per-group *expected calibration error*: ECE is a
  nonnegative plug-in statistic biased upward on small groups. On `cleveland_uci`
  the age 40-49 ECE came back with point 0.076 and interval [0.079, 0.221] — an
  interval sitting entirely above its own point estimate. Both families (every
  `*difference*` gap and `ece`) now carry `descriptive_only = True` in the CI
  table so the warning travels with the number instead of living only here.
- **`degenerate` flags rows with no replicate spread.** Where a group is too
  small or too homogeneous to vary under resampling, every replicate returns the
  same value and the interval collapses to a point (13 of 94 rows on the first
  real `cleveland_uci` run). That is not precision, and the run now warns and
  marks those rows rather than emitting endpoints that look tight.

#### Measured null calibration

`scripts/studies/run_bootstrap_calibration.py` draws cohorts whose predictions
are independent of every sensitive attribute, so every reported finding is a
false one. 20 seeds x 400 rows x 1200 replicates, alpha = 0.05:

| stratify | unadjusted | BH-adjusted | runs with a false finding | max-gap CI excludes zero |
|---|---|---|---|---|
| `group_outcome` (default) | 5.89% | 0.89% | 3/20 | 99.6% |
| `group` | 4.64% | 0.18% | 1/20 | 100.0% |
| `none` | 5.00% | 0.00% | 0/20 | 100.0% |

Readings:

- The **max-gap column is the headline**: on cohorts built with no disparity at
  all, the gap interval excluded zero essentially every time under every scheme.
  Shipping those intervals as a test would have reported every cohort as unfair.
- The pairwise unadjusted rate tracks nominal across all three schemes; the
  spread between them is within Monte Carlo error at 20 seeds (SE ~0.9pp), so
  the default is kept on the conditioning argument rather than on this ranking.
- The BH-adjusted rate — the column the `significant` flag actually uses — stays
  at or below 0.89% everywhere, comfortably inside the 5% it targets.

## Related

- Module map: [modules.md](modules.md)
- Pipeline flow control: [pipeline-flow-control.md](pipeline-flow-control.md)
- Roadmap: [../planning/roadmap.md](../planning/roadmap.md)
