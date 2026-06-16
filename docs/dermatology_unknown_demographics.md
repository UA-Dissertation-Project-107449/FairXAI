# Dermatology — the all-benign "unknown demographics" cohort (PAD-UFES-20)

Status: **data finding**, surfaced by stage 8 (post-prediction fairness assessment).
Relevant to the dissertation methodology section. No code bug — this documents a
structural property of the PAD-UFES-20 test split and how the assessment handles it.

## The finding

In the PAD-UFES-20 test split (682 rows), **228 rows (33%)** have missing
demographic metadata, and they are the *same* rows across two sensitive
attributes:

- `sex == -1` (Unknown): 228 rows
- `fitzpatrick_group == "unknown"`: 228 rows
- overlap of the two: **228** — they are identical rows.

Every one of those 228 rows is **negative** (`y_true == 0`, benign):

```
y_true                0    1
sex
-1                  228    0      <- unknown sex: all benign
 0 (Female)          74  155
 1 (Male)            52  173

y_true                0    1
fitzpatrick_group
I-II                 68  239
III-IV               54   88
V-VI                  4    1
unknown             228    0      <- unknown Fitzpatrick: all benign
```

So **missing demographics perfectly predicts the benign class** in this split.

## Why it matters for fairness metrics

A group with no positives has an **undefined true-positive rate** (TPR = TP/(TP+FN)
with TP+FN = 0). The same holds for FPR in a group with no negatives. If such a
group is left in the group-difference computation, its undefined rate is treated as
0 and **inflates every max-difference delta**.

Concretely, before excluding the unknown group, `resnet18` reported:

| attribute | DP Δ | TPR Δ |
|---|---:|---:|
| sex (incl. Unknown) | 0.455 | 0.729 |
| sex (Female vs Male only) | **0.005** | **0.016** |

The 0.73 "TPR gap" was entirely the unknown-vs-rest artifact. The real
Female-vs-Male gap is ~0.02 — i.e. **the model is essentially fair across sex**.
The genuine signal is a Fitzpatrick **I-II vs III-IV** gap (TPR Δ ≈ 0.23), not
anything involving the unknown cohort.

## How stage 8 handles it

`src/fairxai/fairness/image_assessment.py`:

1. **Min-group gating** drops groups below `min_group_samples` (default 50) — e.g.
   Fitzpatrick `V-VI` (n=5). Reported under `skipped_groups`.
2. **Degenerate-group exclusion** removes groups with no positives or no negatives
   from the group-difference deltas (DP / TPR / FPR / equal-opportunity), while
   **keeping them in the per-group performance table** (with `recall`/`auc` as
   `n/a`, not a misleading 0). Reported under `degenerate_groups`.

Net effect: the unknown cohort stays **visible** (you can see it is 33% of test and
all benign) but does **not corrupt** the headline fairness numbers.

## Confirmed root cause — biopsy-driven missingness (not a pipeline bug)

Traced to the original PAD-UFES-20 metadata (`metadata.csv`, 2298 rows). The
no-demographics rows are all benign **by construction of the dataset**, not by any
FairXAI processing:

- `gender` missing: **804** rows. `fitspatrick` missing: **804** rows. They are the
  **same 804 rows** (`gender.isna() == fitspatrick.isna()` holds exactly).
- **All 804 have `biopsed == False`.** Zero biopsied lesions are missing demographics
  (`gender.isna() & biopsed` = 0).
- Every non-biopsied lesion is benign: `diagnostic ∈ {ACK, SEK, NEV}` for all
  `biopsed == False`. The cancer labels (BCC, SCC, MEL → `skin_cancer = 1`) are
  **always** biopsy-confirmed and therefore fully annotated.

Mechanism: suspicious/cancerous lesions are biopsied and fully recorded; clinically
obvious benign lesions (actinic keratosis, seborrheic keratosis, nevus) are often
not biopsied and ship with missing demographic fields. So in PAD-UFES:

> **missing demographics ⟺ non-biopsied ⟹ benign.**

The standardized CSV maps the missing gender to `sex = -1` / `sex_extended =
"unknown"` and missing Fitzpatrick to `"unknown"`; the all-benign outcome of that
cohort is inherited from the source, faithfully. There is nothing to fix in
`loaders.py` / `preprocessors.py`.

The 804 is the full-dataset cohort; **228** of them fall in the 682-row test split
(the number stage 8 reports).

## Methodological statement for the write-up

Fairness results are reported over groups with adequate support and **both outcome
classes present**. The missing-demographics cohort (a biopsy-sampling artifact of
PAD-UFES, perfectly correlated with the benign class) is described separately and
**excluded from group-difference metrics** because its TPR/FPR are undefined.
Including it would report a spurious ~0.7 TPR gap that is an artifact of the
sampling, not model behaviour.
