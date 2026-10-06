# Dissertation Evidence Check

Snapshot: 2026-10-06, after the cold re-run behind Chapters 6 and 7.

This note records how to read the evidence, not the numbers. The numbers live
in the runs named by `scripts/thesis/runs.yaml` and in the ledger that
`scripts/thesis/build_ledger.py` writes to `output/thesis/chapter6_ledger.md`. Change the
run IDs in `runs.yaml` after a rerun, then rebuild the ledger and figures; do not
copy values into this file.

## Reading Fairness Numbers

- A max-gap interval is descriptive. It is bounded below by zero, so its
  interval excludes zero even on data with no disparity. The test is the
  pairwise group difference with its Benjamini-Hochberg adjustment
  (`<stem>_pairwise_ci.csv`). See
  [decisions.md](../architecture/decisions.md#fairness-intervals-max-gap-cis-are-descriptive-pairwise-differences-are-the-test).
- A mitigation arm is judged against its own baseline with the paired bootstrap
  (`paired_effects.csv`), not by comparing two separate intervals.
- `scripts/studies/run_bootstrap_calibration.py` is the null-calibration study
  behind both rules.
- Missing probability scores make AUC unavailable, not zero. A comparison table
  must never show an AUC drop caused by a missing score.

## Mitigation

- Exponentiated Gradient closes the largest share of most gaps and costs the
  most F1. Treat its gains as a trade, not a free improvement.
- The threshold rule (ThresholdOptimizer) closes less, usually at a much lower
  cost.
- Reweighting and the group-aware resamplers move a gap significantly only when
  it is very large. Grid search is erratic and can widen the gap it targets.
- A constraint on one attribute can move disparity to another criterion or to an
  attribute the arm was not asked about. Read every arm across all attributes and
  criteria, not only the one it constrains.
- Small cohorts such as Cleveland give wide intervals and depend on the age
  bands, so few arms reach significance there.

## Clustering

Clusters are exploratory subgroups, not protected groups. A cluster gap is not
a fairness result on its own: clusters differ in outcome rate by construction,
so parity across them is reached only by ignoring the patient profile. Use
clusters to locate the patients the models cannot separate and to examine error
rates within a cluster. The validity gate only checks the partition internally.

## Figures

`scripts/thesis/make_figures.py` writes the thesis figures to
`output/thesis/figures/`; the full set is listed in
[plots.md](../reference/plots.md#thesis-figures). Caption the cluster-gap figure
(`fig_cluster_gaps`) as exploratory subgroup evidence.
