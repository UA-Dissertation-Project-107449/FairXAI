# Cardiac pipeline — loky "worker stopped" warning (stage 11 combinatorial)

Status: **investigation only, not fixed.** Captures what the warning is, why it appears
where it does, and the candidate fixes. No code changed.

## The warning

```
WARNING - [UserWarning] .../joblib/externals/loky/process_executor.py:782: UserWarning:
A worker stopped while some jobs were given to the executor. This can be caused by a too
short worker timeout or by a memory leak.
  warnings.warn(...)
```

Observed:
- Stage **11 combinatorial** (`scripts/cardiac/combinatorial.py` → `scripts/experiments/run_combinatorial_experiments.py`).
- Fires **once per enabled dataset**.
- Only when **logistic regression** experiments run.

This is **not** the Python 3.12 `fork()` DeprecationWarning seen in the image pipeline. It is a
distinct loky message: a worker **process terminated while it still had tasks assigned**.

## What it actually means

loky (joblib's default process pool) detected that one of its worker processes exited
unexpectedly — return code non-zero or killed by the OS — while the executor still had pending
or in-flight jobs for it. loky transparently relaunches a fresh worker and the batch usually
still completes, which is why the run finishes "successfully" with only a warning. The message
itself lists the two usual triggers:

1. **Memory** — the worker grew too large and was **OOM-killed** by the kernel (or hit a cgroup
   limit), or a "memory leak" (large objects retained across tasks).
2. **Timeout** — the worker idle/exec timeout was too short and loky reaped it mid-flight.

## Why logistic regression, and why once per dataset

Mitigation is **only applied to the logistic-regression baseline**. From
`scripts/experiments/run_combinatorial_experiments.py`:

```
# Mitigation engine currently assumes logistic baseline for pre/in/post mitigation.
# This set is now config-driven (mitigation_supported_model_types in combinatorial.yaml).
```

So in the per-dataset combinatorial matrix, the LR row is the only one that runs the
**fairlearn reduction** in-processing mitigations:

- `ExponentiatedGradient` — `src/fairxai/fairness/mitigation.py:231`
- `GridSearch` — `src/fairxai/fairness/mitigation.py:291`

Both wrap `LogisticRegression` and internally fit **many** LR models (an ensemble / a grid),
holding fitted predictors in memory (`predictors_`). That single experiment is by far the
heaviest in the matrix for memory and time. It runs once per dataset → the warning appears once
per dataset, attached to LR.

## How the parallelism is wired

The only joblib parallelism in the combinatorial stage is the **outer experiment loop**:

```python
# scripts/experiments/run_combinatorial_experiments.py:1628
results = Parallel(n_jobs=n_jobs, verbose=parallel_verbose)(
    delayed(run_single_experiment)(...) for ... )
```

`n_jobs` comes from the combinatorial config (`config.get("n_jobs", 1)`, line ~1446). When
`n_jobs > 1`, loky spawns worker processes, each running one full experiment — **including the
heavy LR fairlearn-reduction experiment**. There is an oversubscription guard for tree models
only:

```python
# _resolve_model_n_jobs: RF/XGBoost get n_jobs=1 under outer parallelism (lines 187-200, 383-395)
return 1 if outer_n_jobs > 1 else -1
```

LR / fairlearn reductions are **not** covered by that guard, and BLAS/OpenMP threads from
numpy/sklearn inside each worker are not pinned either.

> Note: `configs/pipelines/cardiac.yaml` sets `scheduling.parallel_experiments: false`. If the
> combinatorial stage honors that (n_jobs=1), there would be no loky pool and the warning would
> have to come from a nested loky use instead. **First confirmation step is to check the actual
> `n_jobs` the combinatorial run uses** (combinatorial config + log line
> `"Parallel jobs: {n_jobs}"`, ~line 1596). The warning text points squarely at the loky pool, so
> the most likely reality is that the combinatorial stage runs with `n_jobs > 1` from its own
> config, independent of `parallel_experiments`.

## Most likely root cause

A loky worker running the **LR `ExponentiatedGradient` / `GridSearch` experiment** is
**OOM-killed** (memory spike from the ensemble of fitted LR predictors), possibly aggravated by
**thread oversubscription** (each worker spawns BLAS threads × N workers) competing for RAM and
cores. loky relaunches the worker, so results complete, but the spike trips the warning once per
dataset.

Severity: **usually benign** (run completes), but it signals real memory pressure and wasted work
(the killed worker's partial progress is redone). Worth removing for clean, reproducible runs.

## How to confirm

1. **Check parallelism is on:** combinatorial `n_jobs` value + the `"Parallel jobs: N"` log line.
2. **Check for OOM:** `dmesg -T | grep -i -E "killed process|oom"` right after a run, or watch
   `watch -n1 free -g` / a per-process memory monitor during stage 11.
3. **Run the LR mitigation experiment serially:** force `n_jobs=1` for the combinatorial stage —
   if the warning disappears, it is the parallel-worker memory/oversubscription path.
4. **Pin BLAS threads:** run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` —
   if it disappears, oversubscription was a contributor.

## Candidate fixes (not implemented)

Ranked by likely effectiveness / lowest risk:

1. **Pin worker threads** — set `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` /
   `MKL_NUM_THREADS=1` for the combinatorial workers (or wrap the `Parallel` block in
   `threadpoolctl.threadpool_limits(1)`). Removes thread×process oversubscription, the cheapest
   and safest lever.
2. **Extend the oversubscription guard to LR mitigation** — apply the existing
   `_resolve_model_n_jobs` logic (n_jobs=1 under outer parallelism) to the LR base estimator and
   ensure fairlearn reductions are not internally parallel. Directly targets the heavy experiment.
3. **Cap the heavy experiment's memory** — lower `GridSearch` `grid_size`
   (`mitigation.py:291`) and/or `ExponentiatedGradient` iterations, so the retained ensemble of
   predictors is smaller.
4. **Give the pool headroom** — reduce combinatorial `n_jobs` so each worker has more RAM, or
   schedule the LR mitigation experiment on its own (it is the matrix's memory outlier).
5. **Raise the loky timeout** — `Parallel(..., timeout=...)` / loky idle timeout, only if step 2
   in "How to confirm" shows reaping (not OOM) is the trigger.

## Key references

- `scripts/experiments/run_combinatorial_experiments.py:1628` — outer `Parallel(n_jobs=...)`.
- `scripts/experiments/run_combinatorial_experiments.py:187-200, 383-395` — `_resolve_model_n_jobs`
  (tree-model-only oversubscription guard).
- `scripts/experiments/run_combinatorial_experiments.py:66-67` — mitigation = LR baseline only.
- `src/fairxai/fairness/mitigation.py:231, 291` — `ExponentiatedGradient`, `GridSearch`.
