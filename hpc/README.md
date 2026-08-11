# HPC Module

SLURM scripts and bootstrap tooling that run FairXAI analyses on the
Pleiades cluster (IEETA), driven remotely by the WebApp `cluster_gateway`.

## Files

| File | Purpose |
|------|---------|
| `setup_hpc.sh` | Bootstrap (clone/pull + venv + install + work dirs) and `--update` mode |
| `characterize.slurm` | Dataset characterization job → `<job_id>.json` |
| `analysis.slurm` | Generic post-hoc job: binning, clustering, or triage |

## How it fits together

```
WebApp (VM)                         Pleiades (HPC)
-----------                         --------------
cluster_gateway  --scp dataset-->   $HPC_DATASETS_DIR/<job_id>.csv
        |        --sbatch-------->   characterize.slurm / analysis.slurm
        |        <--poll sacct---            |
        |                                   writes
        |        <--scp result----   $HPC_RESULTS_DIR/<job_id>*.json
   apply to Job DB + notify
```

The WebApp runs in `RUN_MODE=hpc`. All heavy analysis runs on the
cluster; the VM only orchestrates (scp + sbatch + poll + scp back) and
persists results. See `cluster_gateway/main.py` and
`cluster_gateway/hpc_runner.py` in the WebApp repo.

## Setup

On the HPC login node (university network only), first bootstrap:

```bash
cd ~/storage                       # -> /beegfs/client/default/storage/proj-datalenzai
FAIRXAI_REPO=<git-url> bash FairXAI/hpc/setup_hpc.sh
```

Later updates:

```bash
bash ~/storage/FairXAI/hpc/setup_hpc.sh --update
```

`setup_hpc.sh` uses the single repo-root `.venv` (per CLAUDE.md),
`module load python/3.11.14-cyf54tg`, and `pip install -e ".[experiment]"`.

Pleiades module names carry a Spack hash and do not survive a cluster rebuild:
the 2026 move to Ubuntu 24.04 retired `python/3.11.7` and `cuda/12.4.0`. When
that happens, run `module spider python`, set `HPC_MODULES`, and re-run this
script — the venv's interpreter is a symlink into the old module's prefix, so
it dies with the module and has to be rebuilt, not just re-pointed.

**cuML is off by default**: there is no rapids/cuml module on Pleiades and
FairXAI falls back to CPU without it. For GPU, pass `--with-cuml` with
`cuda/12.8.1-kzpbyf7` in `HPC_MODULES`.

The script prints the exact `HPC_*` values to copy into the WebApp `.env`.

## SLURM job env contract

Both scripts are parametrized entirely by env vars (passed via
`sbatch --export=ALL,VAR=...`). SLURM jobs start a fresh shell, so each
script activates the venv itself. Neither loads a module unless `HPC_MODULES`
is set; the venv carries its own interpreter.

`characterize.slurm` — required: `DATASET_PATH`, `RESULTS_DIR`.
Optional: `FAIRXAI_VENV`, `HPC_MODULES`, `TARGET_COLUMN`, `INDEX_COLUMN`.
Writes `<RESULTS_DIR>/<dataset-stem>.json`. Characterization only — triage
is submitted separately as `analysis.slurm` with `ANALYSIS_TYPE=triage`.

`analysis.slurm` — required: `ANALYSIS_TYPE` (`binning|clustering|triage`),
`DATASET_PATH`, `RESULT_FILE`, `TARGET_COLUMN`.
`binning` also needs `ATTRIBUTE`, `STRATEGY`; `clustering` also needs
`METHOD` (optional `PCA2D_FILE`); `clustering` and `triage` both accept
`INDEX_COLUMN` and `SENSITIVE_COLUMNS_JSON`. Writes a single JSON file to
`RESULT_FILE`.

`SENSITIVE_COLUMNS_JSON` is a JSON array, not a space-separated list — a column
named `race group` has to survive as one argument. `PCA2D_FILE` holds
`{"points": [...], "feature_columns": [...]}`; the column list is what lets
FairXAI decide whether the stored projection covers the features it clustered
on, and a bare coordinate list reads as "unknown" and is recomputed.

## Resources

Pleiades partitions (`Older_maybe_useful/hpc_info.txt`):

| Node | GPU | Mem | Note |
|------|-----|-----|------|
| `gpu-srv-02` | RTX A2000 | ~14 GB | default `--gres=gpu:nvidia-rtx-a2000`, mem tight |
| `gpu-srv-03` | 2× RTX A6000 | ~248 GB | use `--gres=gpu:nvidia-rtx-a6000` for big datasets |

Storage: `~/storage` → `/beegfs/client/default/storage/proj-datalenzai`.

## Notes

- **One analysis per job** (no batching) — requests are delivered as they
  arrive; this also avoids mixing different users' data in one job.
- `datasets/` and `results/` retention is currently unbounded on beegfs.
  Acceptable for this version; a cleanup policy is future work.
- `RUN_MODE` is set on the WebApp `cluster_gateway` at startup; switch
  modes by restarting that container. Per-request mode switching is a
  possible future enhancement.

## Related

- WebApp integration: `Code/WebApp_DataLenzAI/backend/backend/cluster_gateway/`
- Offline testing without the cluster: `compose.hpc-mock.yaml` (WebApp repo)
- CLI used by these jobs: [../src/fairxai/cli/README.md](../src/fairxai/cli/README.md)
- Integration adapters: [../src/fairxai/integration/README.md](../src/fairxai/integration/README.md)
