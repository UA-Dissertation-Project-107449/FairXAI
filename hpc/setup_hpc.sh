#!/bin/bash
#
# FairXAI HPC bootstrap / update script for the Pleiades cluster (IEETA).
#
# Usage (on the HPC login node, university network only):
#   bash setup_hpc.sh              # full bootstrap: clone/pull + venv + install + dirs
#   bash setup_hpc.sh --update     # fast path: git pull + reinstall package only
#   bash setup_hpc.sh --with-cuml  # also install cuml-cu12 (GPU accel; off by default)
#
# Idempotent: safe to re-run. Echoes the resolved paths needed for the
# WebApp .env (HPC_PROJ_ROOT / HPC_DATASETS_DIR / HPC_RESULTS_DIR /
# HPC_VENV_PATH / HPC_SLURM_DIR) at the end.
set -euo pipefail

# --- config (override via env) ---------------------------------------------
PROJ_ROOT="${HPC_PROJ_ROOT:-$HOME/storage}"   # symlink -> /beegfs/.../proj-datalenzai
FAIRXAI_REPO="${FAIRXAI_REPO:-}"               # git URL; required on first bootstrap
FAIRXAI_HOME="${FAIRXAI_HOME:-$PROJ_ROOT/FairXAI}"
HPC_MODULES="${HPC_MODULES:-python/3.11.7 cuda/12.4.0}"
CUML_VERSION="${CUML_VERSION:-25.2.1}"

UPDATE_ONLY=0
WITH_CUML=0
for arg in "$@"; do
    case "$arg" in
        --update) UPDATE_ONLY=1 ;;
        --with-cuml) WITH_CUML=1 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

echo "==> PROJ_ROOT:    $PROJ_ROOT"
echo "==> FAIRXAI_HOME: $FAIRXAI_HOME"

# --- load modules -----------------------------------------------------------
if command -v module >/dev/null 2>&1; then
    echo "==> module load $HPC_MODULES"
    # shellcheck disable=SC2086
    module load $HPC_MODULES
else
    echo "WARNING: 'module' not found — skipping module load (mock/local env?)" >&2
fi

# --- clone or pull ----------------------------------------------------------
if [ -d "$FAIRXAI_HOME/.git" ]; then
    echo "==> Existing checkout — git pull"
    git -C "$FAIRXAI_HOME" pull --ff-only
else
    if [ -z "$FAIRXAI_REPO" ]; then
        echo "ERROR: $FAIRXAI_HOME is not a git checkout and FAIRXAI_REPO is unset." >&2
        echo "       Set FAIRXAI_REPO=<git url> for the first bootstrap." >&2
        exit 1
    fi
    echo "==> Cloning $FAIRXAI_REPO -> $FAIRXAI_HOME"
    mkdir -p "$(dirname "$FAIRXAI_HOME")"
    git clone "$FAIRXAI_REPO" "$FAIRXAI_HOME"
fi

cd "$FAIRXAI_HOME"

# --- venv (single repo-root .venv, per CLAUDE.md) ---------------------------
VENV_PATH="$FAIRXAI_HOME/.venv"
if [ "$UPDATE_ONLY" -eq 0 ] && [ ! -d "$VENV_PATH" ]; then
    echo "==> Creating venv at $VENV_PATH"
    python3 -m venv "$VENV_PATH"
fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
python3 -m pip install --upgrade pip >/dev/null

echo "==> pip install -e .[experiment]"
pip install -e ".[experiment]"

if [ "$WITH_CUML" -eq 1 ]; then
    echo "==> pip install cuml-cu12==$CUML_VERSION (GPU acceleration)"
    pip install "cuml-cu12==$CUML_VERSION"
fi

# --- work dirs (skip on --update) -------------------------------------------
DATASETS_DIR="$PROJ_ROOT/datasets"
RESULTS_DIR="$PROJ_ROOT/results"
if [ "$UPDATE_ONLY" -eq 0 ]; then
    echo "==> Creating work dirs"
    mkdir -p "$DATASETS_DIR" "$RESULTS_DIR"
fi

# --- smoke test -------------------------------------------------------------
echo "==> Smoke test"
python3 -c "from fairxai.profiling import characterize_dataset; print('import OK')"
fairxai-characterize --help >/dev/null && echo "fairxai-characterize CLI OK"

# --- summary for WebApp .env ------------------------------------------------
cat <<EOF

================ FairXAI HPC ready ================
Fill these into the WebApp .env (cluster_gateway):

  HPC_PROJ_ROOT=$PROJ_ROOT
  HPC_DATASETS_DIR=$DATASETS_DIR
  HPC_RESULTS_DIR=$RESULTS_DIR
  HPC_VENV_PATH=$VENV_PATH
  HPC_SLURM_DIR=$FAIRXAI_HOME/hpc

Also register this host's SSH key on the WebApp side:
  ssh-keyscan <this-hostname> >> known_hosts
===================================================
EOF
