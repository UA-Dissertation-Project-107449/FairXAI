#!/bin/bash
#
# FairXAI HPC bootstrap / update script for the Pleiades cluster (IEETA).
#
# Usage (on the HPC login node, university network only):
#   bash setup_hpc.sh                    # full bootstrap: clone/pull + venv + install + dirs
#   bash setup_hpc.sh --update           # fast path: git pull + reinstall package only
#   bash setup_hpc.sh --with-cuml        # also install cuml-cu12 (needs HPC_MODULES=<cuda module>)
#   bash setup_hpc.sh --branch <name>    # clone or switch to that branch (env: FAIRXAI_BRANCH)
#
# Without --branch the clone lands on the remote's default branch and an
# existing checkout is left on whatever it is already on. With --branch, the
# clone checks that branch out directly and an existing checkout is switched to
# it before pulling, so `--update` keeps following it on later runs.
#
# Idempotent: safe to re-run. Echoes the resolved values needed for the
# WebApp .env (HPC_PROJ_ROOT / HPC_DATASETS_DIR / HPC_RESULTS_DIR /
# HPC_VENV_PATH / HPC_SLURM_DIR / HPC_MODULES) at the end.
set -euo pipefail

# --- config (override via env) ---------------------------------------------
PROJ_ROOT="${HPC_PROJ_ROOT:-$HOME/storage}"   # symlink -> /beegfs/.../proj-datalenzai
FAIRXAI_REPO="${FAIRXAI_REPO:-}"               # git URL; required on first bootstrap
FAIRXAI_HOME="${FAIRXAI_HOME:-$PROJ_ROOT/FairXAI}"
FAIRXAI_BRANCH="${FAIRXAI_BRANCH:-}"           # empty = remote default / leave as-is
# Extra modules to load (e.g. a cuda module for --with-cuml). The python module
# is NOT set here — the script probes for one that works, see below.
HPC_MODULES="${HPC_MODULES:-}"
CUML_VERSION="${CUML_VERSION:-25.2.1}"

UPDATE_ONLY=0
WITH_CUML=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --update) UPDATE_ONLY=1 ;;
        --with-cuml) WITH_CUML=1 ;;
        --branch)
            [ "$#" -ge 2 ] || { echo "ERROR: --branch needs a branch name." >&2; exit 2; }
            FAIRXAI_BRANCH="$2"; shift ;;
        --branch=*) FAIRXAI_BRANCH="${1#--branch=}" ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

echo "==> PROJ_ROOT:    $PROJ_ROOT"
echo "==> FAIRXAI_HOME: $FAIRXAI_HOME"
if [ -n "$FAIRXAI_BRANCH" ]; then
    echo "==> BRANCH:       $FAIRXAI_BRANCH"
fi

# --- clone or pull ----------------------------------------------------------
if [ -d "$FAIRXAI_HOME/.git" ]; then
    # Switch first, then pull, so --update follows the requested branch rather
    # than whatever the checkout happened to be left on.
    if [ -n "$FAIRXAI_BRANCH" ]; then
        current="$(git -C "$FAIRXAI_HOME" rev-parse --abbrev-ref HEAD)"
        if [ "$current" != "$FAIRXAI_BRANCH" ]; then
            echo "==> Switching branch: $current -> $FAIRXAI_BRANCH"
            git -C "$FAIRXAI_HOME" fetch origin "$FAIRXAI_BRANCH"
            git -C "$FAIRXAI_HOME" checkout "$FAIRXAI_BRANCH"
        fi
        # Name the remote branch explicitly. A local branch can be tracking
        # something else entirely — a bare `git pull` would then quietly update
        # from whatever that is, not from the branch that was asked for.
        echo "==> Existing checkout — git pull origin $FAIRXAI_BRANCH"
        # --ff-only refuses after a rebase or force-push upstream. That is on
        # purpose: recover with `git reset --hard origin/<branch>` once you have
        # looked at what diverged, rather than having this script discard work.
        git -C "$FAIRXAI_HOME" pull --ff-only origin "$FAIRXAI_BRANCH"
        git -C "$FAIRXAI_HOME" branch --set-upstream-to="origin/$FAIRXAI_BRANCH" \
            "$FAIRXAI_BRANCH" >/dev/null
    else
        echo "==> Existing checkout — git pull"
        git -C "$FAIRXAI_HOME" pull --ff-only
    fi
else
    if [ -z "$FAIRXAI_REPO" ]; then
        echo "ERROR: $FAIRXAI_HOME is not a git checkout and FAIRXAI_REPO is unset." >&2
        echo "       Set FAIRXAI_REPO=<git url> for the first bootstrap." >&2
        exit 1
    fi
    echo "==> Cloning $FAIRXAI_REPO -> $FAIRXAI_HOME"
    mkdir -p "$(dirname "$FAIRXAI_HOME")"
    if [ -n "$FAIRXAI_BRANCH" ]; then
        # Not --single-branch: the other branches cost nothing to fetch and
        # keep a later `--branch <other>` from needing a fresh clone.
        git clone --branch "$FAIRXAI_BRANCH" "$FAIRXAI_REPO" "$FAIRXAI_HOME"
    else
        git clone "$FAIRXAI_REPO" "$FAIRXAI_HOME"
    fi
fi

cd "$FAIRXAI_HOME"

# --- load modules -----------------------------------------------------------
# After the pull, never before: a module that stops resolving must not be able to
# block the update that removes it. That is exactly how the 2026 rebuild left this
# script unable to fix itself.
if [ -z "${HPC_MODULES//[[:space:]]/}" ]; then
    :
elif ! command -v module >/dev/null 2>&1; then
    echo "WARNING: 'module' not found — skipping module load (mock/local env?)" >&2
else
    echo "==> module load $HPC_MODULES"
    # Not fatal. The venv usually builds fine without whatever failed.
    # shellcheck disable=SC2086
    module load $HPC_MODULES || {
        echo "WARNING: module load failed — run 'module spider <name>' for the" >&2
        echo "         current name on this cluster." >&2
    }
fi

# --- pick an interpreter that can build a venv ------------------------------
# Probed, not hardcoded. Neither source is dependable on Pleiades:
#   - the system python3 is Ubuntu's, which splits ensurepip into a
#     python3.12-venv package that is not installed and needs root;
#   - Spack module names carry a hash that changes on every cluster rebuild,
#     and some are broken — after the 2026 one `python/3.11.14-cyf54tg`
#     resolves while its own dependencies (libxcrypt, util-linux-uuid) do not.
# So: try what is on PATH, then every python module oldest-first (oldest =
# best wheel coverage; a module that fails to load just costs a second).
VENV_MODULE=""   # module that owns the chosen interpreter; empty = python3 on PATH

python_builds_venv() {
    # requires-python is >=3.11, and `python3 -m venv` needs ensurepip.
    python3 -c 'import sys, importlib.util as u
sys.exit(0 if sys.version_info >= (3, 11) and u.find_spec("ensurepip") else 1)' 2>/dev/null
}

list_python_modules() {
    command -v module >/dev/null 2>&1 || return 0
    module spider python 2>&1 \
        | grep -oE 'python/[0-9]+\.[0-9]+[^[:space:]]*' \
        | sort -Vu
}

if python_builds_venv; then
    echo "==> Interpreter: $(command -v python3) ($(python3 -V 2>&1))"
else
    echo "==> python3 on PATH cannot build a venv — probing python modules"
    for candidate in $(list_python_modules || true); do
        module load "$candidate" >/dev/null 2>&1 || { echo "    $candidate: load failed"; continue; }
        if python_builds_venv; then
            VENV_MODULE="$candidate"
            echo "==> Interpreter: module $candidate ($(python3 -V 2>&1))"
            break
        fi
        echo "    $candidate: no ensurepip / too old"
        module unload "$candidate" >/dev/null 2>&1 || true
    done
fi

# --- venv (single repo-root .venv, per CLAUDE.md) ---------------------------
VENV_PATH="$FAIRXAI_HOME/.venv"
# A venv's bin/python3 is a symlink into whatever built it. When that interpreter
# goes away — a retired module, an OS upgrade — activate still succeeds and every
# command after it fails. Rebuild instead, on --update too.
if [ -d "$VENV_PATH" ] && ! "$VENV_PATH/bin/python3" -V >/dev/null 2>&1; then
    echo "==> Venv at $VENV_PATH has no working interpreter — rebuilding"
    rm -rf "$VENV_PATH"
# Same treatment for a pip-less venv: `--without-pip`, or a bootstrap that died
# halfway. Everything below this point needs pip.
elif [ -d "$VENV_PATH" ] && ! "$VENV_PATH/bin/python3" -m pip --version >/dev/null 2>&1; then
    echo "==> Venv at $VENV_PATH has no pip — rebuilding"
    rm -rf "$VENV_PATH"
fi
if [ ! -d "$VENV_PATH" ]; then
    # Checked here because the alternative is a confusing pip resolution
    # failure several minutes into the install.
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
        echo "ERROR: python3 is $(python3 -V 2>&1), FairXAI needs >= 3.11." >&2
        echo "       No python module worked either — check 'module spider python'." >&2
        exit 1
    fi
    echo "==> Creating venv at $VENV_PATH ($(python3 -V 2>&1))"
    if ! python3 -m venv "$VENV_PATH"; then
        # Nothing on this node has ensurepip. Build the venv empty and pull pip
        # into it from pypa — same bytes ensurepip would have unpacked, just
        # over the network instead of from the (missing) system package.
        echo "==> venv creation failed — retrying without pip and bootstrapping it"
        rm -rf "$VENV_PATH"
        python3 -m venv --without-pip "$VENV_PATH"
        curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$VENV_PATH/bin/python3" || {
            echo "ERROR: could not bootstrap pip into $VENV_PATH." >&2
            echo "       No ensurepip and no reachable bootstrap.pypa.io. Load a" >&2
            echo "       python module that has ensurepip, or copy a pip wheel over." >&2
            rm -rf "$VENV_PATH"
            exit 1
        }
    fi
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
fairxai characterize --help >/dev/null && echo "fairxai characterize CLI OK"
fairxai triage --help >/dev/null && echo "fairxai triage CLI OK"

# --- summary for WebApp .env ------------------------------------------------
# A venv built from a module python needs that module loaded to run at all: its
# bin/python3 is a symlink into the Spack prefix and the shared libraries come
# from the module's environment. So the jobs need it too, not just this shell.
JOB_MODULES="$(echo "${HPC_MODULES} ${VENV_MODULE}" | xargs || true)"

cat <<EOF

================ FairXAI HPC ready ================
Checked out: $(git -C "$FAIRXAI_HOME" rev-parse --abbrev-ref HEAD) @ $(git -C "$FAIRXAI_HOME" rev-parse --short HEAD)

Fill these into the WebApp .env (cluster_gateway):

  HPC_PROJ_ROOT=$PROJ_ROOT
  HPC_DATASETS_DIR=$DATASETS_DIR
  HPC_RESULTS_DIR=$RESULTS_DIR
  HPC_VENV_PATH=$VENV_PATH
  HPC_SLURM_DIR=$FAIRXAI_HOME/hpc
  HPC_MODULES=$JOB_MODULES

Also register this host's SSH key on the WebApp side:
  ssh-keyscan <this-hostname> >> known_hosts
===================================================
EOF
