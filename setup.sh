#!/usr/bin/env bash
# setup.sh — Bootstrap the FairXAI development environment.
#
# What it does:
#   1. Checks for Python ≥ 3.10
#   2. Creates a virtual environment (.venv) if it doesn't exist
#   3. Activates the venv and installs requirements.txt
#   4. Prints a success summary
#
# Usage:
#   ./setup.sh              # default — create .venv in project root
#   ./setup.sh /path/to/venv  # custom venv location

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${1:-${SCRIPT_DIR}/.venv}"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
info()  { echo -e "\033[1;34m[setup]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[setup]\033[0m $*"; }
err()   { echo -e "\033[1;31m[setup]\033[0m $*" >&2; }

# ------------------------------------------------------------------
# 1. Locate Python ≥ 3.10
# ------------------------------------------------------------------
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major="${ver%%.*}"
        minor="${ver##*.}"
        if (( major > MIN_PYTHON_MAJOR || (major == MIN_PYTHON_MAJOR && minor >= MIN_PYTHON_MINOR) )); then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    err "Python >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} is required but was not found on PATH."
    err "Install it and re-run this script."
    exit 1
fi

info "Using $PYTHON ($("$PYTHON" --version 2>&1))"

# ------------------------------------------------------------------
# 2. Create virtual environment
# ------------------------------------------------------------------
if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists at ${VENV_DIR}"
else
    info "Creating virtual environment at ${VENV_DIR} ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# ------------------------------------------------------------------
# 3. Activate and install
# ------------------------------------------------------------------
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

info "Upgrading pip ..."
pip install --upgrade pip --quiet

if [[ -f "$REQUIREMENTS" ]]; then
    info "Installing requirements from ${REQUIREMENTS} ..."
    pip install -r "$REQUIREMENTS" --quiet
else
    err "requirements.txt not found at ${REQUIREMENTS} — skipping install."
fi

# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
echo ""
ok "Environment ready!"
ok "  Python : $(python --version 2>&1)"
ok "  venv   : ${VENV_DIR}"
ok "  pip    : $(pip --version 2>&1 | awk '{print $2}')"
echo ""
ok "Activate with:  source ${VENV_DIR}/bin/activate"
