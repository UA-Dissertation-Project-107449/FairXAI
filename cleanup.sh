#!/usr/bin/env bash
# cleanup.sh — Remove generated outputs from the FairXAI workspace.
#
# Default targets (all enabled unless flags say otherwise):
#   output/       — experiment runs, recommendations, …
#   data/processed/ — pre-processed CSVs
#   data/raw/       — downloaded raw data  (data/external/ is NEVER touched)
#   logs/           — execution logs
#
# Flags:
#   --output-only    Only remove output/ (skip data/ and logs/)
#   --keep-latest    Preserve the run pointed to by output/*/latest_run
#   --nuke-env       Also remove the virtual environment (.venv)
#   --dry-run        Show what would be deleted without deleting anything
#   -y / --yes       Skip confirmation prompt
#
# Usage examples:
#   ./cleanup.sh                     # default — remove all four targets
#   ./cleanup.sh --output-only       # only output/
#   ./cleanup.sh --keep-latest       # keep latest run per pipeline
#   ./cleanup.sh --dry-run           # preview only
#   ./cleanup.sh --nuke-env -y       # scorched earth, no prompt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
OUTPUT_ONLY=false
KEEP_LATEST=false
NUKE_ENV=false
DRY_RUN=false
AUTO_YES=false

# ------------------------------------------------------------------
# Parse flags
# ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-only)  OUTPUT_ONLY=true  ;;
        --keep-latest)  KEEP_LATEST=true  ;;
        --nuke-env)     NUKE_ENV=true     ;;
        --dry-run)      DRY_RUN=true      ;;
        -y|--yes)       AUTO_YES=true     ;;
        *)
            echo "Unknown flag: $1" >&2
            echo "Usage: $0 [--output-only] [--keep-latest] [--nuke-env] [--dry-run] [-y]" >&2
            exit 1
            ;;
    esac
    shift
done

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
info()    { echo -e "\033[1;34m[cleanup]\033[0m $*"; }
ok()      { echo -e "\033[1;32m[cleanup]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[cleanup]\033[0m $*"; }
err()     { echo -e "\033[1;31m[cleanup]\033[0m $*" >&2; }

# Remove a path (file or directory). Respects DRY_RUN.
nuke() {
    local target="$1"
    if [[ ! -e "$target" ]]; then
        return
    fi
    if $DRY_RUN; then
        info "[dry-run] would remove: ${target}"
    else
        rm -rf "$target"
        ok "Removed: ${target}"
    fi
}

# ------------------------------------------------------------------
# Build target list
# ------------------------------------------------------------------
TARGETS=()

# output/ — always included
if [[ -d "${SCRIPT_DIR}/output" ]]; then
    if $KEEP_LATEST; then
        # Remove everything inside each pipeline's runs/ EXCEPT the latest
        for pipeline_dir in "${SCRIPT_DIR}"/output/*/; do
            [[ -d "$pipeline_dir" ]] || continue
            pipeline_name="$(basename "$pipeline_dir")"

            # Resolve latest_run symlink
            latest_link="${pipeline_dir}latest_run"
            latest_target=""
            if [[ -L "$latest_link" ]]; then
                latest_target="$(readlink -f "$latest_link")"
            fi

            # Clean runs/
            runs_dir="${pipeline_dir}runs"
            if [[ -d "$runs_dir" ]]; then
                for run in "${runs_dir}"/*/; do
                    run_real="$(readlink -f "$run")"
                    if [[ -n "$latest_target" && "$run_real" == "$latest_target" ]]; then
                        info "Keeping latest run: ${run} (${pipeline_name})"
                    else
                        TARGETS+=("$run")
                    fi
                done
            fi

            # Clean archived_runs/
            archived_dir="${pipeline_dir}archived_runs"
            if [[ -d "$archived_dir" ]]; then
                TARGETS+=("$archived_dir")
            fi

            # Clean recommendations/
            recs_dir="${pipeline_dir}recommendations"
            if [[ -d "$recs_dir" ]]; then
                TARGETS+=("$recs_dir")
            fi

            # Clean run_history.jsonl
            hist_file="${pipeline_dir}run_history.jsonl"
            if [[ -f "$hist_file" ]]; then
                TARGETS+=("$hist_file")
            fi
        done
    else
        TARGETS+=("${SCRIPT_DIR}/output")
    fi
fi

if ! $OUTPUT_ONLY; then
    # data/processed/
    [[ -d "${SCRIPT_DIR}/data/processed" ]] && TARGETS+=("${SCRIPT_DIR}/data/processed")

    # data/raw/
    [[ -d "${SCRIPT_DIR}/data/raw" ]] && TARGETS+=("${SCRIPT_DIR}/data/raw")

    # logs/ — honour --keep-latest for per-run log dirs
    if [[ -d "${SCRIPT_DIR}/logs" ]]; then
        if $KEEP_LATEST; then
            for pipeline_log_dir in "${SCRIPT_DIR}"/logs/*/; do
                [[ -d "$pipeline_log_dir" ]] || continue
                latest_link="${pipeline_log_dir}latest_run"
                latest_target=""
                if [[ -L "$latest_link" ]]; then
                    latest_target="$(readlink -f "$latest_link")"
                fi
                runs_dir="${pipeline_log_dir}runs"
                if [[ -d "$runs_dir" ]]; then
                    for run in "${runs_dir}"/*/; do
                        run_real="$(readlink -f "$run")"
                        if [[ -n "$latest_target" && "$run_real" == "$latest_target" ]]; then
                            info "Keeping latest log run: ${run}"
                        else
                            TARGETS+=("$run")
                        fi
                    done
                fi
            done
        else
            TARGETS+=("${SCRIPT_DIR}/logs")
        fi
    fi
fi

# Optional: virtual environment
if $NUKE_ENV; then
    [[ -d "${SCRIPT_DIR}/.venv" ]] && TARGETS+=("${SCRIPT_DIR}/.venv")
fi

# ------------------------------------------------------------------
# Nothing to do?
# ------------------------------------------------------------------
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    ok "Nothing to clean — workspace is already tidy."
    exit 0
fi

# ------------------------------------------------------------------
# Confirmation
# ------------------------------------------------------------------
echo ""
warn "The following paths will be removed:"
for t in "${TARGETS[@]}"; do
    echo "  ${t}"
done
echo ""

if ! $AUTO_YES && ! $DRY_RUN; then
    read -r -p "Proceed? [y/N] " answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
        info "Aborted."
        exit 0
    fi
fi

# ------------------------------------------------------------------
# Execute
# ------------------------------------------------------------------
for t in "${TARGETS[@]}"; do
    nuke "$t"
done

# Recreate empty directories so the project layout stays intact
if ! $DRY_RUN; then
    mkdir -p "${SCRIPT_DIR}/output"
    if ! $OUTPUT_ONLY; then
        mkdir -p "${SCRIPT_DIR}/data/processed"
        mkdir -p "${SCRIPT_DIR}/data/raw"
        mkdir -p "${SCRIPT_DIR}/logs"
    fi
fi

echo ""
ok "Cleanup complete."
