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
#   --output-only       Only remove output/ (skip data/ and logs/)
#   --keep-latest       Preserve the run pointed to by output/*/latest_run
#   --keep-last N       Keep the N most recent runs (by name); delete older ones
#   --pipeline NAME     Restrict cleanup to a single pipeline (e.g. "cardiac")
#   --nuke-env          Also remove the virtual environment (.venv)
#   --dry-run           Show what would be deleted without deleting anything
#   -y / --yes          Skip confirmation prompt
#
# Usage examples:
#   ./cleanup.sh                                # default — remove all four targets
#   ./cleanup.sh --output-only                  # only output/
#   ./cleanup.sh --keep-latest                  # keep latest run per pipeline
#   ./cleanup.sh --keep-last 3                  # keep 3 most recent runs per pipeline
#   ./cleanup.sh --keep-last 1 --pipeline cardiac   # keep 1 cardiac run only
#   ./cleanup.sh --dry-run                      # preview only
#   ./cleanup.sh --nuke-env -y                  # scorched earth, no prompt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
OUTPUT_ONLY=false
KEEP_LATEST=false
KEEP_LAST=0       # 0 = keep all (disabled)
PIPELINE=""       # empty = all pipelines
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
        --keep-last)
            if [[ -z "${2:-}" || ! "${2}" =~ ^[0-9]+$ ]]; then
                echo "--keep-last requires a non-negative integer argument" >&2
                exit 1
            fi
            KEEP_LAST="$2"
            shift
            ;;
        --pipeline)
            if [[ -z "${2:-}" ]]; then
                echo "--pipeline requires a pipeline name argument" >&2
                exit 1
            fi
            PIPELINE="$2"
            shift
            ;;
        --nuke-env)     NUKE_ENV=true     ;;
        --dry-run)      DRY_RUN=true      ;;
        -y|--yes)       AUTO_YES=true     ;;
        *)
            echo "Unknown flag: $1" >&2
            echo "Usage: $0 [--output-only] [--keep-latest] [--keep-last N] [--pipeline NAME] [--nuke-env] [--dry-run] [-y]" >&2
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

# Collect subdirs from <dir> that should be deleted, keeping the <keep_n>
# newest (sorted by name — timestamp-prefixed names are chronological).
# Appends paths to the global TARGETS array.
_prune_dir_keep_last() {
    local dir="$1"
    local keep_n="$2"
    [[ -d "$dir" ]] || return

    local -a all_entries
    mapfile -t all_entries < <(
        find "$dir" -maxdepth 1 -mindepth 1 -type d | sort
    )

    local total=${#all_entries[@]}
    local to_delete=$(( total - keep_n ))
    [[ $to_delete -lt 0 ]] && to_delete=0

    for (( i = 0; i < to_delete; i++ )); do
        TARGETS+=("${all_entries[$i]}")
    done
}

# After deletions, fix any broken latest_run / latest_study symlinks under
# a pipeline log or output directory.
_fix_broken_pointers() {
    local scan_dir="$1"

    # Fix latest_run symlink
    local latest_link="${scan_dir}/latest_run"
    if [[ -L "$latest_link" ]] && [[ ! -e "$latest_link" ]]; then
        local runs_dir="${scan_dir}/runs"
        local newest=""
        [[ -d "$runs_dir" ]] && newest="$(find "$runs_dir" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)"
        if [[ -n "$newest" ]]; then
            local run_id
            run_id="$(basename "$newest")"
            if $DRY_RUN; then
                info "[dry-run] would update latest_run -> runs/${run_id}"
            else
                rm -f "$latest_link"
                ln -sf "runs/${run_id}" "$latest_link"
                echo "${run_id}" > "${scan_dir}/latest_run.txt"
                ok "Updated latest_run -> runs/${run_id}"
            fi
        else
            if $DRY_RUN; then
                info "[dry-run] would remove broken latest_run symlink (no runs left)"
            else
                rm -f "$latest_link" "${scan_dir}/latest_run.txt"
                ok "Removed broken latest_run symlink (no runs remaining)"
            fi
        fi
    fi

    # Fix latest_study symlinks within studies/{type}/
    local studies_dir="${scan_dir}/studies"
    if [[ -d "$studies_dir" ]]; then
        for study_type_dir in "${studies_dir}"/*/; do
            [[ -d "$study_type_dir" ]] || continue
            local latest_study_link="${study_type_dir}/latest_study"
            if [[ -L "$latest_study_link" ]] && [[ ! -e "$latest_study_link" ]]; then
                local newest_study=""
                newest_study="$(find "$study_type_dir" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)"
                if [[ -n "$newest_study" ]]; then
                    local study_id
                    study_id="$(basename "$newest_study")"
                    if $DRY_RUN; then
                        info "[dry-run] would update latest_study -> ${study_id} in $(basename "$study_type_dir")"
                    else
                        rm -f "$latest_study_link"
                        ln -sf "$study_id" "$latest_study_link"
                        echo "${study_id}" > "${study_type_dir}/latest_study.txt"
                        ok "Updated latest_study -> ${study_id}"
                    fi
                else
                    if $DRY_RUN; then
                        info "[dry-run] would remove broken latest_study symlink in $(basename "$study_type_dir")"
                    else
                        rm -f "$latest_study_link" "${study_type_dir}/latest_study.txt"
                        ok "Removed broken latest_study symlink (no studies remaining)"
                    fi
                fi
            fi
        done
    fi
}

# ------------------------------------------------------------------
# Build target list
# ------------------------------------------------------------------
TARGETS=()
FIX_POINTER_DIRS=()  # dirs to run _fix_broken_pointers on after nuking

# output/ — always included
if [[ -d "${SCRIPT_DIR}/output" ]]; then
    if $KEEP_LATEST; then
        # Remove everything inside each pipeline's runs/ EXCEPT the latest
        for pipeline_dir in "${SCRIPT_DIR}"/output/*/; do
            [[ -d "$pipeline_dir" ]] || continue
            [[ -n "$PIPELINE" && "$(basename "$pipeline_dir")" != "$PIPELINE" ]] && continue
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
            [[ -d "$archived_dir" ]] && TARGETS+=("$archived_dir")

            # Clean recommendations/
            recs_dir="${pipeline_dir}recommendations"
            [[ -d "$recs_dir" ]] && TARGETS+=("$recs_dir")

            # Clean run_history.jsonl
            hist_file="${pipeline_dir}run_history.jsonl"
            [[ -f "$hist_file" ]] && TARGETS+=("$hist_file")
        done

    elif [[ $KEEP_LAST -gt 0 ]]; then
        for pipeline_dir in "${SCRIPT_DIR}"/output/*/; do
            [[ -d "$pipeline_dir" ]] || continue
            [[ -n "$PIPELINE" && "$(basename "$pipeline_dir")" != "$PIPELINE" ]] && continue

            _prune_dir_keep_last "${pipeline_dir}runs" "$KEEP_LAST"
            FIX_POINTER_DIRS+=("$pipeline_dir")
        done

    else
        if [[ -n "$PIPELINE" ]]; then
            [[ -d "${SCRIPT_DIR}/output/${PIPELINE}" ]] && TARGETS+=("${SCRIPT_DIR}/output/${PIPELINE}")
        else
            TARGETS+=("${SCRIPT_DIR}/output")
        fi
    fi
fi

if ! $OUTPUT_ONLY; then
    # data/processed/
    [[ -d "${SCRIPT_DIR}/data/processed" ]] && TARGETS+=("${SCRIPT_DIR}/data/processed")

    # data/raw/
    [[ -d "${SCRIPT_DIR}/data/raw" ]] && TARGETS+=("${SCRIPT_DIR}/data/raw")

    # logs/ — honour --keep-latest / --keep-last for per-run and per-study dirs
    if [[ -d "${SCRIPT_DIR}/logs" ]]; then
        if $KEEP_LATEST; then
            for pipeline_log_dir in "${SCRIPT_DIR}"/logs/*/; do
                [[ -d "$pipeline_log_dir" ]] || continue
                [[ -n "$PIPELINE" && "$(basename "$pipeline_log_dir")" != "$PIPELINE" ]] && continue

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

        elif [[ $KEEP_LAST -gt 0 ]]; then
            for pipeline_log_dir in "${SCRIPT_DIR}"/logs/*/; do
                [[ -d "$pipeline_log_dir" ]] || continue
                [[ -n "$PIPELINE" && "$(basename "$pipeline_log_dir")" != "$PIPELINE" ]] && continue

                # Prune full pipeline runs
                _prune_dir_keep_last "${pipeline_log_dir}runs" "$KEEP_LAST"

                # Prune study runs per study type
                studies_dir="${pipeline_log_dir}studies"
                if [[ -d "$studies_dir" ]]; then
                    for study_type_dir in "${studies_dir}"/*/; do
                        [[ -d "$study_type_dir" ]] || continue
                        _prune_dir_keep_last "$study_type_dir" "$KEEP_LAST"
                    done
                fi

                FIX_POINTER_DIRS+=("$pipeline_log_dir")
            done

        else
            if [[ -n "$PIPELINE" ]]; then
                [[ -d "${SCRIPT_DIR}/logs/${PIPELINE}" ]] && TARGETS+=("${SCRIPT_DIR}/logs/${PIPELINE}")
            else
                TARGETS+=("${SCRIPT_DIR}/logs")
            fi
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

# Fix any broken symlinks left behind by deletions
for fix_dir in "${FIX_POINTER_DIRS[@]}"; do
    _fix_broken_pointers "$fix_dir"
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
