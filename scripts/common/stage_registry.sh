#!/usr/bin/env bash

# Shared adapter between the Python stage catalog and Bash orchestrators.

load_stage_registry() {
    local domain=$1
    local root_dir=$2
    local python_bin=${3:-python3}
    local declarations

    declarations=$(PYTHONPATH="$root_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$python_bin" "$root_dir/scripts/common/export_stage_registry.py" "$domain") || return 1
    eval "$declarations"
}

resolve_stage() {
    local input="${1,,}"
    if [[ "$input" =~ ^[0-9]+$ && -n "${STAGE_NAME[$input]+x}" ]]; then
        echo "$input"
        return
    fi

    local stripped="${input#phase}"
    stripped="${stripped#stage}"
    stripped="${stripped#step}"
    if [[ "$stripped" =~ ^[0-9]+$ && -n "${STAGE_NAME[$stripped]+x}" ]]; then
        echo "$stripped"
        return
    fi

    if [[ -n "${STAGE_NUM[$input]+x}" ]]; then
        echo "${STAGE_NUM[$input]}"
        return
    fi

    echo "ERROR: Unknown stage '$1'. Valid: $STAGE_VALID" >&2
    return 1
}

stage_marker_exists() {
    local checkpoint_dir=$1
    local number=$2
    local marker

    for marker in ${STAGE_MARKERS[$number]}; do
        [[ -f "$checkpoint_dir/$marker" ]] && return 0
    done
    return 1
}

# Record or verify what a run is running on. Checkpoint markers prove a stage
# finished but not what it finished on, so a resume after a config edit or with
# a narrower --datasets silently mixes two experiments under one run ID.
#
#   run_manifest_guard <root_dir> <run_root> <pipeline> <resuming:0|1> \
#                      <datasets_csv> <model_types_csv> <flags_csv> <configs...>
#
# Resuming with different inputs stops the run unless ALLOW_MANIFEST_CHANGE is set.
run_manifest_guard() {
    local root_dir=$1 run_root=$2 pipeline=$3 resuming=$4
    local datasets=$5 model_types=$6 flags=$7
    shift 7

    FAIRXAI_MANIFEST_RESUMING="$resuming" \
    FAIRXAI_MANIFEST_DATASETS="$datasets" \
    FAIRXAI_MANIFEST_MODEL_TYPES="$model_types" \
    FAIRXAI_MANIFEST_FLAGS="$flags" \
    FAIRXAI_MANIFEST_ALLOW_CHANGE="${ALLOW_MANIFEST_CHANGE:-0}" \
    PYTHONPATH="$root_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 - "$root_dir" "$run_root" "$pipeline" "$@" <<'PY'
import os
import sys
from pathlib import Path

from fairxai.pipeline import (
    build_run_manifest,
    compare_run_manifests,
    read_run_manifest,
    write_run_manifest,
)

root, run_root, pipeline, *configs = sys.argv[1:]


def split(name):
    raw = os.environ.get(name, "")
    return [item for item in raw.split(",") if item]


flags = dict(item.split("=", 1) for item in split("FAIRXAI_MANIFEST_FLAGS") if "=" in item)
current = build_run_manifest(
    pipeline,
    split("FAIRXAI_MANIFEST_DATASETS"),
    split("FAIRXAI_MANIFEST_MODEL_TYPES"),
    [Path(c) for c in configs],
    flags,
    project_root=Path(root),
)

resuming = os.environ.get("FAIRXAI_MANIFEST_RESUMING") == "1"
recorded = read_run_manifest(Path(run_root)) if resuming else None

if recorded is None:
    write_run_manifest(Path(run_root), current)
    if resuming:
        print("No run manifest on disk (run predates it); recording this invocation's inputs.")
    sys.exit(0)

differences = compare_run_manifests(recorded, current)
if not differences:
    sys.exit(0)

allow = os.environ.get("FAIRXAI_MANIFEST_ALLOW_CHANGE", "0").lower() in {"1", "true", "yes", "on"}
header = "Resume inputs differ from the ones this run started with:"
print(header, file=sys.stderr)
for line in differences:
    print(f"  {line}", file=sys.stderr)
if not allow:
    print(
        "Refusing to resume: the results would mix two experiments under one run ID.\n"
        "Start a new run, or set ALLOW_MANIFEST_CHANGE=1 to proceed deliberately.",
        file=sys.stderr,
    )
    sys.exit(1)

print("ALLOW_MANIFEST_CHANGE is set; continuing and updating the manifest.", file=sys.stderr)
write_run_manifest(Path(run_root), current)
PY
}
