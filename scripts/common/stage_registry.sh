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
