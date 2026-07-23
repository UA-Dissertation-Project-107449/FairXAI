#!/usr/bin/env python3
"""Emit one FairXAI domain's stage catalog as Bash declarations."""

from __future__ import annotations

import argparse
import shlex

from fairxai.pipeline.stages import get_stages


def _assignment(key: str, value: str | int) -> str:
    return f"[{shlex.quote(str(key))}]={shlex.quote(str(value))}"


def render_shell_registry(domain: str) -> str:
    stages = get_stages(domain)
    stage_num = [_assignment(name, stage.number) for stage in stages for name in stage.all_names]
    stage_name = [_assignment(stage.number, stage.name) for stage in stages]
    stage_markers = [
        _assignment(stage.number, " ".join(stage.marker_filenames)) for stage in stages
    ]
    valid = " ".join(f"{stage.name}({stage.number})" for stage in stages)

    return "\n".join(
        [
            f"declare -gA STAGE_NUM=({' '.join(stage_num)})",
            f"declare -gA STAGE_NAME=({' '.join(stage_name)})",
            f"declare -gA STAGE_MARKERS=({' '.join(stage_markers)})",
            f"declare -ga STAGE_ORDER=({' '.join(str(stage.number) for stage in stages)})",
            f"STAGE_FIRST={stages[0].number}",
            f"STAGE_LAST={stages[-1].number}",
            f"STAGE_VALID={shlex.quote(valid)}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("domain")
    args = parser.parse_args()
    print(render_shell_registry(args.domain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
