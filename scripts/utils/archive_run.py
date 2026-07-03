#!/usr/bin/env python3
"""Archive a pipeline run into ``output/<domain>/archived_runs/``.

Copies (default) or moves a completed run out of the live ``runs/`` directory
into the domain's ``archived_runs/`` directory, optionally under a new name, and
records the operation in ``archive_manifest.json`` so provenance survives the
rename. Prevents runs referenced by the dissertation from being lost to routine
cleanup of ``runs/``.

Manifest is a JSON list; each entry records when the run was archived, its prior
name/id, its new name, source and destination paths, and an optional note.

Examples (from Code/FairXAI/, venv active):

    # Archive a cardiac run under its own id
    python3 scripts/utils/archive_run.py run_20260702_... --domain cardiac

    # Archive under a descriptive name with a note
    python3 scripts/utils/archive_run.py run_20260702_... --domain cardiac \
        --name cardio70k_full_profile --note "Full 70k profiling for chapter 3"

    # Register an already-moved directory (e.g. the manual _ARCHIVED one)
    python3 scripts/utils/archive_run.py --source-path output/_ARCHIVED/dermatology/<x> \
        --domain dermatology --name <x> --note "Pre-existing manual archive" --register-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "archive_manifest.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_manifest(path: Path) -> list[dict]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            print(f"WARNING: corrupt manifest {path}; starting fresh.", file=sys.stderr)
    return []


def _append_manifest(path: Path, entry: dict) -> None:
    manifest = _load_manifest(path)
    manifest.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id", nargs="?", help="Run id under output/<domain>/runs/.")
    p.add_argument("--domain", required=True, help="e.g. cardiac, dermatology")
    p.add_argument("--name", default=None, help="Archived name (default: run id).")
    p.add_argument("--note", default="", help="Free-text note stored in the manifest.")
    p.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help="Archive an explicit directory instead of output/<domain>/runs/<run_id>.",
    )
    p.add_argument("--output-root", type=Path, default=Path("output"), help="Output root dir.")
    p.add_argument("--move", action="store_true", help="Move instead of copy.")
    p.add_argument(
        "--register-only",
        action="store_true",
        help="Only add a manifest entry for an existing archived dir; no copy/move.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing archive.")
    args = p.parse_args(argv)

    archived_dir = args.output_root / args.domain / "archived_runs"
    manifest_path = archived_dir / MANIFEST_NAME

    # Resolve source.
    if args.source_path is not None:
        source = args.source_path
        original_name = source.name
    elif args.run_id:
        source = args.output_root / args.domain / "runs" / args.run_id
        original_name = args.run_id
    else:
        print("ERROR: provide a run_id or --source-path.", file=sys.stderr)
        return 2

    archived_name = args.name or original_name
    dest = archived_dir / archived_name

    if args.register_only:
        if not dest.exists():
            print(
                f"ERROR: --register-only but archived dir absent: {dest}",
                file=sys.stderr,
            )
            return 2
    else:
        if not source.exists():
            print(f"ERROR: source not found: {source}", file=sys.stderr)
            return 2
        if dest.exists():
            if not args.force:
                print(f"ERROR: destination exists: {dest} (use --force).", file=sys.stderr)
                return 2
            shutil.rmtree(dest)
        archived_dir.mkdir(parents=True, exist_ok=True)
        if args.move:
            shutil.move(str(source), str(dest))
        else:
            shutil.copytree(source, dest)

    entry = {
        "archived_at": _now_iso(),
        "domain": args.domain,
        "original_run_id": original_name,
        "archived_name": archived_name,
        "source_path": str(source),
        "archived_path": str(dest),
        "operation": ("register-only" if args.register_only else ("move" if args.move else "copy")),
        "note": args.note,
    }
    _append_manifest(manifest_path, entry)

    print(f"Archived '{original_name}' -> {dest}")
    print(f"Operation: {entry['operation']}")
    print(f"Manifest updated: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
