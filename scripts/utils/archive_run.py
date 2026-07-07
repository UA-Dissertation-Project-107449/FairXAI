#!/usr/bin/env python3
"""Archive a pipeline run into ``output/<domain>/archived_runs/``.

Copies (default) or moves a completed run out of the live ``runs/`` directory
into the domain's ``archived_runs/`` directory, optionally under a new name, and
records the operation in ``archive_manifest.json`` so provenance survives the
rename. Prevents runs referenced by the dissertation from being lost to routine
cleanup of ``runs/``.

The paired per-run log dir (``logs/<domain>/runs/<run_id>``) is archived by
default into ``<archived_run>/_logs/`` so outputs and the logs that produced them
stay together; pass ``--no-logs`` to skip it.

Manifest is a JSON list; each entry records when the run was archived, its prior
name/id, its new name, source and destination paths, the paired log paths, and an
optional note.

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
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MANIFEST_NAME = "archive_manifest.json"


class ManifestError(ValueError):
    """Raised when an existing archive manifest cannot be trusted."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read valid JSON from {path}: {exc}") from exc

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not all(isinstance(entry, dict) for entry in data):
        raise ManifestError(f"expected a JSON object or list of objects in {path}")
    return data


def _append_manifest(path: Path, entry: dict, manifest: list[dict] | None = None) -> None:
    """Append an entry and atomically replace the manifest."""
    if manifest is None:
        manifest = _load_manifest(path)
    manifest.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _validate_component(value: str, label: str) -> str:
    """Require one ordinary path component, never an absolute/traversal path."""
    if not value or value in {".", ".."}:
        raise ValueError(f"{label} must be a non-empty path component")
    if Path(value).is_absolute() or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a single path component: {value!r}")
    return value


def _archive_paired_logs(
    logs_root: Path,
    domain: str,
    run_id: str,
    dest: Path,
    move: bool,
) -> Optional[Path]:
    """Copy (or move) the run's log dir into ``<dest>/_logs`` alongside its output.

    Logs live in the parallel ``logs/<domain>/runs/<run_id>`` tree, keyed by the
    same run id as the output run, and are usually read paired with the outputs
    (the log is the evidence that produced the archived numbers). Returns the
    archived log path, or None when no per-run log dir exists (older runs predate
    structured run-scoped logging).
    """
    log_source = logs_root / domain / "runs" / run_id
    if not log_source.is_dir():
        return None
    log_dest = dest / "_logs"
    if log_dest.exists():
        shutil.rmtree(log_dest)
    if move:
        shutil.move(str(log_source), str(log_dest))
    else:
        shutil.copytree(log_source, log_dest)
    return log_dest


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
    p.add_argument("--logs-root", type=Path, default=Path("logs"), help="Pipeline logs root dir.")
    p.add_argument(
        "--no-logs",
        dest="with_logs",
        action="store_false",
        help="Do not archive the paired logs/<domain>/runs/<run_id> log dir.",
    )
    p.set_defaults(with_logs=True)
    p.add_argument("--move", action="store_true", help="Move instead of copy.")
    p.add_argument(
        "--register-only",
        action="store_true",
        help="Only add a manifest entry for an existing archived dir; no copy/move.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing archive.")
    args = p.parse_args(argv)

    try:
        domain = _validate_component(args.domain, "--domain")
        if args.run_id and args.source_path is None:
            _validate_component(args.run_id, "run_id")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    archived_dir = args.output_root / domain / "archived_runs"
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
    try:
        _validate_component(archived_name, "--name")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    dest = archived_dir / archived_name
    output_root_resolved = args.output_root.resolve()
    archived_dir_resolved = archived_dir.resolve()
    dest_resolved = dest.resolve()
    if not archived_dir_resolved.is_relative_to(output_root_resolved):
        print(
            f"ERROR: archive directory escapes --output-root: {archived_dir}",
            file=sys.stderr,
        )
        return 2
    if not dest_resolved.is_relative_to(archived_dir_resolved):
        print(f"ERROR: destination escapes archive directory: {dest}", file=sys.stderr)
        return 2

    try:
        manifest = _load_manifest(manifest_path)
    except ManifestError as exc:
        print(f"ERROR: corrupt archive manifest; refusing to continue: {exc}", file=sys.stderr)
        return 2

    if args.register_only:
        if not source.is_dir():
            print(
                f"ERROR: --register-only source directory absent: {source}",
                file=sys.stderr,
            )
            return 2
        archived_path = source
    else:
        if not source.is_dir():
            print(f"ERROR: source directory not found: {source}", file=sys.stderr)
            return 2
        source_resolved = source.resolve()
        if (
            source_resolved == dest_resolved
            or dest_resolved.is_relative_to(source_resolved)
            or source_resolved.is_relative_to(dest_resolved)
        ):
            print(
                f"ERROR: source and destination overlap: {source} -> {dest}",
                file=sys.stderr,
            )
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
        archived_path = dest

    # Archive the paired per-run log dir next to the outputs (skipped for
    # register-only, which points at an already-relocated directory).
    logs_archived_path: Optional[Path] = None
    if args.with_logs and not args.register_only:
        logs_archived_path = _archive_paired_logs(
            args.logs_root, domain, original_name, archived_path, args.move
        )

    entry = {
        "archived_at": _now_iso(),
        "domain": domain,
        "original_run_id": original_name,
        "archived_name": archived_name,
        "source_path": str(source),
        "archived_path": str(archived_path),
        "logs_source_path": (
            str(args.logs_root / domain / "runs" / original_name) if args.with_logs else None
        ),
        "logs_archived_path": str(logs_archived_path) if logs_archived_path else None,
        "operation": ("register-only" if args.register_only else ("move" if args.move else "copy")),
        "note": args.note,
    }
    _append_manifest(manifest_path, entry, manifest)

    print(f"Archived '{original_name}' -> {archived_path}")
    print(f"Operation: {entry['operation']}")
    if logs_archived_path:
        print(f"Logs archived: {logs_archived_path}")
    elif args.with_logs and not args.register_only:
        print("Logs: no per-run log dir found; skipped.")
    print(f"Manifest updated: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
