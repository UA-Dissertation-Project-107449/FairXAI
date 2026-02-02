"""Compat shim for stdlib profile.

This file exists in scripts/cardiac and can shadow the stdlib `profile` module
when executing scripts from this directory. Provide a minimal `run`/`runctx`
implementation so libraries like SHAP do not fail when they import `profile`.

Use scripts/cardiac/profile_data.py for the pipeline runner.
"""

from __future__ import annotations

import cProfile
import pstats
from typing import Any, Optional
from pathlib import Path
import runpy
import sys


def run(statement: str, filename: Optional[str] = None, sort: int | str = -1) -> Any:
	prof = cProfile.Profile()
	prof.run(statement)
	if filename:
		prof.dump_stats(filename)
		return prof
	pstats.Stats(prof).sort_stats(sort).print_stats()
	return prof


def runctx(
	statement: str,
	globals_dict: dict,
	locals_dict: dict,
	filename: Optional[str] = None,
	sort: int | str = -1
) -> Any:
	prof = cProfile.Profile()
	prof.runctx(statement, globals_dict, locals_dict)
	if filename:
		prof.dump_stats(filename)
		return prof
	pstats.Stats(prof).sort_stats(sort).print_stats()
	return prof


__all__ = ["run", "runctx"]


if __name__ == "__main__":
	root_dir = Path(__file__).resolve().parents[2]
	target = root_dir / "scripts" / "cardiac" / "profile_data.py"
	sys.argv = [str(target)] + sys.argv[1:]
	runpy.run_path(str(target), run_name="__main__")
