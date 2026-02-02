#!/usr/bin/env python3
"""Cardiac phase runner: compare experiments."""
from pathlib import Path
import runpy
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
TARGET = ROOT_DIR / "scripts" / "analysis" / "compare_experiments.py"

if __name__ == "__main__":
    sys.argv = [str(TARGET)] + sys.argv[1:]
    runpy.run_path(str(TARGET), run_name="__main__")
