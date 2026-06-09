#!/usr/bin/env python3
"""Dermatology phase runner: recommendations."""

import runpy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
TARGET = ROOT_DIR / "scripts" / "common" / "generate_recommendations.py"

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--pipeline" not in args:
        args = ["--pipeline", "dermatology"] + args
    sys.argv = [str(TARGET)] + args
    runpy.run_path(str(TARGET), run_name="__main__")
