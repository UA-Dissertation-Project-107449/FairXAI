#!/usr/bin/env python3
"""Cardiac phase runner: train baseline."""
from pathlib import Path
import runpy
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
TARGET = ROOT_DIR / "scripts" / "common" / "train_baseline.py"

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--pipeline" not in args:
        args = ["--pipeline", "cardiac"] + args
    sys.argv = [str(TARGET)] + args
    runpy.run_path(str(TARGET), run_name="__main__")
