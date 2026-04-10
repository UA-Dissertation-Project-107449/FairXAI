"""Integration test: run_grouping_analysis.py on synthetic processed CSV."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "experiments" / "run_grouping_analysis.py"


@pytest.fixture
def synthetic_processed_csv(tmp_path):
    """Write a 60-row cardiac-like processed CSV with required columns."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "age_group": rng.choice(["<40", "40-49", "50-59", "60+"], size=n),
            "sex": rng.choice(["Female", "Male"], size=n),
            "trestbps": rng.uniform(90, 180, n),
            "chol": rng.uniform(150, 350, n),
            "thalach": rng.uniform(100, 200, n),
            "oldpeak": rng.uniform(0, 4, n),
            "ca": rng.integers(0, 4, size=n).astype(float),
            "heart_disease": rng.integers(0, 2, size=n),
        }
    )
    processed_dir = _ROOT / "data" / "processed" / "cardiac"
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / "test_synthetic_processed.csv"
    df.to_csv(csv_path, index=False)
    yield csv_path
    # Cleanup: remove group_cluster col from original if added
    if csv_path.exists():
        saved = pd.read_csv(csv_path)
        if "group_cluster" in saved.columns:
            saved = saved.drop(columns=["group_cluster"])
            saved.to_csv(csv_path, index=False)
        csv_path.unlink(missing_ok=True)


@pytest.mark.integration
def test_grouping_produces_cluster_assignments(synthetic_processed_csv, tmp_path):
    """run_grouping_analysis.py writes group_cluster back to processed CSV."""
    run_root = tmp_path / "output" / "cardiac" / "runs" / "run_test"
    run_root.mkdir(parents=True, exist_ok=True)

    env = {
        **__import__("os").environ,
        "PYTHONPATH": str(_ROOT / "src"),
    }
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--run-id", "run_test",
            "--datasets", "test_synthetic",
            "--methods", "kmeans",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Script should exit 0 (or 1 only if no datasets found, which we prevent)
    assert result.returncode == 0, (
        f"Script failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # group_cluster must be written back into the processed CSV
    saved = pd.read_csv(synthetic_processed_csv)
    assert "group_cluster" in saved.columns, (
        "group_cluster column not written back to processed CSV"
    )
    assert saved["group_cluster"].notna().all(), "group_cluster has NaN values"
    assert saved["group_cluster"].nunique() >= 2, "Expected >= 2 clusters"


@pytest.mark.integration
def test_grouping_produces_cluster_artifacts(synthetic_processed_csv):
    """run_grouping_analysis.py creates expected output files."""
    # Script writes to _ROOT/output/cardiac/runs/run_test — not tmp_path
    grouping_dir = _ROOT / "output" / "cardiac" / "runs" / "run_test" / "grouping" / "test_synthetic"

    env = {
        **__import__("os").environ,
        "PYTHONPATH": str(_ROOT / "src"),
    }
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--run-id", "run_test",
            "--datasets", "test_synthetic",
            "--methods", "kmeans",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    try:
        assert (grouping_dir / "cluster_assignments.csv").exists()
        assert (grouping_dir / "cluster_diagnostics.csv").exists()
        assert (grouping_dir / "subgroup_profiles.md").exists()

        profiles = (grouping_dir / "subgroup_profiles.md").read_text()
        assert len(profiles) > 0, "subgroup_profiles.md is empty"

        assignments = pd.read_csv(grouping_dir / "cluster_assignments.csv")
        assert "group_cluster" in assignments.columns
        assert len(assignments) > 0
    finally:
        # Clean up output so reruns start fresh
        import shutil
        run_dir = _ROOT / "output" / "cardiac" / "runs" / "run_test"
        if run_dir.exists():
            shutil.rmtree(run_dir)
