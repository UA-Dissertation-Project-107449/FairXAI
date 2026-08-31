"""Where pre-train clustering writes its diagnostics.

A flat ``studies/grouping_pretrain/<dataset>/`` directory is overwritten by the
next run on the same dataset, so an archived run silently acquires a clustering
it never used. These pin the two scopes: owned by a run when the pipeline
invokes it, versioned as a study when a person does.
"""

import subprocess
import sys
from pathlib import Path

_FAIRXAI_ROOT = Path(__file__).parent.parent.parent
_SCRIPT = _FAIRXAI_ROOT / "scripts" / "cardiac" / "cluster_subgroups.py"
_BASH_PIPELINE = _FAIRXAI_ROOT / "scripts" / "cardiac" / "cardiac_pipeline.sh"
_PREFECT_FLOW = _FAIRXAI_ROOT / "flows" / "cardiac_pipeline.py"

sys.path.insert(0, str(_FAIRXAI_ROOT / "src"))


def test_the_script_accepts_a_run_id():
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0
    assert "--run-id" in out.stdout


def test_both_orchestrators_pass_a_run_id():
    """Either call site forgetting this puts the artifacts back in a shared dir."""
    bash = _BASH_PIPELINE.read_text()
    cluster_call = bash[bash.index("cluster_subgroups.py") :][:400]
    assert '--run-id "$RUN_ID"' in cluster_call

    flow = _PREFECT_FLOW.read_text()
    task = flow[flow.index("def cluster_subgroups(") :][:1200]
    assert '"--run-id"' in task


def test_a_run_id_scopes_the_output_under_that_run():
    from fairxai.cli.runner_utils import get_run_root, get_study_root

    base = Path("/tmp/base")
    run_dir = get_run_root(base, "run_20260831_103200") / "grouping_pretrain"
    study_dir = get_study_root(base, "grouping_pretrain", "run_20260831_110000")

    assert run_dir == base / "runs" / "run_20260831_103200" / "grouping_pretrain"
    assert study_dir == base / "studies" / "grouping_pretrain" / "run_20260831_110000"
    # The old unversioned path is exactly what neither scope may resolve to.
    assert run_dir != base / "studies" / "grouping_pretrain"
    assert study_dir != base / "studies" / "grouping_pretrain"


def test_an_idempotent_skip_still_records_provenance(tmp_path):
    """A run that inherits labels must archive where they came from."""
    import pandas as pd

    from fairxai.clustering.grouping_pipeline import _write_inherited_marker

    train = pd.DataFrame({"group_cluster": [0, 0, 1, 1, 1]})
    test = pd.DataFrame({"group_cluster": [0, 1]})
    out_dir = tmp_path / "grouping_pretrain" / "cleveland_uci"

    _write_inherited_marker(
        out_dir, "cleveland_uci", train, test, (tmp_path / "tr.csv", tmp_path / "te.csv")
    )

    import json

    payload = json.loads((out_dir / "inherited.json").read_text())
    assert payload["fitted_here"] is False
    assert payload["train_label_counts"] == {"1": 3, "0": 2}
    assert payload["source_train_split"].endswith("tr.csv")
