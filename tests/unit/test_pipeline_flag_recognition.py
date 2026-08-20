"""Dry-run flag recognition tests for pipeline orchestrators.

These tests validate that new scope flags are recognized by CLI parsers without
running full pipeline workloads.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFECT_FLOW = ROOT / "flows" / "cardiac_pipeline.py"
COMBINATORIAL = ROOT / "scripts" / "experiments" / "run_combinatorial_experiments.py"
BASH_PIPELINE = ROOT / "scripts" / "cardiac" / "cardiac_pipeline.sh"
DERM_BASH_PIPELINE = ROOT / "scripts" / "dermatology" / "dermatology_pipeline.sh"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_prefect_help_lists_scope_flags() -> None:
    result = _run([sys.executable, str(PREFECT_FLOW), "--help"])
    assert result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    assert "--datasets" in output
    assert "--model-types" in output
    assert "--study-mode" in output
    assert "--parallel-studies" in output
    assert "--parallel-experiments" in output
    assert "--max-cores" in output
    assert "--max-samples" in output
    assert "--hpo-search-n-jobs" in output


def test_combinatorial_help_lists_scope_flags() -> None:
    result = _run([sys.executable, str(COMBINATORIAL), "--help"])
    assert result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    assert "--datasets" in output
    assert "--model-types" in output


def test_bash_parser_accepts_scope_flags_before_stage_validation() -> None:
    # Use an invalid stage so execution stops during stage resolution,
    # proving parser accepted the new flags without launching stage scripts.
    result = _run(
        [
            "bash",
            str(BASH_PIPELINE),
            "--datasets",
            "cleveland",
            "--model-types",
            "logistic_regression",
            "--study-mode",
            "auto_safe",
            "--parallel-studies",
            "--parallel-experiments",
            "--max-cores",
            "4",
            "--hpo-search-n-jobs",
            "2",
            "--go-until",
            "invalid_stage_name",
        ]
    )

    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0
    assert "Unknown stage 'invalid_stage_name'" in combined
    assert "Unknown argument '--datasets'" not in combined
    assert "Unknown argument '--model-types'" not in combined
    assert "Unknown argument '--study-mode'" not in combined
    assert "Unknown argument '--parallel-studies'" not in combined
    assert "Unknown argument '--parallel-experiments'" not in combined
    assert "Unknown argument '--max-cores'" not in combined
    assert "Unknown argument '--hpo-search-n-jobs'" not in combined


def test_dermatology_bash_parser_accepts_baseline_flags() -> None:
    for figure_flag in ("--figures", "--no-figures"):
        for group_view_flag in ("--group-views", "--no-group-views"):
            result = _run(
                [
                    "bash",
                    str(DERM_BASH_PIPELINE),
                    "--datasets",
                    "pad_ufes_20",
                    "--model-types",
                    "resnet18",
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--no-pretrained",
                    figure_flag,
                    group_view_flag,
                    "--go-until",
                    "invalid_stage_name",
                ]
            )

            combined = (result.stdout or "") + (result.stderr or "")
            assert result.returncode != 0
            assert "Unknown stage 'invalid_stage_name'" in combined
            assert "Unknown argument '--datasets'" not in combined
            assert "Unknown argument '--model-types'" not in combined
            assert "Unknown argument '--device'" not in combined
            assert "Unknown argument '--epochs'" not in combined
            assert "Unknown argument '--batch-size'" not in combined
            assert "Unknown argument '--no-pretrained'" not in combined
            assert f"Unknown argument '{figure_flag}'" not in combined
            assert f"Unknown argument '{group_view_flag}'" not in combined


def test_prefect_compare_stage_forwards_dataset_scope_to_grouping() -> None:
    source = PREFECT_FLOW.read_text(encoding="utf-8")

    assert "def compare_experiments(run_id: str, datasets:" in source
    assert 'grouping_args.extend(["--datasets", *datasets])' in source
    assert "compare_experiments.submit(run_id, datasets, verbose" in source


def test_prefect_preprocess_forwards_max_samples() -> None:
    source = PREFECT_FLOW.read_text(encoding="utf-8")

    assert 'args.extend(["--max-samples", str(max_samples)])' in source
    assert "resolved_max_samples," in source
    assert "max_samples=args.max_samples" in source


def test_mitigation_stage_receives_model_type_args():
    """Stage 10 was the only stage not forwarding MODEL_TYPES; it must now."""
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "cardiac" / "cardiac_pipeline.sh"
    ).read_text()
    lines = script.splitlines()
    invocations = [i for i, line in enumerate(lines) if "cardiac/mitigation.py" in line]
    assert invocations, "no mitigation.py invocation found"
    for idx in invocations:
        block = "\n".join(lines[idx : idx + 3])
        assert "MODEL_TYPE_ARGS" in block, f"missing MODEL_TYPE_ARGS near line {idx + 1}"


def _prefect_task_source(name: str) -> str:
    """The body of one @task, up to whatever is defined next."""
    source = PREFECT_FLOW.read_text(encoding="utf-8")
    start = source.index(f"def {name}(")
    end = source.index("\ndef ", start + 1)
    return source[start:end]


def test_prefect_mitigation_task_forwards_model_types() -> None:
    """Stage 10's Prefect task must forward --model-types like the bash one does."""
    task_source = _prefect_task_source("compare_mitigation_techniques")

    assert "model_types: Optional[list[str]] = None," in task_source
    assert 'args.extend(["--model-types", *model_types])' in task_source


def test_prefect_mitigation_call_site_passes_model_types() -> None:
    """The task takes model_types positionally, so the submit must name it.

    Adding the parameter without touching the call site would silently bind
    ``verbose`` to ``model_types``.
    """
    source = PREFECT_FLOW.read_text(encoding="utf-8")
    start = source.index("compare_mitigation_techniques.submit(")
    call = source[start : source.index(")", start)]

    assert "model_types" in call


def test_prefect_and_bash_agree_on_which_stages_take_model_types() -> None:
    """Both orchestrators drive the same scripts; the family scope must match.

    Bash appends ``MODEL_TYPE_ARGS`` to a fixed set of phase runners. Any runner
    that gets it there must have a ``--model-types`` branch in the flow task that
    launches the same script, or a flow run silently mitigates a different set of
    families than the identical bash invocation.
    """
    bash = BASH_PIPELINE.read_text(encoding="utf-8").splitlines()
    scripts_with_model_types = set()
    for idx, line in enumerate(bash):
        block = "\n".join(bash[idx : idx + 4])
        if "MODEL_TYPE_ARGS" not in block:
            continue
        for token in block.split():
            if token.endswith(".py") or token.endswith('.py"'):
                scripts_with_model_types.add(Path(token.strip('"')).name)

    flow = PREFECT_FLOW.read_text(encoding="utf-8")
    for name in ("mitigation.py", "combinatorial.py"):
        assert name in scripts_with_model_types, f"bash stopped scoping {name} by family"
        start = flow.index(f'"{name.removesuffix(".py")}.py"')
        task_start = flow.rindex("\ndef ", 0, start)
        task_end = flow.index("\ndef ", start)
        assert (
            'args.extend(["--model-types", *model_types])' in flow[task_start:task_end]
        ), f"flow task launching {name} does not forward --model-types"


def test_dermatology_bash_parser_accepts_explain_toggle() -> None:
    """Stage 10 (explain) is the run's most expensive stage; it must be skippable.

    With cached frozen features the saliency stage dominates wall-clock, so a
    baseline-only run needs to turn it off without editing the pipeline config.
    """
    for explain_flag in ("--explain", "--no-explain"):
        result = _run(
            [
                "bash",
                str(DERM_BASH_PIPELINE),
                explain_flag,
                "--go-until",
                "invalid_stage_name",
            ]
        )

        combined = (result.stdout or "") + (result.stderr or "")
        assert result.returncode != 0
        assert "Unknown stage 'invalid_stage_name'" in combined
        assert f"Unknown argument '{explain_flag}'" not in combined


def test_dermatology_bash_explain_stage_is_gated_on_run_explain() -> None:
    """The flag has to gate the invocation, not just parse.

    ``explain.py`` already no-ops on ``xai.enabled: false``, but only after
    importing torch and resolving the run, so gating in the orchestrator is the
    part that actually saves time.
    """
    source = DERM_BASH_PIPELINE.read_text(encoding="utf-8")

    assert "RUN_EXPLAIN=${RUN_EXPLAIN:-" in source, "no RUN_EXPLAIN env/config default"

    lines = source.splitlines()
    invocations = [i for i, line in enumerate(lines) if "dermatology/explain.py" in line]
    assert invocations, "no explain.py invocation found"
    for idx in invocations:
        preceding = "\n".join(lines[max(0, idx - 6) : idx])
        assert "RUN_EXPLAIN" in preceding, f"explain.py at line {idx + 1} is not gated"


# --- Dermatology Prefect flow parity -----------------------------------------
# The dermatology flow and its bash pipeline drive the same phase runners. These
# tests derive what bash forwards and require the flow to match, so the two
# orchestrators cannot drift the way the cardiac pair did at stage 10.

DERM_PREFECT_FLOW = ROOT / "flows" / "dermatology_pipeline.py"

# Bash builds one array per optional scope flag; each maps to the literal(s) the
# equivalent flow task must put on the command line.
_BASH_ARG_ARRAY_TO_FLAGS = {
    "DATASET_ARGS": ("--datasets",),
    "MODEL_TYPE_ARGS": ("--model-types",),
    "DEVICE_ARGS": ("--device",),
    "EPOCH_ARGS": ("--epochs",),
    "BATCH_ARGS": ("--batch-size",),
    "PRETRAINED_ARGS": ("--pretrained", "--no-pretrained"),
    "FIGURE_ARGS": ("--figures", "--no-figures"),
    "GROUP_VIEW_ARGS": ("--group-views", "--no-group-views"),
    "AUGMENTATION_ARGS": ("--augmentation", "--no-augmentation"),
}


def _derm_bash_invocations() -> dict[str, set[str]]:
    """Map each dermatology phase runner to the bash arg arrays it receives."""
    invocations: dict[str, set[str]] = {}
    for line in DERM_BASH_PIPELINE.read_text(encoding="utf-8").splitlines():
        match = re.search(r"dermatology/([a-z_]+)\.py", line)
        if not match:
            continue
        arrays = set(re.findall(r"\$\{([A-Z_]+)\[@\]\}", line))
        invocations[f"{match.group(1)}.py"] = arrays
    return invocations


def _flow_task_body(flow_source: str, script_name: str) -> str:
    """The @task body that launches *script_name*, up to whatever follows it."""
    marker = f'"{script_name}"'
    start = flow_source.index(marker)
    return flow_source[flow_source.rindex("\ndef ", 0, start) : flow_source.index("\ndef ", start)]


def test_dermatology_prefect_help_lists_scope_flags() -> None:
    result = _run([sys.executable, str(DERM_PREFECT_FLOW), "--help"])
    assert result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    for flag in (
        "--resume-from",
        "--go-until",
        "--run-id",
        "--datasets",
        "--model-types",
        "--device",
        "--epochs",
        "--batch-size",
        "--pretrained",
        "--no-pretrained",
        "--figures",
        "--no-figures",
        "--group-views",
        "--no-group-views",
        "--augmentation",
        "--no-augmentation",
        "--no-recommendations",
        "--explain",
        "--no-explain",
    ):
        assert flag in output, f"dermatology flow --help does not list {flag}"


def test_dermatology_prefect_runs_every_stage_script_bash_runs() -> None:
    """A stage bash runs but the flow does not is a silently shorter pipeline."""
    flow_source = DERM_PREFECT_FLOW.read_text(encoding="utf-8")
    for script_name in _derm_bash_invocations():
        assert f'"{script_name}"' in flow_source, f"flow never launches {script_name}"


def test_dermatology_prefect_forwards_the_same_flags_as_bash() -> None:
    """Every scope flag bash hands a phase runner must reach it from the flow too."""
    flow_source = DERM_PREFECT_FLOW.read_text(encoding="utf-8")
    for script_name, arrays in _derm_bash_invocations().items():
        body = _flow_task_body(flow_source, script_name)
        for array in arrays:
            for flag in _BASH_ARG_ARRAY_TO_FLAGS[array]:
                assert flag in body, f"flow task for {script_name} does not forward {flag}"


def test_dermatology_prefect_stage_numbers_match_the_catalog() -> None:
    """Dermatology numbering is sparse (no 5 or 6); the flow must not renumber."""
    from fairxai.pipeline.stages import DERMATOLOGY_STAGES

    flow_source = DERM_PREFECT_FLOW.read_text(encoding="utf-8")
    gated = {int(n) for n in re.findall(r"_should_run\((\d+)\)", flow_source)}

    assert gated == {stage.number for stage in DERMATOLOGY_STAGES}


def test_dermatology_prefect_explain_stage_is_gated() -> None:
    """The flow needs the same escape hatch from the heaviest stage bash has."""
    flow_source = DERM_PREFECT_FLOW.read_text(encoding="utf-8")

    assert "run_explain" in flow_source
    assert 'os.getenv("RUN_EXPLAIN")' in flow_source, "no RUN_EXPLAIN env fallback"
    assert '"xai"' in flow_source, "explain toggle does not fall back to xai.enabled"


def test_dermatology_bash_parser_accepts_augmentation_toggle() -> None:
    """Augmentation decides whether the frozen-feature cache is usable at all.

    An aug-on vs aug-off comparison has to be expressible as a flag; if it needs a
    config edit between runs, the two runs differ by something the run artifacts
    do not record as a deliberate choice.
    """
    for flag in ("--augmentation", "--no-augmentation"):
        result = _run(["bash", str(DERM_BASH_PIPELINE), flag, "--go-until", "nonexistent_stage"])
        combined = (result.stdout or "") + (result.stderr or "")
        assert f"Unknown argument '{flag}'" not in combined, f"bash pipeline rejects {flag}"


def test_dermatology_bash_forwards_augmentation_only_to_train() -> None:
    """Only the train stage consumes augmentation; the other runners would reject it."""
    invocations = _derm_bash_invocations()
    assert "AUGMENTATION_ARGS" in invocations["train_baseline.py"]
    for script_name, arrays in invocations.items():
        if script_name == "train_baseline.py":
            continue
        assert (
            "AUGMENTATION_ARGS" not in arrays
        ), f"{script_name} must not receive AUGMENTATION_ARGS"
