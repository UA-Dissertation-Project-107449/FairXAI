"""
Pipeline stage definitions, checkpoint management, and flow-control helpers.

Stages are the ordered phases of a cardiac (or future) pipeline.
The orchestrators (Prefect flow / bash script) use these to implement
``--resume-from`` and ``--go-until`` semantics without coupling the
individual analysis scripts to any checkpoint logic.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineStage:
    """Immutable descriptor for one pipeline phase."""

    number: int
    name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""
    # Glob patterns (relative to project root) whose *existence* proves
    # the stage ran.  ``{run_root}`` is replaced at validation time.
    checkpoint_artifacts: tuple[str, ...] = field(default_factory=tuple)

    # --- helpers ----------------------------------------------------------
    @property
    def all_names(self) -> tuple[str, ...]:
        """Return canonical name + all aliases."""
        return (self.name, *self.aliases)

    @property
    def marker_filename(self) -> str:
        return f"{self.number}_{self.name}.done"

    def __str__(self) -> str:  # noqa: D105
        return f"[{self.number}/{len(STAGES)}] {self.name}"


# ---------------------------------------------------------------------------
# Stage registry (order matters!)
# ---------------------------------------------------------------------------

STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        number=1,
        name="load",
        description="Load & standardize raw datasets",
        checkpoint_artifacts=("data/raw/cardiac/*_standardized.csv",),
    ),
    PipelineStage(
        number=2,
        name="profile",
        aliases=("profiling",),
        description="Profile datasets (complexity + fairness)",
        checkpoint_artifacts=("{run_root}/profiling/*_data_profile.json",),
    ),
    PipelineStage(
        number=3,
        name="recommend",
        aliases=("recommendations", "triage"),
        description="Generate fairness triage recommendations",
        checkpoint_artifacts=("{run_root}/recommendations/*_triage.json",),
    ),
    PipelineStage(
        number=4,
        name="preprocess",
        aliases=("preprocessing",),
        description="Split, scale and generate fairness profiles",
        checkpoint_artifacts=("data/processed/cardiac/*/*_train.csv",),
    ),
    PipelineStage(
        number=5,
        name="hpo_study",
        aliases=("hpo",),
        description="Run hyperparameter optimisation study",
        checkpoint_artifacts=(),
    ),
    PipelineStage(
        number=6,
        name="feature_selection_study",
        aliases=("feature_selection", "fs_study"),
        description="Run feature-selection ablation study",
        checkpoint_artifacts=(),
    ),
    PipelineStage(
        number=7,
        name="train",
        aliases=("baseline", "training"),
        description="Train baseline model(s)",
        # NOTE: we rely on the checkpoint marker rather than model artefacts
        # (.pkl) here because experiment stages may produce many models that
        # are *not* persisted individually.
        checkpoint_artifacts=(),
    ),
    PipelineStage(
        number=8,
        name="assess",
        aliases=("fairness", "assessment"),
        description="Assess post-prediction fairness",
        checkpoint_artifacts=(
            "{run_root}/baseline/prediction_fairness/fairness_report.json",
            "{run_root}/baseline/fairness/*_fairness_assessment.json",
        ),
    ),
    PipelineStage(
        number=9,
        name="attribute_binning",
        description="Attribute binning strategy analysis",
        checkpoint_artifacts=(),
    ),
    PipelineStage(
        number=10,
        name="mitigation",
        description="Mitigation technique comparison",
        checkpoint_artifacts=(),
    ),
    PipelineStage(
        number=11,
        name="combinatorial",
        aliases=("combo",),
        description="Combinatorial experiments",
        checkpoint_artifacts=(),
    ),
    PipelineStage(
        number=12,
        name="compare",
        aliases=("comparison",),
        description="Experiment comparison & reporting",
        checkpoint_artifacts=(),
    ),
)

# Fast look-ups (built once at import time)
STAGE_BY_NUMBER: Dict[int, PipelineStage] = {s.number: s for s in STAGES}
STAGE_BY_NAME: Dict[str, PipelineStage] = {}
for _s in STAGES:
    for _n in _s.all_names:
        STAGE_BY_NAME[_n.lower()] = _s


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_stage(identifier: str) -> PipelineStage:
    """
    Resolve a user-supplied stage identifier to a ``PipelineStage``.

    Accepts:
      - A stage name or alias (case-insensitive): ``"profile"``, ``"recommendations"``
      - A stage number (as string): ``"2"``
      - A prefixed form: ``"phase2"``, ``"stage2"``

    Raises ``ValueError`` with a helpful message on unknown input.
    """
    raw = identifier.strip().lower()

    # Try as a plain number
    if raw.isdigit():
        num = int(raw)
        if num in STAGE_BY_NUMBER:
            return STAGE_BY_NUMBER[num]

    # Try stripping common prefixes
    for prefix in ("phase", "stage", "step"):
        if raw.startswith(prefix):
            suffix = raw[len(prefix) :]
            if suffix.isdigit() and int(suffix) in STAGE_BY_NUMBER:
                return STAGE_BY_NUMBER[int(suffix)]

    # Try as a name / alias
    if raw in STAGE_BY_NAME:
        return STAGE_BY_NAME[raw]

    names = ", ".join(f"{s.number}={s.name}" for s in STAGES)
    raise ValueError(f"Unknown pipeline stage '{identifier}'. " f"Valid stages: {names}")


def get_stage_range(
    resume_from: Optional[str] = None,
    go_until: Optional[str] = None,
) -> List[PipelineStage]:
    """
    Return the ordered list of stages to execute.

    Both ``resume_from`` and ``go_until`` are *inclusive*.
    Omitting either implies "first" / "last" respectively.
    """
    start = resolve_stage(resume_from).number if resume_from else 1
    end = resolve_stage(go_until).number if go_until else STAGES[-1].number

    if start > end:
        s, e = resolve_stage(resume_from), resolve_stage(go_until)  # type: ignore[arg-type]
        raise ValueError(
            f"--resume-from ({s.name}, #{s.number}) is after "
            f"--go-until ({e.name}, #{e.number}). Nothing to run."
        )
    return [s for s in STAGES if start <= s.number <= end]


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

_CHECKPOINTS_DIR = ".checkpoints"


def _checkpoints_dir(run_root: Path) -> Path:
    return run_root / _CHECKPOINTS_DIR


def mark_stage_complete(run_root: Path, stage: PipelineStage) -> Path:
    """
    Write a completion marker for *stage* under ``run_root/.checkpoints/``.

    Returns the path to the marker file.
    """
    ckpt_dir = _checkpoints_dir(run_root)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    marker = ckpt_dir / stage.marker_filename
    payload = {
        "stage": stage.name,
        "number": stage.number,
        "completed_at": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n")
    logger.debug("Checkpoint written: %s", marker)
    return marker


def get_completed_stages(run_root: Path) -> List[PipelineStage]:
    """Return the list of stages that have a completion marker on disk."""
    ckpt_dir = _checkpoints_dir(run_root)
    if not ckpt_dir.is_dir():
        return []
    completed = []
    for stage in STAGES:
        if (ckpt_dir / stage.marker_filename).exists():
            completed.append(stage)
    return completed


def validate_prior_stages(
    run_root: Path,
    resume_from: PipelineStage,
    _project_root: Path,
) -> None:
    """
    Validate that every stage *before* ``resume_from`` has a checkpoint marker.

    Resume validation is marker-based by design. Artifact layouts can evolve
    over time, but checkpoint markers provide a stable, stage-level contract.

    Raises ``RuntimeError`` with a detailed message on validation failure.
    """
    prior = [s for s in STAGES if s.number < resume_from.number]
    if not prior:
        return  # resuming from stage 1 — nothing to validate

    errors: list[str] = []
    completed = get_completed_stages(run_root)
    completed_nums = {s.number for s in completed}

    for stage in prior:
        if stage.number not in completed_nums:
            errors.append(
                f"  Stage {stage.number} ({stage.name}): "
                f"no completion marker at "
                f"{_checkpoints_dir(run_root) / stage.marker_filename}"
            )

    if errors:
        detail = "\n".join(errors)
        raise RuntimeError(
            f"Cannot resume from '{resume_from.name}' (stage {resume_from.number}).\n"
            f"The following prior stages failed validation:\n{detail}\n\n"
            f"Run root: {run_root}\n"
            f"Hint: re-run the full pipeline or an earlier --resume-from to "
            f"generate the missing checkpoint markers."
        )
