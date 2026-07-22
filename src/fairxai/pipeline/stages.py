"""Domain-aware pipeline stages, checkpoints, and flow-control helpers."""

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
    legacy_marker_names: tuple[str, ...] = field(default_factory=tuple)
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

    @property
    def marker_filenames(self) -> tuple[str, ...]:
        """Return the canonical marker followed by accepted legacy markers."""
        return (
            self.marker_filename,
            *(f"{self.number}_{name}.done" for name in self.legacy_marker_names),
        )

    def __str__(self) -> str:  # noqa: D105
        # No domain denominator: a stage does not know which subset owns it, and
        # numbering is sparse (dermatology skips 5-6), so "n/total" would lie.
        return f"[stage {self.number}] {self.name}"


# ---------------------------------------------------------------------------
# Domain stage registry (order matters within each domain)
# ---------------------------------------------------------------------------

CARDIAC_STAGES: tuple[PipelineStage, ...] = (
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
        name="tune",
        aliases=("hpo_study", "hpo"),
        legacy_marker_names=("hpo_study",),
        description="Run hyperparameter optimisation study",
        checkpoint_artifacts=(),
    ),
    PipelineStage(
        number=6,
        name="select_features",
        aliases=("feature_selection_study", "feature_selection", "fs_study"),
        legacy_marker_names=("feature_selection_study",),
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
        name="bin_attributes",
        aliases=("attribute_binning", "age_binning"),
        legacy_marker_names=("attribute_binning",),
        description="Attribute binning strategy analysis",
        checkpoint_artifacts=(),
    ),
    PipelineStage(
        number=10,
        name="mitigate",
        aliases=("mitigation",),
        legacy_marker_names=("mitigation",),
        description="Mitigation technique comparison",
        checkpoint_artifacts=(),
    ),
    PipelineStage(
        number=11,
        name="sweep",
        aliases=("combinatorial", "combo"),
        legacy_marker_names=("combinatorial",),
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

DERMATOLOGY_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(number=1, name="load", description="Load & standardize raw datasets"),
    PipelineStage(number=2, name="profile", aliases=("profiling",), description="Profile datasets"),
    PipelineStage(
        number=3,
        name="recommend",
        # "triage" is kept for parity with the cardiac subset and the bash
        # pipelines; it is a stage alias, unrelated to the ``fairxai triage`` CLI.
        aliases=("recommendations", "triage"),
        description="Generate fairness triage recommendations",
    ),
    PipelineStage(
        number=4,
        name="preprocess",
        aliases=("preprocessing",),
        description="Prepare image metadata and dataset splits",
    ),
    PipelineStage(
        number=7,
        name="train",
        aliases=("baseline", "training"),
        description="Train baseline model(s)",
    ),
    PipelineStage(
        number=8,
        name="assess",
        aliases=("fairness", "assessment"),
        description="Assess post-prediction fairness",
    ),
    PipelineStage(
        number=9,
        name="compare",
        aliases=("comparison",),
        description="Compare baseline models",
    ),
    PipelineStage(
        number=10,
        name="explain",
        aliases=("explainability", "xai"),
        description="Generate model explanations",
    ),
    PipelineStage(
        number=11,
        name="mitigate",
        aliases=("mitigation",),
        legacy_marker_names=("mitigation",),
        description="Run mitigation experiments",
    ),
)

PIPELINE_STAGES: dict[str, tuple[PipelineStage, ...]] = {
    "cardiac": CARDIAC_STAGES,
    "dermatology": DERMATOLOGY_STAGES,
}


def _build_stage_indexes(
    stages: tuple[PipelineStage, ...],
) -> tuple[Dict[int, PipelineStage], Dict[str, PipelineStage]]:
    by_number = {stage.number: stage for stage in stages}
    by_name: Dict[str, PipelineStage] = {}
    for stage in stages:
        for name in stage.all_names:
            by_name[name.lower()] = stage
    return by_number, by_name


STAGE_INDEXES = {domain: _build_stage_indexes(stages) for domain, stages in PIPELINE_STAGES.items()}
CARDIAC_STAGE_BY_NUMBER, CARDIAC_STAGE_BY_NAME = STAGE_INDEXES["cardiac"]
DERMATOLOGY_STAGE_BY_NUMBER, DERMATOLOGY_STAGE_BY_NAME = STAGE_INDEXES["dermatology"]

# Backwards-compatible cardiac exports. New internal callers should select a
# domain explicitly through CARDIAC_STAGES/DERMATOLOGY_STAGES or ``domain=``.
STAGES = CARDIAC_STAGES
STAGE_BY_NUMBER = CARDIAC_STAGE_BY_NUMBER
STAGE_BY_NAME = CARDIAC_STAGE_BY_NAME


def get_stages(domain: str = "cardiac") -> tuple[PipelineStage, ...]:
    """Return the ordered stage subset for *domain*."""
    normalized = domain.strip().lower()
    try:
        return PIPELINE_STAGES[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(PIPELINE_STAGES))
        raise ValueError(f"Unknown pipeline domain '{domain}'. Valid domains: {valid}") from exc


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_stage(identifier: str, domain: str = "cardiac") -> PipelineStage:
    """
    Resolve a user-supplied stage identifier to a ``PipelineStage``.

    Accepts:
      - A stage name or alias (case-insensitive): ``"profile"``, ``"recommendations"``
      - A stage number (as string): ``"2"``
      - A prefixed form: ``"phase2"``, ``"stage2"``

    Raises ``ValueError`` with a helpful message on unknown input.
    """
    stages = get_stages(domain)
    stage_by_number, stage_by_name = STAGE_INDEXES[domain.strip().lower()]
    raw = identifier.strip().lower()

    # Try as a plain number
    if raw.isdigit():
        num = int(raw)
        if num in stage_by_number:
            return stage_by_number[num]

    # Try stripping common prefixes
    for prefix in ("phase", "stage", "step"):
        if raw.startswith(prefix):
            suffix = raw[len(prefix) :]
            if suffix.isdigit() and int(suffix) in stage_by_number:
                return stage_by_number[int(suffix)]

    # Try as a name / alias
    if raw in stage_by_name:
        return stage_by_name[raw]

    names = ", ".join(f"{s.number}={s.name}" for s in stages)
    raise ValueError(f"Unknown pipeline stage '{identifier}'. " f"Valid stages: {names}")


def get_stage_range(
    resume_from: Optional[str] = None,
    go_until: Optional[str] = None,
    domain: str = "cardiac",
) -> List[PipelineStage]:
    """
    Return the ordered list of stages to execute.

    Both ``resume_from`` and ``go_until`` are *inclusive*.
    Omitting either implies "first" / "last" respectively.
    """
    stages = get_stages(domain)
    start = resolve_stage(resume_from, domain).number if resume_from else stages[0].number
    end = resolve_stage(go_until, domain).number if go_until else stages[-1].number

    if start > end:
        s = resolve_stage(resume_from, domain)  # type: ignore[arg-type]
        e = resolve_stage(go_until, domain)  # type: ignore[arg-type]
        raise ValueError(
            f"--resume-from ({s.name}, #{s.number}) is after "
            f"--go-until ({e.name}, #{e.number}). Nothing to run."
        )
    return [s for s in stages if start <= s.number <= end]


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


def get_completed_stages(run_root: Path, domain: str = "cardiac") -> List[PipelineStage]:
    """Return the list of stages that have a completion marker on disk."""
    ckpt_dir = _checkpoints_dir(run_root)
    if not ckpt_dir.is_dir():
        return []
    completed = []
    for stage in get_stages(domain):
        if any((ckpt_dir / marker).exists() for marker in stage.marker_filenames):
            completed.append(stage)
    return completed


def validate_prior_stages(
    run_root: Path,
    resume_from: PipelineStage,
    _project_root: Path,
    domain: str = "cardiac",
) -> None:
    """
    Validate that every stage *before* ``resume_from`` has a checkpoint marker.

    Resume validation is marker-based by design. Artifact layouts can evolve
    over time, but checkpoint markers provide a stable, stage-level contract.

    Raises ``RuntimeError`` with a detailed message on validation failure.
    """
    prior = [s for s in get_stages(domain) if s.number < resume_from.number]
    if not prior:
        return  # resuming from stage 1 — nothing to validate

    errors: list[str] = []
    completed = get_completed_stages(run_root, domain=domain)
    completed_nums = {s.number for s in completed}

    for stage in prior:
        if stage.number not in completed_nums:
            errors.append(
                f"  Stage {stage.number} ({stage.name}): "
                f"no completion marker matching "
                f"{', '.join(stage.marker_filenames)} under {_checkpoints_dir(run_root)}"
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
