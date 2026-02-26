"""Pipeline orchestration utilities — stage definitions, checkpointing, and flow control."""

from fairxai.pipeline.stages import (
    STAGES,
    STAGE_BY_NAME,
    STAGE_BY_NUMBER,
    PipelineStage,
    resolve_stage,
    get_stage_range,
    validate_prior_stages,
    mark_stage_complete,
    get_completed_stages,
)

__all__ = [
    "STAGES",
    "STAGE_BY_NAME",
    "STAGE_BY_NUMBER",
    "PipelineStage",
    "resolve_stage",
    "get_stage_range",
    "validate_prior_stages",
    "mark_stage_complete",
    "get_completed_stages",
]
