# Pipeline Module

Pipeline orchestration helpers for staged FairXAI execution.

This module defines stage metadata, stage resolution (`resume-from`/`go-until`),
and checkpoint marker management used by orchestrators.

## Files

| File | Purpose |
|------|---------|
| `stages.py` | Stage registry, resolution helpers, stage-range selection, checkpoint utilities |
| `__init__.py` | Public re-exports of pipeline stage APIs |

## Public API

- Stage registry and lookups:
  - `STAGES`
  - `STAGE_BY_NAME`
  - `STAGE_BY_NUMBER`

- Data model:
  - `PipelineStage`

- Flow control helpers:
  - `resolve_stage(identifier)`
  - `get_stage_range(resume_from=None, go_until=None)`

- Checkpoint helpers:
  - `mark_stage_complete(run_root, stage)`
  - `get_completed_stages(run_root)`
  - `validate_prior_stages(run_root, target_stage, logger=None)`

## Stage Semantics

Stages are ordered and inclusive when selecting ranges:

- `resume_from` starts at the specified stage
- `go_until` stops after the specified stage

Validation errors are raised if a range is invalid (e.g., start after end).

## Checkpoint Contract

Completion markers are written under:

```text
{run_root}/.checkpoints/
```

Each stage writes a marker file named with number and stage name (e.g.
`2_profile.done`). Optional artifact glob checks can also validate stage
completeness.

## Usage Example

```python
from fairxai.pipeline import get_stage_range, resolve_stage

stages = get_stage_range(resume_from="profile", go_until="mitigation")
for stage in stages:
    print(stage.number, stage.name)

stage = resolve_stage("phase3")
print(stage.name)  # recommend
```
