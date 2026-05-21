# Pipeline Module

Stage registry, aliases, checkpoint marker helpers, and resume validation for
FairXAI orchestrators.

## Files

| File | Purpose |
|------|---------|
| `stages.py` | `PipelineStage`, ordered `STAGES`, resolution, ranges, checkpoints |
| `__init__.py` | Public exports |

## Public API

- `STAGES`
- `STAGE_BY_NAME`
- `STAGE_BY_NUMBER`
- `PipelineStage`
- `resolve_stage`
- `get_stage_range`
- `validate_prior_stages`
- `mark_stage_complete`
- `get_completed_stages`

## Current Stages

| # | Name |
|---|------|
| 1 | `load` |
| 2 | `profile` |
| 3 | `recommend` |
| 4 | `preprocess` |
| 5 | `hpo_study` |
| 6 | `feature_selection_study` |
| 7 | `train` |
| 8 | `assess` |
| 9 | `attribute_binning` |
| 10 | `mitigation` |
| 11 | `combinatorial` |
| 12 | `compare` |

## Checkpoint Contract

Checkpoint markers live under:

```text
output/cardiac/runs/<run_id>/.checkpoints/
```

Resume validation is marker-based. Some stages have artifacts, but checkpoint
markers are the stable orchestration contract.

## Usage

```python
from fairxai.pipeline import get_stage_range, resolve_stage

stage = resolve_stage("train")
window = get_stage_range(resume_from="profile", go_until="assess")
```

## Related

- Flow control docs: [../../../docs/architecture/pipeline-flow-control.md](../../../docs/architecture/pipeline-flow-control.md)
- Cheat sheet: [../../../docs/guides/cheat-sheet.md](../../../docs/guides/cheat-sheet.md)
