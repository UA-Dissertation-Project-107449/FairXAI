# FairXAI Documentation

This directory is the documentation hub for the FairXAI research codebase.
Use this page as the reading order before diving into individual module
READMEs under `src/fairxai/`.

## Start Here

| Need | Read |
|------|------|
| Fast command reference | [guides/cheat-sheet.md](guides/cheat-sheet.md) |
| Pipeline stages, resume, and checkpoints | [architecture/pipeline-flow-control.md](architecture/pipeline-flow-control.md) |
| Source module responsibilities | [architecture/modules.md](architecture/modules.md) |
| Experiment JSON/table contracts | [reference/results-schema.md](reference/results-schema.md) |
| Plotting APIs and figure outputs | [reference/plots.md](reference/plots.md) |
| Current implementation status | [planning/roadmap.md](planning/roadmap.md) |
| Dissertation interpretation checkpoint | [research/dissertation-evidence-check.md](research/dissertation-evidence-check.md) |

## Sections

- `architecture/` - repo layout, module dependencies, pipeline control, and design decisions.
- `guides/` - everyday developer and researcher workflows.
- `reference/` - stable contracts for results, plots, and experiment-specific behavior.
- `research/` - dissertation-facing evidence notes and interpretation checkpoints.
- `planning/` - roadmap, deferred work, and implementation status.

## Documentation Rules

- Root `README.md` explains how to install, run, and navigate.
- Folder READMEs explain local purpose, important files, public APIs, config inputs, outputs, and tests.
- `docs/guides/style-guide.md` defines the README/docstring baseline.
- Code, configs, CI workflows, and `fairxai.pipeline.stages.STAGES` are source of truth when docs drift.
