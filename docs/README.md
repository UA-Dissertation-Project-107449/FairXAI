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
| Design decisions and known limits | [architecture/decisions.md](architecture/decisions.md) |
| Running and marking tests | [guides/testing.md](guides/testing.md) |
| Attribute binning experiment | [reference/attribute-binning.md](reference/attribute-binning.md) |
| Dermatology pipeline | [DERMATOLOGY.md](DERMATOLOGY.md) |

## Notes

| Topic | Read |
|-------|------|
| PAD-UFES-20 all-benign "unknown demographics" cohort | [dermatology_unknown_demographics.md](dermatology_unknown_demographics.md) |
| loky "worker stopped" warning in the cardiac sweep (investigation, not fixed) | [cardiac_loky_worker_warning.md](cardiac_loky_worker_warning.md) |
| Architecture diagrams (`.drawio` sources and renders) | [diagrams/README.md](diagrams/README.md) |

## Sections

- `architecture/` - repo layout, module dependencies, pipeline control, and design decisions.
- `guides/` - everyday developer and researcher workflows.
- `reference/` - stable contracts for results, plots, and experiment-specific behavior.
- `research/` - dissertation-facing evidence notes and interpretation checkpoints.
- `planning/` - roadmap, deferred work, and implementation status.
- `diagrams/` - draw.io sources and their PNG/SVG renders.

## Documentation Rules

- Root `README.md` explains how to install, run, and navigate.
- Folder READMEs explain local purpose, important files, public APIs, config inputs, outputs, and tests.
- `docs/guides/style-guide.md` defines the README/docstring baseline.
- Code, configs, CI workflows, and `fairxai.pipeline.stages.STAGES` are source of truth when docs drift.
