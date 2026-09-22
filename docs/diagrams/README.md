# Architecture Diagrams

Source `.drawio` files and their renders. Edit the `.drawio` in
[diagrams.net](https://app.diagrams.net/) (File → Open), then re-export next to
it under the same base name so the embeds in the docs keep working.

F1 ships in three files, one per job. The `.drawio` is the versioned master and
the only one worth editing, the `.png` is what the README embed points at, and
the `.svg` goes in the dissertation. Export settings match the DataLenzAI
figures and are recorded once, in
`Code/WebApp_DataLenzAI/docs/diagrams/README.md`.

## Figures

| ID | Figure | Files |
|----|--------|-------|
| F1 | Stage pipeline, both domains | `F1.drawio`, `F1.drawio.png`, `F1.drawio.svg` |

F1 is embedded in the [root README](../../README.md), above the stage table.

It draws both domains in one figure: the shared `load → profile → recommend`
front end, which is the part DataLenzAI reaches through the CLI, the cardiac
branch running through `tune` and `select_features` to `compare`, and the
dermatology branch, which skips stages 5 and 6 and ends `compare → explain →
mitigate`. The dotted edges are the optional hooks driven by `RUN_GROUPING`,
`RUN_SIMILARITY` and `RUN_AGE_BINNING`.

`src/fairxai/pipeline/stages.py` is the source of truth. If the figure and the
catalog disagree, the catalog is right and the figure needs redrawing.

The DataLenzAI figures (A0 system context, W1–W4) live in that repo, at
`Code/WebApp_DataLenzAI/docs/diagrams/`.

## Last checked

| ID | Last checked | Against | State |
|----|--------------|---------|-------|
| F1 | 2026-09-22 | `src/fairxai/pipeline/stages.py` | Current. Uses the canonical stage names rather than the pre-rename aliases. Re-exported on this date to match the DataLenzAI settings. |

## Not drawn

A module dependency figure (F2, the six layers from `utils`/`cli` up to
`comparison`/`viz`) was specified but never drawn. The same information is in
[`architecture/modules.md`](../architecture/modules.md) as prose and a table,
which is why it stayed unbuilt.
