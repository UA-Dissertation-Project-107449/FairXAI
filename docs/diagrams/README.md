# Architecture Diagrams

Source `.drawio` files and their renders. Edit the `.drawio` in
[diagrams.net](https://app.diagrams.net/) (File → Open), then re-export next to
it under the same base name so the embeds in the docs keep working.

Each figure ships in three files, one per job. The `.drawio` is the versioned
master and the only one worth editing, the `.png` is what Markdown embeds point
at, and the `.svg` goes in the dissertation. Export settings match the
DataLenzAI figures and are recorded once, in
`Code/WebApp_DataLenzAI/docs/diagrams/README.md`.

## Figures

| ID | Figure | Files |
|----|--------|-------|
| F1 | Stage pipeline, both domains | `F1.drawio`, `F1.drawio.png`, `F1.drawio.svg` |
| E1 | Experimental design, both tracks | `E1.drawio`, `E1.drawio.png`, `E1.drawio.svg` |
| D1 | Dataset-processing workflow | `D1.drawio`, `D1.drawio.png`, `D1.drawio.svg` |

### F1, stage pipeline

F1 is embedded in the [root README](../../README.md), above the stage table.

It draws both domains in one figure: the shared `load → profile → recommend`
front end, which is the part DataLenzAI reaches through the CLI, the cardiac
branch running through `tune` and `select_features` to `compare`, and the
dermatology branch, which skips stages 5 and 6 and ends `compare → explain →
mitigate`. Each branch is labeled as its domain's track, and Start and End
markers bound the run. The dotted edges are the optional hooks driven by
`RUN_GROUPING`, `RUN_SIMILARITY` and `RUN_AGE_BINNING`.

`src/fairxai/pipeline/stages.py` is the source of truth. If the figure and the
catalog disagree, the catalog is right and the figure needs redrawing.

### E1 and D1, dissertation figures

E1 and D1 were made for the dissertation rather than for this repo. They are
kept here because they help to interpret the pipeline and show what was
actually run. The `§` marks in E1's boxes refer to dissertation sections and
can be ignored.

### E1, experimental design

E1 reads at the level of the experiments rather than the stages. The cardiac track (Cleveland,
four-site, Cardiovascular) runs the optional tuning, feature-selection and
subgroup-discovery studies that fix one configuration, measures it per group,
then adds uncertainty, the binning re-cut and the trade-off sweep. The
dermatology track (PAD-UFES-20) measures one configuration per group and adds
the post hoc backbone check.

### D1, dataset-processing workflow

D1 runs from the five sources through the harmonization rules R1–R7 to one
canonical table per domain, then splits into the tabular and image tracks and
ends at the record counts each cohort keeps. SCIN is drawn as profiling only.

The DataLenzAI figures (A0 system context, W1–W4) live in that repo, at
`Code/WebApp_DataLenzAI/docs/diagrams/`.

## Last checked

| ID | Last checked | Against | State |
|----|--------------|---------|-------|
| F1 | 2026-10-21 | `src/fairxai/pipeline/stages.py` | Current. Uses the canonical stage names rather than the pre-rename aliases. The band around the shared stages reads "command-invocable from DataLenzAI", since those three are reachable through the CLI rather than run by the web platform itself. Redrawn on this date: Start and End markers added, and the two branches labeled "Cardiac-only track" and "Dermatology-only track". No stage changed. |
| E1 | 2026-10-21 | The final experiment runs | Current. Added to this repo on 2026-10-28. |
| D1 | 2026-10-21 | The final experiment runs | Current. Added to this repo on 2026-10-28. |

## Not drawn

A module dependency figure (F2, the six layers from `utils`/`cli` up to
`comparison`/`viz`) was specified but never drawn. The same information is in
[`architecture/modules.md`](../architecture/modules.md) as prose and a table,
which is why it stayed unbuilt.
