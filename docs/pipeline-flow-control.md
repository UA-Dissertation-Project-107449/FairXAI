# Pipeline Flow Control

Both the **Prefect flow** (`flows/cardiac_pipeline.py`) and the **bash pipeline** (`scripts/cardiac/cardiac_pipeline.sh`) share identical flow-control semantics via `--resume-from` / `--go-until` flags.

---

## Stages

| # | Name           | Aliases                        | Description                              |
|---|----------------|--------------------------------|------------------------------------------|
| 1 | `load`         | —                              | Load & standardize raw datasets          |
| 2 | `profile`      | `profiling`                    | Profile datasets (complexity + fairness) |
| 3 | `recommend`    | `recommendations`, `triage`    | Generate fairness triage recommendations |
| 4 | `preprocess`   | `preprocessing`                | Split, scale, generate fairness profiles |
| 5 | `train`        | `baseline`, `training`         | Train baseline model(s)                  |
| 6 | `assess`       | `fairness`, `assessment`       | Assess post-prediction fairness          |
| 7 | `attribute_binning`  | `age_binning`                  | Attribute binning strategy analysis      |
| 8 | `mitigation`   | —                              | Mitigation technique comparison          |
| 9 | `combinatorial`| `combo`                        | Combinatorial experiments                |
| 10| `compare`      | `comparison`                   | Experiment comparison & reporting        |

Stages can be referenced by **name**, **alias**, or **number** (e.g., `profile`, `profiling`, `2`, `phase2` all resolve to stage 2).

### Stage Dependencies

```
load (1)
  └─ profile (2)
        ├─ recommend (3)       [independent branch]
        └─ preprocess (4)
              └─ train (5)
                    └─ assess (6)
                          ├─ attribute_binning (7)  [optional]
                          ├─ mitigation (8)        [optional]
                          └─ combinatorial (9)     [optional]
                                └─ compare (10)    [optional]
```

---

## Flags

| Flag | Prefect CLI | Bash env var | Description |
|------|-------------|--------------|-------------|
| Resume point | `--resume-from <stage>` | `RESUME_FROM=<stage>` | First stage to execute (inclusive). Triggers artifact validation for prior stages. |
| Stop point | `--go-until <stage>` | `GO_UNTIL=<stage>` | Last stage to execute (inclusive). Stages after this are skipped. |
| Run ID | `--run-id <id>` | `RUN_ID=<id>` | Explicit run ID. On resume without this, defaults to `latest_run` symlink. |
| Dataset scope | `--datasets <d1> [d2 ...]` | — (CLI only) | Optional dataset override propagated to stages. |
| Model scope | `--model-types <m1> [m2 ...]` | — (CLI only) | Optional model-type override (baseline/combinatorial stages). |
| Skip attr binning | `--no-attribute-binning` | `RUN_ATTRIBUTE_BINNING=false` | Skip stage 7 even if in active range. |
| Skip mitigation | `--no-mitigation` | `RUN_MITIGATION=false` | Skip stage 8 even if in active range. |
| Skip combinatorial | `--no-combinatorial` | `RUN_COMBINATORIAL=false` | Skip stage 9 even if in active range. |
| Skip comparison | `--no-comparison` | `RUN_COMPARISON=false` | Skip stage 10 even if in active range. |
| Verbose | `-v` / `--verbose` | `VERBOSE=true` | Verbose logging. |

### Dataset and model override precedence

Override precedence is:

1. CLI flags (`--datasets`, `--model-types`)
2. Config values
3. Code defaults / auto-discovery

No environment-variable override layer is used for dataset/model scope.

---

## Usage Examples

### Run only through profiling (stages 1–2)

```bash
# Bash
GO_UNTIL=profile bash scripts/cardiac/cardiac_pipeline.sh

# Prefect
python flows/cardiac_pipeline.py --go-until profile
```

### Run only through recommendations (stages 1–3)

```bash
GO_UNTIL=recommend bash scripts/cardiac/cardiac_pipeline.sh
python flows/cardiac_pipeline.py --go-until recommend
```

### Run the core pipeline without experiments (stages 1–6)

```bash
GO_UNTIL=assess bash scripts/cardiac/cardiac_pipeline.sh
python flows/cardiac_pipeline.py --go-until assess
```

### Run only Cleveland with selected models

```bash
# Bash orchestrator (CLI flags)
bash scripts/cardiac/cardiac_pipeline.sh \
      --datasets cleveland \
      --model-types logistic_regression xgboost

# Prefect orchestrator
python flows/cardiac_pipeline.py \
      --datasets cleveland \
      --model-types logistic_regression xgboost
```

### Resume partial run with dataset/model scope

```bash
bash scripts/cardiac/cardiac_pipeline.sh \
      --resume-from train \
      --go-until compare \
      --datasets cleveland \
      --model-types logistic_regression

python flows/cardiac_pipeline.py \
      --resume-from train \
      --go-until compare \
      --datasets cleveland \
      --model-types logistic_regression
```

### Resume a failed run from preprocessing

```bash
# With explicit run ID
RESUME_FROM=preprocess RUN_ID=run_20260224_143000_12345_abc123 bash scripts/cardiac/cardiac_pipeline.sh

# Auto-detect latest run
RESUME_FROM=preprocess bash scripts/cardiac/cardiac_pipeline.sh

# Prefect equivalents
python flows/cardiac_pipeline.py --resume-from preprocess --run-id run_20260224_143000_12345_abc
python flows/cardiac_pipeline.py --resume-from preprocess
```

### Resume from training, stop after assessment

```bash
RESUME_FROM=train GO_UNTIL=assess bash scripts/cardiac/cardiac_pipeline.sh
python flows/cardiac_pipeline.py --resume-from train --go-until assess
```

### Run a single stage (e.g., re-run only recommendations)

```bash
RESUME_FROM=recommend GO_UNTIL=recommend bash scripts/cardiac/cardiac_pipeline.sh
python flows/cardiac_pipeline.py --resume-from recommend --go-until recommend
```

---

## Checkpoints

Each stage writes a completion marker on success to:

```
output/cardiac/runs/<run_id>/.checkpoints/<number>_<name>.done
```

Example after a full run:

```
.checkpoints/
├── 1_load.done
├── 2_profile.done
├── 3_recommend.done
├── 4_preprocess.done
├── 5_train.done
├── 6_assess.done
├── 7_attribute_binning.done
├── 8_mitigation.done
├── 9_combinatorial.done
└── 10_compare.done
```

Each `.done` file is JSON with a timestamp and hostname:

```json
{
  "stage": "profile",
  "number": 2,
  "completed_at": "2026-02-24T14:30:00+00:00",
  "hostname": "workstation",
  "pid": 12345
}
```

### Resume validation

When `--resume-from` is set, the orchestrator checks that **every prior stage** has a `.done` marker. If any are missing, the pipeline fails immediately with a clear error message listing the missing stages.

For stages that declare artifact patterns (e.g., stage 1 expects `data/raw/cardiac/*_standardized.csv`), the Prefect flow also verifies at least one file matches each glob. The bash pipeline checks markers only (simpler, lighter).

---

## Design Decisions

1. **`--resume-from` + `--go-until`** over `--start-at` / `--stop-after`: "resume" clearly implies failure recovery with artifact reuse; "go-until" reads naturally as "the last thing to do".

2. **Checkpoint markers on disk** (`.checkpoints/` dir) over a database or JSON state file: simpler, no dependencies, survives across bash and Prefect, easy to inspect with `ls`.

3. **Validation in the orchestrator, not in individual scripts**: the 10 analysis scripts remain decoupled from any checkpoint logic. They don't know they're being orchestrated. This keeps them reusable standalone.

4. **Stage 5 (`train`) uses only the checkpoint marker**, not `.pkl` files — experiment stages produce many models that aren't individually persisted, so file-based artifact checks would be unreliable. The checkpoint marker is the source of truth.

5. **Run ID auto-resolves from `latest_run` symlink** when resuming without an explicit `--run-id`. The symlink is already maintained by `runner_utils.update_latest_pointer()`.

6. **Names + numbers both accepted**: more code, but more user-friendly. `3`, `recommend`, `recommendations`, `triage`, `phase3` all resolve to the same stage.

7. **Optional stages (7–10) have two independent gates**: the range gate (`--resume-from` / `--go-until`) and the feature toggle (`RUN_ATTRIBUTE_BINNING`, `--no-attribute-binning`, etc.). Both must pass for the stage to execute.
