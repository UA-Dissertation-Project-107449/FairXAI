"""Run IDs and paths shared by the thesis scripts, read from runs.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
FAIRXAI = HERE.parents[1]
OUTPUT = FAIRXAI / "output"
CARD = OUTPUT / "cardiac"
DERM = OUTPUT / "dermatology"
DEFAULT_OUT = OUTPUT / "thesis"

_CFG = yaml.safe_load((HERE / "runs.yaml").read_text())

RUN_CARDIAC = _CFG["cardiac"]["uci"]
RUN_C70_SUB = _CFG["cardiac"]["cardio70k_subsample"]
RUN_C70 = _CFG["cardiac"]["cardio70k"]
RUN_DERM_AUG = _CFG["dermatology"]["augmented"]
RUN_DERM_CACHED = _CFG["dermatology"]["cached_no_aug"]

CALIBRATION = OUTPUT / _CFG["studies"]["null_calibration"]
SYNTHETIC = OUTPUT / _CFG["studies"]["synthetic"]
FOUR_SITE_DATA = FAIRXAI / _CFG["data"]["four_site"]
USAB = (FAIRXAI / _CFG["usability"]).resolve()

UCI = ["cleveland_uci", "four_site_uci"]
UCI_MODELS = ["logistic_regression", "random_forest", "svm", "xgboost"]
C70_MODELS = ["logistic_regression", "random_forest", "xgboost"]
# (run, cohorts) for every cardiac cohort the chapter reports.
CARDIAC_COHORTS = ((RUN_CARDIAC, UCI), (RUN_C70, ["cardio70k"]))

# The pairwise tables carry two exact redundancies: equalized_odds/tpr repeats
# equal_opportunity/tpr, and fnr is 1 - tpr. Counting "distinct" drops both.
REDUNDANT = {("equalized_odds", "tpr"), ("equalized_odds", "fnr")}


def split_of(cohort: str) -> str:
    """The prediction split the chapter quotes: held-out for Cardio70k, pooled CV otherwise."""
    return "test" if cohort == "cardio70k" else "cv"


def models_of(cohort: str) -> list[str]:
    return C70_MODELS if cohort == "cardio70k" else UCI_MODELS


def run_dir(run: str) -> Path:
    """A cardiac or dermatology run directory."""
    for base in (CARD, DERM):
        if (base / "runs" / run).is_dir():
            return base / "runs" / run
    raise FileNotFoundError(f"run {run} not under {CARD} or {DERM}")


def study_dir(kind: str, run: str) -> Path:
    """The newest ``studies/<kind>/<run>_<timestamp>`` folder for a cardiac run."""
    found = sorted((CARD / "studies" / kind).glob(f"{run}_*"))
    if not found:
        raise FileNotFoundError(f"no {kind} study for {run}")
    return found[-1]


def rel(path: Path) -> str:
    """A path as quoted in the ledger: relative to the FairXAI root when possible."""
    try:
        return str(path.relative_to(FAIRXAI))
    except ValueError:
        return str(path)
