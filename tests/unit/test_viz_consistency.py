"""Guards on the figure conventions the dissertation figures depend on.

Three things are pinned here, each of which was inconsistent enough to either
crash a notebook or produce a figure that looks wrong next to its neighbours:

* ``PALETTE_DATASET`` must cover the dataset ids the pipeline actually runs.
  Seaborn raises on a dict palette that is missing a hue level, so a stale key
  is not a cosmetic fallback -- it is a hard failure at plot time.
* Every figure in ``fairxai.viz`` must be saved at the same resolution, or the
  low-DPI ones read as blurry beside the rest at print size.
* ``generate_dissertation_plots.py`` must set a global rcParams theme, which is
  the one place typography and layout can be changed for every figure at once.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_VIZ_DIR = _ROOT / "src" / "fairxai" / "viz"
_PLOT_SCRIPT = _ROOT / "scripts" / "studies" / "generate_dissertation_plots.py"

_MIN_DPI = 300


def _configured_dataset_ids() -> set[str]:
    """Every cardiac dataset id the shipped configs name, commented ones included.

    ``cardio70k`` is commented out in ``combinatorial.yaml`` by decision D1, but
    it is still a dataset the study runs explicitly, so its palette entry has to
    survive that comment.
    """
    ids: set[str] = set()

    pipeline = yaml.safe_load((_ROOT / "configs" / "pipelines" / "cardiac.yaml").read_text())
    ids.update(str(d) for d in pipeline["runtime"]["datasets"])

    combinatorial_text = (_ROOT / "configs" / "experiments" / "combinatorial.yaml").read_text()
    combinatorial = yaml.safe_load(combinatorial_text)
    ids.update(str(d) for d in combinatorial["datasets"])
    ids.update(re.findall(r"^\s*#\s*-\s*(\w+)\s*$", combinatorial_text, flags=re.MULTILINE))

    return ids


class TestDatasetPalette:
    def test_palette_covers_every_configured_dataset(self) -> None:
        from fairxai.viz.style import PALETTE_DATASET

        missing = _configured_dataset_ids() - set(PALETTE_DATASET)
        assert not missing, f"PALETTE_DATASET has no colour for {sorted(missing)}"

    def test_positive_rate_plot_runs_with_real_dataset_ids(self) -> None:
        """The regression this guards: a stale key is a crash, not a fallback.

        ``sns.lineplot`` raises ``ValueError: The palette dictionary is missing
        keys`` when a hue level has no entry, so every call of this function on
        real cardiac data failed while the palette was keyed on the old ids.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        from fairxai.notebook_utils.profiling import plot_positive_rates_by_age

        datasets = sorted(_configured_dataset_ids())
        frame = pd.DataFrame(
            [
                {"dataset": dataset, "age_group": age, "prevalence": 0.3}
                for dataset in datasets
                for age in ("40-49", "50-59")
            ]
        )

        result = plot_positive_rates_by_age(frame, ["40-49", "50-59"], show=False)
        assert result is not None
        plt.close("all")

    def test_legacy_dataset_keys_still_resolve(self) -> None:
        """Committed notebooks still pass the pre-rename ids; they must not break."""
        from fairxai.viz.style import PALETTE_DATASET

        for legacy in ("cleveland", "kaggle_heart"):
            assert legacy in PALETTE_DATASET


class TestFigureResolution:
    def test_no_viz_figure_is_saved_below_dissertation_dpi(self) -> None:
        offenders = []
        for path in sorted(_VIZ_DIR.glob("*.py")):
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                for value in re.findall(r"\bdpi=(\d+)", line):
                    if int(value) < _MIN_DPI:
                        offenders.append(f"{path.name}:{lineno} dpi={value}")

        assert not offenders, "figures saved below print resolution: " + ", ".join(offenders)

    def test_viz_modules_save_through_the_shared_helper(self) -> None:
        """``save_figure`` is what applies the shared bounding box and padding.

        A bare ``fig.savefig`` skips ``bbox_inches``/``pad_inches``, so the
        figure crops differently from every other figure in the same chapter.
        """
        offenders = []
        for name in ("fairness.py", "transformations.py"):
            path = _VIZ_DIR / name
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if re.search(r"\bfig\.savefig\(", line):
                    offenders.append(f"{name}:{lineno}")

        assert not offenders, "bare savefig calls bypass save_figure: " + ", ".join(offenders)


class TestDissertationPlotTheme:
    def test_plot_script_sets_a_global_rcparams_theme(self) -> None:
        """One file has to own typography and layout for every generated figure."""
        source = _PLOT_SCRIPT.read_text()
        assert "rcParams" in source, "generate_dissertation_plots.py sets no global theme"

    def test_theme_pins_the_properties_that_cannot_be_set_per_figure(self) -> None:
        source = _PLOT_SCRIPT.read_text()
        for key in ("font.family", "font.size", "axes.grid", "lines.linewidth", "savefig.dpi"):
            assert key in source, f"global theme does not pin {key}"
