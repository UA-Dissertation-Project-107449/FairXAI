"""WebApp integration adapters — thin wrappers that produce JSON-serializable dicts."""

from fairxai.integration.binning import run_binning
from fairxai.integration.characterize import characterize_dataset
from fairxai.integration.clustering import run_clustering
from fairxai.integration.profile import profile_dataset

__all__ = ["characterize_dataset", "profile_dataset", "run_binning", "run_clustering"]
