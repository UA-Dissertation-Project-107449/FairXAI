"""WebApp adapter for dataset characterization — re-exports the profiling function."""

from fairxai.profiling.domain_characterization import characterize_dataset

__all__ = ["characterize_dataset"]
