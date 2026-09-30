"""Pure-Python host streaming pipeline for surface EMG (sEMG) processing.

``semg_dsp`` is the scientific DSP production package, intentionally kept
independent from the SDD orchestrator infrastructure. It hosts the real-time
streaming pipeline stages — signal source acquisition, filtering, windowing,
and pipeline orchestration — with no third-party runtime dependencies.
"""

__version__ = "0.1.0"
