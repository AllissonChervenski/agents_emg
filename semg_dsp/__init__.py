"""semg_dsp: Deterministic streaming digital signal processing for sEMG.

Provides streaming signal ingestion, causal IIR SOS biquad filtering,
and stateful sliding window accumulation with strict bitwise
chunk invariance and scientific oracle validation.
"""

from semg_dsp.source import ChunkData, SampleSource, SyntheticSampleSource
from semg_dsp.filter import CausalSosFilter
from semg_dsp.window import StatefulWindowBuffer
from semg_dsp.pipeline import StreamingPipeline

__all__ = [
    "ChunkData",
    "SampleSource",
    "SyntheticSampleSource",
    "CausalSosFilter",
    "StatefulWindowBuffer",
    "StreamingPipeline",
]
