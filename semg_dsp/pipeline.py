"""End-to-end streaming DSP pipeline coordinator.

Connects a SampleSource, a CausalSosFilter, and a StatefulWindowBuffer
into a cohesive causal streaming processor with strict channel matching,
state continuity, and independent oracle verification.
"""

import numpy as np

from semg_dsp.source import SampleSource
from semg_dsp.filter import CausalSosFilter
from semg_dsp.window import StatefulWindowBuffer

__all__ = ["StreamingPipeline"]


class StreamingPipeline:
    """Streaming DSP pipeline integrating source, biquad filter, and window buffer.

    Attributes:
        source: Streaming sample source producing multichannel chunks.
        filter_stage: Causal IIR SOS filter processing streaming chunks.
        window_stage: Sliding window accumulator slicing continuous filtered streams.
        num_channels: Channel count matching across all three stages.
    """

    def __init__(
        self,
        source: SampleSource,
        filter_stage: CausalSosFilter,
        window_stage: StatefulWindowBuffer,
    ) -> None:
        if not isinstance(source, SampleSource):
            raise TypeError(f"source must implement SampleSource protocol, got {type(source).__name__}")
        if not isinstance(filter_stage, CausalSosFilter):
            raise TypeError(f"filter_stage must be CausalSosFilter, got {type(filter_stage).__name__}")
        if not isinstance(window_stage, StatefulWindowBuffer):
            raise TypeError(f"window_stage must be StatefulWindowBuffer, got {type(window_stage).__name__}")

        src_ch = int(source.num_channels)
        filt_ch = int(filter_stage.num_channels)
        win_ch = int(window_stage.num_channels)

        if not (src_ch == filt_ch == win_ch):
            raise ValueError(
                f"channel count mismatch: source ({src_ch}), filter ({filt_ch}), "
                f"and window ({win_ch}) must have identical channel counts"
            )

        self._source = source
        self._filter = filter_stage
        self._window = window_stage
        self._num_channels = src_ch

    @property
    def source(self) -> SampleSource:
        """Return underlying sample source."""
        return self._source

    @property
    def filter_stage(self) -> CausalSosFilter:
        """Return underlying causal SOS filter."""
        return self._filter

    @property
    def window_stage(self) -> StatefulWindowBuffer:
        """Return underlying stateful window accumulator."""
        return self._window

    @property
    def num_channels(self) -> int:
        """Return number of channels processed by the pipeline."""
        return self._num_channels

    def step(self, num_samples: int) -> list[np.ndarray]:
        """Pull num_samples from source, filter causally, and push into window accumulator.

        Args:
            num_samples: Number of samples to request from the source.

        Returns:
            List of complete window arrays emitted by the window accumulator.
        """
        if isinstance(num_samples, bool) or not isinstance(num_samples, (int, np.integer)):
            raise TypeError(f"num_samples must be an integer, got {type(num_samples).__name__}")
        if int(num_samples) < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}")
        if int(num_samples) == 0:
            return []

        chunk = self._source.read_chunk(int(num_samples))
        filtered_chunk = self._filter.process_chunk(chunk)
        windows = self._window.process_chunk(filtered_chunk)
        return windows

    def finalize(self) -> list[np.ndarray]:
        """Flush final incomplete window according to window buffer partial_policy.

        Returns:
            List of emitted windows upon stream completion.
        """
        return self._window.finalize()

    def reset(self) -> None:
        """Reset all stages to initial state deterministically."""
        self._source.reset()
        self._filter.reset()
        self._window.reset()
