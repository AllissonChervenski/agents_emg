"""Stateful sliding window accumulator for streaming signals.

Maintains causal sliding windows of shape (window_length, num_channels)
with explicit stride control, retaining unconsumed residual samples across
streaming chunk boundaries, and resolving final residuals on finalize().
"""

import numpy as np

from semg_dsp.source import ChunkData

__all__ = ["StatefulWindowBuffer"]


class StatefulWindowBuffer:
    """Stateful multichannel causal sliding window accumulator.

    Collects samples from sequential chunks and emits fixed-length sliding
    windows according to window_length and stride. Unconsumed residual samples
    and overlapping history are strictly retained in a bounded internal buffer
    across chunk boundaries.

    Attributes:
        window_length: Fixed number of time samples per emitted window.
        stride: Number of time samples to advance between successive windows.
        num_channels: Expected number of recording channels.
        partial_policy: Policy for handling incomplete residual samples on stream
            termination via finalize() ("drop" or "pad").
    """

    def __init__(
        self,
        window_length: int,
        stride: int,
        num_channels: int = 1,
        partial_policy: str = "drop",
    ) -> None:
        if isinstance(window_length, bool) or not isinstance(window_length, (int, np.integer)):
            raise TypeError(f"window_length must be an integer, got {type(window_length).__name__}")
        if int(window_length) <= 0:
            raise ValueError(f"window_length must be strictly positive, got {window_length}")

        if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
            raise TypeError(f"stride must be an integer, got {type(stride).__name__}")
        if int(stride) <= 0:
            raise ValueError(f"stride must be strictly positive, got {stride}")
        if int(stride) > int(window_length):
            raise ValueError(f"stride ({stride}) cannot exceed window_length ({window_length})")

        if isinstance(num_channels, bool) or not isinstance(num_channels, (int, np.integer)):
            raise TypeError(f"num_channels must be an integer, got {type(num_channels).__name__}")
        if int(num_channels) < 1:
            raise ValueError(f"num_channels must be at least 1, got {num_channels}")

        valid_policies = {"drop", "pad"}
        if partial_policy not in valid_policies:
            raise ValueError(f"partial_policy must be one of {valid_policies}, got {partial_policy!r}")

        self.window_length = int(window_length)
        self.stride = int(stride)
        self.num_channels = int(num_channels)
        self.partial_policy = partial_policy

        # Bounded internal buffer storage of shape (N, num_channels) in float32
        self._buffer = np.empty((0, self.num_channels), dtype=np.float32)

    @property
    def buffered_samples(self) -> int:
        """Return current number of sequential samples held in buffer."""
        return int(self._buffer.shape[0])

    def reset(self) -> None:
        """Reset internal accumulator, discarding all buffered samples."""
        self._buffer = np.empty((0, self.num_channels), dtype=np.float32)

    def process_chunk(self, chunk: ChunkData | np.ndarray) -> list[np.ndarray]:
        """Ingest chunk, emit all available complete sliding windows, and retain residual.

        Args:
            chunk: Incoming multichannel data chunk of shape (num_samples, num_channels)
                in strict float32.

        Returns:
            List of complete window arrays of shape (window_length, num_channels) in float32.
        """
        if isinstance(chunk, ChunkData):
            raw_data = chunk.data
        elif isinstance(chunk, np.ndarray):
            raw_data = chunk
        else:
            raise TypeError(f"chunk must be ChunkData or np.ndarray, got {type(chunk).__name__}")

        if raw_data.ndim != 2:
            raise ValueError(f"chunk data must be 2D array of shape (num_samples, num_channels), got shape {raw_data.shape}")
        if raw_data.shape[1] != self.num_channels:
            raise ValueError(f"chunk has {raw_data.shape[1]} channels, but buffer configured for {self.num_channels}")
        if raw_data.dtype != np.float32:
            raise TypeError(f"chunk data must have np.float32 dtype, got {raw_data.dtype}")
        if not np.all(np.isfinite(raw_data)):
            raise ValueError("chunk data contains non-finite values (NaN or Inf)")

        if raw_data.shape[0] == 0:
            return []

        windows: list[np.ndarray] = []
        chunk_len = raw_data.shape[0]
        cursor = 0

        while cursor < chunk_len:
            # Buffer memory is strictly bounded by window_length: only ingest what is needed
            needed = self.window_length - self._buffer.shape[0]
            take = min(needed, chunk_len - cursor)
            slice_to_add = raw_data[cursor : cursor + take]
            cursor += take

            if self._buffer.shape[0] == 0:
                self._buffer = slice_to_add.copy()
            else:
                self._buffer = np.concatenate([self._buffer, slice_to_add], axis=0)

            # Emit window as soon as window_length samples are reached, advancing by stride
            if self._buffer.shape[0] == self.window_length:
                windows.append(self._buffer.copy())
                self._buffer = self._buffer[self.stride :].copy()

        return windows

    def finalize(self) -> list[np.ndarray]:
        """Flush final incomplete residual according to partial_policy.

        Returns:
            List containing final padded window if policy is 'pad' and residual exists,
            otherwise empty list.
        """
        remaining = self._buffer.shape[0]
        final_windows: list[np.ndarray] = []

        if remaining > 0 and self.partial_policy == "pad":
            pad_len = self.window_length - remaining
            zero_pad = np.zeros((pad_len, self.num_channels), dtype=np.float32)
            padded_win = np.concatenate([self._buffer, zero_pad], axis=0)
            final_windows.append(padded_win)

        # Clear buffer regardless of policy
        self._buffer = np.empty((0, self.num_channels), dtype=np.float32)
        return final_windows
