"""SampleSource abstract protocol and ChunkData container.

Defines the ingestion contracts for streaming sEMG pipelines without
coupling to concrete hardware, files, or signal synthesis generators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = ["ChunkData", "SampleSource"]


@dataclass(frozen=True, eq=False)
class ChunkData:
    """Immutable multichannel streaming chunk container.

    Attributes:
        data: Multichannel sample array of shape (num_samples, num_channels)
            and strict np.float32 dtype backed by an immutable buffer.
        start_sample_idx: Monotonically increasing sequential sample counter
            referencing the first sample of this chunk.

    Raises:
        ValueError: If ``data`` is not a 2D array, or if ``start_sample_idx``
            is not a non-negative integer.
        TypeError: If ``data`` is not an ndarray, does not have strict
            np.float32 dtype, or ``start_sample_idx`` is not an integer
            (booleans explicitly rejected).

    Note:
        ``eq=False`` on the dataclass decorator is intentional: equality and
        hashing are implemented explicitly below so that signed-zero
        canonicalization stays consistent between ``__eq__`` and ``__hash__``.
    """

    data: np.ndarray
    start_sample_idx: int = 0

    def __post_init__(self) -> None:
        """Enforce strict chunk invariants, deep immutability, and canonical representations."""
        self._validate_data()
        idx = self._normalize_start_sample_idx()

        # Back the chunk with a deeply immutable, canonicalized copy:
        # 1. Defensive copy decouples the caller's buffer and prevents view aliasing.
        # 2. Canonicalize signed zeros (-0.0 -> +0.0) so the bitwise
        #    representation matches numerical equality.
        # 3. np.frombuffer(...) yields a read-only array whose WRITEABLE flag
        #    cannot be re-enabled.
        arr_copy = np.array(self.data, dtype=np.float32, copy=True)
        arr_copy[arr_copy == 0.0] = 0.0
        immutable_data = np.frombuffer(
            arr_copy.tobytes(), dtype=np.float32
        ).reshape(arr_copy.shape)

        object.__setattr__(self, "data", immutable_data)
        object.__setattr__(self, "start_sample_idx", idx)

    def _validate_data(self) -> None:
        """Validate the type, dimensionality, and dtype of ``data``."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(
                f"ChunkData data must be a numpy ndarray, got {type(self.data).__name__}"
            )

        if self.data.ndim != 2:
            raise ValueError(
                "ChunkData data must be a 2D array of shape "
                f"(num_samples, num_channels), got shape {self.data.shape}"
            )

        if self.data.dtype != np.float32:
            raise TypeError(
                "ChunkData data must have np.float32 dtype "
                f"(strict float32 invariant), got {self.data.dtype}"
            )

    def _normalize_start_sample_idx(self) -> int:
        """Validate ``start_sample_idx`` and return it as a plain Python int."""
        if isinstance(self.start_sample_idx, bool) or not isinstance(
            self.start_sample_idx, (int, np.integer)
        ):
            raise TypeError(
                "start_sample_idx must be an integer, "
                f"got {type(self.start_sample_idx).__name__}"
            )

        idx = int(self.start_sample_idx)
        if idx < 0:
            raise ValueError(
                f"start_sample_idx must be a non-negative integer, got {idx}"
            )
        return idx

    @property
    def num_samples(self) -> int:
        """Return the number of time samples in this chunk."""
        return int(self.data.shape[0])

    @property
    def num_channels(self) -> int:
        """Return the number of recording channels in this chunk."""
        return int(self.data.shape[1])

    def __eq__(self, other: object) -> bool:
        """Value equality comparing start_sample_idx and array content."""
        if not isinstance(other, ChunkData):
            return NotImplemented
        return (
            self.start_sample_idx == other.start_sample_idx
            and self.data.shape == other.data.shape
            and self.data.dtype == other.data.dtype
            and bool(np.array_equal(self.data, other.data))
        )

    def __hash__(self) -> int:
        """Hash based on immutable buffer bytes, shape, and start index."""
        return hash(
            (
                self.start_sample_idx,
                self.data.shape,
                self.data.dtype,
                self.data.tobytes(),
            )
        )


class SampleSource(ABC):
    """Abstract base protocol for streaming multichannel sample sources."""

    @abstractmethod
    def read_chunk(self, num_samples: int) -> ChunkData:
        """Read and return the next sequential chunk of samples.

        Implementations MUST maintain strict temporal continuity: the first
        sample of each returned chunk MUST immediately follow the last sample
        of the previous chunk, regardless of varying ``num_samples`` values.

        Args:
            num_samples: Number of samples to read along the temporal axis.

        Returns:
            ChunkData containing (num_samples, num_channels) in float32.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the source state to initial sample index zero."""
        ...
