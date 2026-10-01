"""SampleSource abstract protocol and ChunkData container.

Defines the ingestion contracts for streaming sEMG pipelines without
coupling to concrete hardware, files, or signal synthesis generators.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

__all__ = ["ChunkData", "SampleSource", "SyntheticSampleSource"]


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
    sampling_rate_hz: float = 1000.0

    def __post_init__(self) -> None:
        """Enforce strict chunk invariants, deep immutability, and canonical representations."""
        self._validate_data()
        idx = self._normalize_start_sample_idx()
        rate = self._normalize_sampling_rate_hz()

        # Back the chunk with a deeply immutable, canonicalized copy:
        # 1. Defensive copy decouples the caller's buffer and prevents view aliasing.
        # 2. Canonicalize signed zeros (-0.0 -> +0.0) so the bitwise
        #    representation matches numerical equality.
        # 3. np.frombuffer(...) yields a read-only array whose WRITEABLE flag
        #    cannot be re-enabled.
        arr_copy = np.array(self.data, dtype=np.float32, copy=True)
        arr_copy[arr_copy == 0.0] = 0.0
        arr_copy[np.isnan(arr_copy)] = np.nan
        immutable_data = np.frombuffer(
            arr_copy.tobytes(), dtype=np.float32
        ).reshape(arr_copy.shape)

        object.__setattr__(self, "data", immutable_data)
        object.__setattr__(self, "start_sample_idx", idx)
        object.__setattr__(self, "sampling_rate_hz", rate)

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

    def _normalize_sampling_rate_hz(self) -> float:
        if isinstance(self.sampling_rate_hz, bool) or not isinstance(
            self.sampling_rate_hz, (int, float, np.floating, np.integer)
        ):
            raise TypeError(
                f"sampling_rate_hz must be a float, got {type(self.sampling_rate_hz).__name__}"
            )
        rate = float(self.sampling_rate_hz)
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError(
                f"sampling_rate_hz must be strictly positive and finite, got {rate}"
            )
        return rate

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
            and self.sampling_rate_hz == other.sampling_rate_hz
            and self.data.shape == other.data.shape
            and self.data.dtype == other.data.dtype
            and bool(np.array_equal(self.data, other.data, equal_nan=True))
        )

    def __hash__(self) -> int:
        """Hash based on immutable buffer bytes, shape, and start index."""
        return hash(
            (
                self.start_sample_idx,
                self.sampling_rate_hz,
                self.data.shape,
                self.data.dtype,
                self.data.tobytes(),
            )
        )


class SampleSource(ABC):
    """Abstract base protocol for streaming multichannel sample sources."""

    num_channels: int

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


class SyntheticSampleSource(SampleSource):
    """Deterministic analytical L0 waveform generator for streaming tests.

    Synthesizes multichannel chunks of analytical waveforms (``zeros``,
    ``dc``, ``impulse``, ``step``, ``sine``, ``multi_tone``, ``saturation``)
    in strict ``np.float32`` with guaranteed phase continuity across
    successive :meth:`read_chunk` calls of arbitrary sizes.

    Sample time is computed strictly from a monotonically increasing global
    counter as ``t = (total_samples_emitted + arange(num_samples)) /
    sampling_rate_hz``; phase and sample index are never reset per chunk.

    Attributes:
        waveform: Name of the analytical waveform to synthesize.
        num_channels: Number of identical recording channels per chunk.
        sampling_rate_hz: Sampling rate in Hertz (strictly positive).
        amplitude: Peak amplitude (scalar, or per-component for
            ``multi_tone``).
        frequency_hz: Component frequency in Hertz (scalar, or
            per-component for ``multi_tone``).
        phase_rad: Component initial phase in radians (scalar, or
            per-component for ``multi_tone``).
        dc_offset: Constant output level for the ``dc`` waveform.
        event_sample_idx: Global sample index of the Kronecker impulse
            (``impulse``) or of the unit-step transition (``step``).
        clip_limits: ``(lower, upper)`` amplitude clipping bounds applied
            by the ``saturation`` waveform.
        total_samples_emitted: Monotonically increasing count of samples
            emitted since construction or the last :meth:`reset`.

    Raises:
        ValueError: If ``waveform`` is unknown, ``num_channels`` is not a
            positive integer, ``sampling_rate_hz`` is not strictly positive
            and finite, ``clip_limits`` is not a strictly ordered
            ``(lower, upper)`` pair, ``event_sample_idx`` is negative, or
            the ``multi_tone`` component sequences have mismatched lengths.
        TypeError: If ``num_channels`` or ``event_sample_idx`` is not an
            integer (booleans explicitly rejected).
    """

    _WAVEFORMS = ("zeros", "dc", "impulse", "step", "sine", "multi_tone", "saturation")

    def __init__(
        self,
        waveform: str = "sine",
        num_channels: int = 1,
        sampling_rate_hz: float = 1000.0,
        amplitude: float | Sequence[float] = 1.0,
        frequency_hz: float | Sequence[float] = 10.0,
        phase_rad: float | Sequence[float] = 0.0,
        dc_offset: float = 1.0,
        event_sample_idx: int = 0,
        clip_limits: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """Initialize the synthetic source and validate all parameters."""
        if waveform not in self._WAVEFORMS:
            raise ValueError(
                f"Unknown waveform {waveform!r}; "
                f"expected one of {self._WAVEFORMS}"
            )

        if isinstance(num_channels, bool) or not isinstance(
            num_channels, (int, np.integer)
        ):
            raise TypeError(
                f"num_channels must be an integer, got {type(num_channels).__name__}"
            )
        if int(num_channels) < 1:
            raise ValueError(
                f"num_channels must be a positive integer, got {num_channels}"
            )

        rate = float(sampling_rate_hz)
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError(
                f"sampling_rate_hz must be strictly positive and finite, "
                f"got {sampling_rate_hz}"
            )

        if isinstance(event_sample_idx, bool) or not isinstance(
            event_sample_idx, (int, np.integer)
        ):
            raise TypeError(
                "event_sample_idx must be an integer, "
                f"got {type(event_sample_idx).__name__}"
            )
        if int(event_sample_idx) < 0:
            raise ValueError(
                f"event_sample_idx must be a non-negative integer, "
                f"got {event_sample_idx}"
            )

        if len(clip_limits) != 2:
            raise ValueError(
                f"clip_limits must be an ordered (lower, upper) pair, got {clip_limits}"
            )
        lower = np.float32(clip_limits[0])
        upper = np.float32(clip_limits[1])
        if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
            raise ValueError(
                "clip_limits must be an ordered (lower, upper) pair of finite float32 with "
                f"lower < upper, got ({lower}, {upper})"
            )

        def _to_float32_param(
            val: float | Sequence[float],
        ) -> np.float32 | tuple[np.float32, ...]:
            if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
                return tuple(np.float32(x) for x in val)
            return np.float32(val)

        self.waveform = waveform
        self.num_channels = int(num_channels)
        self.sampling_rate_hz = rate
        self.amplitude = _to_float32_param(amplitude)
        self.frequency_hz = _to_float32_param(frequency_hz)
        self.phase_rad = _to_float32_param(phase_rad)
        self.dc_offset = np.float32(dc_offset)
        self.event_sample_idx = int(event_sample_idx)
        self.clip_limits = (lower, upper)
        self.total_samples_emitted = 0

        self._amplitudes = np.atleast_1d(np.asarray(amplitude, dtype=np.float32))
        self._frequencies = np.atleast_1d(np.asarray(frequency_hz, dtype=np.float32))
        self._phases = np.atleast_1d(np.asarray(phase_rad, dtype=np.float32))
        if waveform == "multi_tone" and not (
            self._amplitudes.size == self._frequencies.size == self._phases.size
        ):
            raise ValueError(
                "multi_tone requires amplitude, frequency_hz and phase_rad "
                "sequences of identical length, got sizes "
                f"{self._amplitudes.size}, {self._frequencies.size} and "
                f"{self._phases.size}"
            )

        # Precompute exact rational period denominator and numerator for periodic phase modulo
        denoms = []
        nums = []
        for f in self._frequencies:
            f_str = str(float(f))
            rate_str = str(float(self.sampling_rate_hz))
            frac = Fraction(f_str) / Fraction(rate_str)
            denoms.append(frac.denominator)
            nums.append(frac.numerator % frac.denominator)
        self._denominators = denoms
        self._numerators = nums

    def read_chunk(self, num_samples: int) -> ChunkData:
        """Generate the next chunk, preserving global phase continuity.

        Args:
            num_samples: Number of samples to generate (strictly positive).

        Returns:
            ChunkData of shape (num_samples, num_channels) in float32 whose
            ``start_sample_idx`` equals the counter value before emission.

        Raises:
            TypeError: If ``num_samples`` is not an integer.
            ValueError: If ``num_samples`` is not strictly positive.
        """
        if isinstance(num_samples, bool) or not isinstance(
            num_samples, (int, np.integer)
        ):
            raise TypeError(
                f"num_samples must be an integer, got {type(num_samples).__name__}"
            )
        if int(num_samples) < 1:
            raise ValueError(
                f"num_samples must be a strictly positive integer, got {num_samples}"
            )

        max_idx = int(np.iinfo(np.int64).max)
        if self.total_samples_emitted > max_idx - int(num_samples):
            raise OverflowError(
                f"Sample index counter {self.total_samples_emitted} + {num_samples} "
                f"exceeds signed 64-bit integer limit ({max_idx})"
            )

        start = self.total_samples_emitted
        idx = start + np.arange(num_samples, dtype=np.int64)
        rate = np.float32(self.sampling_rate_hz)
        t = (np.float32(start) + np.arange(num_samples, dtype=np.float32)) / rate
        mono = self._render(idx, t)

        data = np.empty((num_samples, self.num_channels), dtype=np.float32)
        data[:] = mono[:, np.newaxis]

        self.total_samples_emitted += int(num_samples)
        return ChunkData(data=data, start_sample_idx=start, sampling_rate_hz=float(self.sampling_rate_hz))

    def reset(self) -> None:
        """Reset the global emission counter to zero."""
        self.total_samples_emitted = 0

    @staticmethod
    def _int_fraction_to_float32(r: int, denom: int) -> np.float32:
        """Convert exact arbitrary-precision fraction r / denom to np.float32 with zero float64 promotion."""
        if r == 0:
            return np.float32(0.0)
        shift = denom.bit_length() - r.bit_length()
        mantissa = (r << (24 + shift)) // denom
        return np.ldexp(np.float32(mantissa), -(24 + shift))

    def _compute_phase_cycles(self, idx: np.ndarray, num: int, denom: int) -> np.ndarray:
        """Compute cyclic phase in [0, 1) strictly in float32 with zero integer overflow."""
        B = denom.bit_length()
        if (denom - 1) * num < 2**63 - 1 and B <= 127:
            n_step = ((idx % denom) * num) % denom
            return (n_step.astype(np.float32) / np.float32(denom)).astype(np.float32)
        else:
            return np.fromiter(
                (self._int_fraction_to_float32((int(i) * num) % denom, denom) for i in idx),
                dtype=np.float32,
                count=len(idx),
            )

    def _render(self, idx: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Evaluate the configured waveform at global sample indices/times in strict float32."""
        if self.waveform == "zeros":
            return np.zeros(idx.shape, dtype=np.float32)
        if self.waveform == "dc":
            return np.full(idx.shape, self.dc_offset, dtype=np.float32)
        if self.waveform == "impulse":
            return (idx == self.event_sample_idx).astype(np.float32)
        if self.waveform == "step":
            return (idx >= self.event_sample_idx).astype(np.float32)

        two_pi = np.float32(2.0 * np.pi)

        if self.waveform == "sine":
            cycles = self._compute_phase_cycles(idx, self._numerators[0], self._denominators[0])
            ang = two_pi * cycles + self._phases[0]
            return (self._amplitudes[0] * np.sin(ang)).astype(np.float32)
        if self.waveform == "multi_tone":
            col = np.zeros(idx.shape, dtype=np.float32)
            for a, denom_val, num_val, p in zip(
                self._amplitudes, self._denominators, self._numerators, self._phases
            ):
                cycles = self._compute_phase_cycles(idx, int(num_val), int(denom_val))
                ang = two_pi * cycles + p
                col += a * np.sin(ang)
            return col.astype(np.float32)
        # saturation: sinusoid hard-clipped to clip_limits
        cycles = self._compute_phase_cycles(idx, self._numerators[0], self._denominators[0])
        ang = two_pi * cycles + self._phases[0]
        signal = self._amplitudes[0] * np.sin(ang)
        return np.clip(signal, self.clip_limits[0], self.clip_limits[1]).astype(np.float32)

