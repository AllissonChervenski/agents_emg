"""Causal Second-Order Sections (SOS) biquad streaming filter.

Implements sample-by-sample and chunk-by-chunk IIR filtering using
the Direct Form II Transposed (DF2T) structure with explicit state
persistence across chunks, guaranteeing strict causality and chunk invariance.
"""

from collections.abc import Sequence
import numpy as np

from semg_dsp.source import ChunkData

__all__ = ["CausalSosFilter"]


class CausalSosFilter:
    """Stateful multichannel causal SOS/biquad streaming filter.

    Implements a cascade of Second-Order Sections in Direct Form II Transposed (DF2T):
        y[n]  = b0 * x[n] + z1[n-1]
        z1[n] = b1 * x[n] - a1 * y[n] + z2[n-1]
        z2[n] = b2 * x[n] - a2 * y[n]

    State shape is explicitly (n_sections, 2, num_channels) in strict np.float32.
    Coefficients are validated and normalized to a0 = 1.0 upon initialization.
    """

    def __init__(
        self,
        sos_coefficients: np.ndarray | Sequence[Sequence[float]],
        num_channels: int = 1,
    ) -> None:
        if isinstance(num_channels, bool) or not isinstance(num_channels, (int, np.integer)):
            raise TypeError(f"num_channels must be an integer, got {type(num_channels).__name__}")
        if int(num_channels) < 1:
            raise ValueError(f"num_channels must be a positive integer, got {num_channels}")

        if isinstance(sos_coefficients, np.ndarray) and sos_coefficients.dtype != np.float32:
            raise TypeError(
                f"sos_coefficients ndarray must have np.float32 dtype, got {sos_coefficients.dtype}"
            )

        raw_sos = np.asarray(sos_coefficients, dtype=np.float32)
        if raw_sos.ndim != 2 or raw_sos.shape[1] != 6:
            raise ValueError(f"sos_coefficients must have shape (n_sections, 6), got {raw_sos.shape}")
        if raw_sos.shape[0] < 1:
            raise ValueError("sos_coefficients must contain at least one biquad section")
        if not np.all(np.isfinite(raw_sos)):
            raise ValueError("sos_coefficients must contain only finite numerical values (no NaN or Inf)")

        self.n_sections = int(raw_sos.shape[0])
        self.num_channels = int(num_channels)

        # Normalize coefficients so each section has a0 == 1.0
        norm_sos = np.empty_like(raw_sos, dtype=np.float32)
        for s in range(self.n_sections):
            b0, b1, b2, a0, a1, a2 = raw_sos[s]
            if a0 == 0.0 or not np.isfinite(a0):
                raise ValueError(f"Section {s} has invalid leading denominator coefficient a0={a0}")
            with np.errstate(divide="ignore", over="ignore"):
                inv_a0 = np.float32(1.0) / a0
            if not np.isfinite(inv_a0):
                raise ValueError(
                    f"Section {s} leading coefficient a0={a0} causes reciprocal overflow to Inf"
                )
            norm_sos[s, 0] = b0 * inv_a0
            norm_sos[s, 1] = b1 * inv_a0
            norm_sos[s, 2] = b2 * inv_a0
            norm_sos[s, 3] = np.float32(1.0)
            norm_sos[s, 4] = a1 * inv_a0
            norm_sos[s, 5] = a2 * inv_a0

        if not np.all(np.isfinite(norm_sos)):
            raise ValueError(
                "Normalized SOS coefficients contain non-finite values (overflow or division by near-zero a0)"
            )

        self.sos_coefficients = norm_sos
        self._state = np.zeros((self.n_sections, 2, self.num_channels), dtype=np.float32)

    @property
    def state(self) -> np.ndarray:
        """Return current filter state array of shape (n_sections, 2, num_channels)."""
        return self._state

    def reset(self) -> None:
        """Reset all filter section delay states to initial zero."""
        self._state.fill(0.0)

    def process_chunk(self, chunk: ChunkData | np.ndarray) -> ChunkData | np.ndarray:
        """Process streaming chunk through the causal DF2T biquad cascade.

        Maintains internal delay states across calls. Preserves exact start_sample_idx
        when invoked with ChunkData.
        """
        is_chunk_data = isinstance(chunk, ChunkData)
        if is_chunk_data:
            raw_data = chunk.data
            start_idx = chunk.start_sample_idx
            rate_hz = chunk.sampling_rate_hz
        elif isinstance(chunk, np.ndarray):
            raw_data = chunk
            start_idx = 0
            rate_hz = 1000.0
        else:
            raise TypeError(f"chunk must be ChunkData or np.ndarray, got {type(chunk).__name__}")

        if raw_data.ndim != 2:
            raise ValueError(f"chunk data must be a 2D array of shape (num_samples, num_channels), got {raw_data.shape}")
        if raw_data.shape[1] != self.num_channels:
            raise ValueError(f"chunk has {raw_data.shape[1]} channels, but filter configured for {self.num_channels}")
        if raw_data.dtype != np.float32:
            raise TypeError(f"chunk data must have np.float32 dtype, got {raw_data.dtype}")
        if not np.all(np.isfinite(raw_data)):
            raise ValueError("chunk data contains non-finite values (NaN or Inf)")

        num_samples = raw_data.shape[0]
        if num_samples == 0:
            empty_arr = np.empty((0, self.num_channels), dtype=np.float32)
            return ChunkData(data=empty_arr, start_sample_idx=start_idx, sampling_rate_hz=rate_hz) if is_chunk_data else empty_arr

        out_data = np.empty_like(raw_data, dtype=np.float32)

        # Sample-by-sample DF2T cascade per channel to enforce strict causality and zero crosstalk
        for ch in range(self.num_channels):
            cur_x = raw_data[:, ch].copy()
            for s in range(self.n_sections):
                b0 = self.sos_coefficients[s, 0]
                b1 = self.sos_coefficients[s, 1]
                b2 = self.sos_coefficients[s, 2]
                a1 = self.sos_coefficients[s, 4]
                a2 = self.sos_coefficients[s, 5]

                z1 = self._state[s, 0, ch]
                z2 = self._state[s, 1, ch]
                out_s = np.empty(num_samples, dtype=np.float32)

                for n in range(num_samples):
                    xn = cur_x[n]
                    yn = np.float32(b0 * xn + z1)
                    z1 = np.float32(b1 * xn - a1 * yn + z2)
                    z2 = np.float32(b2 * xn - a2 * yn)
                    out_s[n] = yn

                self._state[s, 0, ch] = z1
                self._state[s, 1, ch] = z2
                cur_x = out_s
            out_data[:, ch] = cur_x

        if is_chunk_data:
            return ChunkData(data=out_data, start_sample_idx=start_idx, sampling_rate_hz=rate_hz)
        return out_data
