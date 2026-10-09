"""4th-order causal Butterworth bandpass filter for baseline preprocessing.

Implements FR-012.1, SC-002, Decision-2:
- 4th-order causal Butterworth bandpass filter (20-450 Hz SOS).
- 50 Hz notch filter is omitted per empirical spectral audit (R_50 = 0.84).
- Direct Form II Transposed (DF2T) execution via semg_dsp.CausalSosFilter.
- PreprocessingConfig contract conforming to preprocessing_config_schema.json.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import scipy.signal as signal

from semg_dsp.filter import CausalSosFilter


@dataclass(frozen=True)
class PreprocessingConfig:
    """Canonical configuration record for baseline causal DSP filtering."""

    filter_type: str = "butterworth_causal_sos"
    order: int = 4
    low_cutoff_hz: float = 20.0
    high_cutoff_hz: float = 450.0
    sampling_rate_hz: float = 2000.0
    notch_50hz_enabled: bool = False
    sos_coefficients_hash: str = ""
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary conforming to preprocessing_config_schema.json."""
        return {
            "filter_type": self.filter_type,
            "order": self.order,
            "low_cutoff_hz": self.low_cutoff_hz,
            "high_cutoff_hz": self.high_cutoff_hz,
            "sampling_rate_hz": self.sampling_rate_hz,
            "notch_50hz_enabled": self.notch_50hz_enabled,
            "sos_coefficients_hash": self.sos_coefficients_hash,
            "config_hash": self.config_hash,
        }


class CausalBandpassFilter:
    """4th-order causal Butterworth bandpass filter wrapping CausalSosFilter."""

    def __init__(
        self,
        low_hz: float = 20.0,
        high_hz: float = 450.0,
        fs: float = 2000.0,
        order: int = 4,
        num_channels: int = 12,
    ) -> None:
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.fs = float(fs)
        self.order = int(order)
        self.num_channels = int(num_channels)

        # Design Butterworth bandpass filter in SOS representation
        raw_sos = signal.butter(
            N=self.order,
            Wn=[self.low_hz, self.high_hz],
            btype="bandpass",
            fs=self.fs,
            output="sos",
        )

        # Normalize each biquad section so a0 == 1.0 in float32
        n_sections = raw_sos.shape[0]
        norm_sos = np.empty_like(raw_sos, dtype=np.float32)
        for s in range(n_sections):
            b0, b1, b2, a0, a1, a2 = raw_sos[s]
            inv_a0 = np.float32(1.0 / a0)
            norm_sos[s, 0] = np.float32(b0 * inv_a0)
            norm_sos[s, 1] = np.float32(b1 * inv_a0)
            norm_sos[s, 2] = np.float32(b2 * inv_a0)
            norm_sos[s, 3] = np.float32(1.0)
            norm_sos[s, 4] = np.float32(a1 * inv_a0)
            norm_sos[s, 5] = np.float32(a2 * inv_a0)

        self._sos_coefficients = norm_sos
        self._filter = CausalSosFilter(
            sos_coefficients=self._sos_coefficients,
            num_channels=self.num_channels,
        )

        # Cryptographic coefficient and configuration hashing
        self._sos_hash = hashlib.sha256(self._sos_coefficients.tobytes()).hexdigest()

        canonical_dict = {
            "filter_type": "butterworth_causal_sos",
            "order": self.order,
            "low_cutoff_hz": self.low_hz,
            "high_cutoff_hz": self.high_hz,
            "sampling_rate_hz": self.fs,
            "notch_50hz_enabled": False,
            "sos_coefficients_hash": self._sos_hash,
        }
        self._config_hash = hashlib.sha256(
            json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self._config = PreprocessingConfig(
            filter_type="butterworth_causal_sos",
            order=self.order,
            low_cutoff_hz=self.low_hz,
            high_cutoff_hz=self.high_hz,
            sampling_rate_hz=self.fs,
            notch_50hz_enabled=False,
            sos_coefficients_hash=self._sos_hash,
            config_hash=self._config_hash,
        )

    def get_sos_coefficients(self) -> np.ndarray:
        """Return the normalized SOS coefficient array of shape (4, 6) in float32."""
        return np.copy(self._sos_coefficients)

    def get_config(self) -> PreprocessingConfig:
        """Return PreprocessingConfig instance describing this filter."""
        return self._config

    def reset(self) -> None:
        """Reset internal filter delay state to zero."""
        self._filter.reset()

    def process(self, signals: np.ndarray) -> np.ndarray:
        """Filter a 2D multichannel signal array of shape (num_samples, num_channels).

        Maintains state across streaming invocations.
        """
        if not isinstance(signals, np.ndarray):
            signals = np.asarray(signals)

        if signals.ndim != 2:
            raise ValueError(
                f"signals must be 2D array of shape (num_samples, num_channels), got {signals.shape}"
            )

        if signals.shape[1] != self.num_channels:
            raise ValueError(
                f"signals has {signals.shape[1]} channels, but filter configured for {self.num_channels}"
            )

        if not np.all(np.isfinite(signals)):
            raise ValueError("Input signal array contains non-finite values (NaN or Inf)")

        sig_float32 = signals.astype(np.float32, copy=False)
        if sig_float32.shape[0] == 0:
            return np.empty((0, self.num_channels), dtype=np.float32)

        # Host-only SciPy acceleration of the same causal DF2T SOS cascade.
        # Preserve the reference filter's state (including array identity) so
        # reset() and chunked streaming keep their original behavior.
        out, final_state = signal.sosfilt(
            self._sos_coefficients,
            sig_float32,
            axis=0,
            zi=self._filter.state,
        )
        self._filter.state[...] = final_state
        return out
