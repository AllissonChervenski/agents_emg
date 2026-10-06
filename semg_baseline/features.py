"""Time-domain feature extraction for 12-channel sEMG signals.

Implements FR-004, SC-002, Decision-2:
- 5 classical sEMG time-domain features per channel:
  1. Mean Absolute Value (MAV)
  2. Root Mean Square (RMS)
  3. Waveform Length (WL)
  4. Zero Crossing count (ZC) with deadzone noise threshold
  5. Slope Sign Change count (SSC) with deadzone noise threshold
- Total feature dimensionality: 12 channels x 5 features = 60 features per window.
- Fully vectorized NumPy calculations complying with TOL-TIME-DOMAIN-FEATURES (1e-12).
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def compute_mav(windows: np.ndarray) -> np.ndarray:
    """Compute Mean Absolute Value across time axis for each channel.

    Args:
        windows: Array of shape (N, W, C) or (W, C).

    Returns:
        Array of shape (N, C) in float64.
    """
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim == 2:
        w = w[None, :, :]
    return np.mean(np.abs(w), axis=1, dtype=np.float64)


def compute_rms(windows: np.ndarray) -> np.ndarray:
    """Compute Root Mean Square across time axis for each channel.

    Args:
        windows: Array of shape (N, W, C) or (W, C).

    Returns:
        Array of shape (N, C) in float64.
    """
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim == 2:
        w = w[None, :, :]
    return np.sqrt(np.mean(w**2, axis=1, dtype=np.float64))


def compute_wl(windows: np.ndarray) -> np.ndarray:
    """Compute Waveform Length (cumulative length of signal waveform) for each channel.

    Args:
        windows: Array of shape (N, W, C) or (W, C).

    Returns:
        Array of shape (N, C) in float64.
    """
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim == 2:
        w = w[None, :, :]
    diffs = np.abs(np.diff(w, axis=1))
    return np.sum(diffs, axis=1, dtype=np.float64)


def compute_zc(windows: np.ndarray, epsilon: float | np.ndarray = 1e-5) -> np.ndarray:
    """Compute Zero Crossing count with deadzone threshold for each channel.

    A zero crossing occurs between consecutive samples x[i] and x[i+1] when:
        ((x[i] > 0 and x[i+1] < 0) or (x[i] < 0 and x[i+1] > 0)) and |x[i+1] - x[i]| >= epsilon.

    Args:
        windows: Array of shape (N, W, C) or (W, C).
        epsilon: Scalar or per-channel array of deadzone thresholds.

    Returns:
        Array of shape (N, C) in float64.
    """
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim == 2:
        w = w[None, :, :]

    x_curr = w[:, :-1, :]
    x_next = w[:, 1:, :]

    sign_change = ((x_curr > 0.0) & (x_next < 0.0)) | ((x_curr < 0.0) & (x_next > 0.0))
    amplitude_diff = np.abs(x_next - x_curr)
    deadzone = amplitude_diff >= epsilon

    zc_events = sign_change & deadzone
    return np.sum(zc_events, axis=1, dtype=np.float64)


def compute_ssc(windows: np.ndarray, epsilon: float | np.ndarray = 1e-5) -> np.ndarray:
    """Compute Slope Sign Change count with deadzone threshold for each channel.

    A slope sign change occurs at sample x[i] between x[i-1] and x[i+1] when:
        ((x[i] > x[i-1] and x[i] > x[i+1]) or (x[i] < x[i-1] and x[i] < x[i+1]))
        and (|x[i] - x[i-1]| >= epsilon or |x[i] - x[i+1]| >= epsilon).

    Args:
        windows: Array of shape (N, W, C) or (W, C).
        epsilon: Scalar or per-channel array of deadzone thresholds.

    Returns:
        Array of shape (N, C) in float64.
    """
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim == 2:
        w = w[None, :, :]

    x_prev = w[:, :-2, :]
    x_curr = w[:, 1:-1, :]
    x_next = w[:, 2:, :]

    peak = (x_curr > x_prev) & (x_curr > x_next)
    trough = (x_curr < x_prev) & (x_curr < x_next)
    extrema = peak | trough

    diff_prev = np.abs(x_curr - x_prev)
    diff_next = np.abs(x_curr - x_next)
    deadzone = (diff_prev >= epsilon) | (diff_next >= epsilon)

    ssc_events = extrema & deadzone
    return np.sum(ssc_events, axis=1, dtype=np.float64)


class TimeDomainFeatureExtractor:
    """Vectorized extractor of 5 classical time-domain features across multi-channel sEMG."""

    def __init__(
        self,
        epsilon: float | np.ndarray | Sequence[float] = 1e-5,
        num_channels: int = 12,
    ) -> None:
        self.num_channels = int(num_channels)
        if isinstance(epsilon, (int, float, np.floating, np.integer)):
            self.epsilon: float | np.ndarray = float(epsilon)
        else:
            eps_arr = np.asarray(epsilon, dtype=np.float64)
            if eps_arr.shape == (self.num_channels,):
                self.epsilon = eps_arr
            elif eps_arr.size == 1:
                self.epsilon = float(eps_arr.item())
            else:
                raise ValueError(
                    f"epsilon array shape must be ({self.num_channels},), got {eps_arr.shape}"
                )

    def extract(self, windows: np.ndarray) -> np.ndarray:
        """Extract 60 time-domain features from a batch of analysis windows.

        Args:
            windows: Array of shape (N, W, C) or (W, C).

        Returns:
            Feature matrix X of shape (N, 60) in float64.
            Columns are ordered as:
            - 0..11: MAV (channels 0..11)
            - 12..23: RMS (channels 0..11)
            - 24..35: WL (channels 0..11)
            - 36..47: ZC (channels 0..11)
            - 48..59: SSC (channels 0..11)
        """
        w = np.asarray(windows, dtype=np.float64)
        if w.ndim == 2:
            w = w[None, :, :]

        if w.ndim != 3:
            raise ValueError(
                f"windows array must have shape (N, window_len, num_channels), got {w.shape}"
            )

        if w.shape[2] != self.num_channels:
            raise ValueError(
                f"windows has {w.shape[2]} channels, but extractor configured for {self.num_channels}"
            )

        if not np.all(np.isfinite(w)):
            raise ValueError("Input windows contain non-finite values (NaN or Inf)")

        mav = compute_mav(w)
        rms = compute_rms(w)
        wl = compute_wl(w)
        zc = compute_zc(w, epsilon=self.epsilon)
        ssc = compute_ssc(w, epsilon=self.epsilon)

        return np.hstack([mav, rms, wl, zc, ssc])
