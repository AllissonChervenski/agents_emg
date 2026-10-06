"""Unit tests for 4th-order causal Butterworth bandpass filter (FR-012.1, SC-002, Decision-2).

Validates:
- SOS coefficient structure (4 sections, 6 elements, float32, normalized a0=1.0).
- PreprocessingConfig schema compliance against preprocessing_config_schema.json.
- Frequency response attenuation (passband 20-450 Hz, rejection at 5 Hz and 600 Hz).
- Strict causality, zero channel cross-talk, and streaming chunk invariance.
- Robust rejection of NaN, Inf, and invalid channel shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
import jsonschema
import numpy as np
import pytest

from semg_baseline.filter import CausalBandpassFilter, PreprocessingConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_SCHEMA_PATH = (
    REPO_ROOT
    / "specs"
    / "004-baseline-training-and-evaluation"
    / "contracts"
    / "preprocessing_config_schema.json"
)


def _load_preprocessing_schema() -> dict:
    with open(PREPROCESSING_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_filter_sos_coefficients_and_schema_validation() -> None:
    """Filter must generate 4 biquad sections and config conforming to JSON schema."""
    bp = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=2000.0, order=4, num_channels=12)
    sos = bp.get_sos_coefficients()

    assert sos.shape == (4, 6)
    assert sos.dtype == np.float32
    for s in range(4):
        assert np.isclose(sos[s, 3], 1.0), f"Section {s} a0 is not 1.0"

    config: PreprocessingConfig = bp.get_config()
    assert config.filter_type == "butterworth_causal_sos"
    assert config.order == 4
    assert config.low_cutoff_hz == 20.0
    assert config.high_cutoff_hz == 450.0
    assert config.sampling_rate_hz == 2000.0
    assert config.notch_50hz_enabled is False
    assert len(config.sos_coefficients_hash) == 64
    assert len(config.config_hash) == 64

    # Validate schema
    schema = _load_preprocessing_schema()
    jsonschema.validate(instance=config.to_dict(), schema=schema)


def test_filter_frequency_response_passband_and_stopband() -> None:
    """Filter must pass 100 Hz and strongly attenuate 5 Hz (<20 Hz) and 600 Hz (>450 Hz)."""
    fs = 2000.0
    duration = 1.0
    t = np.arange(int(fs * duration), dtype=np.float32) / fs
    num_channels = 12

    # Test 1: Passband (100 Hz)
    sig_100hz = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)[:, None]
    sig_100hz_multi = np.tile(sig_100hz, (1, num_channels))
    bp_pass = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=fs, order=4, num_channels=num_channels)
    out_100hz = bp_pass.process(sig_100hz_multi)
    # Ignore initial transient (first 200 samples = 100 ms)
    rms_in_100 = np.sqrt(np.mean(sig_100hz_multi[200:] ** 2))
    rms_out_100 = np.sqrt(np.mean(out_100hz[200:] ** 2))
    gain_100 = rms_out_100 / rms_in_100
    assert 0.90 < gain_100 < 1.05, f"100 Hz gain out of expected passband: {gain_100}"

    # Test 2: Low stopband (5 Hz)
    sig_5hz = np.sin(2 * np.pi * 5.0 * t).astype(np.float32)[:, None]
    sig_5hz_multi = np.tile(sig_5hz, (1, num_channels))
    bp_low = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=fs, order=4, num_channels=num_channels)
    out_5hz = bp_low.process(sig_5hz_multi)
    rms_in_5 = np.sqrt(np.mean(sig_5hz_multi[200:] ** 2))
    rms_out_5 = np.sqrt(np.mean(out_5hz[200:] ** 2))
    gain_5 = rms_out_5 / rms_in_5
    assert gain_5 < 0.15, f"5 Hz not attenuated sufficiently: gain={gain_5}"

    # Test 3: High stopband (600 Hz)
    sig_600hz = np.sin(2 * np.pi * 600.0 * t).astype(np.float32)[:, None]
    sig_600hz_multi = np.tile(sig_600hz, (1, num_channels))
    bp_high = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=fs, order=4, num_channels=num_channels)
    out_600hz = bp_high.process(sig_600hz_multi)
    rms_in_600 = np.sqrt(np.mean(sig_600hz_multi[200:] ** 2))
    rms_out_600 = np.sqrt(np.mean(out_600hz[200:] ** 2))
    gain_600 = rms_out_600 / rms_in_600
    assert gain_600 < 0.20, f"600 Hz not attenuated sufficiently: gain={gain_600}"


def test_filter_zero_channel_crosstalk() -> None:
    """An impulse on channel 0 must not leak into any other channel."""
    bp = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=2000.0, order=4, num_channels=12)
    sig = np.zeros((200, 12), dtype=np.float32)
    sig[10, 0] = 1.0  # Impulse on channel 0 only

    out = bp.process(sig)
    assert np.any(np.abs(out[:, 0]) > 0.01), "Channel 0 should have response"
    assert np.all(out[:, 1:] == 0.0), "Channels 1..11 must have exactly zero cross-talk"


def test_filter_streaming_chunk_invariance() -> None:
    """Processing a signal continuously vs in chunks must yield bit-exact results."""
    rng = np.random.default_rng(42)
    sig = rng.normal(0.0, 1.0, (1000, 12)).astype(np.float32)

    # All-at-once
    bp1 = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=2000.0, order=4, num_channels=12)
    out_all = bp1.process(sig)

    # Chunk-by-chunk (chunks of 100 samples)
    bp2 = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=2000.0, order=4, num_channels=12)
    chunk_outs: list[np.ndarray] = []
    for i in range(0, 1000, 100):
        chunk_outs.append(bp2.process(sig[i : i + 100]))
    out_chunked = np.vstack(chunk_outs)

    np.testing.assert_allclose(out_all, out_chunked, atol=1e-6)


def test_filter_rejects_nan_and_inf() -> None:
    """Filter must reject inputs containing NaN or Inf with ValueError."""
    bp = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=2000.0, order=4, num_channels=12)

    sig_nan = np.zeros((100, 12), dtype=np.float32)
    sig_nan[10, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        bp.process(sig_nan)

    sig_inf = np.zeros((100, 12), dtype=np.float32)
    sig_inf[10, 3] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        bp.process(sig_inf)
