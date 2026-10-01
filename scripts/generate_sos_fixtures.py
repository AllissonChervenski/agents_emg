#!/usr/bin/env python3
"""Generate independent L1 SOS filter golden fixtures for semg_dsp test verification.

Strictly adheres to Constitution Principle VI:
1. Pure scientific calculation using scipy.signal.butter and scipy.signal.sosfilt.
2. NEVER imports semg_dsp or any production code under test.
3. Produces tests/fixtures/dsp/sos_test_filter.npz with deterministic, reproducible arrays:
   - sos: 2-section 4th order bandpass filter coefficients (float32, normalized a0=1.0)
   - input signals, expected outputs, and expected final states for:
     zeros, impulse (isolated on ch0 to test ch1 cross-talk), dc, sine, multi_tone.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import scipy.signal as signal

# Assert zero production DSP imports
for mod in sys.modules:
    assert not mod.startswith("semg_dsp"), f"Principle VI violation: {mod} imported!"


def generate_sos_fixtures(output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = 200
    n_channels = 2
    fs = 1000.0  # Hz

    # 4th order bandpass (2 sections), [20, 450] Hz at fs=1000 Hz (Nyquist = 500 Hz)
    # Wn = [20/500, 450/500] = [0.04, 0.90]
    sos64 = signal.butter(4, [0.04, 0.90], btype="bandpass", output="sos")
    n_sections = sos64.shape[0]
    assert n_sections == 4  # 4th order bandpass has 4 biquad sections

    # Let us also create a 2-section filter (4th order lowpass or 2nd order bandpass)
    # A 2-section bandpass (2nd order = 2 sections):
    sos2_64 = signal.butter(2, [0.05, 0.45], btype="bandpass", output="sos")
    assert sos2_64.shape[0] == 2
    sos = sos2_64.astype(np.float32)

    # Verify a0 == 1.0 for each section
    for s in range(sos.shape[0]):
        assert np.isclose(sos[s, 3], 1.0), f"Section {s} a0 is not 1.0"
        sos[s, 3] = 1.0  # normalize strictly to 1.0

    t64 = np.arange(n_samples, dtype=np.float64) / fs

    # 1. Zeros
    zeros_in = np.zeros((n_samples, n_channels), dtype=np.float32)
    zi_zeros = np.zeros((sos.shape[0], 2, n_channels), dtype=np.float64)
    zeros_out64, zeros_zf64 = signal.sosfilt(sos.astype(np.float64), zeros_in.astype(np.float64), axis=0, zi=zi_zeros)

    # 2. Impulse (isolated on channel 0, channel 1 is completely zeros to test cross-talk)
    impulse_in = np.zeros((n_samples, n_channels), dtype=np.float32)
    impulse_in[10, 0] = 1.0  # impulse at sample 10 on ch0 only
    zi_impulse = np.zeros((sos.shape[0], 2, n_channels), dtype=np.float64)
    impulse_out64, impulse_zf64 = signal.sosfilt(sos.astype(np.float64), impulse_in.astype(np.float64), axis=0, zi=zi_impulse)
    # Channel 1 must be strictly zero
    assert np.all(impulse_out64[:, 1] == 0.0)
    assert np.all(impulse_zf64[:, :, 1] == 0.0)

    # 3. DC / Constant (1.0 on both channels)
    dc_in = np.ones((n_samples, n_channels), dtype=np.float32)
    zi_dc = np.zeros((sos.shape[0], 2, n_channels), dtype=np.float64)
    dc_out64, dc_zf64 = signal.sosfilt(sos.astype(np.float64), dc_in.astype(np.float64), axis=0, zi=zi_dc)

    # 4. Single Sine (50 Hz, well inside bandpass)
    f_sine = 50.0
    sine64 = np.sin(2.0 * np.pi * f_sine * t64)
    sine_in = np.column_stack([sine64, sine64]).astype(np.float32)
    zi_sine = np.zeros((sos.shape[0], 2, n_channels), dtype=np.float64)
    sine_out64, sine_zf64 = signal.sosfilt(sos.astype(np.float64), sine_in.astype(np.float64), axis=0, zi=zi_sine)

    # 5. Multi-tone (10 Hz stopband, 100 Hz passband, 480 Hz stopband)
    multi64 = (
        0.5 * np.sin(2.0 * np.pi * 10.0 * t64)
        + 1.0 * np.sin(2.0 * np.pi * 100.0 * t64)
        + 0.3 * np.sin(2.0 * np.pi * 480.0 * t64)
    )
    multi_in = np.column_stack([multi64, multi64 * 0.5]).astype(np.float32)
    zi_multi = np.zeros((sos.shape[0], 2, n_channels), dtype=np.float64)
    multi_out64, multi_zf64 = signal.sosfilt(sos.astype(np.float64), multi_in.astype(np.float64), axis=0, zi=zi_multi)

    # Save all arrays in compressed npz as strict float32
    np.savez_compressed(
        output_path,
        sos=sos,
        zeros_in=zeros_in,
        zeros_out=zeros_out64.astype(np.float32),
        zeros_zf=zeros_zf64.astype(np.float32),
        impulse_in=impulse_in,
        impulse_out=impulse_out64.astype(np.float32),
        impulse_zf=impulse_zf64.astype(np.float32),
        dc_in=dc_in,
        dc_out=dc_out64.astype(np.float32),
        dc_zf=dc_zf64.astype(np.float32),
        sine_in=sine_in,
        sine_out=sine_out64.astype(np.float32),
        sine_zf=sine_zf64.astype(np.float32),
        multi_in=multi_in,
        multi_out=multi_out64.astype(np.float32),
        multi_zf=multi_zf64.astype(np.float32),
        multi_tone_in=multi_in,
        multi_tone_out=multi_out64.astype(np.float32),
        multi_tone_zf=multi_zf64.astype(np.float32),
    )

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"Generated independent L1 SOS fixture: {output_path}")
    print(f"SHA-256 Digest: {digest}")
    print(f"SOS shape: {sos.shape}, dtype: {sos.dtype}")
    print("Constitution Principle VI Check: Verified zero semg_dsp imports in generator.")
    return digest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    fixture_path = root / "tests" / "fixtures" / "dsp" / "sos_test_filter.npz"
    generate_sos_fixtures(fixture_path)
