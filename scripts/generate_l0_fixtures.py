#!/usr/bin/env python3
"""Generate independent L0 analytical golden fixtures for semg_dsp test verification.

Strictly adheres to Constitution Principle VI:
1. Pure analytical / mathematical formulations in high precision (float64), converted to float32.
2. NEVER imports semg_dsp or any production code under test.
3. Produces tests/fixtures/dsp/l0_analytical_cases.npz with deterministic, reproducible arrays.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

# Assert zero production DSP imports
for mod in sys.modules:
    assert not mod.startswith("semg_dsp"), f"Principle VI violation: {mod} imported!"


def generate_l0_fixtures(output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = 200
    n_channels = 2
    fs = 1000.0  # Hz
    t64 = np.arange(n_samples, dtype=np.float64) / fs  # high-precision time vector

    # 1. Zeros
    zeros_exact = np.zeros((n_samples, n_channels), dtype=np.float32)

    # 2. Constant / DC (offset = 2.5)
    dc_offset = 2.5
    dc_exact = np.full((n_samples, n_channels), dc_offset, dtype=np.float32)

    # 3. Kronecker Impulse (at index 15)
    impulse_idx = 15
    impulse_exact = np.zeros((n_samples, n_channels), dtype=np.float32)
    impulse_exact[impulse_idx, :] = 1.0

    # 4. Unit Step (starts at index 30)
    step_idx = 30
    step_exact = np.zeros((n_samples, n_channels), dtype=np.float32)
    step_exact[step_idx:, :] = 1.0

    # 5. Single Sine: amplitude 1.5, frequency 10.0 Hz, phase pi/6
    # Calculated in float64, then cast to float32
    f_sine = 10.0
    a_sine = 1.5
    phi_sine = np.pi / 6.0
    sine64 = a_sine * np.sin(2.0 * np.pi * f_sine * t64 + phi_sine)
    sine_ref = np.column_stack([sine64, sine64]).astype(np.float32)

    # 6. Multi-Tone: 3 frequencies (5 Hz, 25 Hz, 60 Hz) with distinct amplitudes and phases
    f_tones = [5.0, 25.0, 60.0]
    a_tones = [1.0, 0.5, 0.25]
    phi_tones = [0.0, np.pi / 4.0, np.pi / 3.0]
    multi64 = np.zeros(n_samples, dtype=np.float64)
    for a, f, phi in zip(a_tones, f_tones, phi_tones):
        multi64 += a * np.sin(2.0 * np.pi * f * t64 + phi)
    multi_ref = np.column_stack([multi64, multi64]).astype(np.float32)

    # 7. Clipping / Saturation: sine with amplitude 2.5 clipped to [-1.0, 1.0]
    sat_raw64 = 2.5 * np.sin(2.0 * np.pi * 5.0 * t64)
    sat64 = np.clip(sat_raw64, -1.0, 1.0)
    sat_ref = np.column_stack([sat64, sat64]).astype(np.float32)

    # Metadata dictionary for test parameters
    metadata = {
        "n_samples": n_samples,
        "n_channels": n_channels,
        "fs": fs,
        "dc_offset": dc_offset,
        "impulse_idx": impulse_idx,
        "step_idx": step_idx,
        "f_sine": f_sine,
        "a_sine": a_sine,
        "phi_sine": phi_sine,
        "f_tones": f_tones,
        "a_tones": a_tones,
        "phi_tones": phi_tones,
        "clip_limits": (-1.0, 1.0),
    }

    # Save to npz
    np.savez_compressed(
        output_path,
        zeros=zeros_exact,
        dc=dc_exact,
        impulse=impulse_exact,
        step=step_exact,
        sine=sine_ref,
        multi_tone=multi_ref,
        saturation=sat_ref,
    )

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"Generated independent L0 fixture: {output_path}")
    print(f"SHA-256 Digest: {digest}")
    print("Constitution Principle VI Check: Verified zero semg_dsp imports in generator.")
    return digest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    fixture_path = root / "tests" / "fixtures" / "dsp" / "l0_analytical_cases.npz"
    generate_l0_fixtures(fixture_path)
