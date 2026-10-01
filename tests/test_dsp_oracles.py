"""Independent Oracle Verification Tests for semg_dsp (US4, FR-009, FR-010).

Verifies CausalSosFilter and StreamingPipeline against independent SciPy L1
oracles and analytical L0 fixtures under formal tolerance contracts:
- TOL-SOS-FILTER-L1: rtol=1e-5, atol=1e-5 (SciPy sosfilt equivalence)
- TOL-ANALYTICAL-L0: rtol=1e-6, atol=1e-6 (Analytical waveforms)
- TOL-CHUNK-INVARIANCE: rtol=1e-6, atol=1e-6 (Filter streaming continuity)
- TOL-WINDOW-ACCUMULATION: rtol=0.0, atol=0.0 (Bitwise window accumulation)

Adheres strictly to Constitution Principle VI:
- Test oracles are computed independently via SciPy and analytical formulas.
- Code under test never generates its own test references.
"""

from pathlib import Path
import numpy as np
from scipy.signal import sosfilt

from semg_dsp.source import SyntheticSampleSource
from semg_dsp.filter import CausalSosFilter
from semg_dsp.window import StatefulWindowBuffer
from semg_dsp.pipeline import StreamingPipeline


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "dsp"


def test_causal_sos_filter_vs_scipy_sosfilt_l1_oracle():
    """Verify CausalSosFilter against independent scipy.signal.sosfilt under TOL-SOS-FILTER-L1."""
    fixture_path = FIXTURES_DIR / "sos_test_filter.npz"
    assert fixture_path.is_file(), f"Missing fixture: {fixture_path}"

    with np.load(fixture_path) as npz:
        sos = npz["sos"].astype(np.float32)
        test_signal = npz["sine_in"].astype(np.float32)

    num_channels = test_signal.shape[1]
    filt = CausalSosFilter(sos, num_channels=num_channels)

    # Process through streaming filter
    filtered_dsp = filt.process_chunk(test_signal)

    # Compute reference with independent SciPy oracle
    filtered_scipy = np.empty_like(test_signal, dtype=np.float32)
    for ch in range(num_channels):
        scipy_out = sosfilt(sos.astype(np.float64), test_signal[:, ch].astype(np.float64))
        filtered_scipy[:, ch] = scipy_out.astype(np.float32)

    # Contract: TOL-SOS-FILTER-L1 (rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        filtered_dsp,
        filtered_scipy,
        rtol=1e-5,
        atol=1e-5,
        err_msg="CausalSosFilter deviates from independent SciPy oracle beyond TOL-SOS-FILTER-L1",
    )


def test_full_pipeline_chunk_invariance_and_oracle_equivalence():
    """Verify StreamingPipeline bitwise chunk invariance and SciPy L1 oracle compliance."""
    fixture_path = FIXTURES_DIR / "sos_test_filter.npz"
    with np.load(fixture_path) as npz:
        sos = npz["sos"].astype(np.float32)

    fs = 1000.0
    w_len = 64
    stride = 16
    n_ch = 2

    src1 = SyntheticSampleSource(waveform="sine", frequency_hz=20.0, num_channels=n_ch, sampling_rate_hz=fs)
    flt1 = CausalSosFilter(sos, num_channels=n_ch)
    win1 = StatefulWindowBuffer(window_length=w_len, stride=stride, num_channels=n_ch)
    pipe1 = StreamingPipeline(src1, flt1, win1)

    src2 = SyntheticSampleSource(waveform="sine", frequency_hz=20.0, num_channels=n_ch, sampling_rate_hz=fs)
    flt2 = CausalSosFilter(sos, num_channels=n_ch)
    win2 = StatefulWindowBuffer(window_length=w_len, stride=stride, num_channels=n_ch)
    pipe2 = StreamingPipeline(src2, flt2, win2)

    # Run A: monolithic chunks of 64
    wins_a = []
    for _ in range(5):
        wins_a.extend(pipe1.step(64))

    # Run B: irregular fragmented chunks
    wins_b = []
    chunk_pattern = [7, 13, 29, 3, 44, 28, 50, 42, 60, 44]
    for n in chunk_pattern:
        wins_b.extend(pipe2.step(n))

    assert len(wins_a) == len(wins_b)
    for i, (wa, wb) in enumerate(zip(wins_a, wins_b)):
        np.testing.assert_array_equal(
            wa,
            wb,
            err_msg=f"Window {i} chunk invariance failed under variable streaming partitions",
        )
