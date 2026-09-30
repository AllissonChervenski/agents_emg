# Implementation Plan: 002-dsp-streaming-pipeline

**Branch**: `feat/002-dsp-streaming-pipeline` | **Date**: 2026-09-30 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-dsp-streaming-pipeline/spec.md`

---

## Summary

Implement a host-only, strictly causal, streaming digital signal processing (DSP) pipeline for sEMG signals using synthetic sample generation, Direct Form II Transposed (DF2T) SOS/biquad filtering with explicit state, and deterministic stateful windowing. All operations enforce `numpy.float32` precision, chunk-invariance ($process(X) \approx \sum process(chunk_i)$), and independent numerical verification under Principle VI of the Constitution using analytical L0 signals and SciPy L1 oracles.

---

## Technical Context

- **Language / Environment**: Python 3.10+ (host-only)
- **Primary Runtime Dependencies**: `numpy` (production runtime), standard library (`dataclasses`, `typing`, `enum`)
- **Testing & Oracle Dependencies**: `pytest`, `scipy` (strictly restricted to test oracles/fixtures, never imported in production runtime `semg_dsp`)
- **New Dependencies**: Zero new runtime dependencies. Property-based testing (Hypothesis) is marked `DECISION_REQUIRED` and deferred.
- **Storage**: In-memory streaming state; golden reference vectors stored as immutable fixtures in `tests/fixtures/dsp/`
- **Target Platform**: Host Python execution, designed for future 1:1 structural transposition to C++ / ESP32-S3 (CMSIS-DSP / ESP-DSP compatible layout)
- **Constraints**:
  - Strictly causal runtime (no `filtfilt`, no lookahead, no whole-recording normalization);
  - Single-precision floating point (`float32`) without silent promotion to `float64`;
  - Code under test NEVER generates its own test references (Principle VI);
  - All numerical tasks marked `numeric_sensitive: true` with mutual provider family independence (`test_designer` ≠ `test_validator` ≠ `coder`).

---

## Constitution Check

**Review against Constitution v1.1.0**:

- **I. Python Central Orchestrator & Deterministic Gatekeeper**: **PASS**. All verification relies on deterministic quality gates (`pytest`, `ruff`, `mypy`, `compileall`).
- **II. Strict TDD Lifecycle & Anti-Tampering Protection**: **PASS**. Tasks follow RED → GREEN → REFACTOR. RED tests and `fixture_files` protected via SHA-256 snapshots.
- **III. Non-Destructive Local Reuse**: **PASS**. Host-only, zero subprocesses or model calls at runtime.
- **IV. CLI Ergonomics & Failure Containment**: **PASS**. Clear exceptions (`ValueError`) for non-finite inputs or dimension mismatches.
- **V. SpecKit Traceability**: **PASS**. Full traceability from `FR-001`..`FR-012` to plan decisions and tasks.
- **VI. Scientific and Numerical Oracle Integrity**: **PASS**. Oracles are computed strictly independently via analytical formulas (L0) and SciPy (L1). Golden vectors versioned with registered tolerances.

---

## Research and Design Decisions

| ID | Decision | Rationale | Alternatives Considered |
|---|---|---|---|
| **D-001** | Package layout in dedicated `semg_dsp/` root (`source.py`, `filter.py`, `window.py`, `pipeline.py`). | Separates scientific DSP codebase cleanly from orchestrator engine infrastructure, allowing independent packaging and eventual C++ porting. | Placing under `orchestrator/dsp/` (rejected: couples scientific domain code to orchestration harness). |
| **D-002** | Direct Form II Transposed (DF2T) biquad cascade with explicit state array `(n_sections, 2, n_channels)`. | DF2T minimizes floating-point roundoff noise in single precision, requires only 2 delay elements per biquad section per channel, and supports streaming sample-by-sample without lookahead. | Direct Form I (requires more state storage); Direct Form II non-transposed (more sensitive to coefficient quantization). |
| **D-003** | Strict `np.float32` precision across inputs, state, and outputs. | Directly models target embedded FPU (ESP32-S3 single-precision) and minimizes memory bandwidth. Promotion to `float64` is guarded against at boundary and state updates. | `float64` (rejected: misleading numerical stability that fails when ported to microcontrollers). |
| **D-004** | Dual-tier independent oracle strategy (L0 analytical + L1 SciPy `sosfilt`). | Satisfies Constitution Principle VI. L0 proves fundamental DSP theory (impulse response, step response, DC gain); L1 verifies arbitrary filter responses against established scientific software without self-reference. | Self-generated test assertions (strictly forbidden by Principle VI); pure L0 only (insufficient for complex multi-pole filters). |
| **D-005** | Formal catalog of tolerance contracts (`tolerance_id`). | Eliminates arbitrary ad-hoc tolerances. Every test asserts against an explicitly justified contract (`TOL-ANALYTICAL-L0`, `TOL-SOS-FILTER-L1`, `TOL-CHUNK-INVARIANCE`, `TOL-WINDOW-ACCUMULATION`). | Global hardcoded `rtol=1e-7` (unrealistic for cascaded IIR float32 operations). |
| **D-006** | Chunk-invariance verification harness. | Proves that streaming state is preserved continuously across arbitrary buffer fragmentation ($1$ sample, $7$ samples, $64$ samples) matching monolithic execution. | Testing only fixed chunk sizes (misses boundary accumulator errors). |
| **D-007** | Stateful window buffer with circular/FIFO array and explicit residual policy (`drop` vs `pad`). | Decouples incoming chunk sizes from window stride. Streaming default `drop` avoids edge artifacts from synthetic padding. | Pure batch array slicing (fails in streaming runtime); generator-only approach (lacks explicit state inspectability). |
| **D-008** | Safe no-op REFACTOR for all `numeric_sensitive` tasks. | Protects mathematically delicate implementations from unguided agent modifications that maintain green tests by loosening precision or restructuring operations. | Active refactoring without snapshot harness (high risk of numerical regression). |

---

## Requirement-to-Decision & Test Traceability

| Requirement | Plan Decision | Test Approach & Verification Evidence |
|---|---|---|
| **FR-001, FR-002** (SampleSource & SyntheticSource) | D-001, D-003: Abstract `SampleSource` base class; `SyntheticSampleSource` supporting zeros, DC, impulse, step, sine, multi-tone, saturation. | `tests/test_dsp_source.py`: Assert exact waveform generation, continuity of phase across multiple `read_chunk` calls, `float32` dtype, and error on invalid parameters. |
| **FR-003, FR-004** (Causal SOS Filter & Strict Causality) | D-002, D-003, D-006: `CausalSosFilter` with DF2T state `(n_sections, 2, n_channels)`. No `filtfilt`, no lookahead. | `tests/test_dsp_filter.py`: Assert output matches causal difference equations; verify non-causal outputs are zero before input arrival ($t < t_0$). |
| **FR-005** (Versioned SOS Coefficients) | D-002: Coefficients passed as pre-computed array `(n_sections, 6)`. | `tests/test_dsp_filter.py`: Reject coefficients with invalid shapes or non-float32 dtypes. |
| **FR-006, FR-007** (Stateful Windowing & Partial Policy) | D-007: `StatefulWindowBuffer` managing circular buffer with `window_length` and `stride`, returning `list[np.ndarray]`. | `tests/test_dsp_window.py`: Test variable chunk arrivals ($1$ to $3 \times W$), verify window shapes `(W, C)`, verify residual sample handling under `drop` policy. |
| **FR-008** (Strict `float32` Invariant) | D-003: Explicit casting and dtype checks at ingestion and state update. | `tests/test_dsp_filter.py`, `tests/test_dsp_window.py`: Validate `array.dtype == np.float32` on every returned chunk, window, and internal state. |
| **FR-009, FR-010** (Oracle Integrity & Tolerance Catalog) | D-004, D-005: Independent reference fixtures in `tests/fixtures/dsp/` generated with SciPy / math formulas; verified against tolerance IDs. | `tests/test_dsp_oracles.py`: Compare filter output with `scipy.signal.sosfilt` reference; assert discrepancies are within `TOL-SOS-FILTER-L1`. |
| **FR-011, FR-012** (Numeric Sensitivity & Family Separation) | D-008: Metadata `"numeric_sensitive": true` on tasks; harness enforces `test_designer` ≠ `test_validator` ≠ `coder`. | Orchestrator runner telemetry validates distinct provider families assigned for each role in numerical tasks. |
| **Chunk Invariance** | D-006: Partition signal into random chunks, stream through filter, compare to batch. | `tests/test_dsp_filter.py`: Assert $\max \| y_{\text{batch}} - y_{\text{stream}} \| \le \text{TOL-CHUNK-INVARIANCE}$. |

---

## Data Model & Component Interfaces

### 1. `semg_dsp.source`

```python
class SampleSource(ABC):
    @abstractmethod
    def read_chunk(self, num_samples: int) -> ChunkData: ...

    @abstractmethod
    def reset(self) -> None: ...

@dataclass(frozen=True)
class ChunkData:
    data: np.ndarray          # shape (num_samples, num_channels), dtype float32
    start_sample_idx: int     # sequential sample counter
    sampling_rate_hz: float   # provisional sampling rate (FIXTURE_ONLY)
```

### 2. `semg_dsp.filter`

```python
class CausalSosFilter:
    def __init__(self, sos_coefficients: np.ndarray, num_channels: int):
        # sos_coefficients shape: (n_sections, 6), dtype float32
        # state shape: (n_sections, 2, num_channels), dtype float32
        ...

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        # chunk: (num_samples, num_channels), float32
        # returns: (num_samples, num_channels), float32
        ...

    def reset(self) -> None: ...
```

### 3. `semg_dsp.window`

```python
class StatefulWindowBuffer:
    def __init__(self, window_length: int, stride: int, num_channels: int, partial_policy: str = "drop"):
        ...

    def process_chunk(self, chunk: np.ndarray) -> list[np.ndarray]:
        # returns list of windows: [(window_length, num_channels), ...]
        ...

    def reset(self) -> None: ...
```

### 4. `semg_dsp.pipeline`

```python
class StreamingPipeline:
    def __init__(self, source: SampleSource, filter_stage: CausalSosFilter, window_stage: StatefulWindowBuffer):
        ...

    def step(self, num_samples: int) -> list[np.ndarray]:
        # pulls chunk from source -> filters -> pushes to window buffer -> returns emitted windows
        ...

    def reset(self) -> None: ...
```

---

## Tolerance Catalog (`tolerance_id`)

| Tolerance ID | `rtol` | `atol` | Target Metric | Scientific / Mathematical Rationale |
|---|---|---|---|---|
| `TOL-ANALYTICAL-L0` | `1e-6` | `1e-6` | L0 Analytical signals | Exact single-precision floating point limit for pure mathematical formulas (step, DC, zero). |
| `TOL-SOS-FILTER-L1` | `1e-5` | `1e-5` | Filter output vs SciPy `sosfilt` | Accounts for minor floating-point summation order differences between SciPy C-routine and explicit DF2T loop in `float32`. |
| `TOL-CHUNK-INVARIANCE`| `1e-6` | `1e-6` | Streaming chunks vs batch processing | Internal filter state maintains exact sample-to-sample continuity; discrepancy is zero or limited to machine epsilon. |
| `TOL-WINDOW-ACCUMULATION`| `0.0` | `0.0` | Window buffer sample reproduction | Pure discrete buffer indexing and sliding; values must be bitwise identical. |

---

## Project Structure & File Layout

```text
specs/002-dsp-streaming-pipeline/
├── spec.md                     # Feature requirements & invariants
├── plan.md                     # This architecture & design plan
├── tasks.md                    # Actionable TDD tasks with harness metadata
└── checklists/
    └── requirements.md         # Requirements-quality checklist

semg_dsp/                       # Scientific DSP production code
├── __init__.py                 # Export public pipeline symbols
├── source.py                   # SampleSource and SyntheticSampleSource
├── filter.py                   # CausalSosFilter with DF2T state
├── window.py                   # StatefulWindowBuffer
└── pipeline.py                 # Integrated StreamingPipeline

tests/                          # Test suite & oracles
├── fixtures/dsp/               # Immutable golden vectors (Principle VI)
│   ├── sos_test_filter.npz     # Pre-computed coefficients & independent reference outputs
│   └── l0_analytical_cases.npz # Golden analytical test vectors
├── test_dsp_source.py          # US1 tests
├── test_dsp_filter.py          # US2 tests (causality, DF2T, chunk invariance)
├── test_dsp_window.py          # US3 tests (stateful sliding windows)
├── test_dsp_oracles.py         # US4 tests (SciPy L1 independent oracle validation)
└── test_dsp_pipeline.py        # Integration tests (end-to-end streaming pipeline)
```
