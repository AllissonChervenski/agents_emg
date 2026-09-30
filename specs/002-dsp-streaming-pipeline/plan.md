# Implementation Plan: 002-dsp-streaming-pipeline

**Branch**: `feat/002-dsp-streaming-pipeline` | **Date**: 2026-09-30 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-dsp-streaming-pipeline/spec.md`

---

## Summary

Implement a host-only, strictly causal, streaming digital signal processing (DSP) pipeline for sEMG signals using synthetic sample generation, Direct Form II Transposed (DF2T) SOS/biquad filtering with explicit state, and deterministic stateful windowing. All operations enforce `numpy.float32` precision, chunk-invariance ($\text{concat}(F(C_1), \dots, F(C_k)) \approx F(\text{concat}(C_1, \dots, C_k))$ e identidade ordenada da sequência de janelas), e verificação numérica independente sob o Princípio VI da Constituição usando sinais analíticos L0 e oráculos SciPy L1.

---

## Technical Context

- **Language / Environment**: Python 3.10+ (host-only)
- **Primary Runtime Dependencies**: `numpy` (production runtime), standard library (`dataclasses`, `typing`, `enum`)
- **Testing & Oracle Dependencies**: `pytest`, `scipy` (strictly restricted to test oracles/fixtures, never imported in production runtime `semg_dsp`)
- **New Dependencies**: Zero new runtime dependencies. Property-based testing (Hypothesis) is marked `DECISION_REQUIRED` and deferred.
- **Storage**: In-memory streaming state; golden reference vectors stored as immutable fixtures in `tests/fixtures/dsp/`
- **Target Platform**: Host Python execution, designed for future 1:1 structural transposition to C++ / embedded targets (CMSIS-DSP / standard biquad layout)
- **Constraints**:
  - Strictly causal runtime (no `filtfilt`, no lookahead, no whole-recording normalization);
  - Single-precision floating point (`float32`) without silent promotion to `float64`;
  - Code under test NEVER generates its own test references (Principle VI);
  - O runtime host deve usar buffers internos de tamanho limitado e shapes determinísticos, evitar crescimento não limitado e evitar buffering da gravação completa. Zero dynamic allocation é requisito futuro do port C++/embedded, não uma garantia do runtime Python da Feature 002;
  - Todas as tarefas de transformação e aritmética numérica marcadas como `numeric_sensitive: true` com separação mútua de famílias de provedores (`test_designer` ≠ `test_validator` ≠ `coder`).

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
| **D-002** | Direct Form II Transposed (DF2T) biquad cascade com vetor de estado explícito `(n_sections, 2, n_channels)`. | DF2T minimiza ruído de quantização em precisão simples, consome apenas 2 elementos de atraso por seção/canal e opera amostra a amostra sem lookahead. | Direct Form I (maior uso de memória); Direct Form II canônica não-transposta (mais sensível a quantização). |
| **D-003** | Precisão estrita `np.float32` em entradas, estados e saídas. | `float32` é a referência numérica do pipeline host por ser um dtype explícito, reproduzível e adequado à futura migração para runtimes embedded com precisão simples. Evita consumo excessivo de memória de buffer e discrepâncias de ponto flutuante. | `float64` (rejeitado: mascara estabilidade numérica que falharia em microcontroladores). |
| **D-004** | Dual-tier independent oracle strategy (L0 analytical + L1 SciPy `sosfilt`). | Satisfies Constitution Principle VI. L0 proves fundamental DSP theory (impulse response, step response, DC gain); L1 verifies arbitrary filter responses against established scientific software without self-reference. | Self-generated test assertions (strictly forbidden by Principle VI); pure L0 only (insufficient for complex multi-pole filters). |
| **D-005** | Formal catalog of tolerance contracts (`tolerance_id`) com status `PROVISIONAL`. | Elimina tolerâncias arbitrárias. Cada asserção referencia um contrato explícito (`TOL-ANALYTICAL-L0`, `TOL-SOS-FILTER-L1`, `TOL-CHUNK-INVARIANCE`, `TOL-WINDOW-ACCUMULATION`) documentado com proveniência e justificativa. | Tolerância universal hardcoded `rtol=1e-7` (inviável para filtros IIR em cascata em float32). |
| **D-006** | Chunk-invariance verification harness para filtro e janelador. | Comprova que para o filtro $\text{concat}(F(C_1), \dots, F(C_k)) \approx F(\text{concat}(C_1, \dots, C_k))$ e que para o janelador a sequência de janelas emitida é exatamente idêntica independente do particionamento de chunks. | Testar apenas chunks de tamanho fixo (não detecta erros de acumulador em fronteiras). |
| **D-007** | Stateful window buffer retendo amostras residuais entre chunks com descarte apenas em `finalize()`. | `process_chunk()` NUNCA descarta amostras parciais durante o streaming; elas permanecem no buffer com estado para os próximos chunks. O descarte sob a política `"drop"` ocorre exclusivamente em `finalize()`. | Descartar residual ao fim de cada chunk (quebraria a propriedade de streaming). |
| **D-008** | Safe no-op REFACTOR for all `numeric_sensitive` tasks. | Protects mathematically delicate implementations from unguided agent modifications that maintain green tests by loosening precision or restructuring operations. | Active refactoring without snapshot harness (high risk of numerical regression). |

---

## Requirement-to-Decision & Test Traceability

| Requirement | Plan Decision | Test Approach & Verification Evidence |
|---|---|---|
| **FR-001, FR-002** (SampleSource & SyntheticSource) | D-001, D-003: Abstract `SampleSource` base class; `SyntheticSampleSource` supporting zeros, DC, impulse, step, sine, multi-tone, saturation. | `tests/test_dsp_source.py`: Assert exact waveform generation, continuity of phase across multiple `read_chunk` calls, `float32` dtype, and error on invalid parameters. |
| **FR-003, FR-004** (Causal SOS Filter & Strict Causality) | D-002, D-003, D-006: `CausalSosFilter` with DF2T state `(n_sections, 2, n_channels)`. No `filtfilt`, no lookahead. | `tests/test_dsp_filter.py`: Assert output matches causal difference equations; verify non-causal outputs are zero before input arrival ($t < t_0$). |
| **FR-005** (Versioned SOS Coefficients) | D-002: Coefficients passed as pre-computed array `(n_sections, 6)`. | `tests/test_dsp_filter.py`: Reject coefficients with invalid shapes or non-float32 dtypes. |
| **FR-006, FR-007** (Stateful Windowing & Partial Policy) | D-007: `StatefulWindowBuffer` managing circular buffer with `window_length` and `stride`, returning `list[np.ndarray]` via `process_chunk`, retaining residuals across streaming calls, and applying `drop` only on `finalize()`. | `tests/test_dsp_window.py`: Test variable chunk arrivals ($1$ to $3 \times W$), verify window shapes `(W, C)`, verify residual retention during streaming and safe handling under `finalize()`. |
| **FR-008** (Strict `float32` Invariant) | D-003: Explicit casting and dtype checks at ingestion and state update. | `tests/test_dsp_filter.py`, `tests/test_dsp_window.py`: Validate `array.dtype == np.float32` on every returned chunk, window, and internal state. |
| **FR-009, FR-010** (Oracle Integrity & Tolerance Catalog) | D-004, D-005: Independent reference fixtures in `tests/fixtures/dsp/` generated with SciPy / math formulas; verified against tolerance IDs with `PROVISIONAL` status. | `tests/test_dsp_oracles.py`: Compare filter output with `scipy.signal.sosfilt` reference; assert discrepancies are within `TOL-SOS-FILTER-L1`. |
| **FR-011, FR-012** (Numeric Sensitivity & Family Separation) | D-008: Metadata `"numeric_sensitive": true` on algorithmic tasks; harness enforces `test_designer` ≠ `test_validator` ≠ `coder`. | Orchestrator runner telemetry validates distinct provider families assigned for each role in numerical tasks. |
| **Chunk Invariance** | D-006: Partition continuous signals into variable chunks, stream through filter and window buffer, compare with monolithic execution. | `tests/test_dsp_filter.py`: Assert $\max \| \text{concat}(F(C_i)) - F(\text{concat}(C_i)) \| \le \text{TOL-CHUNK-INVARIANCE}$. `tests/test_dsp_window.py`: Assert identical sequence of emitted windows regardless of chunk partition. |

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

    def finalize(self) -> list[np.ndarray]:
        # handles end-of-stream residuals according to partial_policy
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

    def finalize(self) -> list[np.ndarray]:
        # flushes remaining windows from window buffer
        ...

    def reset(self) -> None: ...
```

---

## Tolerance Catalog (`tolerance_id`)

| Tolerance ID | Runtime Dtype | Comparison Target | `rtol` | `atol` | Status | Scientific / Mathematical Rationale |
|---|---|---|---|---|---|---|
| `TOL-ANALYTICAL-L0` | `float32` | Pure analytical formulas (step, DC, zero, impulse) | `1e-6` | `1e-6` | `PROVISIONAL` | Exact single-precision floating point limit for pure mathematical formulas (24-bit mantissa IEEE 754). |
| `TOL-SOS-FILTER-L1` | `float32` | SciPy high-precision reference (`scipy.signal.sosfilt`) | `1e-5` | `1e-5` | `PROVISIONAL` | Accounts for minor floating-point summation order differences between SciPy C-routine and explicit DF2T loop in `float32`. |
| `TOL-CHUNK-INVARIANCE`| `float32` | Streaming concatenated chunks vs batch execution | `1e-6` | `1e-6` | `PROVISIONAL` | Internal filter state maintains exact sample-to-sample continuity; discrepancy is zero or limited to machine epsilon. |
| `TOL-WINDOW-ACCUMULATION`| `float32` | Window buffer sample reproduction | `0.0` | `0.0` | `PROVISIONAL` | Pure discrete buffer indexing and sliding; values must be bitwise identical. |

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
├── __init__.py                 # Core package layout (exports added progressively)
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
