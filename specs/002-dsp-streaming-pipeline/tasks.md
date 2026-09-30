---
description: "SpecKit task checklist with harness metadata"
---

# Tasks: 002-dsp-streaming-pipeline

**Input**: Design documents from `/specs/002-dsp-streaming-pipeline/`
**Prerequisites**: `spec.md` and `plan.md` (reviewed and compliant with Constitution v1.1.0 Principle VI).

Keep the SpecKit checklist format exactly: `- [ ] T001 [P?] [US1?] Description with file path`.
Each task has adjacent harness metadata. Implementation `allowed_files` lists only exact production paths; the RED TestDesigner declares the task-specific test files and commands before implementation.

## TDD Contract & Numerical Integrity Invariants

- TDD applies to all implementation tasks (T001–T006): each runs RED → GREEN → REFACTOR under the Python orchestrator.
- In RED, the TestDesigner creates task-specific tests in `tests/`, declares the focused test command, and demonstrates `EXPECTED_FAILURE` caused by the missing behavior. The orchestrator snapshots those tests; GREEN and REFACTOR must not edit or weaken them.
- Algorithmic and numerical transformation tasks (T003, T004, T005, T006) are marked with `"numeric_sensitive": true`. The orchestrator enforces strict provider family independence (`test_designer` ≠ `test_validator` e `test_validator` ≠ `coder`). Structural and audit tasks (T001, T002, T007) are marked `"numeric_sensitive": false`.
- Fixtures and reference golden vectors created by test design are declared under `fixture_files` and tracked via SHA-256 snapshots against tampering (`TEST_TAMPERING`).
- Under Constitution Principle VI, test oracles MUST be computed independently (analytical L0 formulas or SciPy offline routines); code under test in `semg_dsp/` MUST NEVER generate its own expected test assertions.
- REFACTOR for `numeric_sensitive` tasks operates as a safe no-op.
- T007 is a non-implementation review task marked `NOT_AUTOMATABLE`.

---

## Phase 1: Setup

**Purpose**: Initialize the core package structure for `semg_dsp` without premature symbol exports.

- [x] T001 [P] Setup core package layout in semg_dsp/__init__.py
  <!-- harness-task {"requirements":["FR-001"],"acceptance_criteria":["AC-010"],"plan_decisions":["D-001"],"dependencies":[],"test_type":"UNIT","allowed_files":["semg_dsp/__init__.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

**Checkpoint**: Package importable and exposes clean namespace without importing nonexistent modules.

---

## Phase 2: Foundational

**Purpose**: Establish base contracts for streaming sample ingestion and data containers.

- [ ] T002 [US1] Implement SampleSource abstract protocol and ChunkData container in semg_dsp/source.py
  <!-- harness-task {"requirements":["FR-001","FR-008"],"acceptance_criteria":["AC-001","AC-005","AC-010"],"plan_decisions":["D-001","D-003"],"dependencies":["T001"],"test_type":"UNIT","allowed_files":["semg_dsp/source.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

**Checkpoint**: `SampleSource` base abstraction ready; `ChunkData` enforces `float32` and temporal metadata.

---

## Phase 3: User Story 1 — Synthetic L0 Analytical Sample Generation (P1)

**Goal**: Generate deterministic multichannel analytical waveforms (zeros, DC, impulse, step, sine, multi-tone, saturation) in `float32` with phase continuity across streaming chunks.

**Independent Test**: Instantiate `SyntheticSampleSource` with analytical test cases; call `read_chunk` across varying chunk lengths (1 to 64 samples); assert exact match against mathematical formulas within `TOL-ANALYTICAL-L0` tolerance and continuous phase.

- [ ] T003 [US1] Implement SyntheticSampleSource with L0 analytical waveforms and phase continuity in semg_dsp/source.py
  <!-- harness-task {"requirements":["FR-002","FR-008","FR-010","FR-011"],"acceptance_criteria":["AC-001","AC-005","AC-006","AC-011"],"plan_decisions":["D-001","D-003","D-004","D-005"],"dependencies":["T002"],"test_type":"UNIT","allowed_files":["semg_dsp/source.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true,"fixture_files":["tests/fixtures/dsp/l0_analytical_cases.npz"]} -->

**Checkpoint**: Story 1 complete; synthetic signal source produces deterministic multichannel streaming chunks.

---

## Phase 4: User Story 2 — Causal SOS/Biquad Streaming Filtering (P1)

**Goal**: Filter streaming chunks using Direct Form II Transposed (DF2T) biquad cascade with explicit state array `(n_sections, 2, n_channels)`, strict causality, and chunk invariance.

**Independent Test**: Filter arbitrary continuous signals in a single batch call versus successive fragmented chunks (1 sample to $N$ samples); verify chunk invariance $\max \| \text{concat}(F(C_i)) - F(\text{concat}(C_i)) \| \le \text{TOL-CHUNK-INVARIANCE}$; assert strict `float32` state and outputs; verify zero lookahead.

- [ ] T004 [US2] Implement CausalSosFilter with Direct Form II Transposed state and chunk-invariance in semg_dsp/filter.py
  <!-- harness-task {"requirements":["FR-003","FR-004","FR-005","FR-008","FR-010","FR-011"],"acceptance_criteria":["AC-002","AC-003","AC-006","AC-011"],"plan_decisions":["D-002","D-003","D-005","D-006"],"dependencies":["T003"],"test_type":"UNIT","allowed_files":["semg_dsp/filter.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true,"fixture_files":["tests/fixtures/dsp/sos_test_filter.npz"]} -->

**Checkpoint**: Story 2 complete; filter operates sample-by-sample and chunk-by-chunk with zero future-sample lookahead and verified chunk invariance.

---

## Phase 5: User Story 3 — Stateful Windowing & Residual Policy (P1)

**Goal**: Slice continuous filtered streams into fixed-length sliding windows `(window_length, num_channels)` in `float32` according to `stride`, maintaining state across chunks with safe residual handling.

**Independent Test**: Stream variable-length chunks into `StatefulWindowBuffer`; verify exact window shapes, correct stride offsets, channel preservation, and retention of residual samples during streaming with safe handling on `finalize()`. Assert that the sequence of emitted windows is identical regardless of input chunk fragmentation.

- [ ] T005 [US3] Implement StatefulWindowBuffer causal sliding window accumulator in semg_dsp/window.py
  <!-- harness-task {"requirements":["FR-006","FR-007","FR-008","FR-010","FR-011"],"acceptance_criteria":["AC-004","AC-005","AC-006","AC-011"],"plan_decisions":["D-003","D-005","D-007"],"dependencies":["T004"],"test_type":"UNIT","allowed_files":["semg_dsp/window.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

**Checkpoint**: Story 3 complete; continuous sample streams are deterministically partitioned into sliding windows without dropping samples between chunks.

---

## Phase 6: User Story 4 — Streaming Pipeline & Independent Oracle Verification (P1)

**Goal**: Connect source, causal filter, and window buffer into an integrated streaming pipeline verified end-to-end against independent SciPy L1 oracles and registered tolerance contracts under Principle VI.

**Independent Test**: Run end-to-end streaming pipeline test with synthetic multi-frequency signal; compare filter output against independent SciPy `sosfilt` reference; verify all deviations meet `TOL-SOS-FILTER-L1`; verify window emission timing and channel integrity.

- [ ] T006 [US4] Implement StreamingPipeline coordinator connecting source, filter, and window stages in semg_dsp/pipeline.py
  <!-- harness-task {"requirements":["FR-008","FR-009","FR-010","FR-011","FR-012"],"acceptance_criteria":["AC-003","AC-005","AC-007","AC-008","AC-009","AC-011"],"plan_decisions":["D-001","D-003","D-004","D-005","D-006"],"dependencies":["T005"],"test_type":"INTEGRATION","allowed_files":["semg_dsp/pipeline.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true,"fixture_files":["tests/fixtures/dsp/sos_test_filter.npz"]} -->

**Checkpoint**: Story 4 complete; end-to-end streaming pipeline verified against independent scientific oracles.

---

## Final Phase: Polish & Cross-Cutting Concerns

- [ ] T007 Final audit of scope boundaries, oracle independence, and Constitution Principle VI
  <!-- harness-task {"requirements":["FR-009","FR-010"],"acceptance_criteria":["AC-008","AC-010","AC-012"],"plan_decisions":["D-004","D-005","D-008"],"dependencies":["T006"],"test_type":"NOT_AUTOMATABLE","justification":"Human review of the final feature diff, verification of Principle VI oracle independence, absence of real dataset assumptions, and confirmation of deterministic quality gates.","alternative_verification":"Review git diff to verify no out-of-scope files were modified; verify no runtime dependency on scipy; verify all numerical tests pass against independent SciPy oracles under registered tolerances; run full quality gate: pytest -q, ruff check, mypy, compileall.","allowed_files":[],"tdd_phases":[],"numeric_sensitive":false} -->

---

## Dependencies & Execution Order

Tasks are ordered strictly by dependency:

```text
T001 (Setup) 
  └─► T002 (SampleSource Base) 
        └─► T003 (SyntheticSource L0) 
              └─► T004 (CausalSosFilter DF2T) 
                    └─► T005 (StatefulWindowBuffer) 
                          └─► T006 (StreamingPipeline E2E) 
                                └─► T007 (Audit & Principle VI Verification)
```

---

## Parallel Execution

Tasks T001–T006 represent sequential pipeline pipeline layers and shared module dependencies. Each task must execute its own RED phase to produce focused, immutable test designs before GREEN implementation.

---

## Implementation Strategy

1. **Setup & Ingestion Foundation (T001, T002)**: Establish `semg_dsp` package layout and streaming data contracts.
2. **Deterministic Waveform Synthesis (T003)**: Provide pure mathematical L0 signal sources for offline and streaming test suites.
3. **Causal Filter Engine (T004)**: Deliver Direct Form II Transposed biquad filtering with explicit state retention and chunk invariance.
4. **Window Buffer Assembly (T005)**: Implement causal sliding buffer with deterministic stride advancement and safe residual sample handling.
5. **Integrated Pipeline & Oracle Validation (T006)**: Connect stages and validate numerically against independent SciPy golden vectors under registered tolerance contracts.
6. **Final Audit (T007)**: Review scope, immutability of test fixtures, and full deterministic verification pass.
