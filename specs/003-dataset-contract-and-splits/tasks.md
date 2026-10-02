---
description: "SpecKit task checklist with harness metadata for Feature 003"
---

# Tasks: 003-dataset-contract-and-splits

**Input**: Design documents from `/specs/003-dataset-contract-and-splits/`  
**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `contracts/` compliant with Constitution v1.1.0.

All tasks follow the SpecKit checklist format: `- [ ] T001 [P?] [US1?] Description with file path`.  
Each task includes adjacent `harness-task` metadata for orchestrator consumption.

---

## TDD Contract & Scope Containment Rules

- TDD applies to all implementation tasks (T001–T008): each runs RED → GREEN → REFACTOR under the Python orchestrator.
- In RED, the TestDesigner writes task-specific tests in `tests/`, demonstrates `EXPECTED_FAILURE`, and Python records SHA-256 snapshots.
- In GREEN, the implementer edits strictly the paths listed in `allowed_files`. Test files are protected against `TEST_TAMPERING`.
- Temporal alignment and numerical validation tasks are marked `"numeric_sensitive": true`. Structural, loader, and manifest tasks are marked `"numeric_sensitive": false`.
- Raw data directory `data/raw/ninapro_db2/` is strictly READ-ONLY. No task may modify or extract archives permanently.
- T009 is a non-implementation review task marked `NOT_AUTOMATABLE`.

---

## Phase 1: Setup

**Purpose**: Initialize the core package structure for `semg_dataset` with clean exports.

- [x] T001 [P] Setup core package layout and exports in semg_dataset/__init__.py
  <!-- harness-task {"requirements":["FR-001"],"acceptance_criteria":["SC-001"],"plan_decisions":["D-001"],"dependencies":[],"test_type":"UNIT","allowed_files":["semg_dataset/__init__.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

**Checkpoint**: `semg_dataset` package is importable and exposes version and namespace.

---

## Phase 2: Foundational

**Purpose**: Establish cryptographic provenance verification and domain contracts.

- [x] T002 [US1] Implement NinaPro DB2 provenance, license terms, and SHA-256 baseline verification in semg_dataset/provenance.py
  <!-- harness-task {"requirements":["FR-002","FR-010"],"acceptance_criteria":["SC-001","SC-006"],"plan_decisions":["D-001"],"dependencies":["T001"],"test_type":"UNIT","allowed_files":["semg_dataset/provenance.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

- [x] T003 [US2] Implement RecordingData, Exercise, Subject, and AlignmentStatus domain entities in semg_dataset/contract.py
  <!-- harness-task {"requirements":["FR-003","FR-008"],"acceptance_criteria":["SC-002"],"plan_decisions":["D-002","D-005"],"dependencies":["T001"],"test_type":"UNIT","allowed_files":["semg_dataset/contract.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

**Checkpoint**: Provenance replaces all `TO_BE_DOCUMENTED` placeholders; base data contracts enforce 12-channel `float32` and zero NaN/Inf.

---

## Phase 3: User Story 1 — Immutable Raw Dataset Ingestion (P1)

**Goal**: Load MATLAB recordings directly from ZIP archives into memory without disk extraction, enforcing read-only constraints and pre-verifying archive hashes.

**Independent Test**: Instantiate `NinaProDB2Loader`, read recording files across subjects, verify that raw directory is unchanged, files are unextracted on disk, and archive hashes are validated.

- [x] T004 [US1] Implement NinaProDB2Loader for in-memory read-only archive streaming in semg_dataset/loader.py
  <!-- harness-task {"requirements":["FR-001","FR-002","FR-003"],"acceptance_criteria":["SC-001","SC-002"],"plan_decisions":["D-001"],"dependencies":["T002","T003"],"test_type":"UNIT","allowed_files":["semg_dataset/loader.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

**Checkpoint**: User Story 1 complete; raw recordings load seamlessly in memory while preserving raw directory immutability.

---

## Phase 4: User Story 2 — PROVE-OR-QUARANTINE Temporal Alignment Engine (P1)

**Goal**: Establish a single canonical timeline per recording. The engine MUST NOT assume in advance that all 18 E3 recordings pass: it must evaluate each recording independently, proving $t=0$ start anchoring before applying `anchor_start_truncate_tail`, and marking unproven, head-shifted, or ambiguous discrepancies as `QUARANTINE`. Synthetic unit tests must differentiate tail truncation ($label[k] \leftrightarrow emg[k]$) from head shift ($label[k] \leftrightarrow emg[k+\Delta]$).

**Independent Test**: Execute alignment on synthetic cases and real recording headers; verify `ACCEPT_AUTO` for proven start-anchored tail mismatches; verify `QUARANTINE` for head shifts and ambiguous cases; assert zero padding, zero interpolation, and zero silent fallback.

- [x] T005 [US2] Implement ProveOrQuarantineEngine with per-recording evidence, anchor_start_truncate_tail, and quarantine logic in semg_dataset/alignment.py
  <!-- harness-task {"requirements":["FR-005"],"acceptance_criteria":["SC-003"],"plan_decisions":["D-002"],"dependencies":["T003","T004"],"test_type":"UNIT","allowed_files":["semg_dataset/alignment.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

**Checkpoint**: User Story 2 complete; all recordings possess synchronized 12-channel sEMG and labels under an audit-proven timeline.

---

## Phase 5: User Story 3 — Frozen Dataset Views (P1)

**Goal**: Provide distinct, frozen views for `refined` (default: `restimulus` + `rerepetition`) and `stimulus` (secondary: `stimulus` + `repetition`), strictly prohibiting cross-pair contamination.

**Independent Test**: Request both views from `RecordingData`; verify that each exports paired labels and unique view hashes; assert that attempting to mix `stimulus` with `rerepetition` raises a `ValueError`.

- [x] T006 [US3] Implement FrozenViewManager and label pair isolation in semg_dataset/contract.py
  <!-- harness-task {"requirements":["FR-004"],"acceptance_criteria":["SC-002"],"plan_decisions":["D-001"],"dependencies":["T003","T005"],"test_type":"UNIT","allowed_files":["semg_dataset/contract.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

**Checkpoint**: User Story 3 complete; label duality is safely preserved with zero risk of cross-pair label noise.

---

## Phase 6: User Story 4 — Dual Partitioning Protocols & Leakage Controls (P2)

**Goal**: Implement both Within-Subject (repetition split: train `[1,3,4,6]`, test `[2,5]`, val deferred) and Cross-Subject (5 rotating folds: 24 train / 8 val / 8 test) with strict zero-leakage enforcement. Test design MUST include positive controls of leakage (deliberate subject overlap, repetition overlap, duplicate units) to prove the leakage validator catches violations.

**Independent Test**: Generate split partitions under both protocols; assert that partition intersections are strictly empty; assert that intentional leakage injections cause test failures; verify that Cross-Subject generates exactly 5 folds across all 40 subjects with frozen subject lists.

- [x] T007 [US4] Implement WithinSubjectSplitter and CrossSubjectSplitter with positive leakage controls in semg_dataset/splits.py
  <!-- harness-task {"requirements":["FR-006"],"acceptance_criteria":["SC-004","SC-005"],"plan_decisions":["D-003"],"dependencies":["T003","T004"],"test_type":"UNIT","allowed_files":["semg_dataset/splits.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

**Checkpoint**: User Story 4 complete; both evaluation protocols are implemented with mathematical leakage-free guarantees.

---

## Phase 7: User Story 5 — REST Auditing & Manifest Generation (P2)

**Goal**: Retain REST (label 0) in recording representations, audit active vs rest samples, associate REST intervals to repetition units, and generate deterministic JSON manifests.

**Independent Test**: Generate `dataset_catalog.json`, `splits_within_subject.json`, and `splits_cross_subject.json`; assert that REST duration and counts are segregated in summary; verify that re-running manifest generation yields identical SHA-256 hashes.

- [ ] T008 [US5] Implement DatasetManifestGenerator and SplitManifestGenerator in semg_dataset/manifest.py
  <!-- harness-task {"requirements":["FR-007","FR-008","FR-009"],"acceptance_criteria":["SC-004","SC-005"],"plan_decisions":["D-004","D-005"],"dependencies":["T005","T006","T007"],"test_type":"UNIT","allowed_files":["semg_dataset/manifest.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

**Checkpoint**: User Story 5 complete; manifests are serialized and validated against JSON schemas.

---

## Final Phase: Polish & Cross-Cutting Concerns

- [ ] T009 Final audit of dataset contracts, PROVE-OR-QUARANTINE reports, split leakage, and Constitution compliance
  <!-- harness-task {"requirements":["FR-011","FR-012"],"acceptance_criteria":["SC-001","SC-002","SC-003","SC-004","SC-005","SC-006"],"plan_decisions":["D-001","D-002","D-003","D-004","D-005"],"dependencies":["T008"],"test_type":"NOT_AUTOMATABLE","justification":"Human and static review of git diff, verification of zero raw file modifications, zero leakage between partitions, and passage of all deterministic quality gates.","alternative_verification":"Verify git status in data/raw/ is untouched; verify all unit tests pass with pytest -q; run ruff, mypy, compileall, and orchestrator verify.","allowed_files":[],"tdd_phases":[],"numeric_sensitive":false} -->

---

## Dependencies & Execution Order

```text
T001 (Setup)
  ├──► T002 (Provenance) ──► T004 (Loader)
  └──► T003 (Contract)   ──► T005 (Alignment Engine) ──► T006 (Frozen Views)
                                  │                            │
                                  ▼                            ▼
                             T007 (Splits) ────────────► T008 (Manifests)
                                                               │
                                                               ▼
                                                         T009 (Final Audit)
```

---

## Implementation Strategy

1. **MVP Ingestion (T001–T004)**: Establish package, verified provenance, and in-memory raw archive reader.
2. **Alignment & View Core (T005, T006)**: Deliver PROVE-OR-QUARANTINE alignment (handling S12 outlier) and frozen view pair isolation.
3. **Partitioning & Manifests (T007, T008)**: Deliver Within-Subject & Cross-Subject 5-fold split engines and JSON manifest generators.
4. **Final Gate & Audit (T009)**: Authoritative verification across all quality gates.
