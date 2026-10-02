# Feature Specification: 003-dataset-contract-and-splits

**Feature Branch**: `feat/003-dataset-contract-and-splits`  
**Created**: 2026-10-02  
**Status**: Clarified  
**Input**: User description: "Criar a Feature 003 dataset-contract-and-splits para integrar formalmente o NinaPro DB2 como dataset sEMG oficial do projeto. A Feature 003 deve consumir somente em modo read-only os arquivos originais localizados em data/raw/ninapro_db2/, preservar e verificar sua proveniência e hashes, implementar um contrato canônico de dados para os 40 sujeitos e 120 recordings MATLAB observados no intake, e representar explicitamente subject, exercise, recording, canais sEMG, sampling rate, labels e repetições."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Immutable Raw Dataset Ingestion & Provenance Verification (Priority: P1)

As a researcher or DSP pipeline developer, I need an automated, immutable reader for the local NinaPro DB2 dataset that verifies cryptographic file integrity (SHA-256) against the intake baseline without ever modifying the raw storage, and resolves all official citation and licensing provenance.

**Why this priority**: Essential foundation. If raw data integrity or provenance cannot be guaranteed deterministically, downstream signal processing and machine learning pipelines cannot be trusted or audited.

**Independent Test**: Can be tested independently by loading archives directly from `data/raw/ninapro_db2/`, verifying archive hashes against `data/manifests/ninapro_db2/file_inventory.jsonl`, and asserting that raw archives are strictly read-only and unextracted on disk.

**Acceptance Scenarios**:
1. **Given** valid ZIP archives in `data/raw/ninapro_db2/`, **When** the dataset loader initializes, **Then** all archive SHA-256 hashes match the baseline inventory, no disk extraction occurs in `data/raw/`, and provenance metadata is fully populated with official citations and URLs.
2. **Given** an attempt to write to or modify any file in `data/raw/ninapro_db2/`, **When** any module executes, **Then** an explicit error is raised preventing mutation.

---

### User Story 2 - Canonical Recording Representation & PROVE-OR-QUARANTINE Alignment (Priority: P1)

As a pipeline engineer, I need each recording (`S<N>_E<M>_A1.mat`) to be represented by a canonical data structure providing synchronized 12-channel `float32` sEMG signals, sampling rate (2000 Hz nominal), exercise identifier, subject identifier, stimulus labels, and repetition counters, governed by an authoritative PROVE-OR-QUARANTINE temporal alignment policy that resolves array length discrepancies.

**Why this priority**: Raw MATLAB files contain length discrepancies in Exercise 3 (18 files, including a 272-sample outlier in `S12_E3_A1.mat`). The alignment engine must prove that the signals share a common start anchor and that the discrepancy is strictly tail truncation before applying `anchor_start_truncate_tail`; otherwise, the recording must be placed in `QUARANTINE`. Silent `min()`, padding, interpolation, and automatic fallback are strictly prohibited.

**Independent Test**: Can be tested independently by evaluating all 120 recordings with the alignment verifier, ensuring that files with verified start-anchoring and tail-only truncation are aligned via `anchor_start_truncate_tail`, and any unverified recording is quarantined with an explicit audit finding.

**Acceptance Scenarios**:
1. **Given** any of the 120 MATLAB recordings in NinaPro DB2, **When** parsed into a canonical recording structure, **Then** the sEMG array has shape `(N, 12)`, dtype `float32`, sampling rate `2000.0 Hz`, zero NaN/Inf, and synchronized labels.
2. **Given** a recording with temporal length mismatch between raw signals and refined labels (such as `S12_E3_A1.mat`), **When** the alignment verifier proves identical start anchoring and tail-only discrepancy, **Then** `anchor_start_truncate_tail` is applied, and the operation is recorded in the alignment audit trail.
3. **Given** a recording where start anchoring cannot be proven or ambiguity exists, **When** the alignment engine executes, **Then** the recording is marked as `QUARANTINE` and excluded from valid datasets.

---

### User Story 3 - Frozen Dataset Views (Priority: P1)

As a machine learning engineer, I need separate, frozen dataset views for ground-truth labels that prevent cross-pair contamination between planned stimulus and activation-refined labels, each with its own deterministic ID and versioning.

**Why this priority**: NinaPro DB2 provides both planned visual stimulus (`stimulus`/`repetition`) and activation-refined labels (`restimulus`/`rerepetition`). Mixing these pairs (e.g. `stimulus` with `rerepetition`) creates catastrophic label noise.

**Independent Test**: Can be tested independently by querying both views and verifying that each view exposes strictly matched label-repetition pairs and unique view hashes.

**Acceptance Scenarios**:
1. **Given** a recording request with `view="refined"` (default), **When** accessed, **Then** labels are strictly `(restimulus, rerepetition)`.
2. **Given** a recording request with `view="stimulus"` (secondary), **When** accessed, **Then** labels are strictly `(stimulus, repetition)`.
3. **Given** any attempt to pair `stimulus` with `rerepetition` or `restimulus` with `repetition`, **When** validated by the contract, **Then** a validation error is immediately raised.

---

### User Story 4 - Dual Leakage-Free Partitioning Protocols (Priority: P2)

As a researcher, I need reproducible, leakage-free dataset partitions supporting two distinct evaluation protocols: Within-Subject (evaluating user-specific gesture models) and Cross-Subject (evaluating cross-user generalization via 5 rotating folds).

**Why this priority**: Rigorous clinical and machine learning benchmarking requires both intra-subject and inter-subject evaluation paradigms with zero data leakage.

**Independent Test**: Can be tested independently by generating split manifests under both protocols and asserting zero entity intersection between partitions.

**Acceptance Scenarios**:
1. **Given** the Within-Subject protocol, **When** splits are generated, **Then** repetitions `[1, 3, 4, 6]` are assigned to `train`, repetitions `[2, 5]` are assigned to `test`, and `validation` is empty (to be derived strictly from training data in Feature 004).
2. **Given** the Cross-Subject protocol, **When** splits are generated, **Then** exactly 5 rotating folds are produced with 24 train subjects, 8 validation subjects, and 8 test subjects per fold, following explicit, frozen subject lists with zero subject overlap between partitions within each fold.
3. **Given** repeated split generation executions, **When** comparing generated manifests, **Then** manifests are bitwise identical (SHA-256 invariant).

---

### User Story 5 - REST Auditing & Native Global Label Hierarchy (Priority: P2)

As an ML pipeline designer, I need REST (label 0) to be preserved in the canonical contract, audited separately for balance, and deterministically assigned to repetition units, while preserving native global gesture labels (1..49) without artificial offsets.

**Why this priority**: NinaPro DB2 labels are already globally unique across exercises (E1=1..17, E2=18..40, E3=41..49). Modifying them with arbitrary offsets introduces bugs, while dropping REST prematurely would distort streaming temporal continuity.

**Independent Test**: Can be tested independently by verifying label distributions across E1, E2, E3 and inspecting the derived local label view.

**Acceptance Scenarios**:
1. **Given** recordings from E1, E2, and E3, **When** inspecting global labels, **Then** E1 contains 1..17, E2 contains 18..40, and E3 contains 41..49, with REST=0 present in all three.
2. **Given** a request for exercise-local labels, **When** accessed, **Then** a derived view provides local indices (E1: 1..17, E2: 1..23, E3: 1..9) without altering raw data.
3. **Given** dataset manifests, **When** generated, **Then** total samples and duration for active gestures versus REST are audited and reported independently.

---

### Edge Cases

- **Outlier Temporal Discrepancy (`S12_E3_A1.mat`)**: In `S12_E3_A1.mat`, `emg`, `stimulus`, and `repetition` have 875,707 samples, while `restimulus` and `rerepetition` have 875,435 samples (272 samples difference, ~136 ms). The system verifies that start indices are synchronized, then applies `anchor_start_truncate_tail` to truncate the trailing 272 samples, logging the operation.
- **Unverified Temporal Mismatch**: If any recording exhibits start-anchoring offset or indeterminate temporal shift, the system marks it as `QUARANTINE` instead of guessing or applying silent truncation.
- **REST Inter-Repetition Transitions**: Segments with label 0 between repetitions are deterministically associated with adjacent repetition blocks to prevent orphan fragments during windowing.
- **Auxiliary Sensor Asymmetry**: E1 and E2 contain `glove` and `inclin`, while E3 contains `force`. The canonical contract handles auxiliary arrays as optional metadata while maintaining strict uniformity across the 12 sEMG channels.
- **Tampering or Mutation Attempt**: Any attempt to write to `data/raw/ninapro_db2/` raises a read-only filesystem error.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST access raw NinaPro DB2 archives in `data/raw/ninapro_db2/` in strict READ-ONLY mode, without permanent disk extraction or modifying files.
- **FR-002**: System MUST verify the SHA-256 checksum of every archive against `data/manifests/ninapro_db2/file_inventory.jsonl` prior to loading data.
- **FR-003**: System MUST provide a canonical in-memory recording contract (`RecordingData`) containing:
  - `subject_id`: str (`"S1"` to `"S40"`)
  - `exercise_id`: str (`"E1"`, `"E2"`, `"E3"`)
  - `emg`: `float32` array with shape `(N, 12)`, strictly devoid of NaN or Inf
  - `sampling_rate_hz`: float (`2000.0`)
  - `stimulus`: `int8` array with shape `(N,)`
  - `repetition`: `int8` array with shape `(N,)`
  - `restimulus`: `int8` array with shape `(N,)`
  - `rerepetition`: `int8` array with shape `(N,)`
  - `provenance`: Dict[str, Any] with recording-level metadata
- **FR-004**: System MUST support frozen dataset views:
  - Default view: `restimulus` + `rerepetition` (refined).
  - Secondary view: `stimulus` + `repetition` (prescribed).
  - System MUST strictly prohibit and reject mixing pairs (`stimulus` with `rerepetition` or `restimulus` with `repetition`).
  - Each view MUST have an explicit identifier, version, and hash.
- **FR-005**: System MUST implement the **PROVE-OR-QUARANTINE** temporal alignment policy:
  - System MUST NOT assume in advance that all 18 E3 recordings pass; each recording MUST be evaluated independently, producing audit evidence that classifies it strictly as `ACCEPT_AUTO` or `QUARANTINE`.
  - System MUST establish a single canonical timeline per recording and verify start-anchoring ($t=0$).
  - If start anchoring is demonstrably proven and length mismatch is confined to trailing samples: system MUST apply `anchor_start_truncate_tail` and record the sample delta in the alignment audit trail.
  - If start anchoring is unproven, ambiguous, head-shifted ($label[k] \leftrightarrow emg[k+\Delta]$), or non-tail: system MUST mark the recording as `QUARANTINE`.
  - System MUST strictly prohibit padding, interpolation, silent `min()`, or silent fallback to stimulus.
- **FR-006**: System MUST implement a **DUAL SPLIT PROTOCOL**:
  - **Protocol 1: Within-Subject (Repetition-Split)**:
    - `train`: repetitions `[1, 3, 4, 6]` across all subjects.
    - `test`: repetitions `[2, 5]` across all subjects.
    - `val`: empty partition in Feature 003 (validation derivation from train is deferred to Feature 004).
  - **Protocol 2: Cross-Subject (Inter-Subject Folds)**:
    - 5 rotating folds across 40 subjects.
    - Exactly 24 train subjects, 8 validation subjects, and 8 test subjects per fold.
    - Explicit, deterministic, and frozen subject assignment lists.
  - Test suites MUST include positive controls of leakage (intentionally inducing overlap of subjects, repetition units, or split boundaries) demonstrating that the split validator unequivocally detects and rejects leakage.
- **FR-007**: System MUST preserve gesture class 0 (REST) in the canonical dataset contract, audit REST duration and sample counts separately from active movements, and define a deterministic association of REST intervals to repetition units.
- **FR-008**: System MUST preserve native global gesture labels (E1 = 1..17, E2 = 18..40, E3 = 41..49) without applying artificial offsets to raw data, and provide an optional derived view for exercise-local indexing (E1: 1..17, E2: 1..23, E3: 1..9).
- **FR-009**: System MUST generate deterministic JSON manifest artifacts documenting dataset inventory, recording catalog, and split partitions (`splits_within_subject.json` and `splits_cross_subject.json`).
- **FR-010**: System MUST document complete NinaPro DB2 provenance, licensing, and fixture constraints:
  - Origin: NinaPro Database 2 (DB2) — Ninapro project
  - Citation: Atzori, M. et al. (2014). "Electromyography data for non-invasive classification of hand, wrist and finger movements." Scientific Data, 1, 140053. DOI: 10.1038/sdata.2014.53
  - Official URL: https://ninapro.hevs.ch/instructions/DB2.html
  - Terms & License: Open access for non-commercial scientific research with attribution; redistribution of raw signals is subject to dataset terms.
  - Fixture Policy: System MUST NOT commit any fixture containing real NinaPro DB2 raw samples to the repository. All unit test fixtures MUST use minimal synthetic NinaPro-like data generated specifically for tests. Real DB2 data is evaluated strictly in separate integration passes against local uncommitted files.
- **FR-011**: System MUST NOT include any model training, neural network definitions, hyperparameter tuning, quantization, C++, or ESP32 embedded deployment logic.
- **FR-012**: System MUST preserve Feature 002 DSP contracts without silent modifications.

### Key Entities

- **`Subject`**: Participant in the database (identified as `S1` through `S40`).
- **`Exercise`**: Recording protocol block (`E1` = 17 finger/wrist gestures; `E2` = 23 functional grasps; `E3` = 9 force patterns).
- **`RecordingData`**: Synchronized in-memory container representing one subject-exercise session with 12-channel sEMG, timing, and paired labels.
- **`DatasetView`**: Formal view encapsulating a specific label pair (`refined` or `stimulus`) with dedicated hashing and immutability.
- **`DatasetManifest`**: Machine-readable catalog recording all 40 subjects, 120 recordings, sample counts, checksums, and provenance metadata.
- **`SplitManifest`**: Machine-readable specification assigning recordings or repetition windows to `train`, `val`, and `test` partitions with zero intersection.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of the 40 NinaPro DB2 subjects and 120 MATLAB recordings are ingestible in read-only mode with zero file mutations in `data/raw/ninapro_db2/`.
- **SC-002**: 100% of ingested sEMG signals strictly adhere to shape `(N, 12)`, `float32` dtype, nominal 2000 Hz, with 0 NaN and 0 Inf values.
- **SC-003**: 100% of the 18 recordings with temporal length discrepancies (including the 272-sample mismatch in S12) are evaluated under PROVE-OR-QUARANTINE, applying `anchor_start_truncate_tail` only upon verified start anchoring.
- **SC-004**: Generated split partitions exhibit exactly 0.0% leakage (0 samples shared between train, validation, and test partitions).
- **SC-005**: All split and dataset manifests are 100% reproducible across independent executions from identical configurations (verified via SHA-256 equality of manifests).
- **SC-006**: Provenance fields are fully documented, with zero remaining `TO_BE_DOCUMENTED` markers in final documentation.

---

## Assumptions

- NinaPro DB2 raw data remains stored in `data/raw/ninapro_db2/` as 40 ZIP files named `DB2_s1.zip` to `DB2_s40.zip`.
- Sampling rate for Delsys Trigno in NinaPro DB2 is nominally 2000.0 Hz, as established in the original publication (Atzori et al., 2014).
- Auxiliary sensor data (`acc`, `glove`, `inclin`, `force`) are parsed or preserved when relevant, but primary contract guarantees focus on the 12 sEMG channels and corresponding labels.
- The feature operates on host Python (3.10+) with NumPy and SciPy; no embedded runtime is targeted in this feature.
- Downstream model architecture and classification (49 vs 50 classes) is governed exclusively by Feature 004.
