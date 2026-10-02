# Feature Specification: 003-dataset-contract-and-splits

**Feature Branch**: `feat/003-dataset-contract-and-splits`  
**Created**: 2026-10-02  
**Status**: Draft  
**Input**: User description: "Criar a Feature 003 dataset-contract-and-splits para integrar formalmente o NinaPro DB2 como dataset sEMG oficial do projeto. A Feature 003 deve consumir somente em modo read-only os arquivos originais localizados em data/raw/ninapro_db2/, preservar e verificar sua proveniência e hashes, implementar um contrato canônico de dados para os 40 sujeitos e 120 recordings MATLAB observados no intake, e representar explicitamente subject, exercise, recording, canais sEMG, sampling rate, labels e repetições."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Immutable Raw Dataset Ingestion & Provenance Verification (Priority: P1)

As a researcher or DSP pipeline developer, I need an automated, immutable reader for the local NinaPro DB2 dataset that verifies cryptographic file integrity (SHA-256) against the intake baseline without ever modifying the raw storage, and resolves all official citation and licensing provenance.

**Why this priority**: Essential foundation. If raw data integrity or provenance cannot be guaranteed deterministically, downstream signal processing and machine learning pipelines cannot be trusted or audited.

**Independent Test**: Can be tested independently by loading archives directly from `data/raw/ninapro_db2/`, verifying archive hashes against `data/manifests/ninapro_db2/file_inventory.jsonl`, and asserting that raw archives are strictly read-only and unextracted on disk.

**Acceptance Scenarios**:
1. **Given** valid ZIP archives in `data/raw/ninapro_db2/`, **When** the dataset loader initializes, **Then** all archive SHA-256 hashes match the baseline inventory, no disk extraction occurs in `data/raw/`, and provenance metadata is fully populated.
2. **Given** an attempt to write to or modify any file in `data/raw/ninapro_db2/`, **When** any module executes, **Then** an explicit error is raised preventing mutation.

---

### User Story 2 - Canonical Recording Representation & Deterministic Alignment (Priority: P1)

As a pipeline engineer, I need each recording (`S<N>_E<M>_A1.mat`) to be represented by a canonical data structure providing synchronized 12-channel `float32` sEMG signals, sampling rate (2000 Hz nominal), exercise identifier, subject identifier, stimulus labels, and repetition counters, governed by an authoritative temporal alignment policy that resolves array length discrepancies.

**Why this priority**: Raw MATLAB files contain slight length discrepancies in Exercise 3 (18 files with 1–5 sample discrepancies, plus one 272-sample outlier in `S12_E3_A1.mat`). The dataset contract must provide deterministic alignment so that implementers cannot invent arbitrary trimming or padding heuristics.

**Independent Test**: Can be tested independently by loading `S12_E3_A1.mat` and verifying that the aligned output matches the exact canonical alignment rule without data corruption or silent sample truncation.

**Acceptance Scenarios**:
1. **Given** any of the 120 MATLAB recordings in NinaPro DB2, **When** parsed into a canonical recording structure, **Then** the sEMG array has shape `(N, 12)`, dtype `float32`, sampling rate `2000.0 Hz`, zero NaN/Inf, and synchronized labels.
2. **Given** a recording with temporal length mismatch between raw signals and refined labels (such as `S12_E3_A1.mat`), **When** loaded via the canonical reader, **Then** the alignment policy specified in CLARIFY is deterministically applied, preserving alignment audit logs.

---

### User Story 3 - Preserved Duality of Planned vs Refined Labels (Priority: P1)

As a gesture recognition researcher, I need access to both visual stimulus labels (`stimulus`, `repetition`) and activation-refined labels (`restimulus`, `rerepetition`) across all recordings, while having an explicit, unambiguous definition of which label serves as the canonical training target.

**Why this priority**: NinaPro DB2 provides both prescribed stimuli and offline-recalculated labels based on glove/movement onsets. Hardcoding either one without clear contract design leads to irreproducible machine learning experiments.

**Independent Test**: Can be tested independently by loading recordings across E1, E2, and E3, and verifying that both label representations are preserved, accessible, and documented.

**Acceptance Scenarios**:
1. **Given** any recording, **When** inspecting available labels, **Then** both `stimulus`/`repetition` and `restimulus`/`rerepetition` are present, correctly typed (`int8`), and tagged.
2. **Given** downstream consumer requests for the canonical ground truth, **When** queried, **Then** the canonical label designated by CLARIFY decision Q1 is returned.

---

### User Story 4 - Leakage-Free Reproducible Dataset Partitioning (Priority: P2)

As a machine learning engineer, I need to partition the 40 subjects / 120 recordings into Train, Validation, and Test sets based on a strictly enforced isolation unit (e.g., subject-independent cross-validation or repetition-split), ensuring zero temporal or subject leakage, complete balance auditing, and deterministic manifest serialization.

**Why this priority**: Data leakage between training and validation/test invalidates scientific evaluation. Manifests must be persistent and reproducible.

**Independent Test**: Can be tested independently by generating splits under a given protocol and asserting that intersection of entities between train, val, and test is strictly empty.

**Acceptance Scenarios**:
1. **Given** the 40 subjects and a designated split protocol, **When** the split generator executes, **Then** every sample belongs to exactly one partition, the partition intersection is empty, and a deterministic `splits.json` manifest is produced.
2. **Given** split manifests, **When** verified across repeated executions with the same configuration, **Then** the generated manifests are bitwise identical.

---

### Edge Cases

- **Outlier Temporal Discrepancy (`S12_E3_A1.mat`)**: In `S12_E3_A1.mat`, `emg`, `stimulus`, and `repetition` have 875,707 samples, while `restimulus` and `rerepetition` have 875,435 samples (272 samples difference, ~136 ms). The system must handle this via the authorized policy without crashing or silent failure.
- **Micro-Discrepancies (1–5 samples)**: 17 other E3 files have 1 to 5 sample differences in refined labels. Must be aligned identically according to the same deterministic rule.
- **Exercise Structural Asymmetry**: Exercises E1 and E2 contain `glove` and `inclin`, while E3 omits them in favor of `force`. The canonical sEMG contract must focus on the universal 12 sEMG channels without failing when auxiliary sensor arrays differ.
- **REST Class Imbalance**: Label 0 (rest between movements) accounts for a large fraction of the continuous stream. The split and manifest contract must audit and report sample counts for rest versus active gestures.
- **Missing or Corrupted Archive**: If an archive is missing or fails SHA-256 verification, the system must immediately abort with an explicit integrity failure.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST access raw NinaPro DB2 archives in `data/raw/ninapro_db2/` in strict READ-ONLY mode, without permanent disk extraction or modifying files.
- **FR-002**: System MUST verify the SHA-256 checksum of every archive against the recorded intake baseline before loading data.
- **FR-003**: System MUST provide a canonical in-memory recording contract (`RecordingData`) containing:
  - `subject_id`: str (e.g., "S1" to "S40")
  - `exercise_id`: str (e.g., "E1", "E2", "E3")
  - `emg`: `float32` array with shape `(N, 12)`, strictly devoid of NaN or Inf
  - `sampling_rate_hz`: float (nominal `2000.0`)
  - `stimulus`: `int8` array with shape `(N,)`
  - `repetition`: `int8` array with shape `(N,)`
  - `restimulus`: `int8` array with shape `(N,)`
  - `rerepetition`: `int8` array with shape `(N,)`
- **FR-004**: System MUST resolve the canonical ground truth label based on [NEEDS CLARIFICATION: Q1 — Canonical label definition: should the default canonical label be visual `stimulus` or refined `restimulus`?].
- **FR-005**: System MUST resolve temporal length discrepancies between `emg`/`stimulus` and `restimulus` according to [NEEDS CLARIFICATION: Q2 — Alignment policy for temporal mismatches: truncate to common length, fallback to stimulus, or dedicated handling for S12 outlier?].
- **FR-006**: System MUST generate reproducible, leak-free splits governed by [NEEDS CLARIFICATION: Q3 — Split protocol: subject-independent (e.g., 28 train / 6 val / 6 test), repetition-independent (within-subject: e.g. reps 1,3,4,6 train / 2 val / 5 test), or dual supported protocols?].
- **FR-007**: System MUST handle gesture class 0 (REST) according to [NEEDS CLARIFICATION: Q4 — REST handling: included as standard class 0 in splits/metrics, or segregated as transition baseline?].
- **FR-008**: System MUST support gesture vocabulary scope according to [NEEDS CLARIFICATION: Q5 — Movement scope: all 49 gestures across E1/E2/E3, or exercise-specific sub-vocabularies (e.g., E1=17 gestures, E2=23 gestures)?].
- **FR-009**: System MUST generate deterministic JSON manifest artifacts documenting dataset summary, recording index, and split partitions.
- **FR-010**: System MUST document complete NinaPro DB2 provenance, replacing all `TO_BE_DOCUMENTED` placeholders with official citations, URLs, and terms.
- **FR-011**: System MUST NOT include any model training, neural network definitions, hyperparameter tuning, quantization, C++, or ESP32 embedded deployment logic.
- **FR-012**: System MUST preserve Feature 002 DSP contracts without silent modifications.

### Key Entities

- **`Subject`**: Participant in the database (identified as `S1` through `S40`).
- **`Exercise`**: Recording protocol block (`E1` = 17 finger/wrist gestures; `E2` = 23 functional grasps; `E3` = 9 force patterns).
- **`RecordingData`**: Synchronized in-memory container representing one subject-exercise session with 12-channel sEMG, timing, and paired labels.
- **`DatasetManifest`**: Machine-readable catalog recording all 40 subjects, 120 recordings, sample counts, checksums, and provenance metadata.
- **`SplitManifest`**: Machine-readable specification assigning recordings or repetition windows to `train`, `val`, and `test` partitions with zero intersection.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of the 40 NinaPro DB2 subjects and 120 MATLAB recordings are ingestible in read-only mode with zero file mutations in `data/raw/ninapro_db2/`.
- **SC-002**: 100% of ingested sEMG signals strictly adhere to shape `(N, 12)`, `float32` dtype, nominal 2000 Hz, with 0 NaN and 0 Inf values.
- **SC-003**: 100% of the 18 recordings with temporal length discrepancies (including the 272-sample mismatch in S12) are aligned deterministically with zero unhandled exceptions.
- **SC-004**: Generated split partitions exhibit exactly 0.0% leakage (0 samples shared between train, validation, and test partitions).
- **SC-005**: All split and dataset manifests are 100% reproducible across independent executions from identical configurations (verified via SHA-256 equality of manifests).
- **SC-006**: Provenance fields `license` and `official_url` are fully documented, with zero remaining `TO_BE_DOCUMENTED` markers in final documentation.

---

## Assumptions

- NinaPro DB2 raw data remains stored in `data/raw/ninapro_db2/` as 40 ZIP files named `DB2_s1.zip` to `DB2_s40.zip`.
- Sampling rate for Delsys Trigno in NinaPro DB2 is nominally 2000.0 Hz, as established in the original publication (Atzori et al., 2014).
- Auxiliary sensor data (`acc`, `glove`, `inclin`, `force`) are parsed or preserved when relevant, but primary contract guarantees focus on the 12 sEMG channels and corresponding labels.
- The feature operates on host Python (3.10+) with NumPy and SciPy; no embedded runtime is targeted in this feature.
