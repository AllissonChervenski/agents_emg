# Feature Specification: 004-baseline-training-and-evaluation

**Feature Branch**: `feat/004-baseline-training-and-evaluation`  
**Created**: 2026-10-02  
**Status**: Draft (Ready for Clarify)  
**Input**: User description: "A Feature 004 deve estabelecer um baseline offline, reproduzível e auditável de classificação de gestos sEMG sobre o NinaPro DB2 usando os contratos congelados da Feature 003. Baseline inicial: classical sEMG time-domain features (MAV, RMS, WL, ZC, SSC) + regularized LDA. Vocabulário primário: 50 classes (REST 0 + gestos 1..49) e secundário: 49 classes ativas. Avaliação within-subject com leave-one-training-repetition-out em [1,3,4,6] e teste único em [2,5]. Avaliação cross-subject nos 5 folds congelados. Contrato formal de TrainingJob com rastreabilidade total de linhagem e hashes científicos. Pré-processamento e thresholds detalhados no CLARIFY."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Time-Domain Feature Extraction on Frozen Dataset Views (Priority: P1)

As a biosignal researcher or ML engineer, I need an automated, causal feature extraction pipeline that slides windows across continuous NinaPro DB2 recordings from the frozen `refined` dataset view (`restimulus` + `rerepetition`) to compute classical time-domain sEMG features (MAV, RMS, WL, ZC, SSC) per channel without temporal leakage or lookahead.

**Why this priority**: Essential foundation. Every downstream classifier depends on reliable, reproducible, and mathematically rigorous feature extraction from the 12-channel sEMG signals.

**Independent Test**: Can be tested independently by taking a synthetic or isolated recording, applying windowing with configurable window length $W$ and stride $S$, extracting the 5 time-domain features across all 12 channels, and verifying that the resulting feature matrix has shape `(num_windows, 60)`, contains zero NaN/Inf, and aligns strictly with the corresponding window ground-truth labels.

**Acceptance Scenarios**:
1. **Given** a continuous `RecordingData` object from Feature 003 in `refined` view, **When** feature extraction executes with window length $W$ and stride $S$, **Then** a 2D feature matrix `X` of shape `(num_windows, 12 * num_features)` and label vector `y` of shape `(num_windows,)` are generated with zero NaN or Inf.
2. **Given** window label assignment, **When** a window spans a transition between REST (0) and an active gesture (1..49), **Then** the window label is assigned deterministically according to the policy resolved in CLARIFY without guessing.
3. **Given** any feature scaling (e.g. z-score), **When** normalization statistics are calculated, **Then** statistics are fitted strictly on the training partition and applied statelessly to validation and test partitions with zero leakage.

---

### User Story 2 - Auditable TrainingJob Contract & Cryptographic Provenance (Priority: P1)

As an ML pipeline auditor or system orchestrator, I need every training and evaluation run to execute as an immutable, structured `TrainingJob` contract that captures full scientific provenance—including raw dataset manifest hash, dataset view hash, alignment record hash, split manifest hash, preprocessing configuration hash, model configuration hash, environment/dependency versions, and evaluation metrics.

**Why this priority**: Scientific and numerical oracle integrity (Constitution Principle VI). Without complete lineage tracking and verifiable hashes, baseline classification results cannot be audited, reproduced, or contrasted against future deep learning architectures.

**Independent Test**: Can be tested independently by launching a mock or unit `TrainingJob`, inspecting its JSON summary artifact, and validating that all lineage hashes, configuration signatures, environment versions, and execution status are recorded and verifiable.

**Acceptance Scenarios**:
1. **Given** an execution of a baseline pipeline run, **When** the job finishes (success or failure), **Then** a structured `TrainingJob` record is persisted to disk containing `job_id`, `git_commit`, all Feature 003 input hashes, configuration hashes, environment versions, execution timings, and deterministic output hashes.
2. **Given** an external provider reporting success, **When** postconditions or output hashes fail validation in Python, **Then** the `TrainingJob` status is recorded as failure, enforcing that provider success does not equate to job success.

---

### User Story 3 - Within-Subject Baseline Training with Leave-One-Repetition-Out Validation (Priority: P1)

As a researcher benchmarking individualized prosthetic control, I need to evaluate the regularized LDA baseline on each of the 40 subjects under the Within-Subject protocol, selecting model configuration using leave-one-training-repetition-out over repetitions `[1, 3, 4, 6]`, followed by a single final fit on all training repetitions `[1, 3, 4, 6]` and evaluation on the frozen test repetitions `[2, 5]`.

**Why this priority**: Represents the canonical within-user evaluation protocol of Atzori et al. (2014). It proves whether classical features + regularized LDA can distinguish the NinaPro DB2 gesture set per user under rigorous leakage-free conditions.

**Independent Test**: Can be tested independently by executing the Within-Subject pipeline on a single subject (e.g. `S1`), verifying that 4 internal validation folds rotate across repetitions `[1, 3, 4, 6]`, that final testing evaluates strictly on repetitions `[2, 5]`, and that no test window is ever exposed during training or hyperparameter selection.

**Acceptance Scenarios**:
1. **Given** a Within-Subject split for subject $S_i$, **When** model selection runs, **Then** exactly 4 validation rounds are evaluated (val 1 / train 3,4,6; val 3 / train 1,4,6; val 4 / train 1,3,6; val 6 / train 1,3,4) without observing test repetitions `[2, 5]`.
2. **Given** the selected baseline configuration, **When** final evaluation executes, **Then** the model is fitted on all windows from repetitions `[1, 3, 4, 6]` and evaluated once on repetitions `[2, 5]`.
3. **Given** evaluation completion, **When** results are computed, **Then** Top-1 Accuracy, Balanced Accuracy, Macro-F1, per-class precision/recall, and confusion matrix are generated.

---

### User Story 4 - Cross-Subject 5-Fold Benchmark Evaluation (Priority: P2)

As a machine learning researcher evaluating model generalization across unseen subjects, I need to run the baseline regularized LDA on the 5 frozen cross-subject folds from Feature 003 (24 train, 8 validation, 8 test subjects per fold) without regenerating or altering the fold partitions.

**Why this priority**: Validates inter-subject transferability of the classical baseline features, providing a sound benchmark for future cross-user domain adaptation models.

**Independent Test**: Can be tested independently by executing the cross-subject training runner on Fold 1, confirming that exactly subjects $G_1 \cup G_2 \cup G_3$ (24) are used for training, $G_4$ (8) for validation, and $G_5$ (8) for testing, with zero cross-subject contamination.

**Acceptance Scenarios**:
1. **Given** the Cross-Subject split manifest from Feature 003, **When** training executes for fold $k \in \{1..5\}$, **Then** the pipeline consumes exactly the frozen subject lists for that fold.
2. **Given** completion across all 5 folds, **When** aggregating metrics, **Then** per-fold metrics and aggregated summary statistics (mean, median, standard deviation) are produced.

---

### User Story 5 - Dual-Vocabulary Reporting: 50-Class Primary vs 49-Class Secondary (Priority: P2)

As a scientific reviewer or clinician, I need evaluation metrics reported primarily on the full 50-class vocabulary (REST class 0 + gestures 1..49) and secondarily on the active-only 49-class vocabulary (gestures 1..49), presented with strict segregation to avoid cross-study confusion.

**Why this priority**: NinaPro DB2 continuous recordings are dominated by REST intervals (~75% of duration). Reporting both 50-class and active-only 49-class metrics allows transparent comparison with both continuous prosthetic controller benchmarks and segmented gesture classification literature.

**Independent Test**: Can be tested independently by inspecting evaluation report summaries to confirm that 50-class and 49-class metrics are computed with distinct denomination, separate confusion matrices, and unmixed class counts.

**Acceptance Scenarios**:
1. **Given** model predictions on a test set, **When** generating primary evaluation reports, **Then** metrics are computed over all 50 classes (0..49), with REST explicitly tracked.
2. **Given** the same predictions, **When** generating secondary evaluation reports, **Then** windows labeled REST (0) are excluded or masked according to the active-only protocol, and metrics are computed strictly over classes 1..49.

---

### Edge Cases

- **Constant / Silent Channels**: What happens if an sEMG channel exhibits near-zero variance (e.g. poor electrode contact)? The feature extraction must guard against division-by-zero in SSC/ZC thresholds or feature scaling using a numerical epsilon ($\epsilon = 10^{-8}$).
- **Short or Truncated Repetitions**: What happens if a specific repetition contains fewer samples than window length $W$? The window buffer finalize policy (`drop`) excludes incomplete windows and logs the dropped duration.
- **Extreme Class Imbalance (REST vs Rare Gestures)**: REST (class 0) comprises the majority of sample time. Regularized LDA priors must be explicitly configured (uniform vs empirical) and evaluated via Balanced Accuracy and Macro-F1 to prevent majority-class collapse.
- **Singular Covariance in LDA**: When extracting multi-channel correlated features or when a feature has low variance, the pooled within-class covariance matrix may be ill-conditioned. Regularization (shrinkage / Ledoit-Wolf or empirical ridge parameter $\lambda$) must be applied to guarantee numerical stability.
- **Provider / Subprocess Failure**: If an agent or worker process fails, times out, or produces malformed JSON, the orchestrator catches the failure, records the partial log in `TrainingJob.stderr_summary`, and marks `execution_status = "FAILED"`.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST strictly consume frozen dataset artifacts and manifests from Feature 003 (`data/manifests/ninapro_db2/`, `semg_dataset`), never reading raw ZIPs directly in training routines or reinterpreting split boundaries.
- **FR-002**: System MUST default to the `refined` dataset view (`restimulus` + `rerepetition`) for ground truth labels, with optional support for the `stimulus` view without cross-pair contamination.
- **FR-003**: System MUST record in every `TrainingJob` artifact the following cryptographic provenance hashes:
  - `dataset_manifest_hash` (raw dataset baseline hash)
  - `dataset_view_hash` (hash of the active view)
  - `alignment_hash` (or alignment verification version)
  - `split_manifest_hash` (within-subject or cross-subject split manifest hash)
  - `protocol_id` (`"within_subject"` or `"cross_subject"`)
  - `fold_id` (identifier of the fold or subject evaluated)
  - `acceptance_package_hash` (Feature 003 acceptance package hash)
- **FR-004**: System MUST extract 5 classical sEMG time-domain features per channel:
  - Mean Absolute Value (MAV)
  - Root Mean Square (RMS)
  - Waveform Length (WL)
  - Zero Crossing count (ZC)
  - Slope Sign Change count (SSC)
  Yielding exactly $12 \times 5 = 60$ features per analysis window.
- **FR-005**: System MUST implement regularized Linear Discriminant Analysis (LDA) as the baseline classifier, supporting covariance shrinkage to guarantee invertible covariance matrices.
- **FR-006**: System MUST adopt `scikit-learn` as a host-only baseline/evaluation dependency [NEEDS CLARIFICATION: governance mechanism and approval protocol for introducing scikit-learn as a dev/host dependency in pyproject.toml].
- **FR-007**: System MUST implement a primary 50-class vocabulary (`REST = 0`, gestures `1..49`) and an isolated secondary 49-class active-only vocabulary (`gestures 1..49`), never merging or confusing the two in reported metrics.
- **FR-008**: System MUST implement the Within-Subject validation protocol using leave-one-training-repetition-out over repetitions `[1, 3, 4, 6]`:
  - Fold 1: Val = 1, Train = `[3, 4, 6]`
  - Fold 2: Val = 3, Train = `[1, 4, 6]`
  - Fold 3: Val = 4, Train = `[1, 3, 6]`
  - Fold 4: Val = 6, Train = `[1, 3, 4]`
  Final model MUST be fitted on all training repetitions `[1, 3, 4, 6]` and evaluated once on test repetitions `[2, 5]`. Random window shuffling across repetitions is strictly prohibited.
- **FR-009**: System MUST execute Cross-Subject evaluation strictly over the 5 frozen folds generated in Feature 003 (24 train, 8 val, 8 test subjects per fold), reporting per-fold and aggregated cross-subject metrics.
- **FR-010**: System MUST report the following evaluation metrics for all benchmark evaluations:
  - Top-1 Accuracy (primary comparability metric)
  - Balanced Accuracy
  - Macro-F1
  - Per-class Precision and Recall
  - Multi-class Confusion Matrix
  - Per-subject breakdown and statistical aggregates (mean, median, standard deviation).
  System MUST NOT claim that Balanced Accuracy or Macro-F1 were part of Atzori's original benchmark without specific bibliographic citation.
- **FR-011**: System MUST define and persist a formal `TrainingJob` contract containing job metadata, configuration hashes, environment versions, timing, execution status, outputs, metrics, and artifact hashes.
- **FR-012**: System MUST resolve in CLARIFY all DSP preprocessing parameters, window size $W$, stride $S$, ZC/SSC deadzone thresholds, boundary window handling, and feature scaling policies prior to PLAN [NEEDS CLARIFICATION: window length W, stride S, and bandpass/notch filtering parameters].
- **FR-013**: Normalization and feature scaling statistics (e.g. mean, std, min, max) MUST be computed strictly on the training partition and applied statelessly to validation and test partitions.
- **FR-014**: System MUST NOT include INT8 quantization, embedded runtime deployment, C++, ESP32, FreeRTOS, Software-in-the-Loop (SIL), or broad deep learning architecture search in this feature.
- **FR-015**: System MUST NOT modify raw NinaPro DB2 data files or alter Feature 003 artifacts.

---

### Key Entities

- **`TrainingJob`**: Canonical execution record capturing all lineage hashes, configuration signatures, environment versions, random seeds, execution status, metrics, and artifact references.
- **`FeatureConfig`**: Immutable configuration specifying window length $W$, stride $S$, feature set (MAV, RMS, WL, ZC, SSC), and deadzone thresholds.
- **`ModelConfig`**: Configuration specifying classifier hyperparameters (regularized LDA shrinkage parameter, solver, priors).
- **`PreprocessingConfig`**: Specification of bandpass and notch filter stages applied via `semg_dsp` prior to windowing.
- **`EvaluationReport`**: Structured container holding per-fold, per-subject, and aggregated benchmark metrics across primary (50-class) and secondary (49-class) vocabularies.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of training and evaluation runs consume frozen Feature 003 manifests and views, logging verifiable SHA-256 hashes with zero runtime reinterpretation of raw archives.
- **SC-002**: Feature extraction across 12 channels produces a deterministic 60-dimensional feature matrix per window with exactly 0.0% NaN or Inf values.
- **SC-003**: 100% of Within-Subject evaluations strictly follow leave-one-repetition-out over `[1, 3, 4, 6]` for internal validation and evaluate exactly once on `[2, 5]` with zero temporal leakage.
- **SC-004**: 100% of Cross-Subject evaluations execute on the 5 frozen folds (24/8/8) with zero subject intersection between partitions.
- **SC-005**: All evaluation runs generate structured metrics (Top-1 Accuracy, Balanced Accuracy, Macro-F1, Confusion Matrix) reported separately for 50-class and 49-class vocabularies.
- **SC-006**: Generated `TrainingJob` artifacts are bitwise reproducible when executed with identical seeds and configurations on identical hardware environments.

---

## Assumptions

- NinaPro DB2 data is already validated and accessible via `semg_dataset` without modifying `data/raw/ninapro_db2/`.
- Sampling rate of sEMG is 2000.0 Hz nominal.
- `scikit-learn` is approved through the established governance mechanism as a host-only dependency for baseline model fitting and metrics computation, and will never be imported or linked in embedded/DSP runtimes.
- Regularized LDA serves as the baseline model; no neural networks or hyperparameter optimization sweeps are in scope for Feature 004.
- All evaluation occurs on host Python (3.10+); no embedded or target hardware deployment is performed.
