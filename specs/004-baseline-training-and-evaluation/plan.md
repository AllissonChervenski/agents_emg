# Implementation Plan: 004-baseline-training-and-evaluation

**Branch**: `feat/004-baseline-training-and-evaluation` | **Date**: 2026-10-06 | **Spec**: [specs/004-baseline-training-and-evaluation/spec.md](spec.md)

**Input**: Feature specification from `/specs/004-baseline-training-and-evaluation/spec.md`, Research Decisions from `research.md`, Data Model from `data-model.md`, and Contracts from `contracts/`.

---

## Summary

Feature 004 establishes an offline, reproducible, and audit-grade baseline for sEMG gesture recognition over the NinaPro DB2 dataset, strictly consuming the frozen dataset contracts, alignments, and splits from Feature 003.

The pipeline comprises:
1. **Causal Preprocessing**: 4th-order causal Butterworth band-pass (20–450 Hz) via [`CausalSosFilter`](file:///home/allisson/Projects/agents_emg/semg_dsp/filter.py). (50 Hz notch omitted based on empirical spectral audit: na amostra auditada de 9 sujeitos cobrindo os 5 folds, não foi observado excesso espectral sistemático em 50 Hz, $R_{50} = 0.84$).
2. **Steady-State Pure Windowing**: Windows of length $W = 200\,\text{ms}$ (400 samples @ 2000 Hz) with stride $S = 50\,\text{ms}$ (100 samples, 75% overlap, 20 Hz decision rate). Exclusion of all transition/mixed windows (REST ↔ ACTIVE, ACTIVE ↔ ACTIVE, and repetition unit crossings).
3. **Time-Domain Feature Extraction**: 5 classical features per channel (MAV, RMS, WL, ZC, SSC) across 12 channels ($60\text{ features/window}$).
4. **Data-Driven Deadzone Thresholds**: $\epsilon[\text{ch}] = R \times \text{RMS\_rest\_train}[\text{ch}]$ with candidate grid $R \in \{0.5, 1.0, 1.5, 2.0, 3.0\}$ selected via internal validation on TRAIN.
5. **Train-Only Custom Z-Score Scaling**: Stateless transformation of validation and test splits with zero-variance protection; `sklearn.preprocessing.StandardScaler` is prohibited.
6. **Regularized LDA Modeling**:
   - Primary baseline (`uniform_prior_baseline`): $P(c) = 1/50$ for all classes $0..49$.
   - Secondary reference (`empirical_prior_reference`): $P(c) = N_c / N_{\text{train}}$ fitted as a distinct model.
   - Analytic Ledoit-Wolf shrinkage (`solver="lsqr", shrinkage="auto"`, improving covariance conditioning and numerical stability).
7. **Model Taxonomy**:
   - Primary: `global_50_class` (unified vocabulary 0..49).
   - Secondary: `exercise_E1` (18 classes), `exercise_E2` (24 classes), `exercise_E3` (10 classes) using derived local labels from Feature 003.
   - Diagnostic: `exercise_id_from_rest` confounding audit.
8. **Neutral Export Boundary & Three-Tier Oracles**:
   - Model parameters exported exclusively as neutral arrays (`classes`, `coef`, `intercept`, `mean`, `scale` in `model_params.npz`) and JSON manifests (including `zero_variance_indices` metadata); zero pickle or sklearn objects. Integrity sealed by sidecar `model_manifest.sha256`.
   - Feature 005 runtime inference executes $\text{scores} = X_{\text{scaled}} @ \text{coef}^T + \text{intercept}$ and $\text{argmax}$ without importing `scikit-learn`.
   - Independent equivalence oracle test: $\text{sklearn.predict}(X) == \text{neutral\_numpy\_predict}(X)$ with decision score tolerances governed by `TOL-LDA-NEUTRAL-SCORES`.
9. **Append-Only TrainingJobs & Formal Contracts**:
   - Persistent job records in `artifacts/feature_004/jobs/<job_id>/` with deterministic `job_fingerprint` (including `dependency_lock_hash`).
   - Authoritative JSON Schemas in `specs/004-baseline-training-and-evaluation/contracts/` featuring state-aware lifecycle validation (`PLANNED`, `RUNNING`, `SUCCEEDED`, `FAILED`).
   - Procedural synthetic fixtures in `tests/fixtures/synthetic_semg.py`, isolating unit test suites from local 17 GiB raw DB2 data.
10. **Shuffled-Label Negative Control**:
   - Verification suite `tests/test_baseline_shuffled_labels.py` asserting chance-level performance under deterministically permuted training targets against a pre-registered statistical threshold.

---

## Technical Context

**Language/Version**: Python 3.10+ (tested on Python 3.14.7)  
**Primary Dependencies**: `numpy` (>=1.26), `scipy` (>=1.12), `scikit-learn` (>=1.5, host-only baseline/evaluation dependency governed by [`docs/adr/ADR-004-001-scikit-learn-host-only.md`](file:///home/allisson/Projects/agents_emg/docs/adr/ADR-004-001-scikit-learn-host-only.md))  
**Governance State**: `CONDITIONAL_PASS` (ADR-004-001 is in status `PROPOSED`; code mutating `pyproject.toml` or executing `sklearn` code is strictly **`BLOCKED`** until human sign-off is committed).  
**Storage**: Strictly read-only consumption of Feature 003 manifests (`data/manifests/ninapro_db2/`) and raw ZIP archives (`data/raw/ninapro_db2/`). TrainingJob artifacts persisted in `artifacts/feature_004/jobs/<job_id>/`; consolidated reports in `reports/feature_004_baseline/`.  
**Testing**: `pytest` (unit, adversarial, oracle verification, and shuffled-label control using procedural synthetic fixtures), `ruff` (linter), `mypy` (strict static typing), `compileall` (syntax verification).  
**Target Platform**: Linux host-only runtime  
**Project Type**: Machine learning baseline and evaluation library (`semg_baseline/` package)  
**Performance Targets**:
- Unit test suite: `pytest -q tests/` completes in $< 5\text{ seconds}$ as an operational developer reference target.
- Feature extraction: $> 500,000\text{ samples/sec}$ across 12 channels [PROVISIONAL / NON-GATING].
- Single-subject Within-Subject baseline fit: $< 3\text{ seconds}$ [PROVISIONAL / NON-GATING].
- Cross-Subject fold fit (24 subjects): $< 30\text{ seconds}$ per fold [PROVISIONAL / NON-GATING].  
**Constraints**:
- Zero modification or re-evaluation of Feature 003 contracts, views, alignments, or split manifests.
- Zero modification to `data/raw/ninapro_db2/`.
- Zero Python pickle, joblib model dumps, or sklearn object serialization in exported models.
- Scikit-learn prohibited in `semg_dsp/` or embedded runtimes.
- Zero data leakage across validation or test splits (leakage invariant on rest RMS, ZC/SSC thresholds, scaling, and priors).
- Implementation and `pyproject.toml` edit blocked until ADR-004-001 transitions to `APPROVED` with human sign-off.  
**Scale/Scope**: 40 subjects, 120 recordings, 50 gesture classes, 5 cross-subject folds, Within-Subject LOTO protocol, estimated steady-state windows subject to pure-window exclusions [PROVISIONAL].

---

## Constitution Check

*GATE: Evaluated against Constitution v1.1.0 (commit `ce75be0`).*

- **Principle I (Python Central Orchestrator & Deterministic Gatekeeper)**: **PASS**. All transitions, training jobs, postconditions, and deterministic verification gates are owned and enforced by Python. LLM/provider success does not equate to job success.
- **Principle II (Strict TDD Lifecycle & Anti-Tampering Protection)**: **PASS**. All implementation tasks run strict RED → GREEN → REFACTOR with SHA-256 test snapshotting. Test designer creates tests with `EXPECTED_FAILURE`; implementer edits only `allowed_files`.
- **Principle III (Non-Destructive Local Reuse & Zero-Subprocess Read Paths)**: **PASS**. Feature 003 contracts, loader, and manifests are reused locally in-memory without subprocess calls or disk extraction.
- **Principle IV (CLI Ergonomics, Text/JSON Duality & Failure Containment)**: **PASS**. Job summaries and reports provide both structured machine-readable JSON and human-readable Markdown.
- **Principle V (SpecKit Traceability & Living Documentation Supremacy)**: **PASS**. Traceability maintained from `spec.md` (FR-001..FR-015), through `research.md`, `data-model.md`, `contracts/`, to `plan.md` and subsequent tasks.
- **Principle VI (Scientific and Numerical Oracle Integrity)**: **PASS**. Three-tier independent oracles (analytical feature oracles, sklearn training reference, neutral NumPy inference). Zero self-generation of reference fixtures; unit tests use deterministic procedural synthesis with known analytical properties.
- **Technical Constraints & Dependency Governance**: **CONDITIONAL_PASS**. Line 51 requirement (*"Features MUST NOT introduce new third-party dependencies without prior governance approval."*) is addressed via persistent Architecture Decision Record [`docs/adr/ADR-004-001-scikit-learn-host-only.md`](file:///home/allisson/Projects/agents_emg/docs/adr/ADR-004-001-scikit-learn-host-only.md), which is currently `PROPOSED`. Implementation tasks that mutate `pyproject.toml` or import `sklearn` are strictly **`BLOCKED`** until human sign-off is committed and verified by an automated governance gate.

---

## Project Structure

### Documentation (this feature)

```text
specs/004-baseline-training-and-evaluation/
├── spec.md              # Feature specification with Q1–Q5 resolutions
├── plan.md              # This file (implementation plan)
├── research.md          # Phase 0 decisions & alternatives
├── data-model.md        # Phase 1 domain entities, invariants & tolerance registry
├── quickstart.md        # Phase 1 usage & validation scenarios
├── checklists/
│   └── requirements.md  # Quality validation checklist
└── contracts/
    ├── training_job_schema.json          # State-aware schema for TrainingJob execution manifests
    ├── preprocessing_config_schema.json  # Schema for bandpass filter configuration (with config_hash)
    ├── neutral_model_manifest_schema.json# Schema for neutral numeric model parameter manifests (5 roles)
    └── evaluation_report_schema.json     # Schema for standardized evaluation reports (traceability & bounds)
```

### Source Code Layout

```text
semg_baseline/
├── __init__.py          # Public package exports
├── lineage.py           # ScientificLineage tracking and Feature 003 hash verification
├── filter.py            # CausalBandpassFilter (4th-order Butterworth 20-450 Hz SOS)
├── window.py            # PureWindowSegmenter (steady-state pure windowing & transition audit)
├── features.py          # TimeDomainFeatureExtractor (MAV, RMS, WL, ZC, SSC for 12 channels)
├── thresholds.py        # ThresholdCalibrator (data-driven ZC/SSC deadzones on train)
├── scaling.py           # FeatureScaler (train-only custom Z-score with zero-variance protection)
├── lda.py               # CanonicalLdaTrainer (regularized LDA, uniform/empirical priors, Ledoit-Wolf)
├── export.py            # NeutralModelExporter (neutral model_params.npz and sidecar manifest generation)
├── predictor.py         # NeutralNumpyPredictor (Feature 005 runtime inference simulation)
├── runner.py            # TrainingJobExecutor (Within-subject LOTO, Cross-subject 5-fold, append-only)
├── evaluator.py         # EvaluationReportGenerator (Top-1, Balanced Acc, Macro-F1, confusion matrix)
└── confounding.py       # ExerciseConfoundingAuditor (exercise_id_from_rest diagnostic)

docs/adr/
└── ADR-004-001-scikit-learn-host-only.md  # Persistent governance record for scikit-learn

tests/
├── fixtures/
│   └── synthetic_semg.py               # Procedural deterministic synthetic RecordingData generator
├── test_baseline_governance_gate.py    # Automated check for ADR approval, allowlists, and fingerprint
├── test_baseline_preprocessing.py      # Causal bandpass filter contracts and frequency response
├── test_baseline_windowing.py          # Steady-state pure windowing, boundary exclusion, transition audit
├── test_baseline_features.py           # Analytical oracles for MAV, RMS, WL, ZC, SSC (shape 60)
├── test_baseline_thresholds.py         # Data-driven ZC/SSC threshold selection within training pool
├── test_baseline_scaling.py            # Train-only custom Z-score scaling and zero-variance handling
├── test_baseline_lda.py                # Uniform vs empirical priors, Ledoit-Wolf analytic shrinkage
├── test_baseline_export_and_oracle.py  # Neutral model export and sklearn == numpy equivalence oracle
├── test_baseline_shuffled_labels.py    # Negative control: chance-level performance under permuted train labels
├── test_baseline_within_subject.py     # Within-subject LOTO validation and final test evaluation
├── test_baseline_cross_subject.py      # Cross-subject 5-fold evaluation on frozen partitions
├── test_baseline_confounding.py        # exercise_id_from_rest diagnostic runner
└── test_baseline_training_job.py       # Append-only immutability, fingerprinting, and state-aware schema gates
```

**Structure Decision**: A dedicated, isolated `semg_baseline/` package alongside `semg_dsp/` and `semg_dataset/`. This preserves clean architectural boundaries, separating baseline ML pipelines and evaluation from causal DSP primitives (`semg_dsp`) and immutable dataset intake (`semg_dataset`).

---

## Architectural Stages & Conceptual Data Flow

```text
Feature 003 Frozen Contracts
(RecordingData / DatasetView / SplitManifest)
          │
          ▼
1. Preprocessing (`semg_baseline.filter`)
   - 4th-order causal Butterworth 20–450 Hz via CausalSosFilter
   - 50 Hz notch omitted per empirical spectral audit (R50 = 0.84)
          │
          ▼
2. Window Eligibility (`semg_baseline.window`)
   - W=400 (200 ms), S=100 (50 ms), 75% overlap, 20 Hz
   - Pure-window policy: 100% same state & unit
   - Discard REST ↔ ACTIVE, ACTIVE ↔ ACTIVE, and unit crossings
          │
          ▼
3. Threshold Calibration (`semg_baseline.thresholds`)
   - Compute RMS_rest_train per channel strictly on TRAIN
   - Evaluate R ∈ {0.5, 1.0, 1.5, 2.0, 3.0} via internal train CV
   - ε[ch] = R * RMS_rest_train[ch]
          │
          ▼
4. Feature Extraction (`semg_baseline.features`)
   - MAV, RMS, WL, ZC(ε), SSC(ε) across 12 channels -> (N, 60)
   - Zero NaN/Inf assertion
          │
          ▼
5. Feature Scaling (`semg_baseline.scaling`)
   - Custom Z-score: z = (x - mean_train) / scale_train
   - Zero-variance handling (scale=1.0)
   - Stateless transform for validation and test
          │
          ▼
6. Regularized LDA Fit (`semg_baseline.lda`)
   - Primary: uniform priors P(c) = 1/K (canonical 50-class)
   - Secondary: empirical priors P(c) = N_c / N_train (distinct fit)
   - Solver="lsqr", shrinkage="auto" (Ledoit-Wolf)
          │
          ├──────────────────────────────────────────┐
          ▼                                          ▼
7. Neutral Export (`semg_baseline.export`)    8. Evaluation (`semg_baseline.evaluator`)
   - Neutral model_params.npz                 - Top-1, Balanced Acc, Macro-F1
   - Sidecar model_manifest.sha256            - Per-class precision/recall [0, 1]
   - Feature 005 runtime boundary             - Confusion matrix & REST metrics
   - Equivalence oracle test (TOL-LDA)        - Confounding diagnostic audit
          │                                   - Negative control verification
          │                                          │
          └─────────────────────┬────────────────────┘
                                ▼
9. TrainingJob Execution Record (`semg_baseline.runner`)
   - artifacts/feature_004/jobs/<job_id>/
   - Deterministic job_fingerprint (including dependency_lock_hash) & manifest.sha256
   - Fail-closed validation against state-aware JSON Schemas
```

---

## Complexity Tracking

> *No constitutional violations. All implementations adhere strictly to core principles.*
