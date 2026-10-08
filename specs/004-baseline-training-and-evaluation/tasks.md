---
description: "SpecKit task checklist with harness metadata for Feature 004"
---

# Tasks: 004-baseline-training-and-evaluation

**Input**: Design documents from `/specs/004-baseline-training-and-evaluation/`  
**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `contracts/` compliant with Constitution v1.1.0.

All tasks follow the SpecKit checklist format: `- [ ] T001 [P?] [US1?] Description with file path`.  
Each task includes adjacent `harness-task` metadata for Python orchestrator consumption.

---

## TDD Contract & Scope Containment Rules

- TDD applies to all implementation tasks (T001–T019): each runs RED → GREEN → REFACTOR under the Python orchestrator.
- In RED, the TestDesigner writes task-specific tests in `tests/`, demonstrates `EXPECTED_FAILURE`, and Python records SHA-256 test snapshots.
- In GREEN, the implementer edits strictly the paths listed in `allowed_files`. Test files are protected against `TEST_TAMPERING`.
- Mathematical, DSP, feature extraction, and ML tasks are marked `"numeric_sensitive": true`. Structural, governance, and manifest tasks are marked `"numeric_sensitive": false`.
- Raw data directory `data/raw/ninapro_db2/` and Feature 003 artifacts are strictly READ-ONLY. No task may modify or reinterpret raw archives.
- **Dependency Governance Constraint**: ADR-004-001 (`scikit-learn`), ADR-004-002 (`jsonschema`), and ADR-004-003 (`tomli`) are `APPROVED / LOCKED` (aggregate dependency governance status `APPROVED`, `all_required_dependencies_approved = true`). T013b is COMPLETE / FORMALLY ACCEPTED, T013d is COMPLETE / ACCEPTED, T014 is COMPLETE / SCIENTIFICALLY ACCEPTED, T015 is COMPLETE / SCIENTIFICALLY ACCEPTED (US3 ACCEPTED). T016 is ACTIVE (RED phase).
- T020 is a non-implementation review task marked `NOT_AUTOMATABLE`.

---

## Phase 1: Setup

**Purpose**: Initialize the core package structure for `semg_baseline`, shared fixtures, and automated governance gate.

- [x] T001 [P] Setup core package layout and public exports in semg_baseline/__init__.py
  <!-- harness-task {"requirements":["FR-001"],"acceptance_criteria":["SC-001"],"plan_decisions":["Decision-1","Decision-5"],"dependencies":[],"test_type":"UNIT","allowed_files":["semg_baseline/__init__.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

- [x] T002 [P] Implement deterministic procedural synthetic RecordingData generator and test fixtures in tests/fixtures/synthetic_semg.py
  <!-- harness-task {"requirements":["FR-011.5"],"acceptance_criteria":["SC-002","SC-006"],"plan_decisions":["Decision-5"],"dependencies":[],"test_type":"UNIT","allowed_files":["tests/fixtures/synthetic_semg.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

- [x] T003 [P] Implement automated dependency governance gate verifying ADR status, signature, allowlists, and fingerprint in semg_baseline/governance.py
  <!-- harness-task {"requirements":["FR-006"],"acceptance_criteria":["SC-001"],"plan_decisions":["Decision-1"],"dependencies":[],"test_type":"UNIT","allowed_files":["semg_baseline/governance.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

**Checkpoint**: `semg_baseline` package is importable, synthetic fixtures isolate tests from 17 GiB raw DB2 data, and governance gate asserts `CONDITIONAL_PASS`.

---

## Phase 2: Foundational

**Purpose**: Establish cryptographic lineage verification, causal DSP filtering, steady-state pure windowing, time-domain features, and train-only calibration.

- [x] T004 [US2] Implement ScientificLineage tracking and Feature 003 cryptographic hash verification in semg_baseline/lineage.py
  <!-- harness-task {"requirements":["FR-001","FR-002","FR-003"],"acceptance_criteria":["SC-001"],"plan_decisions":["Decision-5"],"dependencies":["T001"],"test_type":"UNIT","allowed_files":["semg_baseline/lineage.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

- [x] T005 [P] [US1] Implement 4th-order causal Butterworth bandpass filter (20-450 Hz SOS, notch omitted) in semg_baseline/filter.py
  <!-- harness-task {"requirements":["FR-012.1"],"acceptance_criteria":["SC-002"],"plan_decisions":["Decision-2"],"dependencies":["T001"],"test_type":"UNIT","allowed_files":["semg_baseline/filter.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

- [x] T006 [P] [US1] Implement steady-state pure window segmenter with boundary exclusion and transition audit in semg_baseline/window.py
  <!-- harness-task {"requirements":["FR-012.2","FR-012.3"],"acceptance_criteria":["SC-002"],"plan_decisions":["Decision-2"],"dependencies":["T001"],"test_type":"UNIT","allowed_files":["semg_baseline/window.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

- [x] T007 [P] [US1] Implement analytical time-domain feature extractor (MAV, RMS, WL, ZC, SSC for 12 channels) in semg_baseline/features.py
  <!-- harness-task {"requirements":["FR-004"],"acceptance_criteria":["SC-002"],"plan_decisions":["Decision-2"],"dependencies":["T001","T002"],"test_type":"UNIT","allowed_files":["semg_baseline/features.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

- [x] T008 [US1] Implement data-driven ZC/SSC threshold calibrator on training rest windows in semg_baseline/thresholds.py
  <!-- harness-task {"requirements":["FR-012.4"],"acceptance_criteria":["SC-002"],"plan_decisions":["Decision-2"],"dependencies":["T001","T007"],"test_type":"UNIT","allowed_files":["semg_baseline/thresholds.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

- [x] T009 [US1] Implement train-only custom Z-score feature scaler with zero-variance protection in semg_baseline/scaling.py
  <!-- harness-task {"requirements":["FR-013"],"acceptance_criteria":["SC-002"],"plan_decisions":["Decision-2"],"dependencies":["T001"],"test_type":"UNIT","allowed_files":["semg_baseline/scaling.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

**Checkpoint**: Signal processing primitives and feature extraction execute statelessly with zero lookahead, zero NaN/Inf, and strict train-only threshold/scaling isolation.

---

## Phase 3: User Story 1 — Time-Domain Feature Extraction on Frozen Dataset Views (P1)

**Goal**: Seamlessly chain filtering, pure windowing, threshold calibration, feature extraction, and scaling over continuous `RecordingData` from Feature 003.

**Independent Test**: Execute `BaselineFeaturePipeline` on synthetic recordings; assert feature matrix shape `(N, 60)`, zero NaN/Inf, strict exclusion of transition windows, and train-only calibration.

- [x] T010 [US1] Implement integrated BaselineFeaturePipeline combining stages 1 to 5 in semg_baseline/pipeline.py
  <!-- harness-task {"requirements":["FR-001","FR-004","FR-012","FR-013"],"acceptance_criteria":["SC-001","SC-002"],"plan_decisions":["Decision-2"],"dependencies":["T004","T005","T006","T007","T008","T009"],"test_type":"UNIT","allowed_files":["semg_baseline/pipeline.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

**Checkpoint**: User Story 1 complete; 60-dimensional feature vectors are extracted from frozen dataset views without data leakage.

---

## Phase 4: User Story 2 — Auditable TrainingJob Contract & Cryptographic Provenance (P1)

**Goal**: Establish immutable, append-only `TrainingJob` records in `artifacts/feature_004/jobs/<job_id>/` with state-aware schema validation (`PLANNED`, `RUNNING`, `SUCCEEDED`, `FAILED`), deterministic `job_fingerprint` (including `dependency_lock_hash`), sidecar SHA-256 manifest, neutral parameter export, and neutral NumPy inference simulation.

**Independent Test**: Launch mock training jobs across all 4 lifecycle states; assert JSON schema validation passes per status; verify `job_fingerprint` excludes volatile fields; verify exported `.npz` arrays match descriptors; verify neutral predictor scores match oracle within `TOL-LDA-NEUTRAL-SCORES`.

- [x] T011 [P] [US2] Implement NeutralModelExporter generating model_params.npz and sidecar model_manifest.sha256 in semg_baseline/export.py
  <!-- harness-task {"requirements":["FR-005.5","FR-011.1"],"acceptance_criteria":["SC-001","SC-006"],"plan_decisions":["Decision-3","Decision-5"],"dependencies":["T001"],"test_type":"UNIT","allowed_files":["semg_baseline/export.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

- [x] T012 [P] [US2] Implement NeutralNumpyPredictor for runtime inference simulation without sklearn in semg_baseline/predictor.py
  <!-- harness-task {"requirements":["FR-005.5","FR-005.6"],"acceptance_criteria":["SC-006"],"plan_decisions":["Decision-3"],"dependencies":["T001","T011"],"test_type":"UNIT","allowed_files":["semg_baseline/predictor.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

- [x] T013 [US2] Implement append-only TrainingJobExecutor with state-aware schema validation and deterministic fingerprinting in semg_baseline/runner.py
  <!-- harness-task {"requirements":["FR-003","FR-011.1","FR-011.2","FR-011.3","FR-011.4"],"acceptance_criteria":["SC-001","SC-006"],"plan_decisions":["Decision-5"],"dependencies":["T004","T011"],"test_type":"UNIT","allowed_files":["semg_baseline/runner.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->
  *Note*: IMPLEMENTATION + OPERATIONAL ACCEPTANCE COMPLETE. TrainingJob schema validation operational under governed jsonschema Draft 2020-12 validator.

- [x] T013a [GOV] Harden governance gate with stdlib-only validation and eliminate zero-variance information loss across export/load boundary in semg_baseline/governance.py, semg_baseline/export.py, semg_baseline/predictor.py
  <!-- harness-task {"requirements":["FR-005.5","FR-006.1","FR-006.4","FR-011.1","FR-013"],"acceptance_criteria":["SC-001","SC-002","SC-006"],"plan_decisions":["Decision-1","Decision-2","Decision-3"],"dependencies":["T003","T011","T012"],"test_type":"UNIT","allowed_files":["semg_baseline/governance.py","semg_baseline/export.py","semg_baseline/predictor.py","specs/004-baseline-training-and-evaluation/contracts/neutral_model_manifest_schema.json"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

- [x] T013c [GOV] Finalize aggregate multi-ADR governance status calculation and symbol-level API-surface AST enforcement for jsonschema and tomli in semg_baseline/governance.py and update semg_baseline/runner.py import pattern
  <!-- harness-task {"requirements":["FR-006.1","FR-006.4","FR-006.6","FR-011.1"],"acceptance_criteria":["SC-001"],"plan_decisions":["Decision-1"],"dependencies":["T003","T013","T013a"],"test_type":"UNIT","allowed_files":["semg_baseline/governance.py","semg_baseline/runner.py","tests/test_baseline_governance_gate.py","specs/004-baseline-training-and-evaluation/tasks.md"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

**Checkpoint**: Phase 4 implementation and multi-ADR governance hardening complete; append-only job runner, cryptographic lineage, neutral inference boundary, aggregate governance & API-surface gates are established, and T013b dependency governance transition is fully executed.

---

## Phase 5: User Story 3 — Within-Subject Baseline Training with Leave-One-Repetition-Out Validation (P1)

**Goal**: Train regularized LDA with Ledoit-Wolf shrinkage, uniform priors ($P=1/50$) and empirical priors ($N_c/N_{\text{train}}$) over repetitions `[1, 3, 4, 6]`, evaluating once on test repetitions `[2, 5]`. (T013b formally accepted; T013d complete; T014 is UNBLOCKED and ready for RED phase).

**Independent Test**: Run Within-Subject training on synthetic subject data; assert 4 validation folds evaluate `[1, 3, 4, 6]`; assert final test evaluates strictly on `[2, 5]`; assert equivalence oracle matches neutral numpy predictor within `TOL-LDA-NEUTRAL-SCORES`.

- [x] T013b [GOV] Execute formal dependency governance transition for all three governed dependencies (ADR-004-001 scikit-learn, ADR-004-002 jsonschema, ADR-004-003 tomli): human approval Stage A, cryptographic lock Stage B, authorized pyproject.toml mutation, uv.lock generation, clean-environment installation gate, and automated gate re-verification
  <!-- harness-task {"requirements":["FR-006.1","FR-006.4","FR-006.6","FR-011.1","FR-011.6"],"acceptance_criteria":["SC-001"],"plan_decisions":["Decision-1"],"dependencies":["T013c"],"test_type":"INTEGRATION","allowed_files":["docs/adr/ADR-004-001-scikit-learn-host-only.md","docs/adr/ADR-004-002-jsonschema-host-contract-validation.md","docs/adr/ADR-004-003-tomli-python310-compatibility.md","pyproject.toml","uv.lock","semg_baseline/governance.py","specs/004-baseline-training-and-evaluation/tasks.md"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->
  *Note*: Technical execution and formal governance acceptance COMPLETE under verified Commit A provenance and locked state.

- [x] T013d [GOV] Post-lock consistency and approval-commit provenance verification in docs/adr/ADR-004-002-*, docs/adr/ADR-004-003-*, semg_baseline/governance.py, tests/test_baseline_governance_gate.py, specs/004-baseline-training-and-evaluation/tasks.md
  <!-- harness-task {"requirements":["FR-006.1","FR-006.4","FR-006.6","FR-011.1","FR-011.6"],"acceptance_criteria":["SC-001"],"plan_decisions":["Decision-1"],"dependencies":["T013b"],"test_type":"UNIT","allowed_files":["docs/adr/ADR-004-002-jsonschema-host-contract-validation.md","docs/adr/ADR-004-003-tomli-python310-compatibility.md","semg_baseline/governance.py","tests/test_baseline_governance_gate.py","specs/004-baseline-training-and-evaluation/tasks.md"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->

- [x] T014 [US3] Implement CanonicalLdaTrainer with uniform/empirical priors and Ledoit-Wolf analytic shrinkage in semg_baseline/lda.py
  <!-- harness-task {"requirements":["FR-005.1","FR-005.2","FR-005.3","FR-005.4","FR-006.2","FR-006.3","FR-011.5"],"acceptance_criteria":["SC-005"],"plan_decisions":["Decision-3"],"dependencies":["T010","T013b","T013d"],"test_type":"UNIT","allowed_files":["semg_baseline/lda.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->
  *Note*: Complete under approved test correction. Old RED hash: `5e00b149596da14adb92b009c60ab3ab6d192a3d52f5da6035324e02055df3e8`; New protected RED hash: `24d0e15a43830b49b9d5cd719329d6488f1f93694b9fa860f5a102dc15fef5b4` (erroneous scalar shrinkage assertion removed per Decision-3; adversarial missing/unexpected classes and vocabulary fail-closed contracts added per FR-011.5; score tolerance registry `TOL-LDA-NEUTRAL-SCORES` enforced; zero `sys.modules` eviction in production code). All 257 tests passing.

- [x] T015 [US3] Implement WithinSubjectRunner executing LOTO validation and single test evaluation in semg_baseline/within_subject.py
  <!-- harness-task {"requirements":["FR-005.7","FR-008"],"acceptance_criteria":["SC-003"],"plan_decisions":["Decision-3","Decision-5"],"dependencies":["T010","T013","T014"],"test_type":"UNIT","allowed_files":["semg_baseline/within_subject.py","tests/test_baseline_within_subject.py","specs/004-baseline-training-and-evaluation/tasks.md"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->
  *Note*: Complete under approved test extensions. Original RED hash: `0c578819ef8b1afb701ab187f2e0c4816b08c85f05ce59e616a29cb3fff28857`; Hardening extension hash: `ec8f288fe5a5ce26197a62af294bf2040dafd90ff921b9b7adc4466f739d24d5`; Residual patch extension hash: `cd32401acd041d22a90dbf804619ef544449310a32cbd75f636384f4fbe45eef` (`tests/test_baseline_within_subject.py`). Hardened contracts verified: (1) structural test isolation (repetitions [2, 5] receive zero raw access, filtering, or segmentation during selection; NaN-poisoning immunity on test reps demonstrated); (2) auditable execution event counter incremented strictly after successful evaluation (`test_evaluation_count == 1`); (3) frozen canonical selection metric ('balanced_accuracy' strictly mandated, non-canonical rejected with ValueError, order-independent lowest-R tie-break verified against reversed order [2.0, 1.0, 0.5]); (4) explicit vocabulary enforcement preventing silent class count shrinkage from 50 (default `num_classes=50`). All 12 unit tests and 269 full suite tests passing.

**Checkpoint**: User Story 3 complete / pending human review acceptance; individual user baseline classifier is fitted and validated with zero test leakage.

---

## Phase 6: User Story 4 — Cross-Subject 5-Fold Benchmark Evaluation (P2)

**Goal**: Run baseline regularized LDA on the 5 frozen Cross-Subject folds from Feature 003 (24 train, 8 val, 8 test subjects per fold).

**Independent Test**: Execute cross-subject runner on synthetic multi-subject fixtures; assert exactly 24 train / 8 val / 8 test subjects per fold; assert zero partition overlap.

- [x] T016 [US4] Implement CrossSubjectRunner executing 5-fold evaluation over frozen partitions in semg_baseline/cross_subject.py
  <!-- harness-task {"requirements":["FR-005.7","FR-009","FR-012.4","FR-013"],"acceptance_criteria":["SC-004"],"plan_decisions":["Decision-5","Decision-6"],"dependencies":["T010","T013","T014"],"test_type":"UNIT","allowed_files":["semg_baseline/cross_subject.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->
  *Note*: TDD cycle complete (RED → GREEN → REFACTOR). Protected RED hash intact: `6aad9a45b37a037314abcd8a12fc49e651c63349b1d9460db797b1165da3af58` (`tests/test_baseline_cross_subject.py`). GREEN implemented `CrossSubjectRunner`, `CrossSubjectFoldResult`, `CrossSubjectBenchmarkResult`, `CrossSubjectAggregateMetrics`, and `MetricStats` in `semg_baseline/cross_subject.py`. REFACTOR verified as NO-OP (implementation is minimal, cohesive, and contract-compliant; all numerical invariants preserved). Traceability updated with Decision-6. Allowed files strictly respected.

**Checkpoint**: User Story 4 complete; inter-subject transferability baseline is evaluated across all 5 frozen folds.

---

## Phase 7: User Story 5 — Dual-Vocabulary & Stratified Reporting, Confounding Audit & Negative Control (P2)

**Goal**: Generate standardized evaluation reports covering primary 50-class and secondary active-only 49-class metrics, exercise-stratified secondary baselines (`exercise_E1`, `exercise_E2`, `exercise_E3`), diagnostic confounding audit (`exercise_id_from_rest`), and shuffled-label negative control.

**Independent Test**: Generate structured reports; assert all normalized metrics bounded in `[0.0, 1.0]`; assert REST precision/recall segregated; assert confounding diagnostic reports interpretation; assert shuffled-label test fails if accuracy exceeds chance upper limit.

- [ ] T017 [US5] Implement EvaluationReportGenerator with per-class metrics, rest metrics, bounds [0, 1], and consolidated summaries in semg_baseline/evaluator.py
  <!-- harness-task {"requirements":["FR-007.1","FR-007.2","FR-007.4","FR-010"],"acceptance_criteria":["SC-005"],"plan_decisions":["Decision-4","Decision-5"],"dependencies":["T013"],"test_type":"UNIT","allowed_files":["semg_baseline/evaluator.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":false} -->
  *Note*: RED phase active (contract reconciled with US5, schema, and active confusion matrix `[49, 50]`). Original RED hash: `c7473be97968738956f8dc9f85d50040bf49a8ec87cfdd560eb3e76248b1f3bf`; Reconciled RED hash: `88a4a22f78ccab58887cb96382cde686f2ecc708af4cfbfffd4ee7a5b923bd66` (`tests/test_baseline_evaluator.py`). EXPECTED_FAILURE demonstrated (`ModuleNotFoundError: No module named 'semg_baseline.evaluator'`). Allowed files strictly `["semg_baseline/evaluator.py"]`.

- [ ] T018 [P] [US5] Implement ExerciseConfoundingAuditor predicting exercise_id from rest windows in semg_baseline/confounding.py
  <!-- harness-task {"requirements":["FR-007.3","FR-007.5"],"acceptance_criteria":["SC-005"],"plan_decisions":["Decision-4"],"dependencies":["T010","T014"],"test_type":"UNIT","allowed_files":["semg_baseline/confounding.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

- [ ] T019 [P] [US5] Implement ShuffledLabelNegativeControl testing chance-level performance under permuted train labels in semg_baseline/controls.py
  <!-- harness-task {"requirements":["FR-005.8"],"acceptance_criteria":["SC-005","SC-006"],"plan_decisions":["Decision-5"],"dependencies":["T010","T014","T017"],"test_type":"UNIT","allowed_files":["semg_baseline/controls.py"],"tdd_phases":["RED","GREEN","REFACTOR"],"numeric_sensitive":true} -->

**Checkpoint**: User Story 5 complete; comprehensive reporting, literature-aligned secondary baselines, session bias audit, and negative control verification are operational.

---

## Final Phase: Polish & Cross-Cutting Concerns

- [ ] T020 Final audit of baseline training contracts, JSON schemas, tolerance registry, and Constitution compliance
  <!-- harness-task {"requirements":["FR-001","FR-002","FR-003","FR-004","FR-005","FR-006","FR-007","FR-008","FR-009","FR-010","FR-011","FR-012","FR-013","FR-014","FR-015"],"acceptance_criteria":["SC-001","SC-002","SC-003","SC-004","SC-005","SC-006"],"plan_decisions":["Decision-1","Decision-2","Decision-3","Decision-4","Decision-5","Decision-6"],"dependencies":["T015","T016","T017","T018","T019"],"test_type":"NOT_AUTOMATABLE","justification":"Human and static review of git diff, verification of zero raw file modifications, zero leakage between partitions, passage of all deterministic quality gates, and compliance with Constitution v1.1.0.","alternative_verification":"Verify git status in data/raw/ is untouched; verify all unit tests pass with pytest -q; run ruff, mypy, compileall, and contract schema validators.","allowed_files":[],"tdd_phases":[],"numeric_sensitive":false} -->

---

## Dependencies & Execution Order

```text
T001 (Package Layout) ──┬──► T004 (Lineage) ──┐
                        ├──► T005 (Causal BP) ──┼──► T010 (Pipeline) ──┬──► T015 (Within-Subject) ──┐
                        ├──► T006 (Windowing) ──┤                      ├──► T016 (Cross-Subject)  ──┤
                        ├──► T007 (Features)  ──┤                      ├──► T018 (Confounding)    ──┤
                        ├──► T008 (Thresholds)──┤                      └──► T019 (Shuffled-Label) ──┤
                        └──► T009 (Scaling)   ──┘                                                   │
                                                                                                    ▼
T002 (Fixtures) ─────────────► Used across all tests                                          T020 (Audit)
                                                                                                    ▲
T003 (Governance Gate) ──► T013a ──► T013c ──► T013b ──► T013d ──► T014 (LDA Trainer)───────────────┤
                                        │                                                           │
                                        ▼                                                           │
T011 (Exporter) ─────────────► T012 (Predictor)                                                     │
       │                                                                                            │
       ▼                                                                                            │
T013 (Job Executor) ─────────► T017 (Report Generator) ─────────────────────────────────────────────┘
```

---

## Parallel Opportunities

- **Setup & Infrastructure**: T001, T002, and T003 can execute concurrently.
- **Foundational DSP & Feature Primitives**: T005, T006, and T007 can execute in parallel.
- **Contracts & Model Boundary**: T011 and T012 can execute in parallel.
- **Auditing & Negative Controls**: T018 and T019 can execute in parallel once the core pipeline is operational.

---

## Implementation Strategy

1. **Foundation & Host Contracts (T001–T013c)**: T001–T013 implementation is complete at source level. T013 contains host-side jsonschema integration whose operational acceptance remains governance-blocked until T013b. T013c hardens aggregate multi-ADR governance and symbol-level API-surface AST enforcement. No sklearn-based modeling is executed before T013b.
2. **ADR Sign-Off & Governance Lock Gate (T013b)**: Completed. Human approval Stage A recorded, cryptographic lock Stage B committed, authorized `pyproject.toml` mutation applied, `uv.lock` generated, clean-environment installation verified, all automated verification gates pass. T014 is unblocked.
3. **Modeling & Protocols (T014–T016)**: Implement canonical regularized LDA, Within-Subject LOTO protocol, and Cross-Subject 5-fold evaluation.
4. **Reporting, Diagnostics & Controls (T017–T019)**: Implement dual-vocabulary report generator, exercise confounding auditor, and shuffled-label negative control.
5. **Final Quality Audit (T020)**: Complete deterministic verification, schema validation, and acceptance sign-off.
