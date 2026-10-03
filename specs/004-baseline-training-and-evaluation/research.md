# Phase 0: Research & Architectural Decisions — Feature 004

**Feature**: `004-baseline-training-and-evaluation`  
**Status**: Completed  
**Inputs**: Feature Specification (`spec.md`), CLARIFY Resolutions (Q1–Q5)

---

## Decision 1 (Q1): Hardened Governance for Host-Only `scikit-learn` via Persistent ADR

- **Decision**: Adopt `scikit-learn` strictly as a host-only development and evaluation dependency governed by a persistent Architecture Decision Record at [`docs/adr/ADR-004-001-scikit-learn-host-only.md`](file:///home/allisson/Projects/agents_emg/docs/adr/ADR-004-001-scikit-learn-host-only.md), without amending Constitution v1.1.0 (commit `ce75be0`).
- **Rationale**:
  - Line 51 of `.specify/memory/constitution.md` states: *"Features MUST NOT introduce new third-party dependencies without prior governance approval."* The active Constitution does not define "prior governance approval" as strictly requiring a constitutional amendment.
  - A persistent ADR with an explicit human approval lifecycle (`PROPOSED` → human sign-off with approver ID, date, and git commit → `APPROVED`) sealed by a SHA-256 fingerprint provides audit-grade governance.
  - Scikit-learn is restricted to host-only evaluation: it is strictly prohibited in `semg_dsp/`, dataset ingestion, or embedded runtime code.
  - Import path allowlist: `semg_baseline/**` and Feature 004 tests (`tests/feature_004/**`).
  - API surface allowlist: `sklearn.discriminant_analysis`, `sklearn.covariance`, and `sklearn.metrics`. APIs that generate splits or pipelines (`model_selection`, `Pipeline`, `StandardScaler`) are strictly prohibited; splits originate 100% from Feature 003 frozen manifests.
  - Automated deterministic quality gate verifies ADR status, human signature, version pinning, import paths, API surface, and ADR fingerprint. Any violation results in a blocking `FAIL`.
  - Mutating `pyproject.toml` is blocked until the ADR is committed as `APPROVED` by a human.
- **Alternatives Rejected**:
  - *Amending Constitution to v1.2.0*: Rejected because the Constitution establishes universal operational principles; cataloging specific library dependencies in the Constitution creates unnecessary governance churn.
  - *Ad-hoc import without ADR*: Rejected as a direct violation of Constitution Principle I and Technical Constraints.

---

## Decision 2 (Q2): Causal DSP Filtering, Windowing & Data-Driven Thresholds

- **Decision**:
  1. **DSP Causal Band-Pass**: 4th-order causal Butterworth band-pass filter (20–450 Hz) executed via [`CausalSosFilter`](file:///home/allisson/Projects/agents_emg/semg_dsp/filter.py). Second-order sections (SOS) coefficients, frequency response, and golden references are owned explicitly by Feature 004.
  2. **Omission of 50 Hz Notch Filter**: The 50 Hz notch filter is formally omitted from the baseline. An empirical spectral audit on NinaPro DB2 recordings across all 5 folds demonstrated an average 50 Hz power ratio of **0.84** (< 1.0) compared to flanking bands (45–48 Hz and 52–55 Hz), showing that Delsys Trigno differential active electrodes eliminate power-line hum at the sensor. A notch filter would needlessly attenuate genuine motor unit action potential firing power in the 20–150 Hz band.
  3. **Windowing Invariants**: Nominal sampling rate = 2000.0 Hz; window length $W = 200\,\text{ms}$ (400 samples); stride $S = 50\,\text{ms}$ (100 samples); overlap = 75%; decision rate = 20 Hz. Invariant: No window may cross `repetition_unit` or partition boundaries.
  4. **Steady-State Pure Window Policy**: Majority voting is prohibited. A window is admitted if and only if all 400 samples belong to the exact same canonical state (label) and same repetition unit. Mixed windows (REST ↔ ACTIVE, ACTIVE ↔ ACTIVE, repetition crossings) are excluded from the baseline. Discarded window metrics are logged per subject/exercise for audit.
  5. **Data-Driven ZC/SSC Thresholds**: Deadzone thresholds are computed as $\epsilon[\text{channel}] = R \times \text{RMS\_rest\_train}[\text{channel}]$, with candidate grid $R \in \{0.5, 1.0, 1.5, 2.0, 3.0\}$ selected strictly via training-pool validation. Single $R$ selected per configuration/fold.
  6. **Train-Only Custom Z-Score Scaling**: $z_j = (x_j - \mu_j) / \sigma_j$. Mean and std computed strictly on TRAIN; validation and test transformed statelessly. If $\sigma_j == 0$, scale factor is fixed to $1.0$ and flagged as `zero_variance`. `sklearn.preprocessing.StandardScaler` is prohibited.
  7. **Anti-Leakage Invariant**: Rest RMS, ZC/SSC thresholds, scaling statistics, and shrinkage parameters never depend on validation or test splits.
- **Alternatives Rejected**:
  - *Fixed $\epsilon = 15\,\mu\text{V}$ threshold*: Rejected because sensor impedances and resting noise floors vary across subjects and channels.
  - *Majority voting on transition windows*: Rejected because transient boundary windows contaminate steady-state gesture representation.
  - *Enabling 50 Hz notch by convention*: Rejected by the empirical findings of the spectral audit.

---

## Decision 3 (Q3): LDA Priors, Ledoit-Wolf Covariance Shrinkage & Neutral Export Boundary

- **Decision**:
  1. **Primary Baseline (`uniform_prior_baseline`)**: Canonical 50-class baseline uses uniform class priors $P(c) = 1/50$ for all classes $c \in \{0..49\}$. This removes the empirical advantage of the dominant REST class (~70–75% of duration) from the Bayesian decision rule, allowing direct measurement of intrinsic gesture separability.
  2. **Secondary Reference (`empirical_prior_reference`)**: A distinct model is trained with empirical class frequencies $P(c) = N_c / N_{\text{train}}$ measured strictly on TRAIN to assess sensitivity to continuous class distribution.
  3. **Distinct Model Fits**: The two configurations are executed as distinct fits because priors participate directly in pooled covariance estimation in scikit-learn's LDA.
  4. **Analytic Ledoit-Wolf Shrinkage**: `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` implements the deterministic Ledoit-Wolf shrinkage lemma, guaranteeing well-conditioned, invertible covariance matrices for 60 features without grid-search tuning.
  5. **Shrinkage & Model Evidence**: Does not persist a single scalar $\lambda^*$ (which misrepresents multi-class internal pooling). Persists `classes_`, `means_`, `covariance_`, `coef_`, `intercept_`, and SHA-256 hashes in `TrainingJob`.
  6. **Neutral Export Boundary (Feature 005 Isolation)**: Model artifacts are exported exclusively as neutral numerical arrays (`.npz` containing `classes`, `coef`, `intercept`, `mean`, `scale`) and structured JSON manifests. Pickles, joblib dumps, or sklearn objects are strictly prohibited. Feature 005 runtime inference evaluates $\text{scores} = X_{\text{scaled}} @ \text{coef}^T + \text{intercept}$ followed by $\text{argmax}$ without importing `scikit-learn`.
  7. **Deterministic Equivalence Oracle**: An independent test in Feature 004 asserts $\text{sklearn.predict}(X) == \text{neutral\_numpy\_predict}(X)$ with exact class match and numerical score tolerances.
  8. **Test Split Immutability**: No prior, shrinkage, solver, or hyperparameter selection may access Within-Subject test repetitions `[2, 5]` or Cross-Subject test folds.
- **Alternatives Rejected**:
  - *Post-hoc log-prior adjustment*: Rejected because priors affect covariance shrinkage in scikit-learn.
  - *Pickle / Joblib model dumps*: Rejected as dangerous, non-portable, and introducing runtime coupling to `scikit-learn`.

---

## Decision 4 (Q4): Model Taxonomy & Recording Confounding Audit

- **Decision**:
  1. **Primary Baseline (`global_50_class`)**: Unified 50-class classifier (REST 0 + ACTIVE 1..49) combining windows from E1, E2, and E3 after validation against Feature 003 contracts. Serves as the primary benchmark for full prosthetic control.
  2. **Secondary Baselines (`exercise_E1`, `exercise_E2`, `exercise_E3`)**: Three independent exercise-stratified models (E1: 18 classes, E2: 24 classes, E3: 10 classes) using derived local labels from Feature 003 (`get_local_labels()`). Described in reports strictly as `exercise-stratified secondary baselines` or `literature-aligned exercise-level analysis` (not "exact replication of Atzori et al." unless all historical conditions match).
  3. **Non-Averaging Invariant**: Metrics are reported separately for global and exercise models. Averaging E1, E2, E3 accuracies to substitute for the global baseline is prohibited.
  4. **Diagnostic Confounding Audit (`exercise_id_from_rest`)**: Auxiliary classifier trained to predict `exercise_id \in \{E1, E2, E3\}` strictly from REST windows. Accuracy near chance (~33.3%) confirms absence of session/recording signatures in baseline REST; accuracy substantially above chance indicates recording confounding that must accompany the scientific interpretation of global 50-class results. Recordings are never excluded based on this diagnostic.
  5. **Scientific Hierarchy**: (1) `global_50_class` [Primary], (2) `E1/E2/E3 exercise-stratified` [Secondary], (3) `exercise_id_from_rest` [Diagnostic].
- **Alternatives Rejected**:
  - *Training only per-exercise models*: Rejected because a functional prosthetic controller must recognize all gesture classes from a single unified model.
  - *Discarding REST or treating it as a standard gesture*: Rejected because REST requires segregated tracking to avoid skewing multi-class metrics.

---

## Decision 5 (Q5): Append-Only TrainingJob Contract, Deterministic Fingerprint & Fixtures

- **Decision**:
  1. **Formal Schemas in `contracts/`**: Authoritative JSON Schemas in `specs/004-baseline-training-and-evaluation/contracts/`:
     - `training_job_schema.json`
     - `evaluation_report_schema.json`
     - `neutral_model_manifest_schema.json` (metadata for binary `.npz` arrays: name, dtype, shape, sha256, role)
     - `preprocessing_config_schema.json`
  2. **Append-Only Job Storage**: Persisted in `artifacts/feature_004/jobs/<job_id>/` with files: `job.json`, `config.json`, `model_manifest.json`, `model_params.npz`, `metrics.json`, `environment.json`, `stdout.log`, `stderr.log`, `manifest.sha256`. Completed jobs are immutable and append-only.
  3. **Deterministic `job_fingerprint`**: Computed from scientific inputs: Feature 003 lineage hashes, protocol/fold, preprocessing config, feature config, model config, seed, git commit, dependency lock hash. Volatile fields (timestamp, PID, absolute path) are excluded.
  4. **Git Versioning Policy**: Local job outputs in `artifacts/feature_004/jobs/**` are not committed to Git. Git versions schemas, canonical configs, traceability manifests, and consolidated reports in `reports/feature_004_baseline/`. Zero real DB2 samples in Git.
  5. **Procedural Synthetic Fixtures**: Shared generator in `tests/fixtures/synthetic_semg.py` producing `RecordingData` NinaPro-like (12 channels, 2 kHz, REST, synthetic active classes with distinct channel/frequency/amplitude patterns, repetitions, unit boundaries, seed).
  6. **Adversarial Test Coverage**: Covers zero-variance features, missing class in train, extreme REST imbalance, transition windows, repetition crossings, NaN/Inf, singular covariance, array shape/dtype/hash mismatches, artifact tampering, synthetic leakage.
  7. **Three-Tier Oracles**:
     - *Layer A*: Analytical feature oracles (mathematically exact MAV/RMS/WL/ZC/SSC).
     - *Layer B*: Sklearn training/reference oracle (governed by ADR).
     - *Layer C*: Neutral NumPy inference oracle (pure numerical arrays).
     - Strict assertion: $\text{sklearn.predict}(X) == \text{neutral\_numpy\_predict}(X)$.
  8. **Unit Tests vs Real DB2 Separation**: `pytest -q` uses only synthetic fixtures, does not access `data/raw/ninapro_db2/`, runs fast (<5s target). Full 40-subject training runs via a dedicated orchestrator runner.
  9. **Fail-Closed Integrity**: Discrepancies between manifests, arrays, hashes, schemas, splits, or lineage result in a blocking `FAIL`.
- **Alternatives Rejected**:
  - *Committing training runs or real DB2 data to Git*: Rejected by Constitution Principle VI and repository size limits.
  - *Monolithic single-script execution*: Rejected because modular stages guarantee testability, auditability, and clear separation of concerns.
