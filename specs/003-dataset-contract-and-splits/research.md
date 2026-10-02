# Phase 0: Research & Architectural Decisions — Feature 003

## Decision 1 (Q1): Frozen Dataset Views for Ground Truth
- **Decision**: Implement two explicit, frozen dataset views:
  1. `refined` (default): `restimulus` + `rerepetition`
  2. `stimulus` (secondary): `stimulus` + `repetition`
  - Cross-pair mixing is strictly prohibited by runtime assertions and typing.
  - Each view has a unique ID, hash, and versioning contract.
- **Rationale**: NinaPro DB2 researchers recalculated true movement onsets and offsets using Cyberglove / inclinometer activation, creating `restimulus`/`rerepetition`. In scientific benchmarks (Atzori et al., 2014), the refined label provides significantly cleaner ground truth for transient analysis. However, visual stimulus labels remain useful for evaluating human-in-the-loop latency. Keeping both views frozen with strict pair integrity prevents catastrophic label noise.
- **Alternatives Rejected**:
  - *Single default with dynamic override*: Rejected because dynamic overrides without frozen hashes lead to irreproducible training manifests.
  - *Permitting cross-mixing (e.g. stimulus with rerepetition)*: Rejected as scientifically invalid.

---

## Decision 2 (Q2): PROVE-OR-QUARANTINE Temporal Alignment
- **Decision**: Establish a single canonical timeline per recording governed by the **PROVE-OR-QUARANTINE** policy:
  1. The alignment engine explicitly checks and proves whether signals share the identical start anchor ($t=0$).
  2. If start anchoring is proven and length mismatch is confined strictly to missing tail samples: apply `anchor_start_truncate_tail` (truncating trailing samples to match the shorter length).
  3. If start anchoring cannot be proven, if temporal shift is detected, or if ambiguity exists: mark the recording as `QUARANTINE` and exclude it from downstream processing.
  4. Zero padding, zero interpolation, zero automatic fallback, and zero silent `min()`.
- **Rationale**: 18 Exercise 3 recordings exhibited length differences (17 with 1–5 samples, and `S12_E3_A1.mat` with 272 samples / ~136 ms). The intake analysis proved that `emg`, `stimulus`, and `repetition` start at $t=0$ in complete synchrony with the trigger pulse, and that the missing samples in `restimulus`/`rerepetition` occur exclusively at the very end of the file due to filter border truncation during Ninapro offline post-processing. Because the start anchor is rigorously intact, `anchor_start_truncate_tail` safely aligns the timeline without introducing phase distortion. If any future or unverified file fails this proof, it must be quarantined rather than silently altered.
- **Alternatives Rejected**:
  - *Silent `min()` truncation without proof*: Rejected because it masks potential phase shifts or start misalignments.
  - *Right-padding with zero/rest*: Rejected because synthetic padding creates artificial boundary transients and false resting samples.
  - *Automatic fallback to stimulus*: Rejected because it silently substitutes a completely different label semantics without user visibility.

---

## Decision 3 (Q3): Dual Leakage-Free Partitioning Protocols
- **Decision**: Implement a **Dual Split Protocol Engine**:
  1. **Within-Subject (Repetition-Split)**:
     - `train`: repetitions `[1, 3, 4, 6]` across all 40 subjects
     - `test`: repetitions `[2, 5]` across all 40 subjects
     - `validation`: empty in Feature 003 (Feature 004 will derive validation strictly from `train`).
  2. **Cross-Subject (5-Fold Rotating Subject-Split)**:
     - 40 subjects partitioned into 5 balanced groups of 8:
       - $G_1$: S1–S8
       - $G_2$: S9–S16
       - $G_3$: S17–S24
       - $G_4$: S25–S32
       - $G_5$: S33–S40
     - 5 rotating folds:
       - **Fold 1**: Test = $G_5$ (8), Val = $G_4$ (8), Train = $G_1 \cup G_2 \cup G_3$ (24)
       - **Fold 2**: Test = $G_4$ (8), Val = $G_3$ (8), Train = $G_5 \cup G_1 \cup G_2$ (24)
       - **Fold 3**: Test = $G_3$ (8), Val = $G_2$ (8), Train = $G_4 \cup G_5 \cup G_1$ (24)
       - **Fold 4**: Test = $G_2$ (8), Val = $G_1$ (8), Train = $G_3 \cup G_4 \cup G_5$ (24)
       - **Fold 5**: Test = $G_1$ (8), Val = $G_5$ (8), Train = $G_2 \cup G_3 \cup G_4$ (24)
     - Explicit, deterministic, and frozen subject assignment lists.
- **Rationale**: The Within-Subject protocol replicates the standard evaluation method of Atzori et al. (2014) for evaluating user-customized prosthetic controllers. The Cross-Subject protocol enables testing generic, pre-trained models across unseen users. Providing both with strict zero-leakage guarantees and frozen manifests satisfies all scientific evaluation requirements.
- **Alternatives Rejected**:
  - *Random shuffling of samples/windows*: Rejected because random shuffling causes massive temporal and window-leakage.
  - *Single 70/15/15 subject split*: Rejected because a single fold doesn't evaluate full-population cross-validation stability.

---

## Decision 4 (Q4): REST Class Handling & Segregated Auditing
- **Decision**:
  - Retain REST (`label = 0`) in the canonical recording timeline.
  - Audit REST sample counts and duration separately from active gestures in `dataset_catalog.json`.
  - Deterministically associate inter-repetition REST segments with adjacent repetition units.
  - Defer the modeling choice of 49 classes (gestures only) vs 50 classes (gestures + rest) to Feature 004.
- **Rationale**: Streaming pipelines must handle continuous time, where resting periods naturally separate active contractions. Removing REST during dataset ingestion would destroy the temporal continuity required by causal DSP filters and streaming window buffers. Auditing it separately allows downstream training pipelines in Feature 004 to choose appropriate sampling or weighting strategies without altering the raw contract.
- **Alternatives Rejected**:
  - *Discarding REST at dataset ingestion*: Rejected because it breaks continuous streaming simulation.
  - *Hardcoding 49-class or 50-class mapping in Feature 003*: Rejected because model vocabulary definition is a model-architecture responsibility (Feature 004).

---

## Decision 5 (Q5): Native Global Labels vs Derived Local Labels
- **Decision**:
  - Preserve native global labels (E1: 1..17, E2: 18..40, E3: 41..49, REST: 0) directly from NinaPro DB2.
  - Do NOT apply any offsets or transformations to raw data.
  - Provide an explicit derived helper `to_exercise_local_label(global_label, exercise_id)` that computes local 1-based indices (E1: 1..17, E2: 1..23, E3: 1..9) when exercise-isolated classification is requested.
- **Rationale**: The authors of NinaPro DB2 designed the dataset with globally unique labels across exercises. Preserving this ground truth directly eliminates translation errors while maintaining complete compatibility with the raw `.mat` files.
- **Alternatives Rejected**:
  - *Normalizing all exercises to 1..N*: Rejected because it creates ambiguity across exercises and requires reversible mapping logic.
