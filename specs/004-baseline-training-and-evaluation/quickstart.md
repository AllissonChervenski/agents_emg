# Phase 1: Quickstart Validation Guide — Feature 004

**Feature**: `004-baseline-training-and-evaluation`  
**Purpose**: Developer usage scenarios and validation walkthroughs for baseline training and evaluation.

---

## Prerequisites

- Python 3.10+ (tested on Python 3.14.7)
- Packages: `numpy`, `scipy`
- Optional host dependency: `scikit-learn` (governed by [`docs/adr/ADR-004-001-scikit-learn-host-only.md`](file:///home/allisson/Projects/agents_emg/docs/adr/ADR-004-001-scikit-learn-host-only.md))
- Frozen Feature 003 dataset contracts in `semg_dataset`

---

## Scenario 1: Preprocessing & Steady-State Pure Windowing

```python
from semg_baseline import CausalBandpassFilter, PureWindowSegmenter
from semg_dataset import NinaProDB2Loader

loader = NinaProDB2Loader(raw_dir="data/raw/ninapro_db2")
rec = loader.load_recording(subject_id="S1", exercise_id="E1")
view = rec.get_view("refined")

# 1. Apply Feature 004 4th-order causal Butterworth bandpass (20-450 Hz)
bp_filter = CausalBandpassFilter(low_hz=20.0, high_hz=450.0, fs=2000.0)
filtered_emg = bp_filter.process(view.signals)

# 2. Segment into steady-state pure windows (W=400 samples, S=100 samples)
segmenter = PureWindowSegmenter(window_len=400, stride=100)
windows, labels, audit = segmenter.segment(
    emg=filtered_emg,
    labels=view.labels,
    repetitions=view.repetitions
)

print(f"Admitted pure windows: {windows.shape[0]}")
print(f"Discarded boundary windows: {audit.discarded_window_count}")
assert audit.repetition_crossing_windows_admitted == 0
```

---

## Scenario 2: Data-Driven ZC/SSC Threshold Calibration on Train

```python
from semg_baseline import ThresholdCalibrator

# Calibrate noise deadzone thresholds strictly using TRAIN resting windows
calibrator = ThresholdCalibrator(candidate_grid=[0.5, 1.0, 1.5, 2.0, 3.0])
thresholds = calibrator.calibrate(
    train_emg=train_windows,
    train_labels=train_labels,
    rest_label=0
)

print(f"Selected R multiplier: {thresholds.r_multiplier}")
print(f"Per-channel epsilon: {thresholds.epsilon_channels}")
```

---

## Scenario 3: Time-Domain Feature Extraction & Train-Only Z-Score

```python
from semg_baseline import TimeDomainFeatureExtractor, FeatureScaler

extractor = TimeDomainFeatureExtractor(epsilon=thresholds.epsilon_channels)
X_train_raw = extractor.extract(train_windows)
X_test_raw = extractor.extract(test_windows)

# Fit scaler strictly on TRAIN
scaler = FeatureScaler()
scaler.fit(X_train_raw)

X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)  # Stateless transform

print(f"Feature matrix shape: {X_train.shape}")  # (N_windows, 60)
assert not np.isnan(X_train).any()
```

---

## Scenario 4: Train Canonical LDA & Export Neutral Representation

```python
from semg_baseline import CanonicalLdaTrainer, NeutralModelExporter

trainer = CanonicalLdaTrainer(priors="uniform", solver="lsqr", shrinkage="auto")
lda_model = trainer.fit(X_train, train_labels)

# Export pure numerical parameters for Feature 005 (zero pickle/sklearn objects)
exporter = NeutralModelExporter(output_dir="artifacts/feature_004/jobs/job_demo")
manifest = exporter.export(
    model=lda_model,
    scaler=scaler,
    job_id="job_demo"
)

print(f"Exported neutral manifest: {manifest.manifest_hash}")
```

---

## Scenario 5: Equivalence Oracle Verification

```python
from semg_baseline import NeutralNumpyPredictor

# 1. Prediction using scikit-learn reference
y_pred_sklearn = lda_model.predict(X_test)
scores_sklearn = lda_model.decision_function(X_test)

# 2. Prediction using neutral NumPy predictor (Feature 005 runtime representation)
predictor = NeutralNumpyPredictor.load("artifacts/feature_004/jobs/job_demo")
y_pred_numpy, scores_numpy = predictor.predict(X_test)

# 3. Assert exact mathematical equivalence
np.testing.assert_array_equal(y_pred_sklearn, y_pred_numpy)
np.testing.assert_allclose(scores_sklearn, scores_numpy, atol=1e-7)
print("Equivalence oracle PASS: 100% bit-exact class prediction!")
```

---

## Scenario 6: Generate Full Evaluation Report

```python
from semg_baseline import EvaluationReportGenerator

reporter = EvaluationReportGenerator()
report = reporter.generate(
    y_true=test_labels,
    y_pred=y_pred_numpy,
    model_type="global_50_class",
    protocol_id="within_subject"
)

print(f"Top-1 Accuracy: {report.top1_accuracy:.4f}")
print(f"Balanced Accuracy: {report.balanced_accuracy:.4f}")
print(f"Macro-F1: {report.macro_f1:.4f}")
print(f"Active-only (1..49) Macro-F1: {report.active_only_metrics['macro_f1']:.4f}")
```

---

## Scenario 7: Exercise Confounding Diagnostic (`exercise_id_from_rest`)

```python
from semg_baseline import ExerciseConfoundingAuditor

auditor = ExerciseConfoundingAuditor()
confounding_report = auditor.audit(
    all_rest_windows=all_rest_windows,
    exercise_ids=exercise_ids_of_rest_windows,
    split_manifest=split_manifest
)

print(f"Rest-Exercise Classification Accuracy: {confounding_report.accuracy:.4f}")
print(f"Chance Level: {confounding_report.chance_level:.4f} (33.3%)")
print(f"Confounding Indicated: {confounding_report.confounding_indicated}")
print(f"Interpretation: {confounding_report.scientific_interpretation}")
```
