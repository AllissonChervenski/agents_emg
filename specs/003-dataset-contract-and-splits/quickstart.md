# Phase 1: Quickstart Validation Guide — Feature 003

## Prerequisites

- Python 3.10+
- Dependencies installed: `numpy`, `scipy`
- NinaPro DB2 raw archives in `data/raw/ninapro_db2/` (40 ZIP files, read-only)
- Manifest baseline in `data/manifests/ninapro_db2/`

---

## Scenario 1: Load a Subject Recording in Read-Only Mode

```python
from semg_dataset import NinaProDB2Loader

loader = NinaProDB2Loader(raw_dir="data/raw/ninapro_db2")

# Load Subject 1, Exercise 1
recording = loader.load_recording(subject_id="S1", exercise_id="E1")

print(recording.subject_id)         # 'S1'
print(recording.exercise_id)        # 'E1'
print(recording.emg.shape)          # (N, 12)
print(recording.emg.dtype)          # float32
print(recording.sampling_rate_hz)   # 2000.0
print(recording.alignment.status)   # AlignmentStatus.IDENTICAL
```

---

## Scenario 2: PROVE-OR-QUARANTINE Alignment Verification (Outlier S12)

```python
# Load Subject 12, Exercise 3 (the 272-sample outlier recording)
recording = loader.load_recording(subject_id="S12", exercise_id="E3")

print(recording.alignment.status)
# Output: AlignmentStatus.ANCHOR_START_TRUNCATE_TAIL
print(recording.alignment.delta_samples)
# Output: 272
print(recording.alignment.start_anchored_proven)
# Output: True
assert len(recording.emg) == len(recording.restimulus)
```

---

## Scenario 3: Frozen Dataset Views (Refined vs Stimulus)

```python
# Default view: refined ground truth (restimulus + rerepetition)
view_refined = recording.get_view("refined")
print(view_refined.view_id)       # 'refined'
print(view_refined.labels[:10])   # restimulus slice
print(view_refined.view_hash)     # SHA-256 digest

# Secondary view: prescribed visual stimulus (stimulus + repetition)
view_stimulus = recording.get_view("stimulus")
print(view_stimulus.view_id)      # 'stimulus'
print(view_stimulus.labels[:10])  # stimulus slice
```

---

## Scenario 4: Generate Within-Subject Repetition Splits

```python
from semg_dataset import WithinSubjectSplitter

splitter = WithinSubjectSplitter()
split_manifest = splitter.generate_splits(loader.list_recordings())

print(split_manifest.protocol)                     # 'within_subject'
print(split_manifest.train_repetitions)            # [1, 3, 4, 6]
print(split_manifest.test_repetitions)             # [2, 5]
print(len(split_manifest.val_repetitions))         # 0 (deferred to Feature 004)
```

---

## Scenario 5: Generate Cross-Subject 5-Fold Splits

```python
from semg_dataset import CrossSubjectSplitter

splitter = CrossSubjectSplitter()
cross_manifest = splitter.generate_splits(loader.list_subjects())

print(cross_manifest.protocol)         # 'cross_subject'
print(len(cross_manifest.folds))       # 5
for fold in cross_manifest.folds:
    print(f"Fold {fold.fold_id}: {len(fold.train_subjects)} train, "
          f"{len(fold.val_subjects)} val, {len(fold.test_subjects)} test")
    # Output: 24 train, 8 val, 8 test
    assert len(set(fold.train_subjects) & set(fold.test_subjects)) == 0
```

---

## Scenario 6: Verify Provenance & Manifest Integrity

```python
from semg_dataset import NinaProProvenance

provenance = NinaProProvenance.get_metadata()
print(provenance.citation)
# Atzori et al. (2014). Scientific Data, 1, 140053.
print(provenance.official_url)
# https://ninapro.hevs.ch/instructions/DB2.html
assert "TO_BE_DOCUMENTED" not in str(provenance)
```
