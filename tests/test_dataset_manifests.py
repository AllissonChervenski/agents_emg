"""Tests for Dataset and Split Manifest Generators, REST Auditing, and Bitwise Reproducibility."""

from pathlib import Path

import numpy as np


from semg_dataset.contract import (
    AlignmentRecord,
    AlignmentStatus,
    RecordingData,
)


def create_synthetic_recording(subject_id: str, exercise_id: str) -> RecordingData:
    """Create a minimal synthetic RecordingData for manifest generation testing."""
    n = 1000
    emg = np.random.randn(n, 12).astype(np.float32) * 0.001
    stim = np.zeros(n, dtype=np.int8)
    rep = np.zeros(n, dtype=np.int8)

    # 2 active repetition bursts
    stim[100:300] = 5
    rep[100:300] = 1
    stim[500:700] = 5
    rep[500:700] = 2

    restim = stim.copy()
    rerep = rep.copy()

    alignment = AlignmentRecord(
        status=AlignmentStatus.IDENTICAL,
        original_emg_length=n,
        original_refined_length=n,
        delta_samples=0,
        start_anchored_proven=True,
    )

    return RecordingData(
        subject_id=subject_id,
        exercise_id=exercise_id,
        emg=emg,
        stimulus=stim,
        repetition=rep,
        restimulus=restim,
        rerepetition=rerep,
        alignment=alignment,
    )


def test_generate_dataset_catalog_with_rest_audit(tmp_path: Path) -> None:
    """DatasetManifestGenerator must produce catalog with segregated REST sample counts and duration."""
    from semg_dataset.manifest import DatasetManifestGenerator

    rec1 = create_synthetic_recording("S1", "E1")
    rec2 = create_synthetic_recording("S1", "E2")

    generator = DatasetManifestGenerator()
    catalog_path = tmp_path / "dataset_catalog.json"
    catalog = generator.generate_catalog([rec1, rec2], output_path=catalog_path)

    assert catalog_path.is_file()
    assert "records" in catalog
    assert "rest_summary" in catalog
    assert len(catalog["records"]) == 2

    # Check REST audit metrics
    rest_sum = catalog["rest_summary"]
    assert rest_sum["total_samples"] == 2000
    assert rest_sum["active_samples"] == 800  # 200 * 2 per recording
    assert rest_sum["rest_samples"] == 1200   # 600 * 2 per recording
    assert rest_sum["rest_duration_seconds"] == 0.6  # 1200 / 2000 Hz
    assert rest_sum["active_duration_seconds"] == 0.4 # 800 / 2000 Hz

    # Check recording schema compliance
    rec_entry = catalog["records"][0]
    assert rec_entry["subject_id"] == "S1"
    assert rec_entry["exercise_id"] == "E1"
    assert rec_entry["num_samples"] == 1000
    assert rec_entry["num_channels"] == 12
    assert rec_entry["dtype"] == "float32"
    assert rec_entry["sampling_rate_hz"] == 2000.0
    assert "refined" in rec_entry["views"]
    assert "stimulus" in rec_entry["views"]


def test_generate_within_subject_manifest(tmp_path: Path) -> None:
    """SplitManifestGenerator must generate valid within-subject split JSON."""
    from semg_dataset.manifest import SplitManifestGenerator
    from semg_dataset.splits import WithinSubjectSplitter

    splitter = WithinSubjectSplitter()
    split_manifest = splitter.generate_splits()

    generator = SplitManifestGenerator()
    out_path = tmp_path / "splits_within_subject.json"
    data = generator.export_split_manifest(split_manifest, output_path=out_path)

    assert out_path.is_file()
    assert data["protocol"] == "within_subject"
    assert data["schema_version"] == "1.0"
    assert data["dataset_name"] == "NinaPro DB2"
    assert len(data["manifest_hash"]) == 64
    assert data["within_subject_parameters"]["train_repetitions"] == [1, 3, 4, 6]
    assert data["within_subject_parameters"]["test_repetitions"] == [2, 5]


def test_generate_cross_subject_manifest(tmp_path: Path) -> None:
    """SplitManifestGenerator must generate valid 5-fold cross-subject split JSON."""
    from semg_dataset.manifest import SplitManifestGenerator
    from semg_dataset.splits import CrossSubjectSplitter

    subjects = [f"S{i}" for i in range(1, 41)]
    splitter = CrossSubjectSplitter()
    split_manifest = splitter.generate_splits(subjects)

    generator = SplitManifestGenerator()
    out_path = tmp_path / "splits_cross_subject.json"
    data = generator.export_split_manifest(split_manifest, output_path=out_path)

    assert out_path.is_file()
    assert data["protocol"] == "cross_subject"
    assert len(data["folds"]) == 5
    assert len(data["manifest_hash"]) == 64

    for fold in data["folds"]:
        assert len(fold["train_subjects"]) == 24
        assert len(fold["val_subjects"]) == 8
        assert len(fold["test_subjects"]) == 8


def test_manifest_generation_is_bitwise_reproducible(tmp_path: Path) -> None:
    """Re-generating split manifests must produce identical content and identical manifest_hash."""
    from semg_dataset.manifest import SplitManifestGenerator
    from semg_dataset.splits import WithinSubjectSplitter

    splitter = WithinSubjectSplitter()
    m1 = splitter.generate_splits()
    m2 = splitter.generate_splits()

    gen = SplitManifestGenerator()
    p1 = tmp_path / "manifest1.json"
    p2 = tmp_path / "manifest2.json"

    d1 = gen.export_split_manifest(m1, output_path=p1, timestamp="2026-10-02T00:00:00Z", git_commit="deadbeef")
    d2 = gen.export_split_manifest(m2, output_path=p2, timestamp="2026-10-02T00:00:00Z", git_commit="deadbeef")

    assert d1["manifest_hash"] == d2["manifest_hash"]
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        assert f1.read() == f2.read()


def test_repetition_unit_rest_association() -> None:
    """Inter-repetition REST intervals must be deterministically associated to repetition units."""
    from semg_dataset.manifest import associate_rest_to_repetition_units

    # Sequence: 10 rest, 20 rep1, 10 rest, 20 rep2, 10 rest
    rep = np.zeros(70, dtype=np.int8)
    rep[10:30] = 1
    rep[40:60] = 2

    units = associate_rest_to_repetition_units(rep)
    assert len(units) == 70

    # Initial lead-in rest is unit 0
    assert (units[0:10] == 0).all()
    # Active rep 1 + subsequent inter-repetition rest is unit 1
    assert (units[10:30] == 1).all()
    assert (units[30:40] == 1).all()
    # Active rep 2 + subsequent trailing rest is unit 2
    assert (units[40:60] == 2).all()
    assert (units[60:70] == 2).all()
