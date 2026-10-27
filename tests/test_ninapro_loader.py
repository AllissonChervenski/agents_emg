"""Tests for NinaProDB2Loader in-memory read-only archive streaming using synthetic fixtures."""

import io
import zipfile
from pathlib import Path

import numpy as np
import scipy.io


def create_synthetic_ninapro_zip(zip_path: Path, subject_num: int = 1) -> None:
    """Create a minimal synthetic NinaPro-like ZIP archive without using real dataset samples."""
    n_samples = 200
    emg = np.random.randn(n_samples, 12).astype(np.float32) * 0.001
    stim = np.random.randint(0, 18, size=(n_samples, 1), dtype=np.int8)
    rep = np.random.randint(0, 7, size=(n_samples, 1), dtype=np.int8)

    mat_content = {
        "emg": emg,
        "stimulus": stim,
        "repetition": rep,
        "restimulus": stim.copy(),
        "rerepetition": rep.copy(),
        "subject": np.array([[subject_num]], dtype=np.uint8),
        "exercise": np.array([[1]], dtype=np.uint8),
    }

    mat_bytes = io.BytesIO()
    scipy.io.savemat(mat_bytes, mat_content)
    raw_mat_data = mat_bytes.getvalue()

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"DB2_s{subject_num}/S{subject_num}_E1_A1.mat", raw_mat_data)


def test_loader_discovers_subjects_and_recordings(tmp_path: Path) -> None:
    """Verify NinaProDB2Loader discovers subjects and recordings from synthetic ZIP archives."""
    from semg_dataset.loader import NinaProDB2Loader

    create_synthetic_ninapro_zip(tmp_path / "DB2_s1.zip", 1)
    create_synthetic_ninapro_zip(tmp_path / "DB2_s2.zip", 2)

    loader = NinaProDB2Loader(raw_dir=tmp_path)
    subjects = loader.list_subjects()
    assert subjects == ["S1", "S2"]

    recordings = loader.list_recordings()
    assert ("S1", "E1") in recordings
    assert ("S2", "E1") in recordings


def test_loader_streams_mat_in_memory_without_disk_extraction(tmp_path: Path) -> None:
    """Verify loader reads .mat files directly in memory, leaving raw directory unextracted."""
    from semg_dataset.loader import NinaProDB2Loader

    create_synthetic_ninapro_zip(tmp_path / "DB2_s1.zip", 1)
    loader = NinaProDB2Loader(raw_dir=tmp_path)

    recording = loader.load_recording("S1", "E1")
    assert recording.subject_id == "S1"
    assert recording.exercise_id == "E1"
    assert recording.emg.shape == (200, 12)
    assert recording.emg.dtype == np.float32

    # Verify no uncompressed .mat files exist on disk in the raw directory
    disk_files = list(tmp_path.rglob("*"))
    mat_on_disk = [p for p in disk_files if p.suffix == ".mat"]
    assert len(mat_on_disk) == 0, f"Detected uncompressed .mat files on disk: {mat_on_disk}"
