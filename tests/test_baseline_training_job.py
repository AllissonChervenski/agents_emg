"""Unit tests for TrainingJobExecutor, state-aware schema validation, and append-only jobs (T013).

Covers FR-003, FR-011.1, FR-011.2, FR-011.3, FR-011.4, SC-001, SC-006, Decision-5:
- Deterministic job_fingerprint calculation excluding volatile execution fields.
- State-aware lifecycle validation against training_job_schema.json (PLANNED, RUNNING, SUCCEEDED, FAILED).
- Append-only job persistence in artifacts/feature_004/jobs/<job_id>/.
- Cryptographic integrity sealing via manifest.sha256 sidecar.
- Detection of tampered job artifacts and fail-closed verification.
- Zero scikit-learn import in semg_baseline/runner.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from semg_baseline.export import NeutralModelExporter
from semg_baseline.filter import CausalBandpassFilter
from semg_baseline.lineage import ScientificLineage
from semg_baseline.runner import (
    AppendOnlyViolation,
    TrainingJob,
    TrainingJobExecutor,
    TrainingJobStatus,
    compute_dependency_lock_hash,
    compute_job_fingerprint,
    create_canonical_feature_config,
    create_canonical_model_config,
    get_environment_versions,
    validate_training_job_dict,
    verify_job_directory,
)
from semg_baseline.scaling import FeatureScaler
from semg_baseline.thresholds import ThresholdCalibration


REPO_ROOT = Path(__file__).resolve().parent.parent


def _dummy_lineage() -> ScientificLineage:
    return ScientificLineage(
        raw_dataset_manifest_hash="a" * 64,
        dataset_view_hash="b" * 64,
        alignment_hash="c" * 64,
        split_manifest_hash="d" * 64,
        acceptance_package_hash="e" * 64,
        protocol_id="within_subject",
        partition_id="S1",
        git_commit="abcdef1234567890abcdef1234567890abcdef12",
    )


def _dummy_prep_config() -> dict:
    flt = CausalBandpassFilter()
    return flt.get_config().to_dict()


def _dummy_feature_config() -> dict:
    return create_canonical_feature_config()


def _dummy_model_config() -> dict:
    return create_canonical_model_config(
        vocabulary_type="global_50_class",
        prior_type="uniform",
        num_classes=50,
    )


def test_dependency_lock_hash_computes_sha256() -> None:
    """Dependency lock hash must compute a 64-char hex digest from repo pyproject.toml."""
    lock_hash = compute_dependency_lock_hash(REPO_ROOT)
    assert isinstance(lock_hash, str)
    assert len(lock_hash) == 64
    assert all(c in "0123456789abcdef" for c in lock_hash)


def test_job_fingerprint_deterministic_and_excludes_volatile_fields() -> None:
    """job_fingerprint must be identical for identical scientific inputs and ignore volatile metadata."""
    lineage = _dummy_lineage()
    prep = _dummy_prep_config()
    feat = _dummy_feature_config()
    model = _dummy_model_config()
    lock_hash = "f" * 64

    fp1 = compute_job_fingerprint(
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config_hash=prep["config_hash"],
        feature_config_hash=feat["config_hash"],
        model_config_hash=model["config_hash"],
        seed=42,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
    )

    fp2 = compute_job_fingerprint(
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config_hash=prep["config_hash"],
        feature_config_hash=feat["config_hash"],
        model_config_hash=model["config_hash"],
        seed=42,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
    )

    assert fp1 == fp2
    assert len(fp1) == 64

    # Different seed must change fingerprint
    fp_diff_seed = compute_job_fingerprint(
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config_hash=prep["config_hash"],
        feature_config_hash=feat["config_hash"],
        model_config_hash=model["config_hash"],
        seed=43,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
    )
    assert fp_diff_seed != fp1

    # Different partition must change fingerprint
    fp_diff_partition = compute_job_fingerprint(
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S2",
        preprocessing_config_hash=prep["config_hash"],
        feature_config_hash=feat["config_hash"],
        model_config_hash=model["config_hash"],
        seed=42,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
    )
    assert fp_diff_partition != fp1


def test_schema_validation_planned_job() -> None:
    """PLANNED job must pass schema validation without runtime execution metadata."""
    lineage = _dummy_lineage()
    prep = _dummy_prep_config()
    feat = _dummy_feature_config()
    model = _dummy_model_config()
    lock_hash = "1" * 64

    fp = compute_job_fingerprint(
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config_hash=prep["config_hash"],
        feature_config_hash=feat["config_hash"],
        model_config_hash=model["config_hash"],
        seed=42,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
    )

    job = TrainingJob(
        job_id="job_ws_S1_test",
        job_fingerprint=fp,
        status=TrainingJobStatus.PLANNED,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config=prep,
        feature_config=feat,
        model_config=model,
        seed=42,
    )

    job_dict = job.to_dict()
    is_valid, errors = validate_training_job_dict(job_dict)
    assert is_valid, f"Validation errors: {errors}"
    assert errors == []


def test_schema_validation_running_job() -> None:
    """RUNNING job requires started_at and environment_versions."""
    lineage = _dummy_lineage()
    prep = _dummy_prep_config()
    feat = _dummy_feature_config()
    model = _dummy_model_config()
    lock_hash = "1" * 64

    fp = compute_job_fingerprint(
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config_hash=prep["config_hash"],
        feature_config_hash=feat["config_hash"],
        model_config_hash=model["config_hash"],
        seed=42,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
    )

    job = TrainingJob(
        job_id="job_ws_S1_test",
        job_fingerprint=fp,
        status=TrainingJobStatus.RUNNING,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config=prep,
        feature_config=feat,
        model_config=model,
        seed=42,
        started_at="2026-10-06T00:00:00Z",
        environment_versions=get_environment_versions(),
    )

    is_valid, errors = validate_training_job_dict(job.to_dict())
    assert is_valid, f"Errors: {errors}"

    # Missing started_at must fail RUNNING validation
    bad_dict = job.to_dict()
    bad_dict.pop("started_at")
    bad_valid, bad_errors = validate_training_job_dict(bad_dict)
    assert not bad_valid
    assert any("started_at" in err for err in bad_errors)


def test_schema_validation_failed_job() -> None:
    """FAILED job requires stderr_summary and environment_versions."""
    lineage = _dummy_lineage()
    prep = _dummy_prep_config()
    feat = _dummy_feature_config()
    model = _dummy_model_config()
    lock_hash = "1" * 64

    fp = compute_job_fingerprint(
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config_hash=prep["config_hash"],
        feature_config_hash=feat["config_hash"],
        model_config_hash=model["config_hash"],
        seed=42,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
    )

    job = TrainingJob(
        job_id="job_ws_S1_test",
        job_fingerprint=fp,
        status=TrainingJobStatus.FAILED,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config=prep,
        feature_config=feat,
        model_config=model,
        seed=42,
        started_at="2026-10-06T00:00:00Z",
        finished_at="2026-10-06T00:01:00Z",
        environment_versions=get_environment_versions(),
        stderr_summary="RuntimeError: ill-conditioned matrix",
    )

    is_valid, errors = validate_training_job_dict(job.to_dict())
    assert is_valid, f"Errors: {errors}"

    # Missing stderr_summary must fail FAILED status
    bad_dict = job.to_dict()
    bad_dict.pop("stderr_summary")
    bad_valid, bad_errors = validate_training_job_dict(bad_dict)
    assert not bad_valid
    assert any("stderr_summary" in err for err in bad_errors)


def test_schema_validation_succeeded_job() -> None:
    """SUCCEEDED job requires full metrics, timings, artifact hashes, and calibration."""
    lineage = _dummy_lineage()
    prep = _dummy_prep_config()
    feat = _dummy_feature_config()
    model = _dummy_model_config()
    lock_hash = "1" * 64

    fp = compute_job_fingerprint(
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config_hash=prep["config_hash"],
        feature_config_hash=feat["config_hash"],
        model_config_hash=model["config_hash"],
        seed=42,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
    )

    calib = ThresholdCalibration(
        r_multiplier=1.5,
        rms_rest_train=np.full(12, 0.05),
        epsilon_channels=np.full(12, 0.075),
    )

    job = TrainingJob(
        job_id="job_ws_S1_test",
        job_fingerprint=fp,
        status=TrainingJobStatus.SUCCEEDED,
        git_commit=lineage.git_commit,
        dependency_lock_hash=lock_hash,
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config=prep,
        feature_config=feat,
        model_config=model,
        seed=42,
        started_at="2026-10-06T00:00:00Z",
        finished_at="2026-10-06T00:02:00Z",
        environment_versions=get_environment_versions(),
        threshold_calibration=calib.to_dict(),
        timing_seconds={
            "preprocessing": 0.5,
            "feature_extraction": 1.2,
            "fitting": 0.3,
            "evaluation": 0.4,
            "total": 2.4,
        },
        metrics={
            "top1_accuracy": 0.85,
            "balanced_accuracy": 0.82,
            "macro_f1": 0.81,
        },
        artifact_hashes={
            "model_params_npz": "2" * 64,
            "model_manifest_json": "3" * 64,
            "stdout_log": "4" * 64,
            "stderr_log": "5" * 64,
            "metrics_json": "6" * 64,
        },
        stdout_summary="Training completed successfully in 2.4s",
    )

    is_valid, errors = validate_training_job_dict(job.to_dict())
    assert is_valid, f"Errors: {errors}"

    # Invalid metric out of bounds [0, 1] must fail
    bad_dict = job.to_dict()
    bad_dict["metrics"]["top1_accuracy"] = 1.5
    bad_valid, bad_errors = validate_training_job_dict(bad_dict)
    assert not bad_valid


def test_append_only_executor_persistence_and_manifest(tmp_path: Path) -> None:
    """Executor must persist all job files and seal directory with manifest.sha256."""
    jobs_dir = tmp_path / "artifacts" / "feature_004" / "jobs"
    executor = TrainingJobExecutor(artifacts_dir=jobs_dir, repo_root=REPO_ROOT)

    lineage = _dummy_lineage()
    prep = _dummy_prep_config()
    feat = _dummy_feature_config()
    model = _dummy_model_config()

    job = executor.create_job(
        job_id="job_ws_S1_001",
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config=prep,
        feature_config=feat,
        model_config=model,
        seed=42,
    )

    assert job.status == TrainingJobStatus.PLANNED

    # Mock trained model and scaler for export
    class MockModel:
        classes_ = np.arange(50, dtype=np.int32)
        coef_ = np.ones((50, 60), dtype=np.float64)
        intercept_ = np.zeros(50, dtype=np.float64)

    scaler = FeatureScaler()
    scaler.mean_train = np.zeros(60, dtype=np.float64)
    scaler.scale_train = np.ones(60, dtype=np.float64)
    scaler.zero_variance_indices = []

    calib = ThresholdCalibration(
        r_multiplier=1.5,
        rms_rest_train=np.full(12, 0.05),
        epsilon_channels=np.full(12, 0.075),
    )

    job_dir = executor.finalize_succeeded(
        job=job,
        model=MockModel(),
        scaler=scaler,
        threshold_calibration=calib,
        metrics={
            "top1_accuracy": 0.88,
            "balanced_accuracy": 0.85,
            "macro_f1": 0.84,
        },
        timing_seconds={
            "preprocessing": 0.4,
            "feature_extraction": 1.1,
            "fitting": 0.2,
            "evaluation": 0.3,
            "total": 2.0,
        },
        stdout_log="Everything ran smoothly.\n",
        stderr_log="",
    )

    assert job_dir.is_dir()
    assert isinstance(executor.exporter, NeutralModelExporter)

    # Check files layout
    expected_files = [
        "job.json",
        "config.json",
        "model_manifest.json",
        "model_params.npz",
        "metrics.json",
        "environment.json",
        "stdout.log",
        "stderr.log",
        "manifest.sha256",
    ]
    for fn in expected_files:
        assert (job_dir / fn).is_file(), f"Missing expected job artifact: {fn}"

    # Verify sealed integrity
    is_valid, errors = verify_job_directory(job_dir)
    assert is_valid, f"Verification failed: {errors}"
    assert errors == []


def test_append_only_prevents_overwrite(tmp_path: Path) -> None:
    """Attempting to finalize an already completed job must raise AppendOnlyViolation."""
    jobs_dir = tmp_path / "artifacts" / "feature_004" / "jobs"
    executor = TrainingJobExecutor(artifacts_dir=jobs_dir, repo_root=REPO_ROOT)

    lineage = _dummy_lineage()
    prep = _dummy_prep_config()
    feat = _dummy_feature_config()
    model = _dummy_model_config()

    job = executor.create_job(
        job_id="job_ws_S1_002",
        lineage=lineage,
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config=prep,
        feature_config=feat,
        model_config=model,
        seed=42,
    )

    class MockModel:
        classes_ = np.arange(50, dtype=np.int32)
        coef_ = np.ones((50, 60), dtype=np.float64)
        intercept_ = np.zeros(50, dtype=np.float64)

    scaler = FeatureScaler()
    scaler.mean_train = np.zeros(60, dtype=np.float64)
    scaler.scale_train = np.ones(60, dtype=np.float64)
    scaler.zero_variance_indices = []

    calib = ThresholdCalibration(
        r_multiplier=1.5,
        rms_rest_train=np.full(12, 0.05),
        epsilon_channels=np.full(12, 0.075),
    )

    executor.finalize_succeeded(
        job=job,
        model=MockModel(),
        scaler=scaler,
        threshold_calibration=calib,
        metrics={
            "top1_accuracy": 0.80,
            "balanced_accuracy": 0.80,
            "macro_f1": 0.80,
        },
        timing_seconds={"total": 1.0},
    )

    # Second finalize attempt must fail-closed
    with pytest.raises(AppendOnlyViolation, match="already exists and is sealed"):
        executor.finalize_succeeded(
            job=job,
            model=MockModel(),
            scaler=scaler,
            threshold_calibration=calib,
            metrics={
                "top1_accuracy": 0.80,
                "balanced_accuracy": 0.80,
                "macro_f1": 0.80,
            },
            timing_seconds={"total": 1.0},
        )


def test_verify_job_detects_tampering(tmp_path: Path) -> None:
    """Modifying any artifact in a sealed job directory must trigger verification failure."""
    jobs_dir = tmp_path / "artifacts" / "feature_004" / "jobs"
    executor = TrainingJobExecutor(artifacts_dir=jobs_dir, repo_root=REPO_ROOT)

    job = executor.create_job(
        job_id="job_ws_S1_003",
        lineage=_dummy_lineage(),
        protocol_id="within_subject",
        partition_id="S1",
        preprocessing_config=_dummy_prep_config(),
        feature_config=_dummy_feature_config(),
        model_config=_dummy_model_config(),
        seed=42,
    )

    class MockModel:
        classes_ = np.arange(50, dtype=np.int32)
        coef_ = np.ones((50, 60), dtype=np.float64)
        intercept_ = np.zeros(50, dtype=np.float64)

    scaler = FeatureScaler()
    scaler.mean_train = np.zeros(60, dtype=np.float64)
    scaler.scale_train = np.ones(60, dtype=np.float64)
    scaler.zero_variance_indices = []

    calib = ThresholdCalibration(
        r_multiplier=1.5,
        rms_rest_train=np.full(12, 0.05),
        epsilon_channels=np.full(12, 0.075),
    )

    job_dir = executor.finalize_succeeded(
        job=job,
        model=MockModel(),
        scaler=scaler,
        threshold_calibration=calib,
        metrics={
            "top1_accuracy": 0.80,
            "balanced_accuracy": 0.80,
            "macro_f1": 0.80,
        },
        timing_seconds={"total": 1.0},
    )

    # Tamper with metrics.json
    metrics_file = job_dir / "metrics.json"
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    data["top1_accuracy"] = 0.9999
    metrics_file.write_text(json.dumps(data), encoding="utf-8")

    is_valid, errors = verify_job_directory(job_dir)
    assert not is_valid
    assert any("metrics.json" in err and "mismatch" in err for err in errors)


def test_runner_does_not_import_sklearn() -> None:
    """Importing semg_baseline.runner must never import scikit-learn in an isolated process."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; import semg_baseline.runner; assert 'sklearn' not in sys.modules, f'Leaked: {[m for m in sys.modules if m.startswith(\"sklearn\")]}'",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0
