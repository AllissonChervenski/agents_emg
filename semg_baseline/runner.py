"""Append-only TrainingJobExecutor, state-aware schema validation, and provenance sealing.

Implements FR-003, FR-011.1, FR-011.2, FR-011.3, FR-011.4, SC-001, SC-006, Decision-5:
- Immutable, append-only TrainingJob execution records under artifacts/feature_004/jobs/<job_id>/.
- State-aware JSON schema validation across PLANNED, RUNNING, SUCCEEDED, and FAILED states.
- Deterministic job_fingerprint computation strictly excluding volatile execution fields.
- Cryptographic integrity sealing via manifest.sha256 sidecar.
- Neutral model parameter export boundary integration.
- Host-only zero scikit-learn import compliance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

import jsonschema
import numpy as np

from semg_baseline.export import NeutralModelExporter
from semg_baseline.lineage import ScientificLineage, compute_file_sha256


class AppendOnlyViolation(Exception):
    """Raised when an attempt is made to overwrite an existing sealed job directory."""


class TrainingJobStatus(str, Enum):
    """Lifecycle states of a TrainingJob."""

    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


def compute_dependency_lock_hash(repo_root: Path | str | None = None) -> str:
    """Compute deterministic SHA-256 digest of repository dependency specifications."""
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parent.parent
    )
    for lock_name in ("uv.lock", "poetry.lock", "requirements.lock", "pyproject.toml"):
        candidate = root / lock_name
        if candidate.is_file():
            return compute_file_sha256(candidate)

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        return compute_file_sha256(pyproject)

    raise FileNotFoundError(
        f"No dependency specification or lock file found in repo root: {root}"
    )


def create_canonical_feature_config(
    window_len: int = 400,
    stride: int = 100,
    decision_rate_hz: float = 20.0,
    feature_names: list[str] | None = None,
    num_channels: int = 12,
) -> dict[str, Any]:
    """Create canonical feature extraction configuration dict matching schema."""
    names = feature_names or ["MAV", "RMS", "WL", "ZC", "SSC"]
    total_features = num_channels * len(names)

    canonical_body = {
        "window_length_samples": int(window_len),
        "stride_samples": int(stride),
        "decision_rate_hz": float(decision_rate_hz),
        "feature_names": names,
        "transition_policy": "pure_window_steady_state",
        "num_channels": int(num_channels),
        "total_features_per_window": int(total_features),
    }

    raw_json = json.dumps(canonical_body, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    canonical_body["config_hash"] = config_hash
    return canonical_body


def create_canonical_model_config(
    model_family: str = "regularized_lda",
    solver: str = "lsqr",
    shrinkage: str = "auto",
    prior_type: str = "uniform",
    num_classes: int = 50,
    vocabulary_type: str = "global_50_class",
) -> dict[str, Any]:
    """Create canonical model configuration dict matching schema."""
    canonical_body = {
        "model_family": model_family,
        "solver": solver,
        "shrinkage": shrinkage,
        "prior_type": prior_type,
        "num_classes": int(num_classes),
        "vocabulary_type": vocabulary_type,
    }

    raw_json = json.dumps(canonical_body, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    canonical_body["config_hash"] = config_hash
    return canonical_body


def compute_job_fingerprint(
    lineage: ScientificLineage,
    protocol_id: str,
    partition_id: str,
    preprocessing_config_hash: str,
    feature_config_hash: str,
    model_config_hash: str,
    seed: int,
    git_commit: str,
    dependency_lock_hash: str,
) -> str:
    """Compute deterministic SHA-256 fingerprint from immutable scientific inputs.

    Volatile execution fields (timestamps, process IDs, file paths, logs, timings)
    are strictly excluded.
    """
    canonical_inputs = {
        "lineage": lineage.to_dict(),
        "protocol_id": protocol_id,
        "partition_id": partition_id,
        "preprocessing_config_hash": preprocessing_config_hash,
        "feature_config_hash": feature_config_hash,
        "model_config_hash": model_config_hash,
        "seed": int(seed),
        "git_commit": git_commit,
        "dependency_lock_hash": dependency_lock_hash,
    }

    raw_json = json.dumps(canonical_inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()


def get_environment_versions() -> dict[str, str]:
    """Retrieve runtime dependency versions conforming to EnvironmentVersions schema."""
    try:
        sklearn_ver = importlib.metadata.version("scikit-learn")
    except Exception:
        sklearn_ver = "not_installed"

    try:
        import scipy
        scipy_ver = getattr(scipy, "__version__", "unknown")
    except Exception:
        scipy_ver = "unknown"

    return {
        "python": sys.version.split()[0],
        "numpy": getattr(np, "__version__", "unknown"),
        "scipy": scipy_ver,
        "scikit_learn": sklearn_ver,
    }


def load_training_job_schema(schema_path: Path | str | None = None) -> dict[str, Any]:
    """Load authoritative training_job_schema.json from contracts."""
    if schema_path is not None:
        p = Path(schema_path).resolve()
    else:
        root = Path(__file__).resolve().parent.parent
        p = (
            root
            / "specs"
            / "004-baseline-training-and-evaluation"
            / "contracts"
            / "training_job_schema.json"
        )

    if not p.is_file():
        raise FileNotFoundError(f"training_job_schema.json not found at: {p}")

    return json.loads(p.read_text(encoding="utf-8"))


def validate_training_job_dict(
    job_dict: dict[str, Any], schema: dict[str, Any] | None = None
) -> tuple[bool, list[str]]:
    """Validate a serialized TrainingJob dictionary against training_job_schema.json."""
    loaded_schema = schema or load_training_job_schema()
    validator = jsonschema.Draft202012Validator(loaded_schema)
    errors: list[str] = []

    for err in sorted(validator.iter_errors(job_dict), key=lambda e: e.path):
        loc = ".".join(str(p) for p in err.path) if err.path else "root"
        errors.append(f"[{loc}] {err.message}")

    return len(errors) == 0, errors


@dataclass
class TrainingJob:
    """Represents an immutable scientific training and evaluation run."""

    job_id: str
    job_fingerprint: str
    status: TrainingJobStatus
    git_commit: str
    dependency_lock_hash: str
    lineage: ScientificLineage
    protocol_id: str
    partition_id: str
    preprocessing_config: dict[str, Any]
    feature_config: dict[str, Any]
    model_config: dict[str, Any]
    seed: int
    started_at: str | None = None
    finished_at: str | None = None
    environment_versions: dict[str, str] | None = None
    threshold_calibration: dict[str, Any] | None = None
    timing_seconds: dict[str, float] | None = None
    metrics: dict[str, float] | None = None
    artifact_hashes: dict[str, str] | None = None
    stdout_summary: str | None = None
    stderr_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for schema validation and job.json."""
        d: dict[str, Any] = {
            "job_id": self.job_id,
            "job_fingerprint": self.job_fingerprint,
            "status": self.status.value if isinstance(self.status, TrainingJobStatus) else self.status,
            "git_commit": self.git_commit,
            "dependency_lock_hash": self.dependency_lock_hash,
            "lineage": self.lineage.to_dict() if hasattr(self.lineage, "to_dict") else self.lineage,
            "protocol_id": self.protocol_id,
            "partition_id": self.partition_id,
            "preprocessing_config": self.preprocessing_config,
            "feature_config": self.feature_config,
            "model_config": self.model_config,
            "seed": self.seed,
        }

        if self.started_at is not None:
            d["started_at"] = self.started_at
        if self.finished_at is not None:
            d["finished_at"] = self.finished_at
        if self.environment_versions is not None:
            d["environment_versions"] = self.environment_versions
        if self.threshold_calibration is not None:
            d["threshold_calibration"] = self.threshold_calibration
        if self.timing_seconds is not None:
            d["timing_seconds"] = self.timing_seconds
        if self.metrics is not None:
            d["metrics"] = self.metrics
        if self.artifact_hashes is not None:
            d["artifact_hashes"] = self.artifact_hashes
        if self.stdout_summary is not None:
            d["stdout_summary"] = self.stdout_summary
        if self.stderr_summary is not None:
            d["stderr_summary"] = self.stderr_summary

        return d


def write_manifest_sha256(job_dir: Path) -> Path:
    """Compute SHA-256 for all artifacts in job directory and write manifest.sha256 sidecar."""
    manifest_lines: list[str] = []
    manifest_path = job_dir / "manifest.sha256"

    for file_path in sorted(job_dir.iterdir()):
        if file_path.name == "manifest.sha256" or file_path.is_dir():
            continue
        sha = compute_file_sha256(file_path)
        manifest_lines.append(f"{sha}  {file_path.name}")

    content = "\n".join(manifest_lines) + "\n"
    manifest_path.write_text(content, encoding="utf-8")
    return manifest_path


def verify_job_directory(job_dir: Path | str) -> tuple[bool, list[str]]:
    """Cryptographically verify sealed job directory integrity and schema validity."""
    path = Path(job_dir).resolve()
    errors: list[str] = []

    if not path.is_dir():
        return False, [f"Job directory not found: {path}"]

    manifest_file = path / "manifest.sha256"
    if not manifest_file.is_file():
        return False, [f"Missing sealed manifest: {manifest_file}"]

    # 1. Verify file checksums
    content = manifest_file.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected_sha, fname = parts[0], parts[1].strip()
        target_file = path / fname
        if not target_file.is_file():
            errors.append(f"Referenced artifact missing: {target_file}")
            continue
        actual_sha = compute_file_sha256(target_file)
        if actual_sha != expected_sha:
            errors.append(
                f"Checksum mismatch for '{fname}': expected {expected_sha}, got {actual_sha}"
            )

    # 2. Verify schema compliance of job.json
    job_file = path / "job.json"
    if job_file.is_file():
        try:
            job_dict = json.loads(job_file.read_text(encoding="utf-8"))
            is_valid, schema_errors = validate_training_job_dict(job_dict)
            if not is_valid:
                errors.extend(schema_errors)
        except Exception as e:
            errors.append(f"Failed to parse or validate job.json: {e}")
    else:
        errors.append("job.json missing from sealed directory.")

    return len(errors) == 0, errors


class TrainingJobExecutor:
    """Manages append-only execution, schema validation, and artifact sealing for TrainingJobs."""

    def __init__(
        self,
        artifacts_dir: Path | str | None = None,
        repo_root: Path | str | None = None,
    ) -> None:
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parent.parent
        )
        self.artifacts_dir = (
            Path(artifacts_dir).resolve()
            if artifacts_dir is not None
            else self.repo_root / "artifacts" / "feature_004" / "jobs"
        )
        self.exporter = NeutralModelExporter()

    def get_job_dir(self, job_id: str) -> Path:
        """Return the destination path for a given job_id."""
        return self.artifacts_dir / job_id

    def create_job(
        self,
        job_id: str,
        lineage: ScientificLineage,
        protocol_id: str,
        partition_id: str,
        preprocessing_config: dict[str, Any],
        feature_config: dict[str, Any],
        model_config: dict[str, Any],
        seed: int = 42,
        git_commit: str | None = None,
        dependency_lock_hash: str | None = None,
    ) -> TrainingJob:
        """Create a new TrainingJob in PLANNED state."""
        commit = git_commit or lineage.git_commit or "unknown"
        lock_hash = dependency_lock_hash or compute_dependency_lock_hash(self.repo_root)

        fp = compute_job_fingerprint(
            lineage=lineage,
            protocol_id=protocol_id,
            partition_id=partition_id,
            preprocessing_config_hash=preprocessing_config["config_hash"],
            feature_config_hash=feature_config["config_hash"],
            model_config_hash=model_config["config_hash"],
            seed=seed,
            git_commit=commit,
            dependency_lock_hash=lock_hash,
        )

        job = TrainingJob(
            job_id=job_id,
            job_fingerprint=fp,
            status=TrainingJobStatus.PLANNED,
            git_commit=commit,
            dependency_lock_hash=lock_hash,
            lineage=lineage,
            protocol_id=protocol_id,
            partition_id=partition_id,
            preprocessing_config=preprocessing_config,
            feature_config=feature_config,
            model_config=model_config,
            seed=seed,
        )

        is_valid, errors = validate_training_job_dict(job.to_dict())
        if not is_valid:
            raise ValueError(f"PLANNED job failed schema validation: {errors}")

        return job

    def finalize_succeeded(
        self,
        job: TrainingJob,
        model: Any,
        scaler: Any,
        threshold_calibration: Any,
        metrics: dict[str, float],
        timing_seconds: dict[str, float] | None = None,
        stdout_log: str = "",
        stderr_log: str = "",
    ) -> Path:
        """Export neutral model, write artifacts, validate SUCCEEDED schema, and seal directory."""
        job_dir = self.get_job_dir(job.job_id)

        # Append-only check: prevent overwriting completed sealed job
        if (job_dir / "manifest.sha256").is_file():
            raise AppendOnlyViolation(
                f"Job directory '{job_dir}' already exists and is sealed with manifest.sha256."
            )

        job_dir.mkdir(parents=True, exist_ok=True)

        now_iso = datetime.now(timezone.utc).isoformat()
        if job.started_at is None:
            job.started_at = now_iso
        job.finished_at = now_iso

        # 1. Export neutral model parameters
        taxonomy = job.model_config.get("vocabulary_type", "global_50_class")
        priors = job.model_config.get("prior_type", "uniform")
        self.exporter.export(
            model=model,
            scaler=scaler,
            job_id=job.job_id,
            job_fingerprint=job.job_fingerprint,
            model_taxonomy=taxonomy,
            priors_type=priors,
            output_dir=job_dir,
        )

        # 2. Write logs
        stdout_file = job_dir / "stdout.log"
        stdout_file.write_text(stdout_log, encoding="utf-8")
        stderr_file = job_dir / "stderr.log"
        stderr_file.write_text(stderr_log, encoding="utf-8")

        # 3. Write config and environment
        env_versions = get_environment_versions()
        env_file = job_dir / "environment.json"
        env_file.write_text(json.dumps(env_versions, indent=2), encoding="utf-8")

        config_data = {
            "job_id": job.job_id,
            "job_fingerprint": job.job_fingerprint,
            "protocol_id": job.protocol_id,
            "partition_id": job.partition_id,
            "seed": job.seed,
            "lineage": job.lineage.to_dict() if hasattr(job.lineage, "to_dict") else job.lineage,
            "preprocessing_config": job.preprocessing_config,
            "feature_config": job.feature_config,
            "model_config": job.model_config,
            "git_commit": job.git_commit,
            "dependency_lock_hash": job.dependency_lock_hash,
        }
        config_file = job_dir / "config.json"
        config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        # 4. Write metrics
        metrics_file = job_dir / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        # 5. Build artifact hashes
        artifact_hashes = {
            "model_params_npz": compute_file_sha256(job_dir / "model_params.npz"),
            "model_manifest_json": compute_file_sha256(job_dir / "model_manifest.json"),
            "stdout_log": compute_file_sha256(stdout_file),
            "stderr_log": compute_file_sha256(stderr_file),
            "metrics_json": compute_file_sha256(metrics_file),
        }

        # 6. Normalize calibration dict
        calib_dict = (
            threshold_calibration.to_dict()
            if hasattr(threshold_calibration, "to_dict")
            else dict(threshold_calibration)
        )

        # Ensure metrics contains only schema-allowed keys for JobMetrics
        canonical_job_metrics = {
            "top1_accuracy": float(metrics["top1_accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        }

        # Update job fields
        job.status = TrainingJobStatus.SUCCEEDED
        job.environment_versions = env_versions
        job.threshold_calibration = calib_dict
        job.timing_seconds = timing_seconds or {"total": 0.0}
        job.metrics = canonical_job_metrics
        job.artifact_hashes = artifact_hashes
        job.stdout_summary = stdout_log[:500] if stdout_log else "Completed successfully."

        # Validate against schema
        is_valid, errors = validate_training_job_dict(job.to_dict())
        if not is_valid:
            raise ValueError(f"SUCCEEDED job failed schema validation: {errors}")

        # 7. Write job.json
        job_file = job_dir / "job.json"
        job_file.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

        # 8. Seal directory with manifest.sha256
        write_manifest_sha256(job_dir)

        return job_dir

    def finalize_failed(
        self,
        job: TrainingJob,
        stderr_summary: str,
        stdout_log: str = "",
        stderr_log: str = "",
    ) -> Path:
        """Record diagnostic failure, validate FAILED schema, and seal directory."""
        job_dir = self.get_job_dir(job.job_id)

        if (job_dir / "manifest.sha256").is_file():
            raise AppendOnlyViolation(
                f"Job directory '{job_dir}' already exists and is sealed with manifest.sha256."
            )

        job_dir.mkdir(parents=True, exist_ok=True)

        now_iso = datetime.now(timezone.utc).isoformat()
        if job.started_at is None:
            job.started_at = now_iso
        job.finished_at = now_iso

        # Write logs
        stdout_file = job_dir / "stdout.log"
        stdout_file.write_text(stdout_log, encoding="utf-8")
        stderr_file = job_dir / "stderr.log"
        stderr_file.write_text(stderr_log or stderr_summary, encoding="utf-8")

        env_versions = get_environment_versions()
        env_file = job_dir / "environment.json"
        env_file.write_text(json.dumps(env_versions, indent=2), encoding="utf-8")

        config_data = {
            "job_id": job.job_id,
            "job_fingerprint": job.job_fingerprint,
            "protocol_id": job.protocol_id,
            "partition_id": job.partition_id,
            "seed": job.seed,
            "lineage": job.lineage.to_dict() if hasattr(job.lineage, "to_dict") else job.lineage,
            "preprocessing_config": job.preprocessing_config,
            "feature_config": job.feature_config,
            "model_config": job.model_config,
            "git_commit": job.git_commit,
            "dependency_lock_hash": job.dependency_lock_hash,
        }
        config_file = job_dir / "config.json"
        config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        job.status = TrainingJobStatus.FAILED
        job.environment_versions = env_versions
        job.stderr_summary = stderr_summary

        is_valid, errors = validate_training_job_dict(job.to_dict())
        if not is_valid:
            raise ValueError(f"FAILED job failed schema validation: {errors}")

        job_file = job_dir / "job.json"
        job_file.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

        write_manifest_sha256(job_dir)
        return job_dir
