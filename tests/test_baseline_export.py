"""Unit tests for NeutralModelExporter and neutral parameter manifests (FR-005.5, FR-011.1, SC-001, SC-006).

Validates:
- Generation of model_params.npz with exactly 5 numerical arrays (zero pickle).
- Generation of model_manifest.json conforming to neutral_model_manifest_schema.json.
- Integrity verification via sidecar model_manifest.sha256.
- Tamper detection on both JSON manifest and NPZ binary arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
import jsonschema
import numpy as np
import pytest

from semg_baseline.export import NeutralModelExporter, verify_exported_model
from semg_baseline.scaling import FeatureScaler


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_SCHEMA_PATH = (
    REPO_ROOT
    / "specs"
    / "004-baseline-training-and-evaluation"
    / "contracts"
    / "neutral_model_manifest_schema.json"
)


class MockLdaModel:
    """Mock LDA model containing standard fitted attributes."""

    def __init__(self, num_classes: int = 50, n_features: int = 60) -> None:
        self.classes_ = np.arange(num_classes, dtype=np.int32)
        rng = np.random.default_rng(42)
        self.coef_ = rng.normal(0.0, 1.0, size=(num_classes, n_features)).astype(np.float64)
        self.intercept_ = rng.normal(0.0, 1.0, size=(num_classes,)).astype(np.float64)


def _load_manifest_schema() -> dict:
    with open(MANIFEST_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_neutral_model_export_and_schema_validation(tmp_path: Path) -> None:
    """Exporter must save 5 arrays, manifest matching JSON schema, and sidecar sha256."""
    model = MockLdaModel(num_classes=50, n_features=60)
    scaler = FeatureScaler()
    scaler.fit(np.ones((20, 60), dtype=np.float64))

    job_dir = tmp_path / "job_test_01"
    exporter = NeutralModelExporter(output_dir=job_dir)
    manifest_dict = exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_test_01",
        job_fingerprint="a" * 64,
        model_taxonomy="global_50_class",
        priors_type="uniform",
    )

    # 1. Validate manifest against schema
    schema = _load_manifest_schema()
    jsonschema.validate(instance=manifest_dict, schema=schema)

    # 2. Check files exist
    npz_file = job_dir / "model_params.npz"
    manifest_file = job_dir / "model_manifest.json"
    sidecar_file = job_dir / "model_manifest.sha256"

    assert npz_file.is_file()
    assert manifest_file.is_file()
    assert sidecar_file.is_file()

    # 3. Verify allow_pickle=False loading
    loaded_npz = np.load(npz_file, allow_pickle=False)
    for arr_name in ("classes", "coef", "intercept", "mean", "scale"):
        assert arr_name in loaded_npz
    assert loaded_npz["classes"].dtype == np.int32
    assert loaded_npz["coef"].dtype == np.float64

    # 4. Independent integrity check
    is_valid, errors = verify_exported_model(job_dir)
    assert is_valid is True
    assert errors == []


def test_tampered_manifest_fails_sidecar_verification(tmp_path: Path) -> None:
    """Tampering with model_manifest.json must be caught by sidecar sha256 check."""
    model = MockLdaModel(num_classes=18, n_features=60)
    scaler = FeatureScaler()
    scaler.fit(np.ones((10, 60), dtype=np.float64))

    job_dir = tmp_path / "job_tamper_manifest"
    exporter = NeutralModelExporter(output_dir=job_dir)
    exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_tamper_manifest",
        job_fingerprint="b" * 64,
        model_taxonomy="exercise_E1",
        priors_type="empirical",
    )

    # Tamper with manifest
    manifest_file = job_dir / "model_manifest.json"
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_data["model_taxonomy"] = "exercise_E2"  # Tampered!
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    is_valid, errors = verify_exported_model(job_dir)
    assert is_valid is False
    assert any("manifest checksum mismatch" in e.lower() for e in errors)


def test_tampered_npz_array_fails_array_hash_verification(tmp_path: Path) -> None:
    """Tampering with binary array in model_params.npz must be caught by descriptor hash."""
    model = MockLdaModel(num_classes=10, n_features=60)
    scaler = FeatureScaler()
    scaler.fit(np.ones((10, 60), dtype=np.float64))

    job_dir = tmp_path / "job_tamper_npz"
    exporter = NeutralModelExporter(output_dir=job_dir)
    exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_tamper_npz",
        job_fingerprint="c" * 64,
        model_taxonomy="exercise_E3",
        priors_type="uniform",
    )

    # Tamper with npz array
    npz_file = job_dir / "model_params.npz"
    data = dict(np.load(npz_file, allow_pickle=False))
    data["intercept"] = data["intercept"] + 1.0  # Tampered!
    np.savez(npz_file, **data)

    is_valid, errors = verify_exported_model(job_dir)
    assert is_valid is False
    assert any("sha256 mismatch for array 'intercept'" in e.lower() for e in errors)


def test_invalid_taxonomy_or_priors_rejected() -> None:
    """Exporting with invalid taxonomy or priors_type must raise ValueError."""
    model = MockLdaModel()
    scaler = FeatureScaler()
    scaler.fit(np.ones((5, 60)))

    exporter = NeutralModelExporter()
    with pytest.raises(ValueError, match="Invalid model_taxonomy"):
        exporter.export(
            model=model,
            scaler=scaler,
            job_id="job_bad",
            job_fingerprint="d" * 64,
            model_taxonomy="invalid_taxonomy",
            priors_type="uniform",
        )
