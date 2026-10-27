"""Round-trip oracle tests for zero-variance feature protection across export and neutral inference.

Validates FR-005.5, FR-011.1, FR-013, SC-002, SC-006, Decision-2, Decision-3:
- FeatureScaler zero-variance guard: std <= 1e-12 sets scale=1.0 and z=0.0.
- NeutralModelExporter persists zero_variance_indices in model_manifest.json.
- NeutralModelExporter retains exactly 5 binary arrays in model_params.npz.
- NeutralNumpyPredictor.load() restores zero_variance_indices from manifest.
- Eliminates the scale <= 1e-12 heuristic at inference time.
- Guarantees test features with non-constant values in zero-variance train columns are strictly zeroed.
- Verifies full 8-step oracle test against manual linear discriminant calculation within TOL-LDA-NEUTRAL-SCORES (1e-7).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from semg_baseline.export import NeutralModelExporter, verify_exported_model
from semg_baseline.predictor import NeutralNumpyPredictor
from semg_baseline.scaling import FeatureScaler


class MockLdaModel:
    """Mock LDA model exposing classes_, coef_, and intercept_."""

    def __init__(self, classes: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> None:
        self.classes_ = np.asarray(classes, dtype=np.int32)
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = np.asarray(intercept, dtype=np.float64)


def test_zero_variance_roundtrip_eight_step_oracle(tmp_path: Path) -> None:
    """Execute the mandatory 8-step oracle test for zero-variance preservation.

    Steps:
    1. Synthetic TRAIN with constant feature in col 1 (zero variance).
    2. FeatureScaler.fit(TRAIN), verify scale=1.0 and 1 in scaler.zero_variance_indices.
    3. Export via NeutralModelExporter, verify manifest and exactly 5 arrays.
    4. Load via NeutralNumpyPredictor.load().
    5. Synthetic TEST where col 1 has distinct values (non-constant).
    6. Compare scaler.transform(X_test) vs predictor transformed features.
    7. Compare decision scores and predicted classes against manual linear discriminant oracle.
    8. Assert decision scores within 1e-7 (TOL-LDA-NEUTRAL-SCORES).
    """
    rng = np.random.default_rng(20261006)
    n_features = 6
    n_train = 60
    n_classes = 3

    # Step 1: Synthetic TRAIN with constant feature in column 1
    X_train = rng.normal(loc=2.0, scale=1.5, size=(n_train, n_features))
    constant_val = 7.42
    X_train[:, 1] = constant_val  # Zero variance in column 1

    # Step 2: Fit FeatureScaler and verify protection
    scaler = FeatureScaler()
    scaler.fit(X_train)
    assert 1 in scaler.zero_variance_indices
    assert scaler.scale_train is not None
    assert scaler.mean_train is not None
    assert scaler.scale_train[1] == 1.0
    assert np.isclose(scaler.mean_train[1], constant_val, atol=1e-12)
    X_train_trans = scaler.transform(X_train)
    np.testing.assert_allclose(X_train_trans[:, 1], 0.0, atol=1e-12)

    # Step 3: Export via NeutralModelExporter
    classes = np.array([0, 1, 2], dtype=np.int32)
    coef = rng.normal(loc=0.0, scale=1.0, size=(n_classes, n_features))
    intercept = np.array([0.5, -0.2, 0.1], dtype=np.float64)
    model = MockLdaModel(classes=classes, coef=coef, intercept=intercept)

    job_dir = tmp_path / "job_zero_var_oracle"
    exporter = NeutralModelExporter(output_dir=job_dir)
    manifest = exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_oracle_001",
        job_fingerprint="a" * 64,
        model_taxonomy="exercise_E3",
        priors_type="uniform",
    )

    # Verify manifest contains zero_variance_indices
    assert manifest["zero_variance_indices"] == [1]

    # Verify model_params.npz contains exactly 5 arrays
    npz_data = np.load(job_dir / "model_params.npz", allow_pickle=False)
    assert set(npz_data.files) == {"classes", "coef", "intercept", "mean", "scale"}
    assert len(npz_data.files) == 5

    # Step 4: Load via NeutralNumpyPredictor.load()
    predictor = NeutralNumpyPredictor.load(job_dir)
    assert predictor.zero_variance_indices == [1]

    # Step 5: Synthetic TEST where column 1 has distinct, non-constant values
    n_test = 25
    X_test = rng.normal(loc=0.0, scale=3.0, size=(n_test, n_features))
    # Col 1 varies widely and differs from training constant
    X_test[:, 1] = np.linspace(-50.0, 50.0, n_test)

    # Step 6: Compare scaler.transform(X_test) vs predictor behavior
    X_test_scaled_scaler = scaler.transform(X_test)
    np.testing.assert_allclose(X_test_scaled_scaler[:, 1], 0.0, atol=1e-12)

    scores_from_unscaled = predictor.decision_scores(X_test, is_scaled=False)
    scores_from_scaled = predictor.decision_scores(X_test_scaled_scaler, is_scaled=True)
    np.testing.assert_allclose(scores_from_unscaled, scores_from_scaled, atol=1e-12, rtol=1e-12)

    # Step 7: Compare against manual linear discriminant oracle
    # Oracle computation:
    # 1. Normalize X_test with training mean and scale
    Z_oracle = (X_test - scaler.mean_train) / scaler.scale_train
    # 2. Zero-variance feature is strictly zeroed
    Z_oracle[:, 1] = 0.0
    # 3. Decision scores = Z @ coef.T + intercept
    oracle_scores = Z_oracle @ coef.T + intercept
    oracle_preds = classes[np.argmax(oracle_scores, axis=1)]

    # Step 8: Assert decision scores and predictions within TOL-LDA-NEUTRAL-SCORES (1e-7)
    y_pred, scores = predictor.predict(X_test, is_scaled=False)

    max_score_diff = np.max(np.abs(scores - oracle_scores))
    assert max_score_diff <= 1e-7, f"Max score difference {max_score_diff} exceeded 1e-7 limit"
    np.testing.assert_allclose(scores, oracle_scores, atol=1e-7, rtol=1e-7)
    np.testing.assert_array_equal(y_pred, oracle_preds)


def test_zero_variance_multiple_indices_roundtrip(tmp_path: Path) -> None:
    """Test round-trip when multiple features have zero variance (e.g. cols 0, 2, 4)."""
    rng = np.random.default_rng(42)
    n_features = 5
    n_train = 40
    n_classes = 4

    X_train = rng.normal(size=(n_train, n_features))
    X_train[:, 0] = 1.0  # Constant col 0
    X_train[:, 2] = -3.5  # Constant col 2
    X_train[:, 4] = 0.0  # Constant col 4

    scaler = FeatureScaler()
    scaler.fit(X_train)
    assert scaler.zero_variance_indices == [0, 2, 4]

    model = MockLdaModel(
        classes=np.arange(n_classes, dtype=np.int32),
        coef=rng.normal(size=(n_classes, n_features)),
        intercept=np.zeros(n_classes),
    )

    job_dir = tmp_path / "multi_zero_var_job"
    exporter = NeutralModelExporter(output_dir=job_dir)
    manifest = exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_multi_zero",
        job_fingerprint="b" * 64,
        model_taxonomy="global_50_class",
        priors_type="empirical",
    )
    assert manifest["zero_variance_indices"] == [0, 2, 4]

    predictor = NeutralNumpyPredictor.load(job_dir)
    assert predictor.zero_variance_indices == [0, 2, 4]

    # Test data where cols 0, 2, 4 are active
    X_test = rng.normal(size=(15, n_features))
    X_test_scaled = scaler.transform(X_test)
    np.testing.assert_allclose(X_test_scaled[:, [0, 2, 4]], 0.0, atol=1e-12)

    scores_raw = predictor.decision_scores(X_test, is_scaled=False)
    scores_scaled = predictor.decision_scores(X_test_scaled, is_scaled=True)
    np.testing.assert_allclose(scores_raw, scores_scaled, atol=1e-12)


def test_zero_variance_empty_list_roundtrip(tmp_path: Path) -> None:
    """Test round-trip when no features have zero variance (standard dataset)."""
    rng = np.random.default_rng(99)
    n_features = 4
    n_train = 30
    n_classes = 2

    X_train = rng.normal(loc=5.0, scale=2.0, size=(n_train, n_features))
    scaler = FeatureScaler()
    scaler.fit(X_train)
    assert scaler.zero_variance_indices == []

    model = MockLdaModel(
        classes=np.array([0, 1], dtype=np.int32),
        coef=rng.normal(size=(n_classes, n_features)),
        intercept=np.array([0.1, -0.1]),
    )

    job_dir = tmp_path / "no_zero_var_job"
    exporter = NeutralModelExporter(output_dir=job_dir)
    manifest = exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_no_zero",
        job_fingerprint="c" * 64,
        model_taxonomy="exercise_E1",
        priors_type="uniform",
    )
    assert manifest["zero_variance_indices"] == []

    predictor = NeutralNumpyPredictor.load(job_dir)
    assert predictor.zero_variance_indices == []

    X_test = rng.normal(size=(10, n_features))
    y_pred, scores = predictor.predict(X_test)
    assert y_pred.shape == (10,)
    assert scores.shape == (10, 2)


def test_verify_exported_model_rejects_corrupted_zero_variance_manifest(tmp_path: Path) -> None:
    """verify_exported_model must reject manifests with invalid zero_variance_indices."""
    scaler = FeatureScaler()
    scaler.fit(np.eye(4))
    model = MockLdaModel(
        classes=np.array([0, 1]),
        coef=np.ones((2, 4)),
        intercept=np.zeros(2),
    )
    job_dir = tmp_path / "corrupted_zero_var"
    exporter = NeutralModelExporter(output_dir=job_dir)
    exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_corrupt",
        job_fingerprint="d" * 64,
        model_taxonomy="exercise_E2",
        priors_type="uniform",
    )

    manifest_path = job_dir / "model_manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Test: negative index
    manifest_data["zero_variance_indices"] = [-1]
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    (job_dir / "model_manifest.sha256").write_text(
        f"{__import__('hashlib').sha256(manifest_path.read_bytes()).hexdigest()}  model_manifest.json\n",
        encoding="utf-8",
    )
    valid, errors = verify_exported_model(job_dir)
    assert valid is False
    assert any("non-negative" in err for err in errors)

    # Test: out of bounds index (>= n_features)
    manifest_data["zero_variance_indices"] = [10]
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    (job_dir / "model_manifest.sha256").write_text(
        f"{__import__('hashlib').sha256(manifest_path.read_bytes()).hexdigest()}  model_manifest.json\n",
        encoding="utf-8",
    )
    valid, errors = verify_exported_model(job_dir)
    assert valid is False
    assert any(">= feature dimension" in err for err in errors)

    # Test: duplicates
    manifest_data["zero_variance_indices"] = [1, 1]
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    (job_dir / "model_manifest.sha256").write_text(
        f"{__import__('hashlib').sha256(manifest_path.read_bytes()).hexdigest()}  model_manifest.json\n",
        encoding="utf-8",
    )
    valid, errors = verify_exported_model(job_dir)
    assert valid is False
    assert any("duplicate" in err for err in errors)


def test_verify_exported_model_rejects_non_finite_values(tmp_path: Path) -> None:
    """verify_exported_model must reject NaN or Inf in numerical arrays."""
    scaler = FeatureScaler()
    scaler.fit(np.eye(4))
    model = MockLdaModel(
        classes=np.array([0, 1]),
        coef=np.ones((2, 4)),
        intercept=np.zeros(2),
    )
    job_dir = tmp_path / "nan_test_job"
    exporter = NeutralModelExporter(output_dir=job_dir)
    exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_nan",
        job_fingerprint="e" * 64,
        model_taxonomy="exercise_E1",
        priors_type="uniform",
    )

    npz_path = job_dir / "model_params.npz"
    data = dict(np.load(npz_path, allow_pickle=False))

    # Test NaN in coef
    corrupt_data = dict(data)
    corrupt_coef = corrupt_data["coef"].copy()
    corrupt_coef[0, 0] = np.nan
    corrupt_data["coef"] = corrupt_coef
    np.savez(npz_path, **corrupt_data)

    valid, errors = verify_exported_model(job_dir)
    assert valid is False
    assert any("non-finite values (NaN or Inf)" in err for err in errors)

    # Test Inf in intercept
    corrupt_data = dict(data)
    corrupt_intercept = corrupt_data["intercept"].copy()
    corrupt_intercept[0] = np.inf
    corrupt_data["intercept"] = corrupt_intercept
    np.savez(npz_path, **corrupt_data)

    valid, errors = verify_exported_model(job_dir)
    assert valid is False
    assert any("non-finite values (NaN or Inf)" in err for err in errors)


def test_verify_exported_model_rejects_non_positive_scale(tmp_path: Path) -> None:
    """verify_exported_model must reject scale <= 0.0."""
    scaler = FeatureScaler()
    scaler.fit(np.eye(4))
    model = MockLdaModel(
        classes=np.array([0, 1]),
        coef=np.ones((2, 4)),
        intercept=np.zeros(2),
    )
    job_dir = tmp_path / "scale_test_job"
    exporter = NeutralModelExporter(output_dir=job_dir)
    exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_scale",
        job_fingerprint="f" * 64,
        model_taxonomy="exercise_E1",
        priors_type="uniform",
    )

    npz_path = job_dir / "model_params.npz"
    data = dict(np.load(npz_path, allow_pickle=False))

    corrupt_scale = data["scale"].copy()
    corrupt_scale[1] = 0.0
    data["scale"] = corrupt_scale
    np.savez(npz_path, **data)

    valid, errors = verify_exported_model(job_dir)
    assert valid is False
    assert any("scale contains non-positive values" in err for err in errors)


def test_verify_exported_model_rejects_duplicate_classes(tmp_path: Path) -> None:
    """verify_exported_model must reject duplicate class labels in classes array."""
    scaler = FeatureScaler()
    scaler.fit(np.eye(4))
    model = MockLdaModel(
        classes=np.array([0, 0]),  # Duplicate classes
        coef=np.ones((2, 4)),
        intercept=np.zeros(2),
    )
    job_dir = tmp_path / "dup_classes_job"
    exporter = NeutralModelExporter(output_dir=job_dir)
    exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_dup",
        job_fingerprint="0" * 64,
        model_taxonomy="exercise_E1",
        priors_type="uniform",
    )

    valid, errors = verify_exported_model(job_dir)
    assert valid is False
    assert any("Duplicate class labels" in err for err in errors)


def test_verify_exported_model_rejects_shape_mismatches(tmp_path: Path) -> None:
    """verify_exported_model must reject mismatched shapes between model parameters and scaler."""
    scaler = FeatureScaler()
    scaler.fit(np.eye(4))
    model = MockLdaModel(
        classes=np.array([0, 1, 2]),
        coef=np.ones((2, 4)),  # 2 rows for 3 classes!
        intercept=np.zeros(3),
    )
    job_dir = tmp_path / "shape_mismatch_job"
    exporter = NeutralModelExporter(output_dir=job_dir)
    exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_mismatch",
        job_fingerprint="1" * 64,
        model_taxonomy="exercise_E1",
        priors_type="uniform",
    )

    valid, errors = verify_exported_model(job_dir)
    assert valid is False
    assert any("does not match number of classes" in err for err in errors)


def test_verify_exported_model_rejects_semantic_role_and_array_set_violations(tmp_path: Path) -> None:
    """verify_exported_model must reject swapped semantic roles, extra arrays, or missing arrays."""
    scaler = FeatureScaler()
    scaler.fit(np.eye(4))
    model = MockLdaModel(
        classes=np.array([0, 1]),
        coef=np.ones((2, 4)),
        intercept=np.zeros(2),
    )
    job_dir = tmp_path / "semantic_viol_job"
    exporter = NeutralModelExporter(output_dir=job_dir)
    exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_sem",
        job_fingerprint="2" * 64,
        model_taxonomy="exercise_E1",
        priors_type="uniform",
    )

    manifest_path = job_dir / "model_manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Test: wrong semantic role
    manifest_data["arrays"][0]["semantic_role"] = "discriminant_coefficients"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    (job_dir / "model_manifest.sha256").write_text(
        f"{__import__('hashlib').sha256(manifest_path.read_bytes()).hexdigest()}  model_manifest.json\n",
        encoding="utf-8",
    )

    valid, errors = verify_exported_model(job_dir)
    assert valid is False
    assert any("Semantic role mismatch" in err for err in errors)

    # Test: extra array in npz
    npz_path = job_dir / "model_params.npz"
    data = dict(np.load(npz_path, allow_pickle=False))
    data["extra_array"] = np.array([1, 2, 3])
    np.savez(npz_path, **data)

    valid, errors = verify_exported_model(job_dir)
    assert valid is False
    assert any("model_params.npz arrays must match exactly" in err for err in errors)
