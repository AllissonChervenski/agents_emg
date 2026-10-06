"""Neutral model parameter exporter and integrity verifier.

Implements FR-005.5, FR-011.1, SC-001, SC-006, Decision-3, and Decision-5:
- Exports trained LDA model and scaler parameters to neutral NumPy arrays:
  classes (int32), coef (float64), intercept (float64), mean (float64), scale (float64).
- Saves model_params.npz (strictly allow_pickle=False, zero pickle/sklearn objects).
- Generates model_manifest.json conforming to neutral_model_manifest_schema.json.
- Seals manifest integrity via sidecar model_manifest.sha256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


VALID_TAXONOMIES = {"global_50_class", "exercise_E1", "exercise_E2", "exercise_E3"}
VALID_PRIORS = {"uniform", "empirical"}


class NeutralModelExporter:
    """Exports model and scaler parameters to neutral numerical archives with manifests."""

    def __init__(self, output_dir: Path | str | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def export(
        self,
        model: Any,
        scaler: Any,
        job_id: str,
        job_fingerprint: str,
        model_taxonomy: str,
        priors_type: str,
        output_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Export parameters to neutral model_params.npz, manifest, and sidecar sha256.

        Args:
            model: Trained classifier exposing classes_, coef_, intercept_.
            scaler: Fitted FeatureScaler exposing mean_train, scale_train.
            job_id: Job identifier (e.g. 'job_ws_S1_20261006').
            job_fingerprint: 64-character SHA-256 digest of scientific inputs.
            model_taxonomy: Taxonomy string in VALID_TAXONOMIES.
            priors_type: Prior probability type ('uniform' or 'empirical').
            output_dir: Destination directory.

        Returns:
            Dictionary matching neutral_model_manifest_schema.json.
        """
        if model_taxonomy not in VALID_TAXONOMIES:
            raise ValueError(
                f"Invalid model_taxonomy '{model_taxonomy}'. Must be one of: {sorted(list(VALID_TAXONOMIES))}"
            )

        if priors_type not in VALID_PRIORS:
            raise ValueError(
                f"Invalid priors_type '{priors_type}'. Must be one of: {sorted(list(VALID_PRIORS))}"
            )

        target_dir = Path(output_dir or self.output_dir or ".")
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract pure numerical arrays
        classes = np.asarray(getattr(model, "classes_"), dtype=np.int32)
        coef = np.asarray(getattr(model, "coef_"), dtype=np.float64)
        intercept = np.asarray(getattr(model, "intercept_"), dtype=np.float64)
        mean = np.asarray(getattr(scaler, "mean_train"), dtype=np.float64)
        scale = np.asarray(getattr(scaler, "scale_train"), dtype=np.float64)

        # Ensure correct shapes
        if classes.ndim != 1:
            classes = classes.ravel()
        if intercept.ndim != 1:
            intercept = intercept.ravel()
        if mean.ndim != 1:
            mean = mean.ravel()
        if scale.ndim != 1:
            scale = scale.ravel()

        array_map = {
            "classes": (classes, "class_labels", "int32"),
            "coef": (coef, "discriminant_coefficients", "float64"),
            "intercept": (intercept, "discriminant_intercepts", "float64"),
            "mean": (mean, "scaler_mean", "float64"),
            "scale": (scale, "scaler_scale", "float64"),
        }

        # 2. Build array descriptors
        array_descriptors: list[dict[str, Any]] = []
        for name in ("classes", "coef", "intercept", "mean", "scale"):
            arr, role, dtype_str = array_map[name]
            arr_sha = hashlib.sha256(arr.tobytes()).hexdigest()
            array_descriptors.append(
                {
                    "name": name,
                    "dtype": dtype_str,
                    "shape": list(arr.shape),
                    "sha256": arr_sha,
                    "semantic_role": role,
                }
            )

        manifest_dict: dict[str, Any] = {
            "model_format_version": "1.0.0",
            "job_id": job_id,
            "job_fingerprint": job_fingerprint,
            "model_taxonomy": model_taxonomy,
            "priors_type": priors_type,
            "arrays": array_descriptors,
        }

        # 3. Persist model_params.npz without pickle
        npz_path = target_dir / "model_params.npz"
        np.savez(
            npz_path,
            classes=classes,
            coef=coef,
            intercept=intercept,
            mean=mean,
            scale=scale,
        )

        # 4. Persist model_manifest.json with canonical formatting
        manifest_path = target_dir / "model_manifest.json"
        manifest_bytes = json.dumps(manifest_dict, indent=2, sort_keys=True).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)

        # 5. Persist sidecar model_manifest.sha256
        sidecar_path = target_dir / "model_manifest.sha256"
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        sidecar_path.write_text(f"{manifest_sha}  model_manifest.json\n", encoding="utf-8")

        return manifest_dict


def verify_exported_model(model_dir: Path | str) -> tuple[bool, list[str]]:
    """Cryptographically verify exported neutral model integrity.

    Validates:
    1. model_manifest.json exists and matches sidecar model_manifest.sha256.
    2. model_params.npz exists and loads without pickle.
    3. Every array in model_params.npz matches the manifest dtype, shape, and SHA-256.

    Returns:
        (is_valid, errors)
    """
    path = Path(model_dir).resolve()
    errors: list[str] = []

    manifest_path = path / "model_manifest.json"
    sidecar_path = path / "model_manifest.sha256"
    npz_path = path / "model_params.npz"

    if not manifest_path.is_file():
        return False, [f"model_manifest.json missing at: {manifest_path}"]
    if not sidecar_path.is_file():
        return False, [f"model_manifest.sha256 sidecar missing at: {sidecar_path}"]
    if not npz_path.is_file():
        return False, [f"model_params.npz missing at: {npz_path}"]

    # 1. Verify manifest against sidecar SHA-256
    manifest_bytes = manifest_path.read_bytes()
    computed_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    sidecar_line = sidecar_path.read_text(encoding="utf-8").strip()
    expected_manifest_sha = sidecar_line.split()[0] if sidecar_line else ""

    if computed_manifest_sha != expected_manifest_sha:
        errors.append(
            f"manifest checksum mismatch for model_manifest.json: expected {expected_manifest_sha}, "
            f"got {computed_manifest_sha}"
        )

    # 2. Parse manifest JSON
    try:
        manifest_data = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as e:
        errors.append(f"Failed to parse model_manifest.json: {e}")
        return False, errors

    descriptors = manifest_data.get("arrays", [])
    if len(descriptors) != 5:
        errors.append(f"Expected exactly 5 array descriptors in manifest, found {len(descriptors)}")

    # 3. Load and verify binary arrays in model_params.npz
    try:
        loaded_npz = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        errors.append(f"Failed to load model_params.npz without pickle: {e}")
        return False, errors

    for desc in descriptors:
        name = desc.get("name")
        expected_dtype = desc.get("dtype")
        expected_shape = tuple(desc.get("shape", []))
        expected_sha = desc.get("sha256")

        if name not in loaded_npz:
            errors.append(f"Array '{name}' declared in manifest missing from model_params.npz")
            continue

        arr = loaded_npz[name]
        if str(arr.dtype) != expected_dtype:
            errors.append(
                f"Array '{name}' dtype mismatch: expected {expected_dtype}, got {arr.dtype}"
            )
        if tuple(arr.shape) != expected_shape:
            errors.append(
                f"Array '{name}' shape mismatch: expected {expected_shape}, got {arr.shape}"
            )

        actual_sha = hashlib.sha256(arr.tobytes()).hexdigest()
        if actual_sha != expected_sha:
            errors.append(
                f"sha256 mismatch for array '{name}': expected {expected_sha}, got {actual_sha}"
            )

    return len(errors) == 0, errors
