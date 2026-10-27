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

EXPECTED_ARRAY_NAMES = {"classes", "coef", "intercept", "mean", "scale"}
EXPECTED_SEMANTIC_ROLES = {
    "classes": "class_labels",
    "coef": "discriminant_coefficients",
    "intercept": "discriminant_intercepts",
    "mean": "scaler_mean",
    "scale": "scaler_scale",
}
EXPECTED_DTYPES = {
    "classes": "int32",
    "coef": "float64",
    "intercept": "float64",
    "mean": "float64",
    "scale": "float64",
}


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

        zero_variance_indices = [
            int(i) for i in getattr(scaler, "zero_variance_indices", [])
        ]

        manifest_dict: dict[str, Any] = {
            "model_format_version": "1.0.0",
            "job_id": job_id,
            "job_fingerprint": job_fingerprint,
            "model_taxonomy": model_taxonomy,
            "priors_type": priors_type,
            "zero_variance_indices": zero_variance_indices,
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

    actual_desc_names = [d.get("name") for d in descriptors if isinstance(d, dict)]
    if len(actual_desc_names) != 5 or set(actual_desc_names) != EXPECTED_ARRAY_NAMES:
        errors.append(
            f"Array descriptor names must match exactly {EXPECTED_ARRAY_NAMES}, found {actual_desc_names}"
        )

    for desc in descriptors:
        if isinstance(desc, dict):
            name = desc.get("name")
            role = desc.get("semantic_role")
            dtype_val = desc.get("dtype")
            if name in EXPECTED_SEMANTIC_ROLES and role != EXPECTED_SEMANTIC_ROLES[name]:
                errors.append(
                    f"Semantic role mismatch for '{name}': expected '{EXPECTED_SEMANTIC_ROLES[name]}', got '{role}'"
                )
            if name in EXPECTED_DTYPES and dtype_val != EXPECTED_DTYPES[name]:
                errors.append(
                    f"Declared dtype mismatch for '{name}': expected '{EXPECTED_DTYPES[name]}', got '{dtype_val}'"
                )

    zero_var = manifest_data.get("zero_variance_indices")
    if zero_var is None or not isinstance(zero_var, list) or not all(
        isinstance(i, int) and i >= 0 for i in zero_var
    ):
        errors.append(
            "model_manifest.json must contain 'zero_variance_indices' as a list of non-negative integers"
        )
    elif len(zero_var) != len(set(zero_var)):
        errors.append("model_manifest.json 'zero_variance_indices' contains duplicate indices")

    # 3. Load and verify binary arrays in model_params.npz
    try:
        loaded_npz = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        errors.append(f"Failed to load model_params.npz without pickle: {e}")
        return False, errors

    if set(loaded_npz.files) != EXPECTED_ARRAY_NAMES:
        errors.append(
            f"model_params.npz arrays must match exactly {EXPECTED_ARRAY_NAMES}, found {set(loaded_npz.files)}"
        )

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

    # 4. Numerical and Structural Integrity Checks
    for name in EXPECTED_ARRAY_NAMES:
        if name in loaded_npz:
            arr = loaded_npz[name]
            if not np.all(np.isfinite(arr)):
                errors.append(f"Array '{name}' contains non-finite values (NaN or Inf)")

    if all(name in loaded_npz for name in EXPECTED_ARRAY_NAMES):
        classes = loaded_npz["classes"]
        coef = loaded_npz["coef"]
        intercept = loaded_npz["intercept"]
        mean = loaded_npz["mean"]
        scale = loaded_npz["scale"]

        if classes.ndim != 1:
            errors.append(f"'classes' must be 1D, got ndim={classes.ndim}")
        if len(classes) != len(np.unique(classes)):
            errors.append("Duplicate class labels in 'classes'")

        if coef.ndim != 2:
            errors.append(f"'coef' must be 2D, got ndim={coef.ndim}")

        if intercept.ndim != 1:
            errors.append(f"'intercept' must be 1D, got ndim={intercept.ndim}")

        if mean.ndim != 1:
            errors.append(f"'mean' must be 1D, got ndim={mean.ndim}")

        if scale.ndim != 1:
            errors.append(f"'scale' must be 1D, got ndim={scale.ndim}")

        if np.any(scale <= 0.0):
            errors.append("scale contains non-positive values (scale <= 0)")

        if coef.ndim == 2 and classes.ndim == 1 and coef.shape[0] != len(classes):
            errors.append(f"coef rows ({coef.shape[0]}) does not match number of classes ({len(classes)})")

        if len(intercept) != len(classes):
            errors.append(f"intercept length ({len(intercept)}) does not match number of classes ({len(classes)})")

        if coef.ndim == 2 and len(mean) > 0 and coef.shape[1] != len(mean):
            errors.append(f"coef feature dimension ({coef.shape[1]}) does not match mean dimension ({len(mean)})")

        if coef.ndim == 2 and len(scale) > 0 and coef.shape[1] != len(scale):
            errors.append(f"coef feature dimension ({coef.shape[1]}) does not match scale dimension ({len(scale)})")

        if len(mean) != len(scale):
            errors.append(f"mean dimension ({len(mean)}) does not match scale dimension ({len(scale)})")

        if zero_var is not None and isinstance(zero_var, list):
            feature_dim = len(mean)
            for idx in zero_var:
                if isinstance(idx, int):
                    if idx < 0 or idx >= feature_dim:
                        errors.append(
                            f"zero_variance_indices contains index {idx} out of range [0, {feature_dim - 1}] (>= feature dimension {feature_dim})"
                        )
                    elif idx < len(scale) and np.isfinite(scale[idx]) and scale[idx] != 1.0:
                        errors.append(
                            f"zero-variance feature index {idx} must have scale == 1.0, got {scale[idx]}"
                        )

    return len(errors) == 0, errors
