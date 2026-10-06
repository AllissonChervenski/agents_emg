"""Neutral NumPy predictor for runtime gesture recognition inference simulation.

Implements FR-005.5, FR-005.6, SC-006, Decision-3:
- Pure NumPy linear decision evaluation: scores = X @ coef.T + intercept.
- Discrete classification: y_hat = classes[argmax(scores)].
- Zero scikit-learn dependency or imports at inference runtime.
- Load from neutral exported model directory with cryptographic integrity verification.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from semg_baseline.export import verify_exported_model


class NeutralNumpyPredictor:
    """Pure numerical predictor simulating Feature 005 runtime inference without scikit-learn."""

    def __init__(
        self,
        classes: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
        zero_variance_indices: Sequence[int] | None = None,
    ) -> None:
        self.classes = np.asarray(classes, dtype=np.int32).ravel()
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64).ravel()
        self.mean = np.asarray(mean, dtype=np.float64).ravel()
        self.scale = np.asarray(scale, dtype=np.float64).ravel()
        self.zero_variance_indices = list(zero_variance_indices) if zero_variance_indices else []

        if self.coef.ndim != 2:
            raise ValueError(f"coef must be a 2D array, got shape {self.coef.shape}")
        if self.coef.shape[0] != len(self.classes):
            raise ValueError(
                f"Row count of coef ({self.coef.shape[0]}) must match class count ({len(self.classes)})"
            )
        if len(self.intercept) != len(self.classes):
            raise ValueError(
                f"Length of intercept ({len(self.intercept)}) must match class count ({len(self.classes)})"
            )
        if self.coef.shape[1] != len(self.mean) or self.coef.shape[1] != len(self.scale):
            raise ValueError(
                f"Feature dimension of coef ({self.coef.shape[1]}) must match scaler ({len(self.mean)})"
            )

    @classmethod
    def load(cls, model_dir: Path | str) -> NeutralNumpyPredictor:
        """Load and cryptographically verify neutral model parameters from disk.

        Args:
            model_dir: Directory containing model_params.npz, model_manifest.json, and model_manifest.sha256.

        Returns:
            NeutralNumpyPredictor instance ready for inference.
        """
        path = Path(model_dir).resolve()
        is_valid, errors = verify_exported_model(path)
        if not is_valid:
            raise ValueError(
                f"Integrity verification failed for model directory '{path}': {errors}"
            )

        npz_file = path / "model_params.npz"
        data = np.load(npz_file, allow_pickle=False)

        classes = data["classes"]
        coef = data["coef"]
        intercept = data["intercept"]
        mean = data["mean"]
        scale = data["scale"]

        # Infer any zero-variance indices from scale array
        zero_var = list(np.where(scale <= 1e-12)[0])

        return cls(
            classes=classes,
            coef=coef,
            intercept=intercept,
            mean=mean,
            scale=scale,
            zero_variance_indices=zero_var,
        )

    def decision_scores(self, X: np.ndarray, is_scaled: bool = False) -> np.ndarray:
        """Compute continuous discriminant decision scores for input features.

        Args:
            X: Array of shape (N_samples, n_features).
            is_scaled: If True, X is assumed already normalized.

        Returns:
            Decision scores array of shape (N_samples, n_classes) in float64.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]

        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D feature matrix, got shape {X_arr.shape}")
        if X_arr.shape[1] != self.coef.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.coef.shape[1]}, got {X_arr.shape[1]}"
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Input features contain non-finite values (NaN or Inf)")

        if not is_scaled:
            X_norm = (X_arr - self.mean) / self.scale
            if self.zero_variance_indices:
                X_norm[:, self.zero_variance_indices] = 0.0
        else:
            X_norm = X_arr

        # Matrix decision evaluation: scores = X @ coef.T + intercept
        return X_norm @ self.coef.T + self.intercept

    def predict(
        self, X: np.ndarray, is_scaled: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict gesture class labels and return decision scores.

        Args:
            X: Array of shape (N_samples, n_features).
            is_scaled: If True, input is already scaled.

        Returns:
            Tuple of (y_pred, decision_scores).
        """
        scores = self.decision_scores(X, is_scaled=is_scaled)
        best_indices = np.argmax(scores, axis=1)
        y_pred = self.classes[best_indices]
        return y_pred, scores
