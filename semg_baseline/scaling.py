"""Train-only custom Z-score feature scaler with zero-variance protection.

Implements FR-013, SC-002, Decision-2:
- Custom Z-score feature normalization: z_j = (x_j - mu_j) / sigma_j.
- Fitted strictly on the training partition with stateless transformation of test splits.
- Zero-variance feature protection: if sigma_j <= 1e-12, scale=1.0 and z_j=0.0.
- Strict prohibition of sklearn.preprocessing.StandardScaler or sklearn dependencies.
- Zero data leakage invariant: validation and test splits NEVER contribute to statistics.
"""

from __future__ import annotations

from typing import Any
import numpy as np


class FeatureScaler:
    """Train-only custom Z-score feature normalizer with zero-variance guard."""

    def __init__(self) -> None:
        self.mean_train: np.ndarray | None = None
        self.scale_train: np.ndarray | None = None
        self.zero_variance_indices: list[int] = []

    def fit(self, X: np.ndarray) -> FeatureScaler:
        """Fit scaler parameters (mean and standard deviation) on training feature matrix.

        Args:
            X: Array of shape (N_train, n_features).

        Returns:
            self
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X_arr.shape}")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Training features contain non-finite values (NaN or Inf)")

        mean = np.mean(X_arr, axis=0, dtype=np.float64)
        std = np.std(X_arr, axis=0, ddof=0, dtype=np.float64)

        # Identify zero-variance features (std <= 1e-12)
        zero_var_mask = std <= 1e-12
        scale = np.where(zero_var_mask, 1.0, std)

        self.mean_train = mean
        self.scale_train = scale
        self.zero_variance_indices = [int(i) for i in np.where(zero_var_mask)[0]]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply stateless Z-score normalization using fitted training parameters.

        Args:
            X: Array of shape (N_samples, n_features).

        Returns:
            Normalized array of shape (N_samples, n_features) in float64.
        """
        if self.mean_train is None or self.scale_train is None:
            raise RuntimeError("FeatureScaler is not fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X_arr.shape}")
        if X_arr.shape[1] != len(self.mean_train):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self.mean_train)}, got {X_arr.shape[1]}"
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Input features contain non-finite values (NaN or Inf)")

        Z = (X_arr - self.mean_train) / self.scale_train

        # For zero-variance features, ensure exact 0.0 value
        if self.zero_variance_indices:
            Z[:, self.zero_variance_indices] = 0.0

        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to training data and return transformed feature matrix."""
        return self.fit(X).transform(X)

    def to_dict(self) -> dict[str, Any]:
        """Serialize parameters to dictionary for manifest or export."""
        if self.mean_train is None or self.scale_train is None:
            raise RuntimeError("FeatureScaler is not fitted yet.")
        return {
            "mean": [float(m) for m in self.mean_train],
            "scale": [float(s) for s in self.scale_train],
            "zero_variance_indices": list(self.zero_variance_indices),
        }
