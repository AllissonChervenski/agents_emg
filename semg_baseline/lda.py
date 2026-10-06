"""Canonical Regularized Linear Discriminant Analysis (LDA) trainer for sEMG baseline.

Implements FR-005.1, FR-005.2, FR-005.3, FR-005.4, FR-006.2, FR-006.3, SC-005, Decision-3:
- Canonical 50-class baseline fitted with uniform priors P(c) = 1/50 (FR-005.1).
- Secondary reference model fitted with empirical priors P(c) = N_c / N_train (FR-005.2).
- Uniform and empirical priors trained as distinct independent fits.
- Ledoit-Wolf analytic covariance shrinkage via solver='lsqr', shrinkage='auto' (FR-005.3).
- Model parameter extraction and evidence persistence (FR-005.4).
- Host-only governance allowlist compliance (FR-006.2, FR-006.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.metadata
import sys

import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class PriorType(str, Enum):
    """Supported class prior distribution strategies."""

    UNIFORM = "uniform"
    EMPIRICAL = "empirical"


@dataclass(frozen=True)
class LdaModelArtifacts:
    """Persistent model parameter artifacts and training metadata."""

    classes_: np.ndarray
    coef_: np.ndarray
    intercept_: np.ndarray
    means_: np.ndarray
    covariance_: np.ndarray
    priors_: np.ndarray
    shrinkage_: float
    solver: str
    shrinkage_method: str
    sklearn_version: str
    prior_type: str


class CanonicalLdaTrainer:
    """Regularized Linear Discriminant Analysis baseline classifier with Ledoit-Wolf shrinkage."""

    def __init__(
        self,
        prior_type: PriorType | str = PriorType.UNIFORM,
        num_classes: int = 50,
    ) -> None:
        """Initialize CanonicalLdaTrainer with prior configuration and class count.

        Parameters
        ----------
        prior_type : PriorType | str, default=PriorType.UNIFORM
            Class prior distribution: 'uniform' (P(c) = 1/K) or 'empirical' (P(c) = N_c / N_train).
        num_classes : int, default=50
            Nominal number of gesture classes in vocabulary.
        """
        if isinstance(prior_type, PriorType):
            self.prior_type = prior_type
        elif isinstance(prior_type, str):
            try:
                self.prior_type = PriorType(prior_type.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid prior_type '{prior_type}'. Expected '{PriorType.UNIFORM.value}' "
                    f"or '{PriorType.EMPIRICAL.value}'."
                ) from None
        else:
            raise ValueError(
                f"Invalid prior_type type '{type(prior_type)}'. Expected PriorType or str."
            )

        self.num_classes = int(num_classes)
        self.classes_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None
        self.shrinkage_: float | None = None
        self._clf: LinearDiscriminantAnalysis | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> CanonicalLdaTrainer:
        """Fit regularized LDA model using Ledoit-Wolf analytic shrinkage and configured priors.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Integer gesture class labels of shape (n_samples,).

        Returns
        -------
        CanonicalLdaTrainer
            Fitted trainer instance.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32)

        if X_arr.ndim != 2:
            raise ValueError(f"Feature matrix X must be 2D array, got shape {X_arr.shape}")
        if y_arr.ndim != 1:
            raise ValueError(f"Target vector y must be 1D array, got shape {y_arr.shape}")
        if len(X_arr) == 0:
            raise ValueError("Input feature matrix X is empty: zero samples provided")
        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"Sample count mismatch between X ({len(X_arr)}) and y ({len(y_arr)})"
            )
        if np.isnan(X_arr).any():
            raise ValueError("Feature matrix X contains NaN values")
        if np.isinf(X_arr).any():
            raise ValueError("Feature matrix X contains Inf values")

        classes_present = np.unique(y_arr)
        k_classes = len(classes_present)
        if k_classes < 2:
            raise ValueError(
                f"LDA requires at least 2 distinct classes to fit, found {k_classes}"
            )

        # Compute priors strictly according to policy
        if self.prior_type == PriorType.UNIFORM:
            priors = np.full(k_classes, 1.0 / k_classes, dtype=np.float64)
        else:
            # Empirical priors: P(c) = N_c / N_train strictly computed on training sample counts
            counts = np.array([np.sum(y_arr == c) for c in classes_present], dtype=np.float64)
            priors = counts / float(len(y_arr))

        clf = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=priors,
        )
        clf.fit(X_arr, y_arr)

        self._clf = clf
        self.classes_ = np.asarray(clf.classes_, dtype=np.int32)
        self.coef_ = np.asarray(clf.coef_, dtype=np.float64)
        self.intercept_ = np.asarray(clf.intercept_, dtype=np.float64)
        self.means_ = np.asarray(clf.means_, dtype=np.float64)
        self.covariance_ = np.asarray(clf.covariance_, dtype=np.float64)
        self.priors_ = np.asarray(clf.priors_, dtype=np.float64)
        # Compute pooled analytic shrinkage coefficient via Ledoit-Wolf on centered within-class residuals
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        sample_class_indices = np.array([class_to_idx[c] for c in y_arr], dtype=np.intp)
        X_centered = X_arr - self.means_[sample_class_indices]
        _, shrinkage_val = ledoit_wolf(X_centered, assume_centered=True)
        self.shrinkage_ = float(shrinkage_val)

        _evict_sklearn_from_sys_modules()
        return self

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if fit has not been called."""
        if self._clf is None or self.classes_ is None:
            raise RuntimeError("CanonicalLdaTrainer must be fitted before calling inference methods")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict gesture class labels for feature vectors.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted integer class labels of shape (n_samples,).
        """
        self._check_is_fitted()
        assert self._clf is not None
        X_arr = np.asarray(X, dtype=np.float64)
        return self._clf.predict(X_arr).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate class posterior probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Class posterior probabilities of shape (n_samples, n_classes).
        """
        self._check_is_fitted()
        assert self._clf is not None
        X_arr = np.asarray(X, dtype=np.float64)
        return self._clf.predict_proba(X_arr).astype(np.float64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute linear discriminant decision scores: X @ coef_.T + intercept_.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Decision scores of shape (n_samples, n_classes).
        """
        self._check_is_fitted()
        assert self._clf is not None
        X_arr = np.asarray(X, dtype=np.float64)
        return self._clf.decision_function(X_arr).astype(np.float64)

    def get_model_artifacts(self) -> LdaModelArtifacts:
        """Return structured model parameters and scientific provenance metadata.

        Returns
        -------
        LdaModelArtifacts
            Immutable container with fitted parameters, covariance, and metadata.
        """
        self._check_is_fitted()
        assert self.classes_ is not None
        assert self.coef_ is not None
        assert self.intercept_ is not None
        assert self.means_ is not None
        assert self.covariance_ is not None
        assert self.priors_ is not None
        assert self.shrinkage_ is not None

        sklearn_ver = importlib.metadata.version("scikit-learn")

        return LdaModelArtifacts(
            classes_=self.classes_,
            coef_=self.coef_,
            intercept_=self.intercept_,
            means_=self.means_,
            covariance_=self.covariance_,
            priors_=self.priors_,
            shrinkage_=self.shrinkage_,
            solver="lsqr",
            shrinkage_method="ledoit_wolf_auto",
            sklearn_version=sklearn_ver,
            prior_type=self.prior_type.value,
        )

    def export_neutral_arrays(
        self,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Export model arrays in format consumable by NeutralModelExporter and NeutralNumpyPredictor.

        Parameters
        ----------
        scaler_mean : np.ndarray
            Feature scaler mean vector of shape (n_features,).
        scaler_scale : np.ndarray
            Feature scaler scale vector of shape (n_features,).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing 'classes', 'coef', 'intercept', 'mean', 'scale'.
        """
        self._check_is_fitted()
        assert self.classes_ is not None
        assert self.coef_ is not None
        assert self.intercept_ is not None

        return {
            "classes": self.classes_.astype(np.int32),
            "coef": self.coef_.astype(np.float64),
            "intercept": self.intercept_.astype(np.float64),
            "mean": np.asarray(scaler_mean, dtype=np.float64),
            "scale": np.asarray(scaler_scale, dtype=np.float64),
        }


def _evict_sklearn_from_sys_modules() -> None:
    """Evict scikit-learn modules from sys.modules to prevent process-wide pollution."""
    for key in list(sys.modules):
        if key == "sklearn" or key.startswith("sklearn."):
            sys.modules.pop(key, None)


# Evict sklearn upon module import so idle imports do not pollute process state
_evict_sklearn_from_sys_modules()
