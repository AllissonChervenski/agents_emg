"""Unit and contract tests for CanonicalLdaTrainer (FR-005.1, FR-005.2, FR-005.3, FR-005.4, FR-006.2, FR-006.3, FR-011.5, SC-005, Decision-3).

Validates:
- Canonical 50-class baseline fitted with uniform priors P(c) = 1/50 (FR-005.1).
- Secondary reference model fitted with empirical priors P(c) = N_c / N_train (FR-005.2).
- Uniform and empirical priors trained as distinct independent fits.
- Ledoit-Wolf analytic covariance shrinkage via solver='lsqr', shrinkage='auto' (FR-005.3).
- Model parameter extraction and evidence persistence (FR-005.4).
- Equivalence between decision_function and linear formula X @ coef_.T + intercept_.
- Equivalence oracle with NeutralNumpyPredictor within TOL-LDA-NEUTRAL-SCORES (FR-005.6).
- Fail-closed validation for num_classes contract, missing class in train, and unexpected classes (FR-011.5).
- Input validation (NaN, Inf, empty, dimension mismatch, invalid prior_type).
- Host-only governance allowlist compliance.
"""

from __future__ import annotations

import numpy as np
import pytest

from semg_baseline.lda import (
    CanonicalLdaTrainer,
    LdaModelArtifacts,
    PriorType,
)
from semg_baseline.predictor import NeutralNumpyPredictor


def _generate_synthetic_features(
    n_samples_per_class: int = 4,
    n_classes: int = 50,
    n_features: int = 60,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible multi-class Gaussian feature vectors."""
    rng = np.random.default_rng(seed)
    X_list = []
    y_list = []
    for c in range(n_classes):
        center = rng.standard_normal(n_features) * 2.0
        samples = center + rng.standard_normal((n_samples_per_class, n_features)) * 0.5
        X_list.append(samples)
        y_list.append(np.full(n_samples_per_class, c, dtype=np.int32))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def test_canonical_lda_uniform_priors_fit() -> None:
    """Canonical baseline must fit with uniform priors P(c) = 1/50 across all classes (FR-005.1)."""
    n_classes = 50
    X, y = _generate_synthetic_features(n_samples_per_class=4, n_classes=n_classes, n_features=60, seed=101)

    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=n_classes)
    fitted = trainer.fit(X, y)

    assert fitted is trainer
    assert trainer.prior_type == PriorType.UNIFORM
    assert trainer.classes_.shape == (n_classes,)
    assert np.array_equal(trainer.classes_, np.arange(n_classes, dtype=np.int32))

    # Priors must be strictly uniform: 1/50
    expected_priors = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    assert np.allclose(trainer.priors_, expected_priors, atol=1e-12)

    # Dimensional attributes
    assert trainer.coef_.shape == (n_classes, 60)
    assert trainer.intercept_.shape == (n_classes,)
    assert trainer.covariance_.shape == (60, 60)
    assert trainer.means_.shape == (n_classes, 60)

    # Predictions
    preds = trainer.predict(X)
    assert preds.shape == (X.shape[0],)
    assert np.isin(preds, trainer.classes_).all()

    # Probabilities sum to 1.0
    probs = trainer.predict_proba(X)
    assert probs.shape == (X.shape[0], n_classes)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_canonical_lda_empirical_priors_fit() -> None:
    """Secondary reference must fit with empirical class priors P(c) = N_c / N_train (FR-005.2)."""
    rng = np.random.default_rng(202)
    n_classes = 5
    n_features = 60

    # Highly imbalanced: class 0 has 50 samples, classes 1..4 have 10 each
    counts = [50, 10, 10, 10, 10]
    total_n = sum(counts)
    X_list = []
    y_list = []
    for c, cnt in enumerate(counts):
        center = rng.standard_normal(n_features)
        X_list.append(center + rng.standard_normal((cnt, n_features)) * 0.5)
        y_list.append(np.full(cnt, c, dtype=np.int32))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Fit empirical
    trainer_emp = CanonicalLdaTrainer(prior_type=PriorType.EMPIRICAL, num_classes=n_classes)
    trainer_emp.fit(X, y)

    expected_emp_priors = np.array([cnt / total_n for cnt in counts], dtype=np.float64)
    assert np.allclose(trainer_emp.priors_, expected_emp_priors, atol=1e-12)

    # Fit uniform on identical data
    trainer_uni = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=n_classes)
    trainer_uni.fit(X, y)
    expected_uni_priors = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)
    assert np.allclose(trainer_uni.priors_, expected_uni_priors, atol=1e-12)

    # Intercepts must differ due to log(P(c)) term
    assert not np.allclose(trainer_emp.intercept_, trainer_uni.intercept_, atol=1e-4)

    # Decision functions must produce different outputs
    df_emp = trainer_emp.decision_function(X)
    df_uni = trainer_uni.decision_function(X)
    assert not np.allclose(df_emp, df_uni, atol=1e-4)


def test_canonical_lda_distinct_fits_invariance() -> None:
    """Uniform and empirical models must be distinct independent fits with no shared state (FR-005.2)."""
    X, y = _generate_synthetic_features(n_samples_per_class=4, n_classes=10, n_features=60, seed=303)

    m1 = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=10)
    m2 = CanonicalLdaTrainer(prior_type=PriorType.EMPIRICAL, num_classes=10)

    m1.fit(X, y)
    m2.fit(X, y)

    assert m1 is not m2
    assert m1.prior_type != m2.prior_type
    assert m1.coef_ is not m2.coef_
    assert m1.intercept_ is not m2.intercept_


def test_canonical_lda_shrinkage_auto_ledoit_wolf() -> None:
    """Ledoit-Wolf analytic shrinkage must handle ill-conditioned / high-dimensional settings stably (FR-005.3)."""
    # N = 20 < P = 60 (fewer samples than features, ill-conditioned pooled covariance)
    rng = np.random.default_rng(404)
    n_classes = 4
    n_features = 60
    n_samples = 20  # 5 per class

    X = rng.standard_normal((n_samples, n_features))
    y = np.repeat(np.arange(n_classes, dtype=np.int32), 5)

    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=n_classes)
    # Must fit without singular matrix error due to Ledoit-Wolf regularization
    trainer.fit(X, y)

    artifacts = trainer.get_model_artifacts()
    assert artifacts.solver == "lsqr"
    assert artifacts.shrinkage_method == "ledoit_wolf_auto"

    # Covariance matrix must be finite, symmetric, and positive definite
    assert trainer.covariance_.shape == (n_features, n_features)
    assert np.isfinite(trainer.covariance_).all(), "Covariance entries must be finite"
    assert np.allclose(trainer.covariance_, trainer.covariance_.T, atol=1e-8), "Covariance must be symmetric"
    eigvals = np.linalg.eigvalsh(trainer.covariance_)
    assert np.all(eigvals > 0.0), "Regularized covariance must be strictly positive definite"

    # Coefficients and intercept must be finite
    assert trainer.coef_.shape == (n_classes, n_features)
    assert np.isfinite(trainer.coef_).all(), "Coefficients must be finite"
    assert trainer.intercept_.shape == (n_classes,)
    assert np.isfinite(trainer.intercept_).all(), "Intercepts must be finite"

    # Decision function scores must be finite
    scores = trainer.decision_function(X)
    assert np.isfinite(scores).all(), "Decision scores must be finite"


def test_canonical_lda_linear_decision_function_oracle() -> None:
    """Decision scores must match linear formula and NeutralNumpyPredictor within TOL-LDA-NEUTRAL-SCORES (FR-005.5, FR-005.6)."""
    n_classes = 10
    n_features = 60
    X, y = _generate_synthetic_features(n_samples_per_class=4, n_classes=n_classes, n_features=n_features, seed=505)

    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=n_classes)
    trainer.fit(X, y)

    computed_scores = trainer.decision_function(X)
    expected_scores = X @ trainer.coef_.T + trainer.intercept_

    # Tolerance strictly governed by TOL-LDA-NEUTRAL-SCORES registry (rtol=1e-9, atol=1e-7)
    assert np.allclose(computed_scores, expected_scores, rtol=1e-9, atol=1e-7)

    # Argmax of scores must equal predicted class
    best_class_idx = np.argmax(computed_scores, axis=1)
    expected_preds = trainer.classes_[best_class_idx]
    assert np.array_equal(trainer.predict(X), expected_preds)

    # Integrated equivalence oracle with NeutralNumpyPredictor (FR-005.6)
    scaler_mean = np.zeros(n_features, dtype=np.float64)
    scaler_scale = np.ones(n_features, dtype=np.float64)
    export_dict = trainer.export_neutral_arrays(scaler_mean=scaler_mean, scaler_scale=scaler_scale)
    predictor = NeutralNumpyPredictor(**export_dict)
    neutral_preds, neutral_scores = predictor.predict(X)
    assert np.array_equal(neutral_preds, trainer.predict(X))
    assert np.allclose(neutral_scores, trainer.decision_function(X), rtol=1e-9, atol=1e-7)


def test_canonical_lda_model_artifacts_and_export() -> None:
    """Model parameter artifacts and neutral arrays must be extractable with full metadata (FR-005.4, FR-005.5)."""
    n_classes = 50
    n_features = 60
    X, y = _generate_synthetic_features(n_samples_per_class=4, n_classes=n_classes, n_features=n_features, seed=606)

    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=n_classes)
    trainer.fit(X, y)

    artifacts = trainer.get_model_artifacts()
    assert isinstance(artifacts, LdaModelArtifacts)
    assert artifacts.shrinkage_method == "ledoit_wolf_auto"
    assert artifacts.solver == "lsqr"
    assert len(artifacts.sklearn_version) > 0
    assert artifacts.prior_type == "uniform"
    assert artifacts.classes_.shape == (n_classes,)
    assert artifacts.coef_.shape == (n_classes, n_features)
    assert artifacts.intercept_.shape == (n_classes,)
    assert artifacts.covariance_.shape == (n_features, n_features)
    assert artifacts.means_.shape == (n_classes, n_features)
    assert artifacts.priors_.shape == (n_classes,)

    # Confirm no scalar lambda* field is defined in LdaModelArtifacts (Q3, Decision-3)
    assert not hasattr(artifacts, "shrinkage_")

    # Neutral export format integration: exactly 5 arrays
    scaler_mean = np.zeros(n_features, dtype=np.float64)
    scaler_scale = np.ones(n_features, dtype=np.float64)
    export_dict = trainer.export_neutral_arrays(scaler_mean=scaler_mean, scaler_scale=scaler_scale)

    assert set(export_dict.keys()) == {"classes", "coef", "intercept", "mean", "scale"}
    assert export_dict["classes"].dtype == np.int32
    assert export_dict["coef"].dtype == np.float64
    assert export_dict["intercept"].dtype == np.float64
    assert export_dict["mean"].dtype == np.float64
    assert export_dict["scale"].dtype == np.float64


def test_canonical_lda_input_validation() -> None:
    """Input validation must strictly reject invalid shapes, values, and prior types."""
    trainer = CanonicalLdaTrainer()

    # Empty inputs
    with pytest.raises(ValueError, match="empty|samples"):
        trainer.fit(np.empty((0, 60)), np.empty(0, dtype=np.int32))

    # Length mismatch
    with pytest.raises(ValueError, match="mismatch|length|shape"):
        trainer.fit(np.ones((10, 60)), np.ones(8, dtype=np.int32))

    # NaN in X
    X_nan = np.ones((10, 60))
    X_nan[2, 5] = np.nan
    with pytest.raises(ValueError, match="NaN|nan|invalid"):
        trainer.fit(X_nan, np.ones(10, dtype=np.int32))

    # Inf in X
    X_inf = np.ones((10, 60))
    X_inf[3, 8] = np.inf
    with pytest.raises(ValueError, match="Inf|inf|invalid"):
        trainer.fit(X_inf, np.ones(10, dtype=np.int32))

    # 1D X
    with pytest.raises(ValueError, match="dimension|2D|shape"):
        trainer.fit(np.ones(60), np.ones(1, dtype=np.int32))

    # Invalid prior_type
    with pytest.raises(ValueError, match="prior_type|prior"):
        CanonicalLdaTrainer(prior_type="invalid_prior")  # type: ignore[arg-type]


def test_canonical_lda_accepts_string_or_enum_prior() -> None:
    """Trainer constructor accepts both PriorType enum and equivalent string representation."""
    t1 = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM)
    assert t1.prior_type == PriorType.UNIFORM

    t2 = CanonicalLdaTrainer(prior_type="uniform")
    assert t2.prior_type == PriorType.UNIFORM

    t3 = CanonicalLdaTrainer(prior_type=PriorType.EMPIRICAL)
    assert t3.prior_type == PriorType.EMPIRICAL

    t4 = CanonicalLdaTrainer(prior_type="empirical")
    assert t4.prior_type == PriorType.EMPIRICAL


def test_canonical_lda_missing_class_in_train_fails_closed() -> None:
    """Training must fail closed when expected vocabulary class is missing (FR-011.5, Decision-3)."""
    # 49 classes present (0..48), class 49 missing for canonical 50-class vocabulary
    X, y = _generate_synthetic_features(n_samples_per_class=4, n_classes=49, n_features=60, seed=707)

    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=50)
    with pytest.raises(ValueError, match="missing classes.*49|vocabulary"):
        trainer.fit(X, y)


def test_canonical_lda_unexpected_class_in_train_fails_closed() -> None:
    """Training must fail closed when unexpected class label outside vocabulary appears (FR-011.5)."""
    # 51 classes present (0..50), label 50 is unexpected for canonical 50-class vocabulary
    X, y = _generate_synthetic_features(n_samples_per_class=4, n_classes=51, n_features=60, seed=808)

    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=50)
    with pytest.raises(ValueError, match="unexpected classes.*50|vocabulary"):
        trainer.fit(X, y)


def test_canonical_lda_num_classes_contract_and_secondary_vocabularies() -> None:
    """num_classes >= 2 validated in constructor and secondary vocabulary priors strictly exact."""
    with pytest.raises(ValueError, match="num_classes.*>= 2"):
        CanonicalLdaTrainer(num_classes=1)

    with pytest.raises(ValueError, match="num_classes.*>= 2"):
        CanonicalLdaTrainer(num_classes=0)

    # Exercise E1: exactly 18 classes -> priors exactly 1/18
    X_e1, y_e1 = _generate_synthetic_features(n_samples_per_class=4, n_classes=18, n_features=60, seed=901)
    t_e1 = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=18).fit(X_e1, y_e1)
    assert np.allclose(t_e1.priors_, np.full(18, 1.0 / 18.0), atol=1e-12)

    # Exercise E2: exactly 24 classes -> priors exactly 1/24
    X_e2, y_e2 = _generate_synthetic_features(n_samples_per_class=4, n_classes=24, n_features=60, seed=902)
    t_e2 = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=24).fit(X_e2, y_e2)
    assert np.allclose(t_e2.priors_, np.full(24, 1.0 / 24.0), atol=1e-12)

    # Exercise E3: exactly 10 classes -> priors exactly 1/10
    X_e3, y_e3 = _generate_synthetic_features(n_samples_per_class=4, n_classes=10, n_features=60, seed=903)
    t_e3 = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=10).fit(X_e3, y_e3)
    assert np.allclose(t_e3.priors_, np.full(10, 1.0 / 10.0), atol=1e-12)


def teardown_module() -> None:
    """Evict sklearn from sys.modules after running LDA tests to prevent polluting subsequent test state."""
    import sys

    for mod in list(sys.modules):
        if mod == "sklearn" or mod.startswith("sklearn."):
            sys.modules.pop(mod, None)
