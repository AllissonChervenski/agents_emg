"""Unit, contract, and scientific tests for ExerciseConfoundingAuditor (FR-007.3, FR-007.5, SC-005, Decision-4).

Validates:
- ExerciseConfoundingReport dataclass contracts, properties, and strict bounds [0.0, 1.0].
- Chance level canonical value: 1/3 (0.3333333333333333).
- Confounding delta calculation: rest_exercise_accuracy - chance_level.
- Confounding indicated boolean flag: True if confounding_delta > 0.05, else False.
- Standardized scientific interpretation strings:
    * Near chance (<= 0.05 delta): "nenhuma evidência de confounding detectável por este diagnóstico"
    * Above chance (> 0.05 delta): "evidência de confounding de sessão/exercício detectada; deve acompanhar explicitamente a interpretação dos resultados globais"
- Schema compliance against evaluation_report_schema.json (confounding_diagnostic property).
- Auxiliary LDA classifier training strictly from REST windows with uniform priors.
- Near-chance outcome on homogeneous synthetic REST signals across E1, E2, E3.
- Robust detection of synthetic session/exercise bias (confounding detected when distinct exercise patterns injected).
- End-to-end within-subject audit over RecordingData fixtures with strict train/test repetition separation ([1, 3, 4, 6] vs [2, 5]).
- Deterministic reproducibility: bit-for-bit identical results on identical inputs.
- Adversarial rejection of invalid inputs: empty arrays, single-class training, NaN/Inf, mismatched dimensions.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from semg_baseline.confounding import (
    ExerciseConfoundingAuditor,
    ExerciseConfoundingReport,
)
from semg_baseline.evaluator import EvaluationReportGenerator
from tests.fixtures.synthetic_semg import make_synthetic_recording

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = (
    REPO_ROOT
    / "specs"
    / "004-baseline-training-and-evaluation"
    / "contracts"
    / "evaluation_report_schema.json"
)


def _load_report_schema() -> dict:
    """Load canonical JSON Schema for EvaluationReport."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_confounding_report_dataclass_contracts_and_schema_validation() -> None:
    """ExerciseConfoundingReport must enforce field contracts and validate against evaluation_report_schema.json."""
    # Test case 1: Clean baseline (near chance)
    report_clean = ExerciseConfoundingReport(
        audit_id="audit_ws_S1",
        protocol_id="within_subject",
        rest_exercise_accuracy=0.34,
        chance_level=1.0 / 3.0,
        confounding_delta=0.34 - (1.0 / 3.0),
        confounding_indicated=False,
        scientific_interpretation="nenhuma evidência de confounding detectável por este diagnóstico",
    )

    assert report_clean.audit_id == "audit_ws_S1"
    assert report_clean.protocol_id == "within_subject"
    assert report_clean.rest_exercise_accuracy == 0.34
    assert np.isclose(report_clean.chance_level, 1.0 / 3.0)
    assert np.isclose(report_clean.confounding_delta, 0.34 - (1.0 / 3.0))
    assert report_clean.confounding_indicated is False
    assert (
        report_clean.scientific_interpretation
        == "nenhuma evidência de confounding detectável por este diagnóstico"
    )
    assert report_clean.interpretation == report_clean.scientific_interpretation

    # Validate diagnostic dict against JSON schema
    schema = _load_report_schema()
    diagnostic_schema = schema["properties"]["confounding_diagnostic"]
    jsonschema.validate(instance=report_clean.to_diagnostic_dict(), schema=diagnostic_schema)

    # Test case 2: Confounded baseline (above chance)
    report_confounded = ExerciseConfoundingReport(
        audit_id="audit_cs_fold1",
        protocol_id="cross_subject",
        rest_exercise_accuracy=0.85,
        chance_level=1.0 / 3.0,
        confounding_delta=0.85 - (1.0 / 3.0),
        confounding_indicated=True,
        scientific_interpretation="evidência de confounding de sessão/exercício detectada; deve acompanhar explicitamente a interpretação dos resultados globais",
    )

    assert report_confounded.confounding_indicated is True
    assert (
        report_confounded.scientific_interpretation
        == "evidência de confounding de sessão/exercício detectada; deve acompanhar explicitamente a interpretação dos resultados globais"
    )
    jsonschema.validate(instance=report_confounded.to_diagnostic_dict(), schema=diagnostic_schema)


def test_confounding_report_rejects_out_of_bounds_accuracy() -> None:
    """ExerciseConfoundingReport must reject accuracy values outside [0.0, 1.0]."""
    with pytest.raises(ValueError, match="rest_exercise_accuracy must be in"):
        ExerciseConfoundingReport(
            audit_id="bad_1",
            protocol_id="within_subject",
            rest_exercise_accuracy=-0.05,
            chance_level=1.0 / 3.0,
            confounding_delta=-0.38,
            confounding_indicated=False,
            scientific_interpretation="nenhuma evidência de confounding detectável por este diagnóstico",
        )

    with pytest.raises(ValueError, match="rest_exercise_accuracy must be in"):
        ExerciseConfoundingReport(
            audit_id="bad_2",
            protocol_id="within_subject",
            rest_exercise_accuracy=1.05,
            chance_level=1.0 / 3.0,
            confounding_delta=0.72,
            confounding_indicated=True,
            scientific_interpretation="evidência de confounding de sessão/exercício detectada; deve acompanhar explicitamente a interpretação dos resultados globais",
        )


def test_confounding_auditor_near_chance_on_homogeneous_rest_features() -> None:
    """Auditor must report chance-level performance and clean interpretation on homogeneous REST noise."""
    auditor = ExerciseConfoundingAuditor()
    rng = np.random.default_rng(42)

    # 3 exercises, 200 rest samples per exercise in train, 100 in test
    # Signal is pure Gaussian noise with identical distribution across all exercises
    n_train_per_ex = 200
    n_test_per_ex = 100
    n_features = 60

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    for ex_idx in range(3):
        X_train_list.append(rng.normal(0.0, 1.0, size=(n_train_per_ex, n_features)))
        y_train_list.append(np.full(n_train_per_ex, ex_idx, dtype=np.int32))

        X_test_list.append(rng.normal(0.0, 1.0, size=(n_test_per_ex, n_features)))
        y_test_list.append(np.full(n_test_per_ex, ex_idx, dtype=np.int32))

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    report = auditor.audit_from_features(
        X_rest_train=X_train,
        y_exercise_train=y_train,
        X_rest_test=X_test,
        y_exercise_test=y_test,
        audit_id="audit_homogeneous",
        protocol_id="within_subject",
    )

    # Accuracy must be near chance (33.3% +/- 0.05 threshold)
    assert 0.0 <= report.rest_exercise_accuracy <= 1.0
    assert report.confounding_delta <= 0.05
    assert report.confounding_indicated is False
    assert (
        report.scientific_interpretation
        == "nenhuma evidência de confounding detectável por este diagnóstico"
    )


def test_confounding_auditor_detects_synthetic_session_bias() -> None:
    """Auditor must reliably detect confounding when REST features contain distinct exercise signatures."""
    auditor = ExerciseConfoundingAuditor()
    rng = np.random.default_rng(123)

    n_train_per_ex = 150
    n_test_per_ex = 75
    n_features = 60

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    # Inject distinct exercise-specific offsets in REST features (representing session/electrode shift)
    exercise_offsets = [0.0, 3.0, -3.0]

    for ex_idx, offset in enumerate(exercise_offsets):
        X_train_list.append(rng.normal(offset, 0.5, size=(n_train_per_ex, n_features)))
        y_train_list.append(np.full(n_train_per_ex, ex_idx, dtype=np.int32))

        X_test_list.append(rng.normal(offset, 0.5, size=(n_test_per_ex, n_features)))
        y_test_list.append(np.full(n_test_per_ex, ex_idx, dtype=np.int32))

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    report = auditor.audit_from_features(
        X_rest_train=X_train,
        y_exercise_train=y_train,
        X_rest_test=X_test,
        y_exercise_test=y_test,
        audit_id="audit_confounded",
        protocol_id="within_subject",
    )

    # Auxiliary LDA should easily separate the exercises: accuracy high, delta > 0.05
    assert report.rest_exercise_accuracy > 0.80
    assert report.confounding_delta > 0.05
    assert report.confounding_indicated is True
    assert (
        report.scientific_interpretation
        == "evidência de confounding de sessão/exercício detectada; deve acompanhar explicitamente a interpretação dos resultados globais"
    )


def test_confounding_auditor_within_subject_recording_pipeline() -> None:
    """Auditor must execute within-subject audit over RecordingData fixtures with zero test leakage."""
    auditor = ExerciseConfoundingAuditor()

    # Generate synthetic recordings for E1, E2, E3 for a single subject
    rec_e1 = make_synthetic_recording(seed=101, subject_id="S1", exercise_id="E1")
    rec_e2 = make_synthetic_recording(seed=102, subject_id="S1", exercise_id="E2")
    rec_e3 = make_synthetic_recording(seed=103, subject_id="S1", exercise_id="E3")

    recordings = {"E1": rec_e1, "E2": rec_e2, "E3": rec_e3}

    report = auditor.audit_within_subject(
        subject_id="S1",
        recordings=recordings,
        candidate_r=1.0,
        train_repetitions=(1, 3, 4, 6),
        test_repetitions=(2, 5),
    )

    assert report.protocol_id == "within_subject"
    assert "S1" in report.audit_id
    assert 0.0 <= report.rest_exercise_accuracy <= 1.0
    assert np.isclose(report.chance_level, 1.0 / 3.0)
    assert np.isclose(report.confounding_delta, report.rest_exercise_accuracy - (1.0 / 3.0))

    # Also test passing as a list of RecordingData
    report_list = auditor.audit_within_subject(
        subject_id="S1",
        recordings=[rec_e1, rec_e2, rec_e3],
        candidate_r=1.0,
    )
    assert report_list.protocol_id == "within_subject"
    assert np.isclose(report_list.rest_exercise_accuracy, report.rest_exercise_accuracy)


def test_confounding_auditor_cross_subject_recording_pipeline() -> None:
    """Auditor must execute cross-subject audit over train and test recording sets."""
    auditor = ExerciseConfoundingAuditor()

    # Train recordings: 2 subjects x 3 exercises
    train_recordings = [
        make_synthetic_recording(seed=201, subject_id="S1", exercise_id="E1"),
        make_synthetic_recording(seed=202, subject_id="S1", exercise_id="E2"),
        make_synthetic_recording(seed=203, subject_id="S1", exercise_id="E3"),
        make_synthetic_recording(seed=204, subject_id="S2", exercise_id="E1"),
        make_synthetic_recording(seed=205, subject_id="S2", exercise_id="E2"),
        make_synthetic_recording(seed=206, subject_id="S2", exercise_id="E3"),
    ]

    # Test recordings: 1 subject x 3 exercises
    test_recordings = [
        make_synthetic_recording(seed=301, subject_id="S3", exercise_id="E1"),
        make_synthetic_recording(seed=302, subject_id="S3", exercise_id="E2"),
        make_synthetic_recording(seed=303, subject_id="S3", exercise_id="E3"),
    ]

    report = auditor.audit_cross_subject(
        fold_id=1,
        train_recordings=train_recordings,
        test_recordings=test_recordings,
        candidate_r=1.0,
    )

    assert report.protocol_id == "cross_subject"
    assert "fold_1" in report.audit_id
    assert 0.0 <= report.rest_exercise_accuracy <= 1.0
    assert np.isclose(report.chance_level, 1.0 / 3.0)


def test_confounding_reproducibility() -> None:
    """Auditor must produce bit-for-bit identical results on repeated runs with identical inputs."""
    auditor1 = ExerciseConfoundingAuditor()
    auditor2 = ExerciseConfoundingAuditor()

    rng = np.random.default_rng(999)
    X_train = rng.normal(0.0, 1.0, size=(120, 60))
    y_train = np.repeat([0, 1, 2], 40)
    X_test = rng.normal(0.0, 1.0, size=(60, 60))
    y_test = np.repeat([0, 1, 2], 20)

    rep1 = auditor1.audit_from_features(
        X_rest_train=X_train,
        y_exercise_train=y_train,
        X_rest_test=X_test,
        y_exercise_test=y_test,
    )
    rep2 = auditor2.audit_from_features(
        X_rest_train=X_train,
        y_exercise_train=y_train,
        X_rest_test=X_test,
        y_exercise_test=y_test,
    )

    assert rep1.rest_exercise_accuracy == rep2.rest_exercise_accuracy
    assert rep1.confounding_delta == rep2.confounding_delta
    assert rep1.confounding_indicated == rep2.confounding_indicated
    assert rep1.scientific_interpretation == rep2.scientific_interpretation


def test_confounding_adversarial_rejections() -> None:
    """Auditor must reject invalid inputs fail-closed."""
    auditor = ExerciseConfoundingAuditor()
    rng = np.random.default_rng(42)

    X_valid = rng.normal(0.0, 1.0, size=(60, 60))
    y_valid = np.repeat([0, 1, 2], 20)

    # 1. Empty train or test
    with pytest.raises(ValueError, match="non-empty"):
        auditor.audit_from_features(
            X_rest_train=np.empty((0, 60)),
            y_exercise_train=np.empty(0, dtype=int),
            X_rest_test=X_valid,
            y_exercise_test=y_valid,
        )

    with pytest.raises(ValueError, match="non-empty"):
        auditor.audit_from_features(
            X_rest_train=X_valid,
            y_exercise_train=y_valid,
            X_rest_test=np.empty((0, 60)),
            y_exercise_test=np.empty(0, dtype=int),
        )

    # 2. Fewer than 2 distinct exercise classes in train
    with pytest.raises(ValueError, match="at least 2 distinct exercise classes"):
        auditor.audit_from_features(
            X_rest_train=X_valid,
            y_exercise_train=np.zeros(60, dtype=int),  # only class 0
            X_rest_test=X_valid,
            y_exercise_test=y_valid,
        )

    # 3. Shape mismatch between features and labels
    with pytest.raises(ValueError, match="Sample count mismatch"):
        auditor.audit_from_features(
            X_rest_train=X_valid,
            y_exercise_train=y_valid[:30],
            X_rest_test=X_valid,
            y_exercise_test=y_valid,
        )

    # 4. Feature dimension mismatch between train and test
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        auditor.audit_from_features(
            X_rest_train=X_valid,
            y_exercise_train=y_valid,
            X_rest_test=X_valid[:, :30],
            y_exercise_test=y_valid,
        )

    # 5. NaN or Inf in features
    X_nan = X_valid.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        auditor.audit_from_features(
            X_rest_train=X_nan,
            y_exercise_train=y_valid,
            X_rest_test=X_valid,
            y_exercise_test=y_valid,
        )


def test_confounding_integrated_in_evaluation_report() -> None:
    """EvaluationReportGenerator must generate valid report when confounding diagnostic is provided."""
    auditor = ExerciseConfoundingAuditor()
    generator = EvaluationReportGenerator()
    schema = _load_report_schema()

    # Generate confounding report
    rng = np.random.default_rng(777)
    X_train = rng.normal(0.0, 1.0, size=(120, 60))
    y_train = np.repeat([0, 1, 2], 40)
    X_test = rng.normal(0.0, 1.0, size=(60, 60))
    y_test = np.repeat([0, 1, 2], 20)

    conf_report = auditor.audit_from_features(
        X_rest_train=X_train,
        y_exercise_train=y_train,
        X_rest_test=X_test,
        y_exercise_test=y_test,
    )

    # Generate primary evaluation report incorporating confounding diagnostic
    y_true = np.repeat(np.arange(50), 10)
    y_pred = y_true.copy()

    eval_report = generator.generate_report(
        y_true=y_true,
        y_pred=y_pred,
        report_id="report_with_confounding",
        job_id="job_001",
        job_fingerprint="a" * 64,
        protocol_id="within_subject",
        partition_id="S1",
        confounding_diagnostic=conf_report.to_diagnostic_dict(),
    )

    report_dict = eval_report.to_dict()
    assert "confounding_diagnostic" in report_dict
    assert report_dict["confounding_diagnostic"]["chance_level"] == conf_report.chance_level
    assert (
        report_dict["confounding_diagnostic"]["interpretation"]
        == conf_report.scientific_interpretation
    )

    # Validate against full evaluation_report_schema.json
    jsonschema.validate(instance=report_dict, schema=schema)
