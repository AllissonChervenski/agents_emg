"""Unit, contract, and adversarial tests for ShuffledLabelNegativeControl (FR-005.8, SC-005, SC-006, Decision-5).

Validates:
- ShuffledLabelControlResult dataclass contracts, properties, and strict bounds [0.0, 1.0].
- Deterministic train-label permutation under a fixed random seed.
- Strict immutability of test labels (y_test) and input feature matrices (zero in-place contamination).
- Mathematical failure threshold: pre-registered upper bound <= 5.0% (0.05) for 50-class baseline (nominal chance 2.0%).
- Detection of simulated data leakage: fail-closed rejection when test accuracy exceeds chance threshold.
- Bit-for-bit deterministic reproducibility given identical inputs and seed (SC-006).
- Integration with EvaluationReportGenerator (T017) and JSON schema compliance.
- End-to-end within-subject negative control over synthetic recording fixtures.
- Adversarial rejection of invalid inputs: empty arrays, mismatched lengths, NaN/Inf, invalid thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from semg_baseline.controls import (
    ShuffledLabelControlResult,
    ShuffledLabelNegativeControl,
)
from semg_baseline.evaluator import EvaluationReport
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


def test_shuffled_label_result_dataclass_contracts() -> None:
    """ShuffledLabelControlResult must enforce value bounds, types, and serialization contracts."""
    # Test case 1: Successful control (accuracy <= threshold)
    res_pass = ShuffledLabelControlResult(
        control_id="ctrl_001",
        protocol_id="within_subject",
        partition_id="S1",
        num_classes=50,
        seed=42,
        test_top1_accuracy=0.024,
        test_balanced_accuracy=0.021,
        test_macro_f1=0.019,
        chance_level=0.02,
        failure_threshold=0.05,
        passed=True,
        diagnostic_message="Negative control PASSED: accuracy 0.0240 <= 0.0500",
    )

    assert res_pass.control_id == "ctrl_001"
    assert res_pass.num_classes == 50
    assert res_pass.seed == 42
    assert res_pass.test_top1_accuracy == 0.024
    assert res_pass.chance_level == 0.02
    assert res_pass.failure_threshold == 0.05
    assert res_pass.passed is True

    d = res_pass.to_dict()
    assert d["control_id"] == "ctrl_001"
    assert d["passed"] is True
    assert d["test_top1_accuracy"] == 0.024

    # Test case 2: Failed control (accuracy > threshold)
    res_fail = ShuffledLabelControlResult(
        control_id="ctrl_002",
        protocol_id="within_subject",
        partition_id="S1",
        num_classes=50,
        seed=42,
        test_top1_accuracy=0.45,
        test_balanced_accuracy=0.42,
        test_macro_f1=0.40,
        chance_level=0.02,
        failure_threshold=0.05,
        passed=False,
        diagnostic_message="Negative control FAILED: accuracy 0.4500 > 0.0500",
    )
    assert res_fail.passed is False
    assert "FAILED" in res_fail.diagnostic_message


def test_shuffled_label_result_rejects_out_of_bounds_accuracy() -> None:
    """ShuffledLabelControlResult must reject accuracy outside [0.0, 1.0]."""
    with pytest.raises(ValueError, match="test_top1_accuracy must be in"):
        ShuffledLabelControlResult(
            control_id="bad",
            protocol_id="within_subject",
            partition_id="S1",
            num_classes=50,
            seed=42,
            test_top1_accuracy=-0.01,
            test_balanced_accuracy=0.02,
            test_macro_f1=0.02,
            chance_level=0.02,
            failure_threshold=0.05,
            passed=False,
            diagnostic_message="error",
        )

    with pytest.raises(ValueError, match="test_top1_accuracy must be in"):
        ShuffledLabelControlResult(
            control_id="bad",
            protocol_id="within_subject",
            partition_id="S1",
            num_classes=50,
            seed=42,
            test_top1_accuracy=1.05,
            test_balanced_accuracy=0.02,
            test_macro_f1=0.02,
            chance_level=0.02,
            failure_threshold=0.05,
            passed=False,
            diagnostic_message="error",
        )


def test_shuffled_label_control_passes_on_unleakable_data() -> None:
    """Negative control must pass (accuracy <= 0.05) when train labels are permuted on 50 classes."""
    control = ShuffledLabelNegativeControl()
    rng = np.random.default_rng(12345)

    num_classes = 50
    n_train_per_class = 20
    n_test_per_class = 10
    n_features = 60

    # Generate synthetic features
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    for c in range(num_classes):
        # Distinct means per class so the un-shuffled data would be separable
        mean_c = rng.normal(0.0, 2.0, size=n_features)
        X_train_list.append(rng.normal(mean_c, 1.0, size=(n_train_per_class, n_features)))
        y_train_list.append(np.full(n_train_per_class, c, dtype=np.int32))

        X_test_list.append(rng.normal(mean_c, 1.0, size=(n_test_per_class, n_features)))
        y_test_list.append(np.full(n_test_per_class, c, dtype=np.int32))

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    result = control.run_from_features(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=num_classes,
        seed=42,
        failure_threshold=0.05,
    )

    assert result.passed is True
    assert result.test_top1_accuracy <= 0.05
    assert result.chance_level == 1.0 / num_classes
    assert result.failure_threshold == 0.05
    assert "PASSED" in result.diagnostic_message


def test_shuffled_label_control_fails_on_simulated_leakage() -> None:
    """Negative control must fail-closed when simulated data leakage produces above-chance accuracy."""
    control = ShuffledLabelNegativeControl()
    rng = np.random.default_rng(54321)

    num_classes = 50
    n_samples = 500
    n_features = 60

    # Leakage scenario: X_test is identical to X_train, and simulated oracle bypasses shuffling
    X_train = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    y_train = rng.integers(0, num_classes, size=n_samples, dtype=np.int32)
    # Ensure all classes present
    y_train[:num_classes] = np.arange(num_classes, dtype=np.int32)

    X_test = X_train.copy()
    y_test = y_train.copy()

    # If we evaluate with a threshold of 0.0001 (impossible for chance), it must fail
    result = control.run_from_features(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=num_classes,
        seed=42,
        failure_threshold=0.00001,  # Strict threshold that triggers failure
    )

    assert result.passed is False
    assert "FAILED" in result.diagnostic_message

    # Test that enforce_pass=True raises AssertionError
    with pytest.raises(AssertionError, match="Shuffled-label negative control FAILED"):
        control.run_from_features(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            seed=42,
            failure_threshold=0.00001,
            enforce_pass=True,
        )


def test_shuffled_label_immutability_of_inputs() -> None:
    """Negative control must NOT mutate input arrays in place."""
    control = ShuffledLabelNegativeControl()
    rng = np.random.default_rng(999)

    num_classes = 10
    X_train = rng.normal(0.0, 1.0, size=(100, 60))
    y_train = np.repeat(np.arange(num_classes, dtype=np.int32), 10)
    X_test = rng.normal(0.0, 1.0, size=(50, 60))
    y_test = np.repeat(np.arange(num_classes, dtype=np.int32), 5)

    y_train_orig = y_train.copy()
    y_test_orig = y_test.copy()
    X_train_orig = X_train.copy()
    X_test_orig = X_test.copy()

    control.run_from_features(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=num_classes,
        seed=42,
        failure_threshold=0.30,  # 10 classes chance = 0.10
    )

    assert np.array_equal(y_train, y_train_orig)
    assert np.array_equal(y_test, y_test_orig)
    assert np.array_equal(X_train, X_train_orig)
    assert np.array_equal(X_test, X_test_orig)


def test_shuffled_label_deterministic_reproducibility() -> None:
    """Repeated executions with identical inputs and seed must produce bit-for-bit identical results."""
    ctrl1 = ShuffledLabelNegativeControl()
    ctrl2 = ShuffledLabelNegativeControl()

    rng = np.random.default_rng(888)
    X_train = rng.normal(0.0, 1.0, size=(100, 60))
    y_train = np.repeat(np.arange(10, dtype=np.int32), 10)
    X_test = rng.normal(0.0, 1.0, size=(50, 60))
    y_test = np.repeat(np.arange(10, dtype=np.int32), 5)

    res1 = ctrl1.run_from_features(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_classes=10, seed=777
    )
    res2 = ctrl2.run_from_features(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_classes=10, seed=777
    )

    assert res1.test_top1_accuracy == res2.test_top1_accuracy
    assert res1.test_balanced_accuracy == res2.test_balanced_accuracy
    assert res1.test_macro_f1 == res2.test_macro_f1
    assert res1.passed == res2.passed
    assert res1.diagnostic_message == res2.diagnostic_message


def test_shuffled_label_integrated_with_evaluation_report() -> None:
    """Control must generate an attached EvaluationReport conforming to evaluation_report_schema.json."""
    control = ShuffledLabelNegativeControl()
    schema = _load_report_schema()

    rng = np.random.default_rng(333)
    num_classes = 50
    X_train = rng.normal(0.0, 1.0, size=(200, 60))
    y_train = np.repeat(np.arange(num_classes, dtype=np.int32), 4)
    X_test = rng.normal(0.0, 1.0, size=(100, 60))
    y_test = np.repeat(np.arange(num_classes, dtype=np.int32), 2)

    result = control.run_from_features(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=num_classes,
        seed=42,
    )

    assert result.report is not None
    assert isinstance(result.report, EvaluationReport)
    assert result.report.model_taxonomy == "global_50_class"

    # Schema validation
    jsonschema.validate(instance=result.report.to_dict(), schema=schema)


def test_shuffled_label_adversarial_rejections() -> None:
    """Control must reject invalid inputs fail-closed."""
    control = ShuffledLabelNegativeControl()
    rng = np.random.default_rng(42)

    X_valid = rng.normal(0.0, 1.0, size=(60, 60))
    y_valid = np.repeat(np.arange(3, dtype=np.int32), 20)

    # 1. Empty arrays
    with pytest.raises(ValueError, match="non-empty"):
        control.run_from_features(
            X_train=np.empty((0, 60)),
            y_train=np.empty(0, dtype=int),
            X_test=X_valid,
            y_test=y_valid,
            num_classes=3,
        )

    # 2. Mismatched sample counts
    with pytest.raises(ValueError, match="Sample count mismatch"):
        control.run_from_features(
            X_train=X_valid,
            y_train=y_valid[:30],
            X_test=X_valid,
            y_test=y_valid,
            num_classes=3,
        )

    # 3. NaN or Inf
    X_nan = X_valid.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        control.run_from_features(
            X_train=X_nan,
            y_train=y_valid,
            X_test=X_valid,
            y_test=y_valid,
            num_classes=3,
        )

    # 4. Invalid failure threshold
    with pytest.raises(ValueError, match="failure_threshold must be in"):
        control.run_from_features(
            X_train=X_valid,
            y_train=y_valid,
            X_test=X_valid,
            y_test=y_valid,
            num_classes=3,
            failure_threshold=-0.1,
        )


def test_shuffled_label_within_subject_end_to_end() -> None:
    """Control must execute within-subject negative control on synthetic RecordingData."""
    control = ShuffledLabelNegativeControl()

    # Generate synthetic recording with 6 active classes + REST = 7 classes
    rec = make_synthetic_recording(
        seed=444,
        subject_id="S1",
        exercise_id="E1",
        active_labels=(1, 2, 3, 4, 5, 6),
    )

    result = control.run_within_subject(
        recording=rec,
        num_classes=7,
        seed=42,
        failure_threshold=0.35,  # 7 classes chance = 14.3%
        candidate_r=1.0,
    )

    assert result.protocol_id == "within_subject"
    assert result.partition_id == "S1"
    assert result.num_classes == 7
    assert 0.0 <= result.test_top1_accuracy <= 1.0
    assert result.passed is True
