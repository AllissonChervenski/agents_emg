"""Unit, contract, and adversarial tests for EvaluationReportGenerator (FR-007, FR-010, SC-005, Decision-4, Decision-5).

Validates:
- Standardized structured evaluation reports with full TrainingJob traceability.
- 50-class primary report: classes 0..49 with explicit REST class 0 tracking.
- 49-class active-only secondary report: strict re-computation over active subset (y_true in 1..49)
  proving that primary 50-class != secondary 49-class active-only.
- All evaluation metrics (top-1 accuracy, balanced accuracy, macro-F1) bounded strictly in [0.0, 1.0].
- Per-class precision, recall, F1, and support maintaining explicit class identities.
- Canonical confusion matrix shape preservation (50x50 for global, 49x49 for active-only,
  18x18 for E1, 24x24 for E2, 10x10 for E3) even under absent classes in predictions.
- Strict segregation of REST metrics (precision, recall, false alarm rate) in 50-class reports;
  never mixed into active-only reports.
- Exercise-stratified secondary baselines: exercise_E1 (18 classes), exercise_E2 (24 classes),
  exercise_E3 (10 classes).
- Multi-partition aggregation: ConsolidatedEvaluationReport with mean, median, and std.
- Schema compliance against specs/004-baseline-training-and-evaluation/contracts/evaluation_report_schema.json.
- Deterministic reproducibility: bit-for-bit identical outputs and confusion matrix SHA-256 hashes.
- Adversarial rejection of mismatched lengths, out-of-vocabulary labels, invalid taxonomies,
  empty arrays, non-integer/NaN inputs, and zero-active subsets.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from semg_baseline.evaluator import (
    ActiveOnlyMetric,
    ClassMetric,
    ConsolidatedEvaluationReport,
    EvaluationReport,
    EvaluationReportGenerator,
    RestMetric,
)

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


def test_evaluator_primary_50_class_report_structure_and_bounds() -> None:
    """Primary 50-class evaluation report must satisfy all bounds, metrics, and schema contracts."""
    generator = EvaluationReportGenerator()
    schema = _load_report_schema()

    # Synthetic predictions with 50 classes: 20 samples per class = 1000 samples
    rng = np.random.default_rng(42)
    y_true = np.repeat(np.arange(50, dtype=np.int32), 20)
    # Simulate realistic predictions with some errors
    y_pred = y_true.copy()
    noisy_indices = rng.choice(len(y_true), size=100, replace=False)
    y_pred[noisy_indices] = rng.integers(0, 50, size=100)

    report = generator.generate_report(
        y_true=y_true,
        y_pred=y_pred,
        report_id="report_ws_S1_global50_20261008",
        job_id="job_ws_S1_20261008",
        job_fingerprint="a" * 64,
        protocol_id="within_subject",
        partition_id="S1",
        model_taxonomy="global_50_class",
        priors_type="uniform",
    )

    assert isinstance(report, EvaluationReport)
    assert report.model_taxonomy == "global_50_class"
    assert report.protocol_id == "within_subject"
    assert report.partition_id == "S1"
    assert report.priors_type == "uniform"

    # Normalized metrics bounded in [0.0, 1.0]
    assert 0.0 <= report.top1_accuracy <= 1.0
    assert 0.0 <= report.balanced_accuracy <= 1.0
    assert 0.0 <= report.macro_f1 <= 1.0

    # Per-class metrics: all 50 classes explicitly present
    assert len(report.per_class_metrics) == 50
    for c in range(50):
        c_str = str(c)
        assert c_str in report.per_class_metrics
        cm = report.per_class_metrics[c_str]
        assert isinstance(cm, ClassMetric)
        assert 0.0 <= cm.precision <= 1.0
        assert 0.0 <= cm.recall <= 1.0
        assert 0.0 <= cm.f1 <= 1.0
        assert cm.support == 20

    # REST metrics segregation
    assert isinstance(report.rest_metrics, RestMetric)
    assert 0.0 <= report.rest_metrics.precision <= 1.0
    assert 0.0 <= report.rest_metrics.recall <= 1.0
    assert 0.0 <= report.rest_metrics.false_alarm_rate <= 1.0

    # Active-only summary metrics
    assert isinstance(report.active_only_metrics, ActiveOnlyMetric)
    assert 0.0 <= report.active_only_metrics.top1_accuracy <= 1.0
    assert 0.0 <= report.active_only_metrics.macro_f1 <= 1.0

    # Confusion matrix dimension & SHA-256 hash
    assert report.confusion_matrix.shape == (50, 50)
    assert report.confusion_matrix_shape == [50, 50]
    assert len(report.confusion_matrix_hash) == 64
    assert all(c in "0123456789abcdef" for c in report.confusion_matrix_hash)

    # Validates against JSON Schema
    report_dict = report.to_dict()
    jsonschema.validate(instance=report_dict, schema=schema)


def test_evaluator_active_only_49_class_report_recalculated_not_just_sliced() -> None:
    """Active-only report MUST be recomputed strictly over active subset (50-class != 49-class active-only)."""
    generator = EvaluationReportGenerator()

    # Scenario demonstrating mathematical inequality:
    # 50 REST samples (class 0), 50 Active samples (class 1)
    # y_true: [0]*50 + [1]*50
    # y_pred: [0]*40 + [1]*10  (on REST: 40 correct, 10 false alarms)
    #         [0]*20 + [1]*30  (on Active: 20 missed active / predicted REST, 30 correct)
    y_true = np.array([0] * 50 + [1] * 50, dtype=np.int32)
    y_pred = np.array([0] * 40 + [1] * 10 + [0] * 20 + [1] * 30, dtype=np.int32)

    # Overall 50-class accuracy: (40 + 30) / 100 = 70.0%
    report_50 = generator.generate_report(
        y_true=y_true,
        y_pred=y_pred,
        report_id="report_test_50",
        job_id="job_test_50",
        job_fingerprint="b" * 64,
        protocol_id="within_subject",
        partition_id="S1",
        model_taxonomy="global_50_class",
        priors_type="uniform",
    )
    assert report_50.top1_accuracy == pytest.approx(0.70, abs=1e-6)

    # Active-only secondary report:
    # Must filter out y_true == 0 samples entirely.
    # Active samples: 50. Of those, 30 are correctly predicted as 1, 20 are erroneously predicted as 0.
    # Active-only accuracy: 30 / 50 = 60.0% != 70.0%
    active_report = generator.generate_active_only_report(
        y_true=y_true,
        y_pred=y_pred,
        report_id="report_test_active49",
        job_id="job_test_50",
        job_fingerprint="b" * 64,
        protocol_id="within_subject",
        partition_id="S1",
        priors_type="uniform",
    )

    assert active_report.top1_accuracy == pytest.approx(0.60, abs=1e-6)
    assert report_50.top1_accuracy != active_report.top1_accuracy

    # Active-only confusion matrix has canonical shape (49, 49) representing classes 1..49
    assert active_report.confusion_matrix.shape == (49, 49)
    assert active_report.confusion_matrix_shape == [49, 49]

    # Embedded active_only_metrics in 50-class report matches recomputed active-only report
    assert report_50.active_only_metrics.top1_accuracy == pytest.approx(0.60, abs=1e-6)


def test_evaluator_rest_metrics_segregation_and_false_alarm_rate() -> None:
    """REST metrics must explicitly isolate class 0 precision, recall, and false alarm rate."""
    generator = EvaluationReportGenerator()

    # 100 REST samples (class 0), 100 Active samples (class 1)
    # y_pred predicts:
    # 90 REST correctly as 0, 10 REST as 1 (TP_rest = 90, FN_rest = 10)
    # 20 Active falsely as 0, 80 Active correctly as 1 (FP_rest = 20, TN_rest = 80)
    y_true = np.array([0] * 100 + [1] * 100, dtype=np.int32)
    y_pred = np.array([0] * 90 + [1] * 10 + [0] * 20 + [1] * 80, dtype=np.int32)

    report = generator.generate_report(
        y_true=y_true,
        y_pred=y_pred,
        report_id="report_test_rest",
        job_id="job_test_rest",
        job_fingerprint="c" * 64,
        protocol_id="within_subject",
        partition_id="S1",
        model_taxonomy="global_50_class",
        priors_type="uniform",
    )

    # Precision_rest = TP / (TP + FP) = 90 / (90 + 20) = 90 / 110
    expected_prec = 90.0 / 110.0
    # Recall_rest = TP / (TP + FN) = 90 / (90 + 10) = 90 / 100 = 0.90
    expected_rec = 0.90
    # False Alarm Rate = FP_rest / Total_active = 20 / 100 = 0.20
    expected_far = 0.20

    assert report.rest_metrics.precision == pytest.approx(expected_prec, abs=1e-6)
    assert report.rest_metrics.recall == pytest.approx(expected_rec, abs=1e-6)
    assert report.rest_metrics.false_alarm_rate == pytest.approx(expected_far, abs=1e-6)


def test_evaluator_confusion_matrix_shape_canonical_preservation() -> None:
    """Confusion matrix shape must remain canonical even if classes are absent from test predictions."""
    generator = EvaluationReportGenerator()

    # Small test fixture with only classes 0, 1, 2 present
    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    y_pred = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)

    report = generator.generate_report(
        y_true=y_true,
        y_pred=y_pred,
        report_id="report_test_sparse",
        job_id="job_test_sparse",
        job_fingerprint="d" * 64,
        protocol_id="within_subject",
        partition_id="S1",
        model_taxonomy="global_50_class",
        priors_type="uniform",
    )

    # Matrix MUST be 50x50, not 3x3
    assert report.confusion_matrix.shape == (50, 50)
    assert report.confusion_matrix_shape == [50, 50]
    # Classes 3..49 must have zero support and zero confusion entries
    assert report.confusion_matrix[3:, :].sum() == 0
    assert report.per_class_metrics["49"].support == 0
    assert report.per_class_metrics["49"].precision == 0.0
    assert report.per_class_metrics["49"].recall == 0.0


def test_evaluator_exercise_stratified_baselines() -> None:
    """Secondary exercise-stratified baselines must have appropriate taxonomy and matrix shapes."""
    generator = EvaluationReportGenerator()
    schema = _load_report_schema()

    configs = [
        ("exercise_E1", 18),  # REST + 17 gestures = 18 classes
        ("exercise_E2", 24),  # REST + 23 gestures = 24 classes
        ("exercise_E3", 10),  # REST + 9 gestures = 10 classes
    ]

    for taxonomy, num_classes in configs:
        y_true = np.repeat(np.arange(num_classes, dtype=np.int32), 5)
        y_pred = y_true.copy()

        rep = generator.generate_report(
            y_true=y_true,
            y_pred=y_pred,
            report_id=f"report_{taxonomy}",
            job_id=f"job_{taxonomy}",
            job_fingerprint="e" * 64,
            protocol_id="within_subject",
            partition_id="S1",
            model_taxonomy=taxonomy,
            priors_type="uniform",
        )

        assert rep.model_taxonomy == taxonomy
        assert rep.confusion_matrix.shape == (num_classes, num_classes)
        assert rep.confusion_matrix_shape == [num_classes, num_classes]
        assert len(rep.per_class_metrics) == num_classes
        assert rep.top1_accuracy == pytest.approx(1.0, abs=1e-6)

        # Validates against schema
        jsonschema.validate(instance=rep.to_dict(), schema=schema)


def test_evaluator_consolidated_report_multi_partition_aggregation() -> None:
    """ConsolidatedEvaluationReport must aggregate multiple partitions with mean, median, std."""
    generator = EvaluationReportGenerator()

    # Create 5 fold reports with varying accuracies
    reports: list[EvaluationReport] = []
    for fold_id, acc in enumerate([0.80, 0.82, 0.85, 0.88, 0.90], start=1):
        y_true = np.repeat(np.arange(50, dtype=np.int32), 10)
        # Construct synthetic predictions giving exact accuracy
        y_pred = y_true.copy()
        n_errors = int(len(y_true) * (1.0 - acc))
        y_pred[:n_errors] = (y_true[:n_errors] + 1) % 50

        rep = generator.generate_report(
            y_true=y_true,
            y_pred=y_pred,
            report_id=f"report_fold_{fold_id}",
            job_id=f"job_fold_{fold_id}",
            job_fingerprint="f" * 64,
            protocol_id="cross_subject",
            partition_id=f"fold_{fold_id}",
            model_taxonomy="global_50_class",
            priors_type="uniform",
        )
        reports.append(rep)

    consolidated = generator.generate_consolidated_report(
        reports=reports,
        consolidated_id="consolidated_cross_subject_5folds",
        protocol_id="cross_subject",
        model_taxonomy="global_50_class",
    )

    assert isinstance(consolidated, ConsolidatedEvaluationReport)
    assert len(consolidated.reports) == 5
    assert len(consolidated.subject_or_fold_breakdown) == 5

    # Check aggregate statistics
    accuracies = [r.top1_accuracy for r in reports]
    assert consolidated.aggregate_accuracy.mean == pytest.approx(float(np.mean(accuracies)), abs=1e-6)
    assert consolidated.aggregate_accuracy.median == pytest.approx(float(np.median(accuracies)), abs=1e-6)
    assert consolidated.aggregate_accuracy.std == pytest.approx(float(np.std(accuracies)), abs=1e-6)

    assert 0.0 <= consolidated.aggregate_balanced_accuracy.mean <= 1.0
    assert 0.0 <= consolidated.aggregate_macro_f1.mean <= 1.0


def test_evaluator_adversarial_mismatched_lengths() -> None:
    """Mismatched y_true and y_pred lengths must fail closed with ValueError."""
    generator = EvaluationReportGenerator()
    y_true = np.array([0, 1, 2], dtype=np.int32)
    y_pred = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="[Ll]ength mismatch|sample count mismatch"):
        generator.generate_report(
            y_true=y_true,
            y_pred=y_pred,
            report_id="rep_fail",
            job_id="job_fail",
            job_fingerprint="0" * 64,
            protocol_id="within_subject",
            partition_id="S1",
            model_taxonomy="global_50_class",
        )


def test_evaluator_adversarial_out_of_vocabulary_labels() -> None:
    """Labels outside the configured vocabulary must fail closed with ValueError."""
    generator = EvaluationReportGenerator()

    # 1. Label 50 in global_50_class (only 0..49 allowed)
    y_true = np.array([0, 1, 50], dtype=np.int32)
    y_pred = np.array([0, 1, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="[Vv]ocabulary|[Oo]ut of [Bb]ounds|invalid class"):
        generator.generate_report(
            y_true=y_true,
            y_pred=y_pred,
            report_id="rep_fail_oov",
            job_id="job_fail",
            job_fingerprint="0" * 64,
            protocol_id="within_subject",
            partition_id="S1",
            model_taxonomy="global_50_class",
        )

    # 2. Negative label
    y_true_neg = np.array([0, -1, 2], dtype=np.int32)
    y_pred_neg = np.array([0, 1, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="[Vv]ocabulary|[Oo]ut of [Bb]ounds|invalid class"):
        generator.generate_report(
            y_true=y_true_neg,
            y_pred=y_pred_neg,
            report_id="rep_fail_neg",
            job_id="job_fail",
            job_fingerprint="0" * 64,
            protocol_id="within_subject",
            partition_id="S1",
            model_taxonomy="global_50_class",
        )


def test_evaluator_adversarial_invalid_taxonomy() -> None:
    """Invalid taxonomy string must fail closed with ValueError."""
    generator = EvaluationReportGenerator()
    y_true = np.array([0, 1], dtype=np.int32)
    y_pred = np.array([0, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="[Tt]axonomy|invalid model_taxonomy"):
        generator.generate_report(
            y_true=y_true,
            y_pred=y_pred,
            report_id="rep_fail_tax",
            job_id="job_fail",
            job_fingerprint="0" * 64,
            protocol_id="within_subject",
            partition_id="S1",
            model_taxonomy="invalid_taxonomy_name",
        )


def test_evaluator_adversarial_empty_arrays() -> None:
    """Empty label or prediction arrays must fail closed with ValueError."""
    generator = EvaluationReportGenerator()
    y_true = np.empty(0, dtype=np.int32)
    y_pred = np.empty(0, dtype=np.int32)

    with pytest.raises(ValueError, match="[Ee]mpty|zero samples"):
        generator.generate_report(
            y_true=y_true,
            y_pred=y_pred,
            report_id="rep_fail_empty",
            job_id="job_fail",
            job_fingerprint="0" * 64,
            protocol_id="within_subject",
            partition_id="S1",
            model_taxonomy="global_50_class",
        )


def test_evaluator_adversarial_non_integer_or_nan() -> None:
    """Non-integer, NaN, or Inf values in labels must fail closed with ValueError."""
    generator = EvaluationReportGenerator()
    y_true = np.array([0.0, np.nan, 2.0])
    y_pred = np.array([0.0, 1.0, 2.0])

    with pytest.raises((ValueError, TypeError), match="[Nn]a[Nn]|[Ff]inite|[Ii]nteger"):
        generator.generate_report(
            y_true=y_true,  # type: ignore[arg-type]
            y_pred=y_pred,  # type: ignore[arg-type]
            report_id="rep_fail_nan",
            job_id="job_fail",
            job_fingerprint="0" * 64,
            protocol_id="within_subject",
            partition_id="S1",
            model_taxonomy="global_50_class",
        )


def test_evaluator_adversarial_no_active_samples_in_active_only() -> None:
    """Generating active-only report with 100% REST samples must fail closed."""
    generator = EvaluationReportGenerator()
    y_true = np.zeros(20, dtype=np.int32)  # All REST
    y_pred = np.zeros(20, dtype=np.int32)

    with pytest.raises(ValueError, match="[Nn]o active samples|active-only"):
        generator.generate_active_only_report(
            y_true=y_true,
            y_pred=y_pred,
            report_id="rep_fail_no_active",
            job_id="job_fail",
            job_fingerprint="0" * 64,
            protocol_id="within_subject",
            partition_id="S1",
        )


def test_evaluator_determinism() -> None:
    """Identical inputs must yield bit-for-bit identical metrics and confusion matrix hash."""
    generator = EvaluationReportGenerator()
    rng = np.random.default_rng(123)
    y_true = rng.integers(0, 50, size=500, dtype=np.int32)
    y_pred = rng.integers(0, 50, size=500, dtype=np.int32)

    rep1 = generator.generate_report(
        y_true=y_true,
        y_pred=y_pred,
        report_id="rep_det_1",
        job_id="job_det",
        job_fingerprint="7" * 64,
        protocol_id="within_subject",
        partition_id="S1",
        model_taxonomy="global_50_class",
    )

    rep2 = generator.generate_report(
        y_true=y_true,
        y_pred=y_pred,
        report_id="rep_det_1",
        job_id="job_det",
        job_fingerprint="7" * 64,
        protocol_id="within_subject",
        partition_id="S1",
        model_taxonomy="global_50_class",
    )

    assert rep1.top1_accuracy == rep2.top1_accuracy
    assert rep1.balanced_accuracy == rep2.balanced_accuracy
    assert rep1.macro_f1 == rep2.macro_f1
    assert rep1.confusion_matrix_hash == rep2.confusion_matrix_hash
    np.testing.assert_array_equal(rep1.confusion_matrix, rep2.confusion_matrix)
    assert rep1.to_dict() == rep2.to_dict()
