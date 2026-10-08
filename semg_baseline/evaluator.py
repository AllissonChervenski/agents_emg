"""Standardized structured evaluation reports and dual-vocabulary report generator.

Implements FR-007, FR-010, SC-005, Decision-4, Decision-5:
- Full TrainingJob traceability: report_id, job_id, job_fingerprint, protocol_id, partition_id, model_taxonomy, priors_type.
- Canonical model taxonomies: global_50_class (50 classes), exercise_E1 (18 classes),
  exercise_E2 (24 classes), exercise_E3 (10 classes).
- Strict separation: active_only_49_class is an evaluation view derived from global_50_class predictions,
  NOT a separate trained model taxonomy.
- Dual-vocabulary reporting:
    * Primary: full vocabulary (0..K-1) with explicit REST class 0 tracking.
    * Active-only view: derived over active subset (y_true > 0), proving 50-class primary != 49-class active-only.
- Active-to-rest misclassifications (y_true > 0, y_pred == 0) are strictly preserved as errors:
    * Reflected in active Top-1 Accuracy, Balanced Accuracy, and Macro-F1.
    * Preserved in rectangular active confusion matrix of shape [K-1, K] where column 0 indexes
      predictions to REST, ensuring row sum equals class support and zero active samples are lost.
    * active_to_rest_count explicitly recorded.
- REST metrics (precision, recall, false_alarm_rate = 1.0 - recall_rest) validated for every
  taxonomy containing REST=0 (global_50_class, exercise_E1, exercise_E2, exercise_E3).
- Bounded metrics: all normalized rates strictly in [0.0, 1.0].
- Canonical dimension preservation: confusion matrices maintain canonical shapes even when
  classes are absent in predictions.
- Consolidated multi-partition aggregation: ConsolidatedEvaluationReport with mean, median, std.
- Deterministic reproducibility: bit-for-bit identical outputs and SHA-256 confusion matrix hashes.
- Fail-closed validation for mismatched lengths, out-of-vocabulary labels, empty arrays,
  non-integers/NaNs, and zero active samples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from semg_baseline.cross_subject import MetricStats


CANONICAL_TAXONOMIES: dict[str, int] = {
    "global_50_class": 50,
    "exercise_E1": 18,
    "exercise_E2": 24,
    "exercise_E3": 10,
}


@dataclass(frozen=True)
class ClassMetric:
    """Performance metrics for an individual gesture class."""

    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching JSON Schema."""
        return {
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "support": int(self.support),
        }


@dataclass(frozen=True)
class RestMetric:
    """Specialized evaluation metrics for the resting baseline state (class 0)."""

    precision: float
    recall: float
    false_alarm_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching JSON Schema."""
        return {
            "precision": float(self.precision),
            "recall": float(self.recall),
            "false_alarm_rate": float(self.false_alarm_rate),
        }


@dataclass(frozen=True)
class ActiveOnlyMetric:
    """Derived evaluation metrics computed strictly over the active gesture subset (y_true > 0)."""

    top1_accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class_metrics: dict[str, ClassMetric]
    confusion_matrix: np.ndarray
    confusion_matrix_shape: list[int]
    confusion_matrix_hash: str
    active_to_rest_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching evaluation_report_schema.json."""
        return {
            "top1_accuracy": float(self.top1_accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "macro_f1": float(self.macro_f1),
            "per_class_metrics": {k: v.to_dict() for k, v in self.per_class_metrics.items()},
            "confusion_matrix_shape": list(self.confusion_matrix_shape),
            "confusion_matrix_hash": str(self.confusion_matrix_hash),
            "active_to_rest_count": int(self.active_to_rest_count),
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Standardized evaluation report with TrainingJob traceability for a single run/partition."""

    report_id: str
    job_id: str
    job_fingerprint: str
    protocol_id: str
    partition_id: str
    model_taxonomy: str
    priors_type: str
    top1_accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class_metrics: dict[str, ClassMetric]
    rest_metrics: RestMetric
    active_only_metrics: ActiveOnlyMetric
    confusion_matrix: np.ndarray
    confusion_matrix_shape: list[int]
    confusion_matrix_hash: str
    subject_or_fold_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    confounding_diagnostic: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary strictly conforming to evaluation_report_schema.json."""
        d: dict[str, Any] = {
            "report_id": self.report_id,
            "job_id": self.job_id,
            "job_fingerprint": self.job_fingerprint,
            "protocol_id": self.protocol_id,
            "partition_id": self.partition_id,
            "model_taxonomy": self.model_taxonomy,
            "priors_type": self.priors_type,
            "top1_accuracy": float(self.top1_accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "macro_f1": float(self.macro_f1),
            "per_class_metrics": {k: v.to_dict() for k, v in self.per_class_metrics.items()},
            "rest_metrics": self.rest_metrics.to_dict(),
            "active_only_metrics": self.active_only_metrics.to_dict(),
            "confusion_matrix_shape": list(self.confusion_matrix_shape),
            "confusion_matrix_hash": str(self.confusion_matrix_hash),
            "subject_or_fold_breakdown": {
                k: {
                    "top1_accuracy": float(v["top1_accuracy"]),
                    "balanced_accuracy": float(v["balanced_accuracy"]),
                    "macro_f1": float(v["macro_f1"]),
                }
                for k, v in self.subject_or_fold_breakdown.items()
            },
        }
        if self.confounding_diagnostic is not None:
            d["confounding_diagnostic"] = self.confounding_diagnostic
        return d


@dataclass(frozen=True)
class ConsolidatedEvaluationReport:
    """Aggregated evaluation report across multiple subjects or folds."""

    consolidated_id: str
    protocol_id: str
    model_taxonomy: str
    reports: list[EvaluationReport]
    subject_or_fold_breakdown: dict[str, dict[str, float]]
    aggregate_accuracy: MetricStats
    aggregate_balanced_accuracy: MetricStats
    aggregate_macro_f1: MetricStats
    aggregate_active_accuracy: MetricStats
    aggregate_active_macro_f1: MetricStats


class EvaluationReportGenerator:
    """Generates canonical EvaluationReport and ConsolidatedEvaluationReport instances."""

    def __init__(self) -> None:
        pass

    def _validate_inputs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_taxonomy: str,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Validate input arrays, lengths, types, and vocabulary bounds fail-closed."""
        if model_taxonomy not in CANONICAL_TAXONOMIES:
            raise ValueError(
                f"Invalid model_taxonomy '{model_taxonomy}'. Must be one of "
                f"{list(CANONICAL_TAXONOMIES.keys())}."
            )
        num_classes = CANONICAL_TAXONOMIES[model_taxonomy]

        # Type & finiteness validation
        try:
            yt = np.asarray(y_true)
            yp = np.asarray(y_pred)
        except Exception as e:
            raise ValueError(f"Failed to convert inputs to numpy arrays: {e}") from e

        if yt.ndim != 1 or yp.ndim != 1:
            raise ValueError(
                f"y_true and y_pred must be 1D arrays; got shapes {yt.shape} and {yp.shape}"
            )

        if len(yt) != len(yp):
            raise ValueError(
                f"Length mismatch: len(y_true)={len(yt)} != len(y_pred)={len(yp)}"
            )

        if len(yt) == 0:
            raise ValueError("Input arrays y_true and y_pred are empty: zero samples provided")

        if not np.all(np.isfinite(yt)) or not np.all(np.isfinite(yp)):
            raise ValueError("y_true or y_pred contains non-finite values (NaN or Inf)")

        # Ensure integer values
        if not np.issubdtype(yt.dtype, np.integer):
            if not np.all(np.equal(np.mod(yt, 1), 0)):
                raise ValueError("y_true contains non-integer float values")
            yt = yt.astype(np.int32)
        else:
            yt = yt.astype(np.int32)

        if not np.issubdtype(yp.dtype, np.integer):
            if not np.all(np.equal(np.mod(yp, 1), 0)):
                raise ValueError("y_pred contains non-integer float values")
            yp = yp.astype(np.int32)
        else:
            yp = yp.astype(np.int32)

        # Vocabulary bounds validation
        if np.any(yt < 0) or np.any(yt >= num_classes):
            oov = np.unique(yt[(yt < 0) | (yt >= num_classes)]).tolist()
            raise ValueError(
                f"Out of bounds labels in y_true for taxonomy '{model_taxonomy}' "
                f"(expected [0, {num_classes - 1}]): invalid classes {oov}"
            )

        if np.any(yp < 0) or np.any(yp >= num_classes):
            oov = np.unique(yp[(yp < 0) | (yp >= num_classes)]).tolist()
            raise ValueError(
                f"Out of bounds labels in y_pred for taxonomy '{model_taxonomy}' "
                f"(expected [0, {num_classes - 1}]): invalid classes {oov}"
            )

        if not np.any(yt != 0):
            raise ValueError(
                "No active samples present in y_true (100% REST). "
                "Active-only evaluation view requires at least one active gesture sample."
            )

        return yt, yp, num_classes

    def _compute_active_only_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
    ) -> ActiveOnlyMetric:
        """Compute derived evaluation metrics strictly over active gestures (y_true > 0)."""
        active_mask = y_true != 0
        if not np.any(active_mask):
            raise ValueError(
                "No active samples present in y_true (100% REST). "
                "Active-only evaluation view requires at least one active gesture sample."
            )

        yt_active = y_true[active_mask]
        yp_active = y_pred[active_mask]
        n_active = len(yt_active)

        # Count active samples mistakenly predicted as REST (column 0)
        active_to_rest_count = int(np.sum(yp_active == 0))

        # Active Top-1 Accuracy: exact match over all active samples
        active_top1_acc = float(np.sum(yt_active == yp_active) / float(n_active))

        # Rectangular active confusion matrix: shape [num_classes - 1, num_classes]
        # Rows: true active classes 1..num_classes-1 (0-indexed as row = class - 1)
        # Cols: predicted classes 0..num_classes-1 (col 0 = REST, cols 1..K-1 = active)
        n_active_classes = num_classes - 1
        active_cm = np.zeros((n_active_classes, num_classes), dtype=np.int32)

        for c in range(1, num_classes):
            row_idx = c - 1
            mask_c = yt_active == c
            if np.any(mask_c):
                preds_c = yp_active[mask_c]
                for p in range(num_classes):
                    active_cm[row_idx, p] = int(np.sum(preds_c == p))

        cm_shape = [n_active_classes, num_classes]
        cm_hash = hashlib.sha256(active_cm.tobytes()).hexdigest()

        # Compute per-class precision, recall, f1, support for active classes 1..num_classes-1
        per_class: dict[str, ClassMetric] = {}
        recalls_list: list[float] = []
        f1s_list: list[float] = []

        for c in range(1, num_classes):
            row_idx = c - 1
            c_str = str(c)
            support_c = int(active_cm[row_idx, :].sum())

            tp_c = float(active_cm[row_idx, c])
            fp_c = float(np.sum(active_cm[:, c]) - tp_c)
            fn_c = float(support_c - tp_c)

            prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0.0 else 0.0
            rec = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0.0 else 0.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0

            per_class[c_str] = ClassMetric(
                precision=float(prec),
                recall=float(rec),
                f1=float(f1),
                support=support_c,
            )

            if support_c > 0:
                recalls_list.append(rec)
            f1s_list.append(f1)

        active_bal_acc = float(np.mean(recalls_list)) if recalls_list else 0.0
        active_macro_f1 = float(np.mean(f1s_list)) if f1s_list else 0.0

        return ActiveOnlyMetric(
            top1_accuracy=active_top1_acc,
            balanced_accuracy=active_bal_acc,
            macro_f1=active_macro_f1,
            per_class_metrics=per_class,
            confusion_matrix=active_cm,
            confusion_matrix_shape=cm_shape,
            confusion_matrix_hash=cm_hash,
            active_to_rest_count=active_to_rest_count,
        )

    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        report_id: str,
        job_id: str,
        job_fingerprint: str,
        protocol_id: str,
        partition_id: str,
        model_taxonomy: str = "global_50_class",
        priors_type: str = "uniform",
        subject_or_fold_breakdown: Mapping[str, Mapping[str, float]] | None = None,
        confounding_diagnostic: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Generate standardized EvaluationReport covering primary and derived active-only metrics."""
        yt, yp, num_classes = self._validate_inputs(y_true, y_pred, model_taxonomy)

        # Primary multi-class confusion matrix: shape [num_classes, num_classes]
        all_classes = np.arange(num_classes, dtype=np.int32)
        cm = confusion_matrix(yt, yp, labels=all_classes).astype(np.int32)
        cm_shape = [num_classes, num_classes]
        cm_hash = hashlib.sha256(cm.tobytes()).hexdigest()

        # Primary global metrics
        top1_acc = float(accuracy_score(yt, yp))
        bal_acc = float(balanced_accuracy_score(yt, yp))
        m_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))

        # Per-class metrics
        per_class_metrics: dict[str, ClassMetric] = {}
        for c in range(num_classes):
            c_str = str(c)
            support_c = int(cm[c, :].sum())
            tp_c = float(cm[c, c])
            fp_c = float(np.sum(cm[:, c]) - tp_c)
            fn_c = float(support_c - tp_c)

            prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0.0 else 0.0
            rec = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0.0 else 0.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0

            per_class_metrics[c_str] = ClassMetric(
                precision=float(prec),
                recall=float(rec),
                f1=float(f1),
                support=support_c,
            )

        # REST metrics (class 0 is REST for all canonical taxonomies)
        n_rest = int(cm[0, :].sum())
        tp_rest = float(cm[0, 0])
        fp_rest = float(np.sum(cm[1:, 0]))
        fn_rest = float(n_rest - tp_rest)

        prec_rest = tp_rest / (tp_rest + fp_rest) if (tp_rest + fp_rest) > 0.0 else 0.0
        rec_rest = tp_rest / float(n_rest) if n_rest > 0 else 0.0
        # false_alarm_rate = 1.0 - recall_rest (# REST predicted as any active / # total REST)
        far_rest = fn_rest / float(n_rest) if n_rest > 0 else 0.0

        rest_metrics = RestMetric(
            precision=float(prec_rest),
            recall=float(rec_rest),
            false_alarm_rate=float(far_rest),
        )

        # Derived active-only view
        active_only = self._compute_active_only_metrics(yt, yp, num_classes)

        # Form subject_or_fold_breakdown dict
        breakdown_dict: dict[str, dict[str, float]] = {}
        if subject_or_fold_breakdown is not None:
            for part_k, part_v in subject_or_fold_breakdown.items():
                breakdown_dict[part_k] = {
                    "top1_accuracy": float(part_v.get("top1_accuracy", 0.0)),
                    "balanced_accuracy": float(part_v.get("balanced_accuracy", 0.0)),
                    "macro_f1": float(part_v.get("macro_f1", 0.0)),
                }

        return EvaluationReport(
            report_id=str(report_id),
            job_id=str(job_id),
            job_fingerprint=str(job_fingerprint),
            protocol_id=str(protocol_id),
            partition_id=str(partition_id),
            model_taxonomy=str(model_taxonomy),
            priors_type=str(priors_type),
            top1_accuracy=top1_acc,
            balanced_accuracy=bal_acc,
            macro_f1=m_f1,
            per_class_metrics=per_class_metrics,
            rest_metrics=rest_metrics,
            active_only_metrics=active_only,
            confusion_matrix=cm,
            confusion_matrix_shape=cm_shape,
            confusion_matrix_hash=cm_hash,
            subject_or_fold_breakdown=breakdown_dict,
            confounding_diagnostic=confounding_diagnostic,
        )

    def generate_consolidated_report(
        self,
        reports: Sequence[EvaluationReport],
        consolidated_id: str,
        protocol_id: str,
        model_taxonomy: str,
    ) -> ConsolidatedEvaluationReport:
        """Aggregate multiple partition reports into a ConsolidatedEvaluationReport."""
        if not reports:
            raise ValueError("reports list cannot be empty for consolidated report")

        for r in reports:
            if r.model_taxonomy != model_taxonomy:
                raise ValueError(
                    f"Model taxonomy mismatch in consolidated report: expected '{model_taxonomy}', "
                    f"got report with '{r.model_taxonomy}'. Averaging across disparate taxonomies is prohibited."
                )
            if r.protocol_id != protocol_id:
                raise ValueError(
                    f"Protocol mismatch in consolidated report: expected '{protocol_id}', "
                    f"got report with '{r.protocol_id}'."
                )

        accuracies = [r.top1_accuracy for r in reports]
        bal_accs = [r.balanced_accuracy for r in reports]
        macro_f1s = [r.macro_f1 for r in reports]
        active_accs = [r.active_only_metrics.top1_accuracy for r in reports]
        active_f1s = [r.active_only_metrics.macro_f1 for r in reports]

        breakdown: dict[str, dict[str, float]] = {
            r.partition_id: {
                "top1_accuracy": float(r.top1_accuracy),
                "balanced_accuracy": float(r.balanced_accuracy),
                "macro_f1": float(r.macro_f1),
            }
            for r in reports
        }

        agg_acc = MetricStats(
            mean=float(np.mean(accuracies)),
            median=float(np.median(accuracies)),
            std=float(np.std(accuracies, ddof=0)),
        )
        agg_bal = MetricStats(
            mean=float(np.mean(bal_accs)),
            median=float(np.median(bal_accs)),
            std=float(np.std(bal_accs, ddof=0)),
        )
        agg_f1 = MetricStats(
            mean=float(np.mean(macro_f1s)),
            median=float(np.median(macro_f1s)),
            std=float(np.std(macro_f1s, ddof=0)),
        )
        agg_active_acc = MetricStats(
            mean=float(np.mean(active_accs)),
            median=float(np.median(active_accs)),
            std=float(np.std(active_accs, ddof=0)),
        )
        agg_active_f1 = MetricStats(
            mean=float(np.mean(active_f1s)),
            median=float(np.median(active_f1s)),
            std=float(np.std(active_f1s, ddof=0)),
        )

        return ConsolidatedEvaluationReport(
            consolidated_id=str(consolidated_id),
            protocol_id=str(protocol_id),
            model_taxonomy=str(model_taxonomy),
            reports=list(reports),
            subject_or_fold_breakdown=breakdown,
            aggregate_accuracy=agg_acc,
            aggregate_balanced_accuracy=agg_bal,
            aggregate_macro_f1=agg_f1,
            aggregate_active_accuracy=agg_active_acc,
            aggregate_active_macro_f1=agg_active_f1,
        )


__all__ = [
    "ActiveOnlyMetric",
    "ClassMetric",
    "ConsolidatedEvaluationReport",
    "EvaluationReport",
    "EvaluationReportGenerator",
    "RestMetric",
]
