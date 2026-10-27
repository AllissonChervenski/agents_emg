"""Governed host-only LDA factory for Feature 004 SYNTHETIC acceptance checks.

This helper is not the canonical 50-class scientific training implementation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from semg_baseline.governance import assert_sklearn_allowed

if TYPE_CHECKING:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def make_validation_lda() -> LinearDiscriminantAnalysis:
    """Instantiate the pre-existing demonstration model after governance approval."""
    assert_sklearn_allowed()
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
