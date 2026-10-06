"""Tests for semg_baseline package initialization and public namespace (T001)."""

import sys


def test_baseline_package_importable_and_has_version() -> None:
    """Verify semg_baseline is importable and defines __version__ string."""
    import semg_baseline

    assert hasattr(semg_baseline, "__version__")
    assert isinstance(semg_baseline.__version__, str)
    assert len(semg_baseline.__version__) > 0


def test_baseline_package_namespace_strict() -> None:
    """Verify semg_baseline defines __all__ containing only actually existing symbols."""
    import semg_baseline

    assert hasattr(semg_baseline, "__all__")
    assert isinstance(semg_baseline.__all__, list)
    for symbol in semg_baseline.__all__:
        assert hasattr(semg_baseline, symbol), f"Exported symbol '{symbol}' not found on semg_baseline"

    # Enforce that future modules are not stubbed or eagerly exported prematurely in T001
    future_symbols = [
        "CausalBandpassFilter",
        "PureWindowSegmenter",
        "TimeDomainFeatureExtractor",
        "ThresholdCalibrator",
        "FeatureScaler",
        "CanonicalLdaTrainer",
        "NeutralModelExporter",
        "NeutralNumpyPredictor",
        "TrainingJobExecutor",
        "EvaluationReportGenerator",
        "ExerciseConfoundingAuditor",
    ]
    for sym in future_symbols:
        assert sym not in semg_baseline.__all__, f"Future symbol '{sym}' must not be exported in T001"


def test_baseline_package_no_sklearn_import_side_effect() -> None:
    """Verify that importing semg_baseline does not import scikit-learn."""
    # Ensure clean state for check
    import semg_baseline  # noqa: F401

    assert "sklearn" not in sys.modules, "semg_baseline must not import sklearn during package initialization"
