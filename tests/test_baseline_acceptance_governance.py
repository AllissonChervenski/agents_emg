"""Contract tests for allowed imports in Feature 004 acceptance script."""
from pathlib import Path

from semg_baseline.governance import verify_imports_in_source

ROOT = Path(__file__).resolve().parents[1]


def test_feature004_validator_imports_follow_dependency_governance() -> None:
    script = ROOT / 'scripts' / 'validate_feature_004.py'
    allowed, violations = verify_imports_in_source(
        script.read_text(encoding='utf-8'), 'scripts/validate_feature_004.py'
    )
    assert allowed, violations


def test_feature004_lda_factory_imports_are_in_approved_scope() -> None:
    helper = ROOT / 'semg_baseline' / 'acceptance_models.py'
    allowed, violations = verify_imports_in_source(
        helper.read_text(encoding='utf-8'), 'semg_baseline/acceptance_models.py'
    )
    assert allowed, violations


def test_feature004_lda_factory_preserves_validation_model_config() -> None:
    from semg_baseline.acceptance_models import make_validation_lda

    clf = make_validation_lda()
    assert clf.solver == 'lsqr'
    assert clf.shrinkage == 'auto'
