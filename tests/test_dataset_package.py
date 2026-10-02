"""Tests for semg_dataset package initialization and public namespace."""


def test_package_importable_and_has_version() -> None:
    """Verify semg_dataset is importable and defines __version__."""
    import semg_dataset

    assert hasattr(semg_dataset, "__version__")
    assert isinstance(semg_dataset.__version__, str)
    assert len(semg_dataset.__version__) > 0


def test_package_exposes_expected_namespace() -> None:
    """Verify semg_dataset defines __all__ with expected core symbols."""
    import semg_dataset

    assert hasattr(semg_dataset, "__all__")
    assert isinstance(semg_dataset.__all__, list)
