"""Tests for semg_dsp core package layout (T001).

Requirements: FR-001
Acceptance criteria: AC-010
"""

import importlib.util


def test_semg_dsp_package_exists():
    """Verify that semg_dsp package exists and is discoverable."""
    spec = importlib.util.find_spec("semg_dsp")
    assert spec is not None, "semg_dsp package must exist and be importable"


def test_semg_dsp_import():
    """Verify that semg_dsp imports cleanly without errors."""
    import semg_dsp  # noqa: F401
