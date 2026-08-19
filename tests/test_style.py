"""Tests for Mantpy's explicit Matplotlib style preset."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import matplotlib as mpl
import pytest

import mantpy as mt

_PUBLICATION_STYLE: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.4,
    "lines.linewidth": 0.4,
    "patch.linewidth": 0.4,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
}


@pytest.fixture
def _rc_snapshot():
    """Restore the process-wide Matplotlib configuration after each test."""
    saved = mpl.rcParams.copy()
    try:
        yield
    finally:
        mpl.rcParams.update(saved)


def test_import_does_not_mutate_rcparams() -> None:
    """A fresh package import must leave all Matplotlib settings untouched."""
    code = textwrap.dedent(
        """
        import sys
        import matplotlib as mpl

        before = mpl.rcParams.copy()
        import mantpy
        after = mpl.rcParams.copy()

        assert before == after
        assert "matplotlib.pyplot" not in sys.modules
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_style_has_one_generic_public_entry_point() -> None:
    """The opt-in style API should expose only its neutral public name."""
    assert mt.style.__all__ == ["apply_publication_style"]
    assert not hasattr(mt.style, "apply_import_defaults")


def test_apply_publication_style_pins_canonical_keys(_rc_snapshot) -> None:
    """The explicit helper must retain the established plotting values."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(_PUBLICATION_STYLE)
    expected = {key: mpl.rcParams[key] for key in _PUBLICATION_STYLE}

    mpl.rcParams.update(mpl.rcParamsDefault)
    applied = mt.style.apply_publication_style()
    actual = {key: mpl.rcParams[key] for key in _PUBLICATION_STYLE}

    assert applied == _PUBLICATION_STYLE
    assert actual == expected


def test_overrides_apply_after_preset(_rc_snapshot) -> None:
    """Caller overrides should take precedence over the preset."""
    applied = mt.style.apply_publication_style(**{"axes.linewidth": 0.6})

    assert applied["axes.linewidth"] == 0.6
    assert mpl.rcParams["axes.linewidth"] == 0.6
