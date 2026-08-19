"""Import-order tests for the public and private plotting modules."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "order",
    [
        ("mantpy.pl", "mantpy._plot_helpers"),
        ("mantpy._plot_helpers", "mantpy.pl"),
    ],
)
def test_plotting_modules_import_in_any_order(order: tuple[str, ...]) -> None:
    """Composite helpers must not rely on a partially initialised module."""
    code = f"""
import importlib

for name in {order!r}:
    importlib.import_module(name)

import mantpy as mt
for name in (
    "niche_bubble",
    "plot_marker_otsu_composite",
):
    assert callable(getattr(mt.pl, name))

from mantpy import _plot_helpers
assert callable(_plot_helpers.plot_marker_otsu_composite)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
