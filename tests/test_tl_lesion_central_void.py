"""Tests for :func:`mantpy.tl.lesion_central_void` and its NamedTuple result.

Critical contract: the new ``CentralVoidResult`` NamedTuple must unpack
positionally exactly like the legacy 5-tuple, so existing notebook code
that does ``area, *_ = lesion_central_void(...)`` or full 5-tuple
unpacking continues to work bit-identically.
"""

from __future__ import annotations

import numpy as np
import pytest

# gudhi is an optional dep; skip the whole module if missing.
pytest.importorskip("gudhi")

import mantpy as mt
from mantpy.tl import CentralVoidResult


def _doughnut_coords(r_in: float = 10.0, r_out: float = 30.0, n: int = 40) -> np.ndarray:
    """Two concentric rings of points — produces a clear central void."""
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, n)
    inner = np.column_stack([r_in * np.cos(angles), r_in * np.sin(angles)])
    angles2 = rng.uniform(0, 2 * np.pi, n)
    outer = np.column_stack([r_out * np.cos(angles2), r_out * np.sin(angles2)])
    return np.vstack([inner, outer])


def test_result_is_named_tuple():
    coords = _doughnut_coords()
    result = mt.tl.lesion_central_void(coords, radius=40.0)
    assert isinstance(result, CentralVoidResult)
    assert isinstance(result, tuple)  # NamedTuple is a tuple subclass


def test_legacy_partial_unpack_compatible():
    """``area, *_ = lesion_central_void(...)`` must still work."""
    coords = _doughnut_coords()
    area, *_ = mt.tl.lesion_central_void(coords, radius=40.0)
    assert isinstance(area, float)
    assert area >= 0.0


def test_legacy_full_unpack_compatible():
    """5-tuple unpacking must yield exactly the legacy fields, in order."""
    coords = _doughnut_coords()
    result = mt.tl.lesion_central_void(coords, radius=40.0)
    area, comp, kept_tris, kept_edges, d_tri = result
    assert area == result.area
    assert comp is result.component
    assert kept_tris is result.kept_tris
    assert kept_edges is result.kept_edges
    assert d_tri is result.d_tri


def test_attribute_access():
    coords = _doughnut_coords()
    r = mt.tl.lesion_central_void(coords, radius=40.0)
    # Attribute access is the new preferred API.
    assert isinstance(r.area, float)
    assert isinstance(r.component, list)
    assert isinstance(r.kept_tris, set)
    assert isinstance(r.kept_edges, set)
    assert r.d_tri is not None


def test_degenerate_too_few_points_returns_sentinel_tuple():
    """<4 points returns the same (0.0, [], set(), set(), None) sentinel."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    result = mt.tl.lesion_central_void(coords, radius=10.0)
    area, comp, kept_tris, kept_edges, d_tri = result
    assert area == 0.0
    assert comp == []
    assert kept_tris == set()
    assert kept_edges == set()
    assert d_tri is None
