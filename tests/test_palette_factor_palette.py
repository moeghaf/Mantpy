"""Tests for :func:`mantpy.palette.factor_palette`."""

from __future__ import annotations

import pytest

import mantpy as mt


def test_condition_hue_two_values():
    pal = mt.palette.factor_palette(["Naive", "Infected"], scheme="condition_hue")
    assert pal == {
        "Naive": mt.palette.CONDITION_HUE_PALETTE["cool"],
        "Infected": mt.palette.CONDITION_HUE_PALETTE["warm"],
    }


def test_genotype_luminance_two_values():
    pal = mt.palette.factor_palette(["KO", "WT"], scheme="genotype_luminance")
    assert pal == {
        "KO": mt.palette.GENOTYPE_LUMINANCE_PALETTE["dark"],
        "WT": mt.palette.GENOTYPE_LUMINANCE_PALETTE["light"],
    }


def test_order_is_preserved():
    """The first value gets the first role colour regardless of alphabetics."""
    pal_a = mt.palette.factor_palette(["A", "B"], scheme="condition_hue")
    pal_b = mt.palette.factor_palette(["B", "A"], scheme="condition_hue")
    assert pal_a["A"] == pal_b["B"]
    assert pal_a["B"] == pal_b["A"]


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unknown scheme"):
        mt.palette.factor_palette(["x"], scheme="not_a_scheme")


def test_too_many_values_raises():
    with pytest.raises(ValueError, match="has only 2 roles"):
        mt.palette.factor_palette(["a", "b", "c"], scheme="condition_hue")


def test_single_value_works():
    """Schemes accept fewer values than their role count."""
    pal = mt.palette.factor_palette(["Solo"], scheme="condition_hue")
    assert pal == {"Solo": mt.palette.CONDITION_HUE_PALETTE["cool"]}
