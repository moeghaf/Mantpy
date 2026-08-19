"""Tests for mantpy._plot_helpers.plot_delta_masked."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.image import AxesImage

import mantpy as mt


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close("all")


def _toy_inputs(K=4, seed=0):
    rng = np.random.default_rng(seed)
    delta = rng.normal(size=(K, K))
    p_fdr = rng.uniform(0, 1, size=(K, K))
    # Force one cell to be highly significant for the star-annotation test.
    p_fdr[0, 1] = 1e-4
    delta[0, 1] = 2.5
    return delta, p_fdr


class TestPanelDeltaMasked:
    def test_descriptive_mode_needs_no_p_values_or_annotations(self):
        delta = np.array([[0.0, 1.5], [-0.5, 0.0]])
        fig, ax = plt.subplots()
        im = mt.pl.plot_delta_masked(
            ax,
            delta,
            None,
            palette={0: "#000", 1: "#777"},
            K=2,
            mask_nonsignificant=False,
        )

        np.testing.assert_array_equal(np.asarray(im.get_array()), delta)
        assert not ax.texts

    def test_returns_image(self):
        delta, p_fdr = _toy_inputs()
        palette = {k: f"#{k * 30:02x}{k * 40:02x}{k * 50:02x}" for k in range(4)}
        fig, ax = plt.subplots()
        im = mt.pl.plot_delta_masked(
            ax,
            delta,
            p_fdr,
            palette=palette,
            K=4,
        )
        assert isinstance(im, AxesImage)

    def test_insignificant_cells_are_nan(self):
        # Cells with p_fdr >= alpha_fdr are masked to NaN.
        K = 3
        delta = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        p_fdr = np.full((K, K), 0.5)  # nothing significant
        p_fdr[1, 2] = 0.001  # one significant cell
        palette = {0: "#000", 1: "#777", 2: "#fff"}
        fig, ax = plt.subplots()
        im = mt.pl.plot_delta_masked(
            ax,
            delta,
            p_fdr,
            palette=palette,
            K=K,
            alpha_fdr=0.05,
        )
        arr = im.get_array()  # numpy.ma.MaskedArray
        # All cells masked except (1,2).
        assert np.ma.is_masked(arr[0, 0])
        assert np.ma.is_masked(arr[2, 2])
        assert not np.ma.is_masked(arr[1, 2])
        assert arr[1, 2] == 6.0

    def test_star_annotations(self):
        K = 2
        delta = np.array([[0.0, 2.5], [0.0, 0.0]])
        # p<0.001 -> ***, p<0.01 -> **, p<alpha_fdr -> *
        p_fdr = np.array([[1.0, 1e-4], [1.0, 1.0]])
        palette = {0: "#000", 1: "#fff"}
        fig, ax = plt.subplots()
        mt.pl.plot_delta_masked(
            ax,
            delta,
            p_fdr,
            palette=palette,
            K=K,
        )
        # Look for the *** text annotation on the significant cell.
        texts = [t.get_text() for t in ax.texts]
        assert any("***" in t for t in texts)
        assert any("+2.5" in t for t in texts)

    def test_tick_label_colors(self):
        delta, p_fdr = _toy_inputs(K=3)
        palette = {0: "#aa0000", 1: "#00aa00", 2: "#0000aa"}
        fig, ax = plt.subplots()
        mt.pl.plot_delta_masked(
            ax,
            delta,
            p_fdr,
            palette=palette,
            K=3,
        )
        for k, lab in enumerate(ax.get_yticklabels()):
            assert lab.get_color() == palette[k]
            assert lab.get_fontweight() == "bold"
