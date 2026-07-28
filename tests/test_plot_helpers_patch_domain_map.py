"""Tests for ``mt.pl.patch_domain_map``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mantpy as mt


class TestPatchDomainMap:
    def _patches(self):
        ys = np.array([0, 0, 1, 1])
        xs = np.array([0, 1, 0, 1])
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], float)
        return ys, xs, colors

    def test_returns_axis_and_draws_image(self):
        ys, xs, colors = self._patches()
        fig, ax = plt.subplots()
        out = mt.pl.patch_domain_map(ax, ys, xs, colors, shape=(2, 2), crop=False)
        assert out is ax
        assert len(ax.images) == 1
        plt.close(fig)

    def test_crop_shrinks_canvas(self):
        ys = np.array([5, 6])
        xs = np.array([5, 6])
        colors = np.array([[1, 0, 0], [0, 1, 0]], float)
        fig, ax = plt.subplots()
        mt.pl.patch_domain_map(ax, ys, xs, colors, shape=(20, 20), crop=True, crop_pad=1)
        # cropped canvas is far smaller than the full 20x20
        assert ax.images[0].get_array().shape[0] < 20
        plt.close(fig)

    def test_edge_color_and_title(self):
        ys, xs, colors = self._patches()
        fig, ax = plt.subplots()
        mt.pl.patch_domain_map(ax, ys, xs, colors, shape=(2, 2), edge_color="#123456", title="map")
        assert ax.get_title() == "map"
        plt.close(fig)
