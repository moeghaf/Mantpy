"""Tests for the scale-bar option of ``mt.pl.channel_overlay_on_neighbours``.

The overlay draws a raw-channel IMC composite restricted to the ECM patches a
cell type touches. When ``scalebar_um`` is supplied together with ``pixel_size``
(micrometres per pixel) an anchored size bar is added to the axes; supplying
``scalebar_um`` without ``pixel_size`` is an error, and omitting both leaves the
axes bar-free.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import mantpy as mt


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close("all")


def _make_overlay_roi(n_ecm: int = 12):
    """Minimal AnnData with a cell-ECM graph + ECM patches, plus a (C,H,W) image."""
    rng = np.random.default_rng(0)
    H = W = 40
    G = nx.Graph()
    cell_xy = [(8, 8), (10, 9), (30, 30), (31, 28)]
    for i, (x, y) in enumerate(cell_xy):
        ct = "AEC" if i < 2 else "OTHER"
        G.add_node(f"cell_{i}", node_type="cell", cell_type=ct, x=float(x), y=float(y))
    px = rng.integers(4, W - 4, size=n_ecm).astype(float)
    py = rng.integers(4, H - 4, size=n_ecm).astype(float)
    patches = pd.DataFrame({"x": px, "y": py, "feat_mean_0": rng.random(n_ecm)})
    for j in range(n_ecm):
        G.add_node(f"ecm_{j}", node_type="ecm", ecm_cluster=int(j % 3))
    for i in range(2):  # connect the two AEC cells to the first three ECM patches
        for j in range(3):
            G.add_edge(f"cell_{i}", f"ecm_{j}")
    a = ad.AnnData(np.zeros((4, 3), dtype=np.float32))
    a.uns["cell_ecm_graph"] = G
    a.uns["ecm_patches"] = patches
    img = rng.integers(0, 500, size=(3, H, W)).astype(np.uint16)
    return a, img


def _n_size_bars(ax) -> int:
    return sum(isinstance(c, AnchoredSizeBar) for c in ax.get_children())


def test_scalebar_drawn_when_pixel_size_and_length_given():
    a, img = _make_overlay_roi()
    fig, ax = plt.subplots()
    out = mt.pl.channel_overlay_on_neighbours(
        a,
        img=img,
        channels=[0],
        cell_type="AEC",
        channel_names=["HABP"],
        pixel_size=2.0,
        scalebar_um=20,
        ax=ax,
        show=False,
    )
    assert out is ax
    assert _n_size_bars(ax) == 1


def test_no_scalebar_by_default():
    a, img = _make_overlay_roi()
    fig, ax = plt.subplots()
    mt.pl.channel_overlay_on_neighbours(
        a,
        img=img,
        channels=[0],
        cell_type="AEC",
        channel_names=["HABP"],
        ax=ax,
        show=False,
    )
    assert _n_size_bars(ax) == 0


def test_scalebar_requires_pixel_size():
    a, img = _make_overlay_roi()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="pixel_size"):
        mt.pl.channel_overlay_on_neighbours(
            a,
            img=img,
            channels=[0],
            cell_type="AEC",
            channel_names=["HABP"],
            scalebar_um=20,
            ax=ax,
            show=False,
        )
