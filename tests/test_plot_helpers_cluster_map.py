"""Tests for mantpy._plot_helpers.plot_cluster_map."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

import mantpy as mt


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close("all")


def _toy_adata(seed=0):
    """Synthetic AnnData with a 10-node ECM graph + 3 clusters."""
    rng = np.random.default_rng(seed)
    a = ad.AnnData(
        X=rng.normal(size=(10, 2)).astype(np.float32),
        var=pd.DataFrame({"is_ecm": [True, False]}, index=["m0", "m1"]),
    )
    G = nx.Graph()
    coords = rng.uniform(0, 100, size=(10, 2))
    # 3 clusters + 1 background (-1) node.
    cluster_assign = [0, 0, 1, 1, 1, 2, 2, 2, 5, -1]
    for i, (x, y) in enumerate(coords):
        G.add_node(f"n{i}", x=float(x), y=float(y), ecm_cluster=cluster_assign[i])
    # A simple line graph.
    for i in range(9):
        G.add_edge(f"n{i}", f"n{i + 1}")
    a.uns["ecm_graph"] = G
    return a


class TestPanelClusterMap:
    def test_returns_axes(self):
        a = _toy_adata()
        palette = {0: "#DC050C", 1: "#228833", 2: "#4477AA", 5: "#66CCEE"}
        fig, ax = plt.subplots()
        out = mt.pl.plot_cluster_map(
            ax,
            a,
            palette=palette,
            best_k=3,
            target_cluster=5,
        )
        assert out is ax
        assert isinstance(out, Axes)

    def test_facecolor_dark(self):
        a = _toy_adata()
        palette = {0: "#000", 1: "#111", 2: "#222", 5: "#333"}
        fig, ax = plt.subplots()
        mt.pl.plot_cluster_map(
            ax,
            a,
            palette=palette,
            best_k=3,
            target_cluster=5,
        )
        # Dark backdrop is part of the figure's identity.
        assert ax.get_facecolor() == (
            13 / 255,
            13 / 255,
            13 / 255,
            1.0,
        )

    def test_target_cluster_scatter_present(self):
        # target_cluster=5 has one node (n8); we expect at least one
        # scatter collection from the target-cluster pass.
        a = _toy_adata()
        palette = {0: "#000", 1: "#111", 2: "#222", 5: "#333"}
        fig, ax = plt.subplots()
        mt.pl.plot_cluster_map(
            ax,
            a,
            palette=palette,
            best_k=3,
            target_cluster=5,
        )
        # Multiple collections from different cluster passes — just verify
        # that scatter was used at all (i.e. PathCollection instances exist).
        from matplotlib.collections import PathCollection

        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) >= 1

    def test_background_can_be_hidden(self):
        a = _toy_adata()
        palette = {0: "#000", 1: "#111", 2: "#222", 5: "#333"}
        fig, ax = plt.subplots()
        mt.pl.plot_cluster_map(
            ax,
            a,
            palette=palette,
            best_k=3,
            target_cluster=5,
            show_background=False,
        )
        from matplotlib.collections import PathCollection

        plotted_nodes = sum(
            len(collection.get_offsets())
            for collection in ax.collections
            if isinstance(collection, PathCollection)
        )
        assert plotted_nodes == 9
        assert len(ax.lines) == 8

    def test_no_ticks(self):
        a = _toy_adata()
        palette = {0: "#000", 1: "#111", 2: "#222", 5: "#333"}
        fig, ax = plt.subplots()
        mt.pl.plot_cluster_map(
            ax,
            a,
            palette=palette,
            best_k=3,
            target_cluster=5,
        )
        assert list(ax.get_xticks()) == []
        assert list(ax.get_yticks()) == []
