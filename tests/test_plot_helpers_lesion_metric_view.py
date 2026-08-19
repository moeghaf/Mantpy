"""Tests for mantpy._plot_helpers.plot_lesion_metric_view."""

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


def _toy_adata_with_lesion(seed=0):
    """Synthetic AnnData with a clear ECM-5 lesion cluster ringed by other clusters."""
    rng = np.random.default_rng(seed)
    # 20 target-cluster nodes in a tight blob (lesion).
    target = rng.normal(loc=(50, 50), scale=4, size=(20, 2))
    # 30 non-target nodes scattered around it (background).
    bg = rng.uniform(0, 100, size=(30, 2))
    coords = np.vstack([target, bg])
    cluster_labels = np.array([5] * 20 + list(rng.integers(0, 3, size=30)), dtype=int)

    a = ad.AnnData(
        X=rng.normal(size=(len(coords), 2)).astype(np.float32),
        obs=pd.DataFrame(index=[str(i) for i in range(len(coords))]),
        var=pd.DataFrame({"is_ecm": [True, True]}, index=["m0", "m1"]),
    )
    G = nx.Graph()
    for i, (x, y) in enumerate(coords):
        G.add_node(int(i), x=float(x), y=float(y), ecm_cluster=int(cluster_labels[i]))
    a.uns["ecm_graph"] = G
    return a


class TestPanelLesionMetricView:
    def test_returns_axes_n_nodes(self):
        a = _toy_adata_with_lesion()
        palette = {0: "#000", 1: "#111", 2: "#222", 5: "#66CCEE"}
        fig, ax = plt.subplots()
        out = mt.pl.plot_lesion_metric_view(
            ax,
            a,
            "n_nodes",
            palette=palette,
            radius_px=40,
            target_cluster=5,
            best_k=3,
        )
        assert out is ax
        assert isinstance(out, Axes)

    def test_white_background(self):
        a = _toy_adata_with_lesion()
        palette = {0: "#000", 1: "#111", 2: "#222", 5: "#66CCEE"}
        fig, ax = plt.subplots()
        mt.pl.plot_lesion_metric_view(
            ax,
            a,
            "n_nodes",
            palette=palette,
            radius_px=40,
            target_cluster=5,
            best_k=3,
        )
        assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)

    @pytest.mark.parametrize("metric", ["n_nodes", "mean_degree", "max_k_core"])
    def test_all_metrics_render(self, metric):
        a = _toy_adata_with_lesion()
        palette = {0: "#000", 1: "#111", 2: "#222", 5: "#66CCEE"}
        fig, ax = plt.subplots()
        mt.pl.plot_lesion_metric_view(
            ax,
            a,
            metric,
            palette=palette,
            radius_px=40,
            target_cluster=5,
            best_k=3,
        )
        from matplotlib.collections import PathCollection

        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        # At least the background scatter + one metric scatter should exist.
        assert len(scatters) >= 1

    def test_no_ticks(self):
        a = _toy_adata_with_lesion()
        palette = {0: "#000", 1: "#111", 2: "#222", 5: "#66CCEE"}
        fig, ax = plt.subplots()
        mt.pl.plot_lesion_metric_view(
            ax,
            a,
            "n_nodes",
            palette=palette,
            radius_px=40,
            target_cluster=5,
            best_k=3,
        )
        assert list(ax.get_xticks()) == []
        assert list(ax.get_yticks()) == []

    def test_central_hole_area_runs(self):
        pytest.importorskip("gudhi", reason="requires mantpy[topology]")
        # Build a doughnut-shaped lesion so central_hole_area has something to find.
        rng = np.random.default_rng(1)
        # Ring of points around (50, 50), radius ~ 15, jitter ~ 1.
        thetas = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        ring = np.column_stack([50 + 15 * np.cos(thetas), 50 + 15 * np.sin(thetas)])
        ring += rng.normal(0, 0.3, ring.shape)
        coords = ring
        labels = np.array([5] * len(ring))
        a = ad.AnnData(
            X=rng.normal(size=(len(coords), 2)).astype(np.float32),
            obs=pd.DataFrame(index=[str(i) for i in range(len(coords))]),
            var=pd.DataFrame({"is_ecm": [True, True]}, index=["m0", "m1"]),
        )
        G = nx.Graph()
        for i, (x, y) in enumerate(coords):
            G.add_node(int(i), x=float(x), y=float(y), ecm_cluster=int(labels[i]))
        a.uns["ecm_graph"] = G

        palette = {5: "#66CCEE"}
        fig, ax = plt.subplots()
        # Should not raise on this doughnut configuration.
        mt.pl.plot_lesion_metric_view(
            ax,
            a,
            "central_hole_area",
            palette=palette,
            radius_px=12,
            target_cluster=5,
            best_k=6,
        )
        assert ax is not None
