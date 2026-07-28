"""Tests for mantpy._plot_helpers.plot_mean_composition."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

import mantpy as mt


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close("all")


def _toy_group_mean(group_order, ecm_cols):
    # Each group sums to 1 across the 4 ecm clusters.
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.1, 0.9, size=(len(group_order), len(ecm_cols)))
    norm = raw / raw.sum(axis=1, keepdims=True)
    return pd.DataFrame(norm, index=group_order, columns=ecm_cols)


class TestPanelMeanComposition:
    def test_returns_axes(self):
        group_order = ["Naive_KO", "Naive_WT", "Infected_KO", "Infected_WT"]
        ecm_cols = [0, 1, 5, 6]
        palette = {0: "#DC050C", 1: "#228833", 5: "#66CCEE", 6: "#999933"}
        gm = _toy_group_mean(group_order, ecm_cols)
        fig, ax = plt.subplots()
        out = mt.pl.plot_mean_composition(
            ax,
            gm,
            ecm_cols,
            palette=palette,
            group_order=group_order,
            target_cluster=5,
        )
        assert out is ax
        assert isinstance(out, Axes)

    def test_xlim_matches_group_order(self):
        group_order = ["a", "b", "c", "d"]
        ecm_cols = [0, 1]
        palette = {0: "#000000", 1: "#ffffff"}
        gm = pd.DataFrame(
            [[0.5, 0.5]] * 4,
            index=group_order,
            columns=ecm_cols,
        )
        fig, ax = plt.subplots()
        mt.pl.plot_mean_composition(
            ax,
            gm,
            ecm_cols,
            palette=palette,
            group_order=group_order,
            target_cluster=1,
        )
        assert ax.get_xlim() == (-0.5, len(group_order) - 0.5)
        assert ax.get_ylim() == (0.0, 1.0)

    def test_target_cluster_has_heavier_edge(self):
        # The target_cluster bar should have linewidth=1.4 while others
        # are 0.6.  We render bars in ``ecm_cols`` order, one stack per
        # group_order entry; the bar.patches list has K * G entries
        # in column-major order — every patch with cluster == target
        # should have linewidth == 1.4.
        group_order = ["g1", "g2"]
        ecm_cols = [0, 5]
        palette = {0: "#000000", 5: "#66CCEE"}
        gm = pd.DataFrame([[0.3, 0.7], [0.4, 0.6]], index=group_order, columns=ecm_cols)
        fig, ax = plt.subplots()
        mt.pl.plot_mean_composition(
            ax,
            gm,
            ecm_cols,
            palette=palette,
            group_order=group_order,
            target_cluster=5,
        )
        # bars are added in cluster order: first 2 patches are cluster 0
        # (groups g1 and g2), next 2 are cluster 5.
        patches = ax.patches
        assert len(patches) == 4
        assert patches[0].get_linewidth() == pytest.approx(0.6)
        assert patches[1].get_linewidth() == pytest.approx(0.6)
        assert patches[2].get_linewidth() == pytest.approx(1.4)
        assert patches[3].get_linewidth() == pytest.approx(1.4)

    def test_legend_present(self):
        group_order = ["g"]
        ecm_cols = [0, 1, 2]
        palette = {0: "#000000", 1: "#777777", 2: "#ffffff"}
        gm = pd.DataFrame([[0.3, 0.3, 0.4]], index=group_order, columns=ecm_cols)
        fig, ax = plt.subplots()
        mt.pl.plot_mean_composition(
            ax,
            gm,
            ecm_cols,
            palette=palette,
            group_order=group_order,
            target_cluster=0,
        )
        leg = ax.get_legend()
        assert leg is not None
        assert leg.get_title().get_text() == "ECM cluster"
