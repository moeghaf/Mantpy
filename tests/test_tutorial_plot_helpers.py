"""Tests for generic plotting helpers used in the tutorials."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import mantpy as mt


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close("all")


def test_categorical_palette_and_tidy_plotters() -> None:
    palette = mt.pl.categorical_palette(["B", "A", "B"], colors=["#111111", "#222222"])
    assert palette == {"B": "#111111", "A": "#222222"}

    ablation = pd.DataFrame(
        {
            "model": ["cell", "cell", "ecm", "ecm"],
            "roc_auc": [0.9, 1.0, 0.6, 0.7],
        }
    )
    ax = mt.pl.cross_compartment_ablation_bars(
        ablation,
        order=("cell", "ecm"),
        figsize=(3, 3),
        show=False,
    )
    assert len(ax.patches) == 2

    per_roi = pd.DataFrame(
        {
            "roi": ["long_r1", "long_r1", "long_r2", "long_r2"],
            "cluster": [0, 1, 0, 1],
            "log2_enr": [-1.0, 2.0, -0.5, 1.5],
        }
    )
    ax = mt.pl.cell_ecm_enrichment_per_roi(
        per_roi,
        sample_labels="suffix",
        xlabel="ROI",
        show=False,
    )
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["r1", "r2"]
    assert ax.get_xlabel() == "ROI"


def test_classifier_roc_can_create_its_own_figure() -> None:
    curves = pd.DataFrame(
        {
            "fold": ["r1", "r1", "r1", "r2", "r2", "r2"],
            "kind": ["roc"] * 6,
            "x": [0, 0.2, 1, 0, 0.1, 1],
            "y": [0, 0.9, 1, 0, 0.8, 1],
        }
    )
    summary = pd.DataFrame({"fold": ["r1", "r2"], "roc_auc": [0.9, 0.85], "pr_auc": [0.8, 0.75]})
    ax = mt.pl.classifier_roc(curves=curves, summary=summary, figsize=(3, 3), show=False)
    assert ax.get_xlabel() == "False positive rate"


def test_heatmap_and_cluster_comparison_accept_complete_plot_settings() -> None:
    matrix = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B"],
            "cluster": [0, 1, 0, 1],
            "log2_enr": [-1.0, 2.0, 0.5, -0.4],
            "p_fdr": [0.1, 0.001, 0.2, 0.3],
        }
    )
    ax = mt.pl.cell_ecm_enrichment_heatmap(
        matrix,
        cluster_labels={0: "zero", 1: "one"},
        xtick_rotation=30,
        figsize=(3, 3),
        show=False,
    )
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["zero", "one"]

    cells = ad.AnnData(
        X=np.empty((3, 0), dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B"]}, index=["c0", "c1", "c2"]),
    )
    cells.obsm["spatial"] = np.array([[0, 0], [2, 0], [1, 2]], dtype=np.float32)
    patches = pd.DataFrame(
        {
            "x": [0.5, 1.5, 1.0],
            "y": [0.5, 0.5, 1.5],
            "ecm_cluster": [0, 1, 1],
            "denoised_cluster": [0, 0, 1],
            "feat_0": [0.0, 1.0, 1.1],
        }
    )
    cells.uns["ecm_patches"] = patches.copy()
    ecm = ad.AnnData(X=np.empty((0, 0), dtype=np.float32))
    ecm.uns["ecm_patches"] = patches
    axes = mt.pl.ecm_cluster_comparison(
        cells,
        ecm,
        graph_kwargs={
            "cell_k": 1,
            "cell_Dmax": 5,
            "ecm_edge_method": "grid",
            "cell_ecm_k": 1,
            "cell_ecm_Dmax": 5,
        },
        palette={0: "#111111", 1: "#999999"},
        titles=("before", "after"),
        figsize=(5, 2.5),
        show=False,
    )
    assert [axis.get_title() for axis in axes] == ["before", "after"]
