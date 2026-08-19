"""Tests for ``mt.pl.cell_ecm_enrichment_heatmap``.

The heatmap consumes the tidy frame from ``mt.tl.cell_ecm_enrichment_matrix``
(columns cell_type, cluster, log2_enr, p_fdr) and renders a cell-type x cluster
diverging image with q-stars. These tests check that it draws onto a supplied
axis, lays out the right grid, honours row ordering and row highlighting,
annotates with stars, and validates its input columns.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import mantpy as mt


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close("all")


def _matrix_frame() -> pd.DataFrame:
    """A small 3-cell-type x 3-cluster tidy enrichment frame."""
    rows = []
    data = {
        "AEC": {0: (-1.5, 0.2), 1: (2.1, 1e-4), 2: (-0.4, 0.4)},
        "MAC": {0: (1.8, 1e-3), 1: (-0.9, 0.1), 2: (0.1, 0.8)},
        "FIB": {0: (0.2, 0.6), 1: (0.3, 0.5), 2: (1.2, 0.02)},
    }
    for ct, by_cl in data.items():
        for cl, (l2, q) in by_cl.items():
            rows.append({"cell_type": ct, "cluster": cl, "log2_enr": l2, "p_fdr": q})
    return pd.DataFrame(rows)


def test_draws_on_supplied_axis_with_correct_grid():
    fig, ax = plt.subplots()
    out = mt.pl.cell_ecm_enrichment_heatmap(_matrix_frame(), ax=ax, show=False)
    assert out is ax
    # One image, 3 columns (clusters) x 3 rows (cell types).
    assert len(ax.images) == 1
    arr = ax.images[0].get_array()
    assert arr.shape == (3, 3)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["ECM 0", "ECM 1", "ECM 2"]
    assert {t.get_text() for t in ax.get_yticklabels()} == {"AEC", "MAC", "FIB"}


def test_order_by_cluster_sorts_rows_descending():
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_heatmap(
        _matrix_frame(),
        ax=ax,
        order_by_cluster=1,
        show=False,
    )
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    # AEC has the highest ECM-1 enrichment (2.1) -> first row (top).
    assert ylabels[0] == "AEC"


def test_highlight_rows_recolors_label():
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_heatmap(
        _matrix_frame(),
        ax=ax,
        highlight_rows="AEC",
        highlight_color="#D55E00",
        show=False,
    )
    aec = [t for t in ax.get_yticklabels() if t.get_text() == "AEC"][0]
    assert aec.get_fontweight() == "bold"


def test_annotations_include_stars():
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_heatmap(_matrix_frame(), ax=ax, show=False)
    texts = [t.get_text() for t in ax.texts]
    # AEC/ECM1 q=1e-4 -> "2.1***" ; FIB/ECM2 q=0.02 -> "1.2*".
    assert any(t.startswith("2.1") and "***" in t for t in texts)
    assert any(t.startswith("1.2") and t.endswith("*") and "**" not in t for t in texts)


def test_cluster_labels_override():
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_heatmap(
        _matrix_frame(),
        ax=ax,
        show=False,
        cluster_labels={0: "alveolar", 1: "BM", 2: "perivasc"},
    )
    assert [t.get_text() for t in ax.get_xticklabels()] == ["alveolar", "BM", "perivasc"]


def test_missing_column_raises():
    bad = _matrix_frame().drop(columns="log2_enr")
    with pytest.raises(KeyError, match="log2_enr"):
        mt.pl.cell_ecm_enrichment_heatmap(bad, show=False)


def test_creates_own_figure_when_no_ax():
    out = mt.pl.cell_ecm_enrichment_heatmap(_matrix_frame(), show=False)
    assert out.images[0].get_array().shape == (3, 3)


def test_integration_with_enrichment_matrix():
    """End-to-end: feed the real tl output straight into the plotter."""
    import anndata as ad
    import networkx as nx

    def make_roi(seed):
        rng = np.random.default_rng(seed)
        G = nx.Graph()
        for i in range(12):
            G.add_node(f"cell_{i}", node_type="cell", cell_type="AEC")
        for i in range(12, 24):
            G.add_node(f"cell_{i}", node_type="cell", cell_type="OTHER")
        nodes = {0: [], 1: [], 2: []}
        eid = 0
        for cl in (0, 1, 2):
            for _ in range(20):
                G.add_node(f"ecm_{eid}", node_type="ecm", ecm_cluster=cl)
                nodes[cl].append(f"ecm_{eid}")
                eid += 1
        for i in range(12):
            for t in rng.choice(nodes[1], size=3, replace=False):
                G.add_edge(f"cell_{i}", t)
        for i in range(12, 24):
            for t in rng.choice(nodes[0] + nodes[2], size=3, replace=False):
                G.add_edge(f"cell_{i}", t)
        a = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
        a.uns["cell_ecm_graph"] = G
        return a

    cohort = {"a": make_roi(0), "b": make_roi(1)}
    mat = mt.tl.cell_ecm_enrichment_matrix(cohort, K_ecm=3, n_perm=200)
    fig, ax = plt.subplots()
    out = mt.pl.cell_ecm_enrichment_heatmap(mat, ax=ax, highlight_rows="AEC", show=False)
    assert out.images[0].get_array().shape == (2, 3)  # 2 cell types x 3 clusters
