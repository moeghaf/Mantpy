"""Tests for building on a graph squidpy already made, rather than a second one.

`sq.gr.spatial_neighbors` and `mt.gr.build_cell_graph` both want to own
`obsp['spatial_connectivities']`. Before `connectivity_key`, mantpy ignored an
existing graph and silently wrote a different one beside it, so downstream
`sq.gr.*` results came from a graph the user never asked for.
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import mantpy as mt
from mantpy._constants import EDGE_TYPE_CC

sq = pytest.importorskip("squidpy")


def _cells(n: int = 200, seed: int = 0) -> ad.AnnData:
    """A plain scverse AnnData — no mantpy provenance."""
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.random((n, 12), dtype=np.float32))
    a.obsm["spatial"] = rng.random((n, 2)) * 100
    a.obs["cell_type"] = rng.choice(["A", "B", "C"], n)
    return a


def _undirected(a: ad.AnnData, key: str = "spatial_connectivities") -> int:
    m = sp.csr_matrix(a.obsp[key])
    return int((m.maximum(m.T) != 0).nnz // 2)


def test_adopts_a_squidpy_graph_edge_for_edge():
    a = _cells()
    sq.gr.spatial_neighbors(a, coord_type="generic", n_neighs=6)
    expected = _undirected(a)

    mt.gr.build_cell_graph(a, connectivity_key="spatial_connectivities")

    g = a.uns["cell_graph_nx"]
    assert g.number_of_edges() == expected


def test_adopting_is_not_the_same_as_rebuilding():
    """Guards the test above: if mantpy's own builder happened to produce the
    same graph, the assertion would pass without adopting anything."""
    a, b = _cells(), _cells()
    sq.gr.spatial_neighbors(a, coord_type="generic", n_neighs=6)
    mt.gr.build_cell_graph(a, connectivity_key="spatial_connectivities")
    mt.gr.build_cell_graph(b, k=5)
    assert a.uns["cell_graph_nx"].number_of_edges() != b.uns["cell_graph_nx"].number_of_edges()


def test_adopting_does_not_warn_about_clobbering():
    """Republishing the graph we just adopted is a no-op in substance: mantpy
    stores the symmetrised, distance-weighted form, so the matrix differs
    numerically while describing the same topology."""
    a = _cells()
    sq.gr.spatial_neighbors(a, coord_type="generic", n_neighs=6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mt.gr.build_cell_graph(a, connectivity_key="spatial_connectivities")
    assert not [x for x in w if "Replacing an existing" in str(x.message)]


def test_rebuilding_over_a_squidpy_graph_does_warn():
    a = _cells()
    sq.gr.spatial_neighbors(a, coord_type="generic", n_neighs=6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mt.gr.build_cell_graph(a, k=5)
    assert [x for x in w if "Replacing an existing" in str(x.message)]


def test_squidpy_knn_asymmetry_does_not_lose_edges():
    """squidpy's kNN adjacency is directed; taking triu without symmetrising
    first would silently drop roughly half the neighbours."""
    a = _cells()
    sq.gr.spatial_neighbors(a, coord_type="generic", n_neighs=6)
    raw = sp.csr_matrix(a.obsp["spatial_connectivities"])
    assert (raw != raw.T).nnz > 0, "fixture no longer asymmetric; test is vacuous"

    mt.gr.build_cell_graph(a, connectivity_key="spatial_connectivities")
    assert a.uns["cell_graph_nx"].number_of_edges() == _undirected(a, "cell_graph")


def test_missing_key_names_the_fix():
    a = _cells()
    with pytest.raises(ValueError, match="spatial_neighbors"):
        mt.gr.build_cell_graph(a, connectivity_key="spatial_connectivities")


def test_ensure_cell_ecm_graph_threads_it_through():
    """The one-shot entry point is what a tutorial actually calls."""
    rng = np.random.default_rng(0)
    a = _cells()
    sq.gr.spatial_neighbors(a, coord_type="generic", n_neighs=6)
    expected = _undirected(a)

    img = rng.random((3, 100, 100)).astype(np.float32)
    e = mt.io.read_ecm_image(img, marker_names=["m1", "m2", "m3"], sample_id="roi1")
    mt.pp.extract_ecm_patches(e, img, patch_size=10, ecm_K=3, features=["mean"])
    a.uns["ecm_patches"] = e.uns["ecm_patches"]

    mt.gr.ensure_cell_ecm_graph(
        a,
        cell_connectivity_key="spatial_connectivities",
        ecm_edge_method="grid",
        ecm_grid_connectivity=8,
        cell_ecm_k=5,
        cell_ecm_Dmax=15.0,
    )
    g = a.uns["cell_ecm_graph"]
    cc = sum(1 for _, _, d in g.edges(data=True) if d.get("edge_type") == EDGE_TYPE_CC)
    assert cc == expected
