"""Tests for :func:`mantpy.gr.largest_component_radius_reconnect`.

Equivalence contract: the returned record must match what the notebook's
inline ``max(comps_all, key=len)`` re-implementation produces — the
schistosoma narrative figure relies on this for the panel-e example
visualisation columns and the shared-vmax precompute.
"""

from __future__ import annotations

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import mantpy as mt


def _make_adata_with_ecm_graph(coords: np.ndarray, cluster_labels: np.ndarray) -> ad.AnnData:
    """Build a minimal AnnData whose ``uns['ecm_graph']`` carries the right node attrs."""
    n = len(coords)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, x=float(coords[i, 0]), y=float(coords[i, 1]), ecm_cluster=int(cluster_labels[i]))
    adata = ad.AnnData(
        X=np.zeros((n, 1), dtype=np.float32),
        obs=pd.DataFrame(index=[str(i) for i in range(n)]),
    )
    adata.uns["ecm_graph"] = G
    return adata


def _notebook_inline_largest(adata, target_cluster: int, radius: float):
    """Replicate the notebook's inline ``max(comps_all, key=len)`` recipe verbatim."""
    G = adata.uns["ecm_graph"]
    xy = {v: (d["x"], d["y"]) for v, d in G.nodes(data=True)}
    cl = {v: int(d.get("ecm_cluster", -1)) for v, d in G.nodes(data=True)}
    target = [v for v, c in cl.items() if c == target_cluster]
    if len(target) < 2:
        return None
    coords = np.array([xy[v] for v in target], dtype=float)
    n = len(target)
    G_rad = nx.Graph()
    G_rad.add_nodes_from(range(n))
    for i, j in cKDTree(coords).query_pairs(r=radius, output_type="ndarray"):
        G_rad.add_edge(int(i), int(j))
    comps_all = [sorted(c) for c in nx.connected_components(G_rad)]
    if not comps_all:
        return {"n_nodes": 1, "coords": coords[:1]}
    biggest = sorted(max(comps_all, key=len))
    return {"n_nodes": len(biggest), "coords": coords[biggest]}


def test_matches_notebook_inline_largest():
    """Library helper must produce the same coords + n_nodes as the notebook recipe."""
    rng = np.random.default_rng(0)
    # Three clusters of target points with radius 5 (so each cluster reconnects internally).
    cluster_a = rng.normal(0, 1, size=(30, 2))
    cluster_b = rng.normal(50, 1, size=(20, 2))
    cluster_c = rng.normal(100, 1, size=(15, 2))
    bg = rng.normal(200, 5, size=(10, 2))  # background, cluster=0
    coords = np.vstack([cluster_a, cluster_b, cluster_c, bg])
    labels = np.concatenate(
        [
            np.full(30, 5, dtype=int),
            np.full(20, 5, dtype=int),
            np.full(15, 5, dtype=int),
            np.full(10, 0, dtype=int),
        ]
    )
    adata = _make_adata_with_ecm_graph(coords, labels)

    lib = mt.gr.largest_component_radius_reconnect(adata, target_cluster=5, radius=5.0)
    inline = _notebook_inline_largest(adata, target_cluster=5, radius=5.0)

    assert lib is not None
    assert lib["n_nodes"] == inline["n_nodes"] == 30
    # Coord arrays should be identical up to row order; we rely on the same
    # underlying iteration order, so compare bit-for-bit.
    np.testing.assert_array_equal(lib["coords"], inline["coords"])


def test_singleton_target_returns_one_node_record():
    """A single target-cluster point yields a 1-node 0-edge record (no None)."""
    coords = np.array([[0.0, 0.0]], dtype=float)
    labels = np.array([5], dtype=int)
    adata = _make_adata_with_ecm_graph(coords, labels)
    rec = mt.gr.largest_component_radius_reconnect(adata, target_cluster=5, radius=10.0)
    assert rec is not None
    assert rec["n_nodes"] == 1
    assert rec["n_edges"] == 0


def test_returns_none_when_target_absent():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    labels = np.array([0, 0], dtype=int)
    adata = _make_adata_with_ecm_graph(coords, labels)
    assert mt.gr.largest_component_radius_reconnect(adata, target_cluster=5, radius=10.0) is None


def test_tie_break_matches_first_seen():
    """When two components have equal sizes, both Python's ``max`` and
    our helper return the first one encountered in iteration order."""
    rng = np.random.default_rng(0)
    # Two equal clusters of size 10
    cluster_a = rng.normal(0, 0.5, size=(10, 2))
    cluster_b = rng.normal(50, 0.5, size=(10, 2))
    coords = np.vstack([cluster_a, cluster_b])
    labels = np.full(20, 5, dtype=int)
    adata = _make_adata_with_ecm_graph(coords, labels)

    lib = mt.gr.largest_component_radius_reconnect(adata, target_cluster=5, radius=5.0)
    inline = _notebook_inline_largest(adata, target_cluster=5, radius=5.0)
    np.testing.assert_array_equal(lib["coords"], inline["coords"])
