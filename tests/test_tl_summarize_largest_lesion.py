"""Tests for :func:`mantpy.tl.summarize_largest_lesion`."""

from __future__ import annotations

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gudhi")

import mantpy as mt
from mantpy.tl import LesionSummary


def _make_adata_with_ecm_graph(coords: np.ndarray, cluster_labels: np.ndarray) -> ad.AnnData:
    n = len(coords)
    G = nx.Graph()
    for i in range(n):
        G.add_node(
            i,
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            ecm_cluster=int(cluster_labels[i]),
        )
    a = ad.AnnData(
        X=np.zeros((n, 1), dtype=np.float32),
        obs=pd.DataFrame(index=[str(i) for i in range(n)]),
    )
    a.uns["ecm_graph"] = G
    return a


def _doughnut(r_in=10.0, r_out=30.0, n=40):
    rng = np.random.default_rng(0)
    a1 = rng.uniform(0, 2 * np.pi, n)
    inner = np.column_stack([r_in * np.cos(a1), r_in * np.sin(a1)])
    a2 = rng.uniform(0, 2 * np.pi, n)
    outer = np.column_stack([r_out * np.cos(a2), r_out * np.sin(a2)])
    return np.vstack([inner, outer])


def test_returns_named_tuple():
    coords = _doughnut()
    a = _make_adata_with_ecm_graph(coords, np.full(len(coords), 5, dtype=int))
    summary = mt.tl.summarize_largest_lesion(a, target_cluster=5, radius=40)
    assert isinstance(summary, LesionSummary)
    assert summary.n_nodes == len(coords)


def test_returns_none_for_empty_target_cluster():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    a = _make_adata_with_ecm_graph(coords, np.array([0, 0]))
    summary = mt.tl.summarize_largest_lesion(a, target_cluster=5, radius=40)
    assert summary is None


def test_attributes_are_reasonable():
    coords = _doughnut()
    a = _make_adata_with_ecm_graph(coords, np.full(len(coords), 5, dtype=int))
    summary = mt.tl.summarize_largest_lesion(a, target_cluster=5, radius=40)
    assert summary.n_nodes > 0
    assert summary.max_degree >= 0
    assert summary.max_k_core >= 0
    assert summary.hole_area >= 0.0


def test_iterates_as_4_tuple():
    """LesionSummary is a NamedTuple — positional unpack works."""
    coords = _doughnut()
    a = _make_adata_with_ecm_graph(coords, np.full(len(coords), 5, dtype=int))
    summary = mt.tl.summarize_largest_lesion(a, target_cluster=5, radius=40)
    n, deg, kcore, hole = summary
    assert n == summary.n_nodes
    assert deg == summary.max_degree
    assert kcore == summary.max_k_core
    assert hole == summary.hole_area
