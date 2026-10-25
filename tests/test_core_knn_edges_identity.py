"""S01 regression tests: identity-aware self-exclusion in kNN edge construction.

Covers :func:`mantpy._core._edges.knn_edges`, its use as the fewer-than-three-
point fallback of :func:`mantpy._core._edges.delaunay_edges`, and the public
builders :func:`mantpy.gr.build_ecm_graph`, :func:`mantpy.gr.build_ecm_graphs`
and :func:`mantpy.gr.build_cell_graph`.

Declared semantics:

* Each point selects its ``min(k, n - 1)`` nearest *other* points.  The query
  point is excluded by identity, so distinct coincident points remain valid
  neighbours at distance 0 and no self-loop is produced.
* Selections are merged by undirected union (an edge exists if either endpoint
  selected the other), so final degrees can exceed ``k``.
* A selection with distance ``> Dmax`` is dropped; ``== Dmax`` is kept.
* Candidates tied at the boundary distance are taken in scikit-learn's returned
  order, which these tests do not pin down: tied cases are checked with
  admissibility and budget properties rather than exact edge sets.

The non-tied oracle computes all pairwise distances with NumPy and does not use
a neighbour-search library.  The legacy-rule comparison reproduces the previous
"drop the first returned neighbour" selection only on inputs where the point
itself is provably returned first (no duplicate coordinates), to show that such
inputs are unaffected.
"""

from __future__ import annotations

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import mantpy as mt
from mantpy._core._edges import delaunay_edges, knn_edges


def _pairwise(xy):
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((diff**2).sum(-1))


def _as_dict(edges):
    out = {}
    for i, j, dist in edges:
        key = (min(i, j), max(i, j))
        assert key not in out, f"edge {key} listed twice"
        out[key] = dist
    return out


def _oracle_knn_edges(xy, k, Dmax=None):
    """Exact kNN union; valid only when each k-th neighbour distance is untied."""
    n = len(xy)
    D = _pairwise(xy)
    out = {}
    for i in range(n):
        others = sorted((j for j in range(n) if j != i), key=lambda j: D[i, j])
        for j in others[: min(k, n - 1)]:
            if Dmax is not None and D[i, j] > Dmax:
                continue
            out.setdefault((min(i, j), max(i, j)), D[i, j])
    return out


def _check_admissible(xy, k, edges, Dmax=None):
    """Tie-robust properties of the declared semantics."""
    n = len(xy)
    D = _pairwise(xy)
    kk = min(k, n - 1)
    E = _as_dict(edges)
    assert all(i != j for i, j in E), "self-loop present"
    for (i, j), dist in E.items():
        assert dist == pytest.approx(D[i, j], abs=1e-12)
        if Dmax is not None:
            assert dist <= Dmax
    if kk <= 0:
        assert not E
        return E
    # r[i]: distance of the kk-th nearest *other* point of i.
    r = np.array([np.sort(np.delete(D[i], i))[kk - 1] for i in range(n)])
    for i, j in E:  # every edge must be selectable by at least one endpoint
        assert D[i, j] <= r[i] + 1e-12 or D[i, j] <= r[j] + 1e-12, (i, j)
    for i in range(n):  # every strictly nearer eligible point must be present
        for j in range(n):
            if j != i and D[i, j] < r[i] - 1e-12 and (Dmax is None or D[i, j] <= Dmax):
                assert (min(i, j), max(i, j)) in E, f"{j} is strictly within the {kk}-NN radius of {i}"
    assert len(E) <= n * kk  # at most kk selections per node before the union
    if Dmax is None:  # each node keeps its kk distinct selections after the union
        degree = np.zeros(n, dtype=int)
        for i, j in E:
            degree[i] += 1
            degree[j] += 1
        assert np.all(degree >= kk), degree
    return E


def _legacy_rule_without_duplicates(xy, k, Dmax=None):
    """Previous selection rule (skip the first returned neighbour), valid only
    when every point is returned as its own first neighbour."""
    from sklearn.neighbors import NearestNeighbors

    n = len(xy)
    dist, ind = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="ball_tree").fit(xy).kneighbors(xy)
    assert np.array_equal(ind[:, 0], np.arange(n)), "precondition: self returned first"
    edges, seen = [], set()
    for i in range(n):
        for pos in range(1, ind.shape[1]):
            j, d = int(ind[i, pos]), float(dist[i, pos])
            if Dmax is not None and d > Dmax:
                continue
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append((i, j, d))
    return edges


# ---------------------------------------------------------------------------
# Duplicate coordinates (failing before the S01 correction)
# ---------------------------------------------------------------------------


def test_two_coincident_points_give_one_zero_length_edge():
    edges = knn_edges(np.array([[0.0, 0.0], [0.0, 0.0]]), k=1)
    assert _as_dict(edges) == {(0, 1): 0.0}


def test_coincident_pair_with_other_points():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    E = _check_admissible(xy, 1, knn_edges(xy, k=1))
    assert E[(0, 1)] == 0.0  # distinct coincident points are neighbours


def test_coincident_triple_and_all_duplicate_cloud():
    xy = np.array([[3.0, 3.0]] * 3 + [[0.0, 0.0], [5.0, 1.0]])
    E = _check_admissible(xy, 2, knn_edges(xy, k=2))
    assert {(0, 1), (0, 2), (1, 2)} <= set(E)
    cloud = np.zeros((5, 2))
    E = _check_admissible(cloud, 2, knn_edges(cloud, k=2))
    assert all(d == 0.0 for d in E.values())


@pytest.mark.parametrize("seed", range(8))
def test_partly_duplicate_integer_coordinates(seed):
    rng = np.random.default_rng(seed)
    xy = rng.integers(0, 6, size=(int(rng.integers(3, 25)), 2)).astype(float)
    k = int(rng.integers(1, 6))
    _check_admissible(xy, k, knn_edges(xy, k=k))


def test_duplicates_with_dmax_and_k_above_available():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    E = _check_admissible(xy, 2, knn_edges(xy, k=2, Dmax=1.0), Dmax=1.0)
    assert set(E) == {(0, 1), (0, 2), (1, 2)}  # the far point is pruned by Dmax
    E = _check_admissible(xy, 10, knn_edges(xy, k=10))
    assert set(E) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}  # k capped at n - 1


# ---------------------------------------------------------------------------
# Ties, oracle and boundary behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_symmetric_ties_on_regular_grid(k):
    xy = np.array([[x, y] for x in range(5) for y in range(5)], dtype=float)
    edges = knn_edges(xy, k=k)
    _check_admissible(xy, k, edges)
    assert edges == _legacy_rule_without_duplicates(xy, k)  # tie order unchanged


@pytest.mark.parametrize("seed", range(5))
def test_matches_pairwise_oracle_on_untied_points(seed):
    rng = np.random.default_rng(100 + seed)
    xy = rng.random((int(rng.integers(3, 40)), 2)) * 10.0
    k = int(rng.integers(1, 7))
    Dmax = None if seed % 2 == 0 else float(np.quantile(_pairwise(xy)[np.triu_indices(len(xy), 1)], 0.3))
    got = _as_dict(knn_edges(xy, k=k, Dmax=Dmax))
    want = _oracle_knn_edges(xy, k, Dmax)
    assert set(got) == set(want)
    for key, dist in want.items():
        assert got[key] == pytest.approx(dist, abs=1e-12)
    assert knn_edges(xy, k=k, Dmax=Dmax) == _legacy_rule_without_duplicates(xy, k, Dmax)


def test_dmax_boundary_is_inclusive():
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 10.0]])  # |01| = 5, |12| = sqrt(45), |02| = 10
    assert _as_dict(knn_edges(xy, k=1, Dmax=5.0)) == {(0, 1): 5.0}
    assert _as_dict(knn_edges(xy, k=1, Dmax=4.999)) == {}
    assert _as_dict(knn_edges(xy, k=1)) == {(0, 1): 5.0, (1, 2): pytest.approx(np.sqrt(45.0))}


def test_k_at_or_above_available_neighbours_gives_complete_graph():
    xy = np.array([[0.0, 0.0], [1.0, 0.2], [0.3, 2.0], [2.5, 1.5]])
    complete = {(i, j) for i in range(4) for j in range(i + 1, 4)}
    assert set(_as_dict(knn_edges(xy, k=3))) == complete
    assert set(_as_dict(knn_edges(xy, k=50))) == complete


def test_empty_singleton_and_two_node_inputs():
    assert knn_edges(np.zeros((0, 2)), k=3) == []
    assert knn_edges(np.array([[1.0, 2.0]]), k=3) == []
    assert _as_dict(knn_edges(np.array([[0.0, 0.0], [3.0, 4.0]]), k=1)) == {(0, 1): 5.0}
    assert knn_edges(np.random.default_rng(0).random((5, 2)), k=0) == []
    with pytest.raises(ValueError):
        knn_edges(np.random.default_rng(0).random((5, 2)), k=-1)


def test_delaunay_fallback_for_fewer_than_three_points():
    assert _as_dict(delaunay_edges(np.array([[0.0, 0.0], [0.0, 0.0]]))) == {(0, 1): 0.0}
    assert _as_dict(delaunay_edges(np.array([[0.0, 0.0], [3.0, 4.0]]))) == {(0, 1): 5.0}
    assert delaunay_edges(np.array([[0.0, 0.0]])) == []


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

_DUP_XY = np.array([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0], [0.0, 0.0], [5.0, 1.0]])


def _ecm_adata(xy):
    patches = pd.DataFrame(
        {"x": xy[:, 0], "y": xy[:, 1], "ecm_cluster": np.arange(len(xy)) % 2, "feat_0": np.arange(len(xy), dtype=float)}
    )
    adata = ad.AnnData(X=np.zeros((len(xy), 1), dtype=np.float32))
    adata.uns["ecm_patches"] = patches
    return adata


def test_build_ecm_graph_knn_has_no_self_loops_with_duplicate_patches():
    adata = _ecm_adata(_DUP_XY)
    mt.gr.build_ecm_graph(adata, k=2, edge_method="knn")
    G = adata.uns["ecm_graph"]
    assert nx.number_of_selfloops(G) == 0
    assert G.number_of_nodes() == len(_DUP_XY)
    assert G.has_edge("ecm_0", "ecm_1") and G.edges["ecm_0", "ecm_1"]["distance"] == 0.0
    D = _pairwise(_DUP_XY)
    for u, v, dist in G.edges(data="distance"):
        assert dist == pytest.approx(D[int(u.split("_")[1]), int(v.split("_")[1])], abs=1e-12)


def test_build_ecm_graphs_delaunay_two_coincident_patches():
    adatas = {"s": _ecm_adata(_DUP_XY[:2])}
    mt.gr.build_ecm_graphs(adatas, edge_method="delaunay")
    G = adatas["s"].uns["ecm_graph"]
    assert nx.number_of_selfloops(G) == 0
    assert set(G.edges()) == {("ecm_0", "ecm_1")}


def test_build_cell_graph_knn_has_no_diagonal_entries_with_duplicate_centroids():
    adata = ad.AnnData(X=np.zeros((len(_DUP_XY), 1), dtype=np.float32))
    adata.obsm["spatial"] = _DUP_XY.copy()
    mt.gr.build_cell_graph(adata, k=2)
    A = adata.obsp["cell_graph"].tocoo()
    assert not np.any(A.row == A.col), "stored diagonal entry (self-loop)"
    assert nx.number_of_selfloops(adata.uns["cell_graph_nx"]) == 0
