"""M01 regression tests: consistent edge counting in cluster co-occurrence.

Covers :func:`mantpy.tl.cluster_cooccurrence` and
:func:`mantpy.tl.cluster_cooccurrence_permutation_test`.

Specified statistic (the documented abundance-product reference is retained):

* ``O[a, b]`` counts ordered endpoint pairs ``(u, v)`` with ``label(u) == a``
  and ``label(v) == b``.  Every undirected edge ``{u, v}`` with ``u != v``
  contributes both ``(u, v)`` and ``(v, u)``; a self-loop ``(u, u)``
  contributes once (adjacency-matrix convention, ``O = L^T A L``).  Edge
  weights are ignored; directed graphs are rejected.
* ``T = O.sum()`` and ``p[a] = n_a / sum(n)`` over all nodes whose label lies
  in ``[0, K)`` (isolated nodes included; other labels and their edges are
  excluded).
* ``E = T * outer(p, p)``.  This abundance product is *not* the exact
  finite-population random-label expectation, so a complete graph is not
  exactly neutral under it.
* ``L = log2((O + eps) / E)``; cells with ``E == 0`` are returned as 0.
  ROIs in one group are pooled by summing ``O`` and ``n`` before normalising.

Hand-enumerated cases spell out ``O`` and ``E`` explicitly.  The generic
oracle enumerates all ordered node pairs and never calls Mantpy's counting
code.  Before the M01 correction, a same-cluster edge added 1 to the diagonal
while a cross-cluster edge added 1 to each mirrored off-diagonal cell, so the
cases containing same-cluster edges fail on the uncorrected implementation.
"""

from __future__ import annotations

import itertools

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import mantpy as mt

EPS = 1e-9
PATH_ABAB = np.log2((np.array([[2.0, 1.0], [1.0, 2.0]]) + EPS) / 1.5)


def _adata(edges, labels, name="roi", *, graph_cls=nx.Graph, weights=None):
    G = graph_cls()
    for i, lbl in enumerate(labels):
        G.add_node(i, ecm_cluster=int(lbl))
    for k, (u, v) in enumerate(edges):
        if weights is None:
            G.add_edge(u, v)
        else:
            G.add_edge(u, v, weight=float(weights[k]), inv_distance=float(weights[k]))
    adata = ad.AnnData(
        X=np.zeros((len(labels), 1), dtype=np.float32),
        obs=pd.DataFrame(index=[f"{name}_{i}" for i in range(len(labels))]),
    )
    adata.uns["ecm_graph"] = G
    adata.uns["sample_id"] = name
    return adata


def _log2_ratio(O, n_per_cluster):
    """Abundance-product reference applied to explicit counts (no Mantpy code)."""
    O = np.asarray(O, dtype=float)
    n = np.asarray(n_per_cluster, dtype=float)
    p = n / max(n.sum(), 1.0)
    E = O.sum() * np.outer(p, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log2((O + EPS) / np.where(E > 0, E, np.nan))
    return np.where(np.isfinite(L), L, 0.0)


def _oracle_counts(n_nodes, edges, labels, K):
    """Brute-force enumeration over every ordered node pair ``(i, j)``."""
    undirected = {frozenset((u, v)) for u, v in edges if u != v}
    loops = {u for u, v in edges if u == v}
    O = np.zeros((K, K))
    for i, j in itertools.product(range(n_nodes), repeat=2):
        li, lj = labels[i], labels[j]
        if not (0 <= li < K and 0 <= lj < K):
            continue
        if (i == j and i in loops) or (i != j and frozenset((i, j)) in undirected):
            O[li, lj] += 1
    n = [sum(1 for lbl in labels if lbl == a) for a in range(K)]
    return O, n


def _oracle(rois, K):
    """Oracle log-enrichment for one ROI or several pooled ROIs."""
    O_tot, n_tot = np.zeros((K, K)), np.zeros(K)
    for n_nodes, edges, labels in rois:
        O, n = _oracle_counts(n_nodes, edges, labels, K)
        O_tot += O
        n_tot += n
    return _log2_ratio(O_tot, n_tot)


def _random_roi(rng, n_nodes, K, p_edge, p_background=0.1):
    labels = [int(x) for x in rng.integers(0, K, size=n_nodes)]
    labels = [-1 if rng.random() < p_background else lbl for lbl in labels]
    edges = [(i, j) for i, j in itertools.combinations(range(n_nodes), 2) if rng.random() < p_edge]
    return n_nodes, edges, labels


# ---------------------------------------------------------------------------
# Hand-enumerated cases
# ---------------------------------------------------------------------------


def test_path_graph_hand_enumerated():
    # 0-1-2-3 labelled A, A, B, B.  Ordered pairs: (0,1),(1,0) -> AA x2;
    # (1,2) -> AB; (2,1) -> BA; (2,3),(3,2) -> BB x2.
    # O = [[2, 1], [1, 2]], T = 6, p = (1/2, 1/2), E = 1.5 in every cell.
    got = mt.tl.cluster_cooccurrence([_adata([(0, 1), (1, 2), (2, 3)], [0, 0, 1, 1])], n_clusters=2)
    np.testing.assert_allclose(got, PATH_ABAB, rtol=0, atol=1e-9)
    assert got[0, 0] == pytest.approx(np.log2(4 / 3), abs=1e-8)
    assert got[0, 1] == pytest.approx(np.log2(2 / 3), abs=1e-8)


def test_complete_graph_is_not_neutral_under_abundance_reference():
    # K4 labelled A, A, B, B: AA edge 0-1 -> 2 ordered pairs, BB edge 2-3 -> 2,
    # four AB edges -> 4 (A,B) + 4 (B,A).  O = [[2, 4], [4, 2]], T = 12, E = 3.
    # The abundance product includes self-pairs, so same-cluster cells of a
    # complete graph fall below the reference (log2 2/3), cross-cluster cells
    # above it (log2 4/3).
    adata = _adata(list(itertools.combinations(range(4), 2)), [0, 0, 1, 1])
    got = mt.tl.cluster_cooccurrence([adata], n_clusters=2)
    expected = np.log2((np.array([[2.0, 4.0], [4.0, 2.0]]) + EPS) / 3.0)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


def test_unequal_sizes_isolate_and_empty_cells():
    # Star centred on node 0 (A) with leaves 1 (B), 2 (B), 3 (C); node 4 (A) is
    # isolated but counts towards abundance.  Ordered pairs: (A,B) x2, (B,A) x2,
    # (A,C) x1, (C,A) x1.  n = (2, 2, 1) -> p = (0.4, 0.4, 0.2); T = 6.
    got = mt.tl.cluster_cooccurrence([_adata([(0, 1), (0, 2), (0, 3)], [0, 1, 1, 2, 0])], n_clusters=3)
    O = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    E = 6.0 * np.array([[0.16, 0.16, 0.08], [0.16, 0.16, 0.08], [0.08, 0.08, 0.04]])
    np.testing.assert_allclose(got, np.log2((O + EPS) / E), rtol=0, atol=1e-9)
    # Empty cells with a positive expectation take the documented pseudocount
    # value log2(eps / E) (about -29.8 here), not 0.
    assert got[1, 1] == pytest.approx(np.log2(EPS / 0.96), abs=1e-9)


def test_absent_cluster_row_and_column_are_zero():
    got = mt.tl.cluster_cooccurrence([_adata([(0, 1), (1, 2), (2, 3)], [0, 0, 1, 1])], n_clusters=3)
    np.testing.assert_array_equal(got[2, :], 0.0)
    np.testing.assert_array_equal(got[:, 2], 0.0)
    np.testing.assert_allclose(got[:2, :2], PATH_ABAB, rtol=0, atol=1e-9)


def test_graph_without_edges_returns_zeros():
    got = mt.tl.cluster_cooccurrence([_adata([], [0, 1, 1])], n_clusters=2)
    np.testing.assert_array_equal(got, np.zeros((2, 2)))


def test_out_of_range_labels_are_excluded_from_counts_and_abundance():
    # Path A, A, B, B plus node 4 labelled -1 (background) joined to node 3 and
    # node 5 labelled 7 (>= K) joined to node 0: both are excluded.
    adata = _adata([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5)], [0, 0, 1, 1, -1, 7])
    got = mt.tl.cluster_cooccurrence([adata], n_clusters=2)
    np.testing.assert_allclose(got, PATH_ABAB, rtol=0, atol=1e-9)
    # ``ignore_negative_labels`` is retained for compatibility; labels outside
    # [0, K) can never index the matrix, so False gives the same result.
    same = mt.tl.cluster_cooccurrence([adata], n_clusters=2, ignore_negative_labels=False)
    np.testing.assert_array_equal(same, got)


def test_self_loop_contributes_one_ordered_pair():
    # Path A, A, B, B plus a self-loop on node 0 (adjacency diagonal entry).
    # O = [[2 + 1, 1], [1, 2]], T = 7, E = 7/4.
    adata = _adata([(0, 1), (1, 2), (2, 3), (0, 0)], [0, 0, 1, 1])
    got = mt.tl.cluster_cooccurrence([adata], n_clusters=2)
    np.testing.assert_allclose(got, np.log2((np.array([[3.0, 1.0], [1.0, 2.0]]) + EPS) / 1.75), rtol=0, atol=1e-9)


def test_edge_weights_do_not_change_the_unit_count_statistic():
    edges, labels = [(0, 1), (1, 2), (2, 3), (1, 3)], [0, 0, 1, 1]
    plain = mt.tl.cluster_cooccurrence([_adata(edges, labels)], n_clusters=2)
    weighted = mt.tl.cluster_cooccurrence([_adata(edges, labels, weights=[0.1, 5.0, 2.0, 0.7])], n_clusters=2)
    np.testing.assert_array_equal(plain, weighted)


@pytest.mark.parametrize("graph_cls", [nx.DiGraph, nx.MultiDiGraph])
def test_directed_graphs_are_rejected(graph_cls):
    adata = _adata([(0, 1), (1, 2)], [0, 0, 1], graph_cls=graph_cls)
    with pytest.raises(ValueError, match="undirected"):
        mt.tl.cluster_cooccurrence([adata], n_clusters=2)
    other = _adata([(0, 1), (1, 2)], [0, 0, 1], name="roi2", graph_cls=graph_cls)
    with pytest.raises(ValueError, match="undirected"):
        mt.tl.cluster_cooccurrence_permutation_test(
            {"roi": adata, "roi2": other}, group_a_rois=["roi"], group_b_rois=["roi2"], n_clusters=2, n_perm=5
        )


# ---------------------------------------------------------------------------
# Generic oracle, pooling and invariances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_matches_bruteforce_oracle_on_random_graphs(seed):
    rng = np.random.default_rng(seed)
    K = int(rng.integers(2, 6))
    roi = _random_roi(rng, int(rng.integers(5, 30)), K, p_edge=float(rng.uniform(0.05, 0.6)))
    got = mt.tl.cluster_cooccurrence([_adata(roi[1], roi[2])], n_clusters=K)
    np.testing.assert_allclose(got, _oracle([roi], K), rtol=0, atol=1e-9)
    np.testing.assert_array_equal(got, got.T)


def test_pooled_rois_equal_oracle_on_pooled_counts():
    rng = np.random.default_rng(11)
    rois = [_random_roi(rng, n, 4, 0.3) for n in (12, 20, 7)]
    adatas = [_adata(r[1], r[2], name=f"r{i}") for i, r in enumerate(rois)]
    np.testing.assert_allclose(mt.tl.cluster_cooccurrence(adatas, n_clusters=4), _oracle(rois, 4), rtol=0, atol=1e-9)


def test_node_renumbering_invariance_and_cluster_renaming_equivariance():
    rng = np.random.default_rng(5)
    n_nodes, edges, labels = _random_roi(rng, 25, 4, 0.25, p_background=0.0)
    base = mt.tl.cluster_cooccurrence([_adata(edges, labels)], n_clusters=4)
    perm = rng.permutation(n_nodes)
    edges_p = [(int(perm[u]), int(perm[v])) for u, v in edges]
    labels_p = [0] * n_nodes
    for old, new in enumerate(perm):
        labels_p[int(new)] = labels[old]
    renumbered = mt.tl.cluster_cooccurrence([_adata(edges_p, labels_p)], n_clusters=4)
    np.testing.assert_allclose(renumbered, base, rtol=0, atol=1e-12)
    pi = np.array([2, 0, 3, 1])
    renamed = mt.tl.cluster_cooccurrence([_adata(edges, [int(pi[lbl]) for lbl in labels])], n_clusters=4)
    np.testing.assert_allclose(renamed[np.ix_(pi, pi)], base, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Permutation-test path
# ---------------------------------------------------------------------------


def _cohort(rng, n_per_group, K):
    adatas, rois = {}, {}
    for g in ("a", "b"):
        for i in range(n_per_group):
            name = f"{g}{i}"
            roi = _random_roi(rng, int(rng.integers(10, 25)), K, float(rng.uniform(0.15, 0.5)), p_background=0.05)
            rois[name] = roi
            adatas[name] = _adata(roi[1], roi[2], name=name)
    return adatas, rois


def test_permutation_delta_matches_summary_function_and_oracle():
    rng = np.random.default_rng(21)
    adatas, rois = _cohort(rng, 3, 3)
    A, B = ["a0", "a1", "a2"], ["b0", "b1", "b2"]
    r = mt.tl.cluster_cooccurrence_permutation_test(
        adatas, group_a_rois=A, group_b_rois=B, n_clusters=3, n_perm=10, seed=0
    )
    summary = mt.tl.cluster_cooccurrence([adatas[n] for n in A], n_clusters=3) - mt.tl.cluster_cooccurrence(
        [adatas[n] for n in B], n_clusters=3
    )
    oracle = _oracle([rois[n] for n in A], 3) - _oracle([rois[n] for n in B], 3)
    np.testing.assert_allclose(r.delta_obs, summary, rtol=0, atol=1e-12)
    np.testing.assert_allclose(r.delta_obs, oracle, rtol=0, atol=1e-9)


def test_permutation_pvalues_agree_with_exact_allocation_enumeration():
    """Two ROIs per group give C(4, 2) = 6 equally likely group-A allocations.

    The Monte Carlo two-sided p-value must estimate the exact tail fraction
    computed from the corrected oracle over all six allocations.  The fixture
    (seed 2) is one where the uncorrected counting gives a different exact
    tail in cells (0, 1) and (1, 0) (6/6 instead of 4/6), so p-values are
    covered by the regression and are not assumed to be invariant.
    """
    rng = np.random.default_rng(2)
    adatas, rois = _cohort(rng, 2, 3)
    names = ["a0", "a1", "b0", "b1"]
    obs = _oracle([rois["a0"], rois["a1"]], 3) - _oracle([rois["b0"], rois["b1"]], 3)
    allocations = list(itertools.combinations(names, 2))
    tails = np.zeros((3, 3))
    for grp in allocations:
        rest = [n for n in names if n not in grp]
        d = _oracle([rois[n] for n in grp], 3) - _oracle([rois[n] for n in rest], 3)
        tails += np.abs(d) >= np.abs(obs) - 1e-12
    p_exact = tails / len(allocations)
    n_perm = 4000
    r = mt.tl.cluster_cooccurrence_permutation_test(
        adatas, group_a_rois=["a0", "a1"], group_b_rois=["b0", "b1"], n_clusters=3, n_perm=n_perm, seed=1
    )
    se = np.sqrt(p_exact * (1.0 - p_exact) / n_perm)
    assert np.all(np.abs(r.p_uncorr - p_exact) <= 5.0 * se + 2.0 / (n_perm + 1))
