"""Identity exclusion regressions for patch graphs and domain adjacency."""

from __future__ import annotations

import importlib.util

import anndata as ad
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

import mantpy as mt


def _edges(coords, k, builder):
    if builder == "patch":
        graph = mt.gr.build_patch_graph(coords, k=k, standardize_edges=False)
        directed = list(map(tuple, graph.edge_index.T))
        assert len(directed) == len(set(directed))
        assert graph.n_directed_edges == 2 * graph.n_edges
        assert {(v, u) for u, v in directed} == set(directed)
        return {tuple(sorted(pair)) for pair in directed}
    adjacency = mt.tl._knn_adjacency(coords, k)
    assert adjacency.format == "csr"
    assert (adjacency != adjacency.T).nnz == 0
    assert set(adjacency.data) == {1.0}
    return {tuple(sorted(pair)) for pair in zip(*adjacency.nonzero(), strict=True)}


@pytest.mark.parametrize("builder", ["patch", "adjacency"])
@pytest.mark.parametrize(
    "coords,k",
    [
        ([[0, 0], [0, 0]], 1),
        ([[0, 0], [0, 0], [0, 0], [3, 0], [7, 0], [20, 0]], 2),
        ([[0, 0]] * 6, 2),
        ([[0, 0]] * 6, 5),
    ],
)
def test_duplicate_identities_never_become_self_edges(builder, coords, k):
    pairs = _edges(np.asarray(coords), k, builder)
    assert all(u != v for u, v in pairs)
    assert len(pairs) <= len(coords) * k
    # An undirected union may exceed degree k; each original row still selected k.
    assert all(sum(i in pair for pair in pairs) >= k for i in range(len(coords)))
    if k == len(coords) - 1:
        assert pairs == {(i, j) for i in range(len(coords)) for j in range(i + 1, len(coords))}
    if len(coords) == 2:
        assert pairs == {(0, 1)}


@pytest.mark.parametrize("builder", ["patch", "adjacency"])
@pytest.mark.parametrize("position", ["first", "middle", "last", "absent"])
def test_controlled_ties_keep_real_sklearn_identity_layer(monkeypatch, builder, position):
    n, k = 8, 2
    rows = np.array([[i, (i + 1) % n, (i + 2) % n] for i in range(n)])
    rows[0] = {"first": [0, 1, 2], "middle": [1, 0, 2], "last": [1, 2, 0], "absent": [1, 2, 3]}[position]
    original_fit = NearestNeighbors.fit

    class ControlledTree:
        def query(self, points, n_neighbors, return_distance):
            assert n_neighbors == k + 1
            return (np.zeros(rows.shape), rows.copy()) if return_distance else rows.copy()

    def fit(self, points, y=None):
        assert self.n_neighbors == k + 1  # retain the auto-algorithm fit heuristic
        original_fit(self, points, y)
        self._tree = ControlledTree()
        return self

    monkeypatch.setattr(NearestNeighbors, "fit", fit)
    actual = _edges(np.zeros((n, 2)), k, builder)
    expected_rows = [[(i + 1) % n, (i + 2) % n] for i in range(n)]
    expected_rows[0] = [2, 3] if position == "absent" else [1, 2]
    expected = {tuple(sorted((i, j))) for i, row in enumerate(expected_rows) for j in row}
    assert actual == expected


@pytest.mark.parametrize("builder", ["patch", "adjacency"])
@pytest.mark.parametrize("n,k", [(0, 1), (1, 1), (5, 0), (5, -1), (5, 5), (5, 6)])
def test_strict_k_contract_is_retained(builder, n, k):
    with pytest.raises(ValueError, match="expected 1 <= k"):
        _edges(np.zeros((n, 2)), k, builder)


@pytest.mark.parametrize("builder", ["patch", "adjacency"])
@pytest.mark.parametrize("n,k", [(6, 1), (8, 2), (8, 3), (11, 4), (12, 5), (12, 11)])
def test_untied_points_match_independent_distance_oracle(builder, n, k):
    points = np.random.default_rng(n).normal(size=(n, 2))
    squared = ((points[:, None] - points[None]) ** 2).sum(axis=2)
    np.fill_diagonal(squared, np.inf)
    expected = {tuple(sorted((i, j))) for i, row in enumerate(np.argsort(squared, axis=1)[:, :k]) for j in row}
    assert _edges(points, k, builder) == expected


@pytest.mark.parametrize("builder", ["patch", "adjacency"])
@pytest.mark.parametrize("k", [1, 3, 6, 7, 8, 15])
def test_unique_tied_grid_preserves_legacy_fit_and_selection(builder, k):
    points = np.asarray([(x, y) for x in range(4) for y in range(4)], dtype=float)
    old = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points, return_distance=False)[:, 1:]
    expected = {tuple(sorted((i, j))) for i, row in enumerate(old) for j in row}
    assert _edges(points, k, builder) == expected


def test_patch_graph_duplicate_nodes_attrs_and_anndata_storage():
    points = np.array([[0, 0], [0, 0], [4, 0], [4, 0]], dtype=float)
    data = ad.AnnData(np.arange(8, dtype=np.float32).reshape(4, 2))
    data.obsm["spatial"] = points
    graph = mt.gr.build_patch_graph(data, k=3, edge_features=["distance", "log_distance"], standardize_edges=False)
    assert graph.n_edges == 6
    assert graph.n_directed_edges == 12
    assert all(u != v for u, v in graph.edge_index.T)
    expected_distance = np.linalg.norm(points[graph.edge_index[1]] - points[graph.edge_index[0]], axis=1)
    np.testing.assert_allclose(graph.edge_attr[:, 0], expected_distance)
    np.testing.assert_allclose(graph.edge_attr[:, 1], np.log1p(expected_distance))
    np.testing.assert_array_equal(data.obsp["patch_connectivities"].toarray(), np.ones((4, 4)) - np.eye(4))
    np.testing.assert_array_equal(data.uns["patch"]["edge_index"], graph.edge_index)
    np.testing.assert_array_equal(data.uns["patch"]["edge_attr"], graph.edge_attr)


@pytest.mark.skipif(importlib.util.find_spec("torch_geometric") is None, reason="requires mantpy[gnn]")
def test_duplicate_patch_graph_to_pyg_keeps_mirrored_attributes():
    data = ad.AnnData(np.ones((4, 2), dtype=np.float32))
    data.obsm["spatial"] = np.array([[0, 0], [0, 0], [4, 0], [4, 0]], dtype=float)
    graph = mt.gr.build_patch_graph(data, k=3, edge_features=["distance"], standardize_edges=False)
    pyg = mt.gr.to_pyg(data, graph_key="patch")
    np.testing.assert_array_equal(pyg.edge_index.numpy(), graph.edge_index)
    np.testing.assert_array_equal(pyg.edge_attr.numpy(), graph.edge_attr)
    assert pyg.edge_index.shape == (2, 12)
    assert not np.any(pyg.edge_index[0].numpy() == pyg.edge_index[1].numpy())


def test_continuity_pruning_retains_distinct_coincident_nodes():
    points = np.array([[0, 0], [0, 0], [0, 5], [0, 5]], dtype=float)
    foreground = np.zeros((6, 1), dtype=bool)
    foreground[[0, 5]] = True
    result = mt.gr.build_patch_graph(
        points,
        k=3,
        pixel_coords=(points[:, 1], points[:, 0]),
        foreground=foreground,
        edge_features=["distance", "continuity"],
        standardize_edges=False,
    )
    assert {tuple(sorted(pair)) for pair in result.edge_index.T} == {(0, 1), (2, 3)}
    np.testing.assert_array_equal(result.edge_attr, [[0, 1]] * 4)


def test_duplicate_clusters_have_hand_calculated_modularity_and_anydim_adjacency():
    points = np.array([[0, 0, 0], [0, 0, 0], [10, 0, 0], [10, 0, 0]], dtype=float)
    expected = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    np.testing.assert_array_equal(mt.tl._knn_adjacency(points, 1).toarray(), expected)
    assert mt.tl.graph_modularity(np.array([0, 0, 1, 1]), coords=points, k=1) == pytest.approx(0.5)


def test_union_degree_can_exceed_selection_k():
    points = np.array([[0, 0], [1, 0], [-1.1, 0], [0, 1.2], [0, -1.3]])
    for builder in ("patch", "adjacency"):
        edges = _edges(points, 1, builder)
        assert edges == {(0, 1), (0, 2), (0, 3), (0, 4)}
