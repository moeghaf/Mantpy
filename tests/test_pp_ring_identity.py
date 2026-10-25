"""Perimeter graph identity exclusion and feature-oracle regressions."""

from __future__ import annotations

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from mantpy.pp import _compute_islet_graph_feat_11d, extract_structure_ecm


def _features(points, values=None, flags=None):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(points)
    return _compute_islet_graph_feat_11d(
        np.ones((3, 3), dtype=np.uint8),
        points[:, 0],
        points[:, 1],
        np.ones(n) if flags is None else np.asarray(flags),
        np.arange(n) % 2 if values is None else np.asarray(values),
    )


def _edge_feature_oracle(n, pairs, values):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(pairs)
    sizes = [len(component) for component in nx.connected_components(graph)]
    degree = np.array([graph.degree(i) for i in range(n)], dtype=float)
    p = degree[degree > 0] / degree.sum()
    entropy = -np.sum(p * np.log(p)) / np.log(max(len(p), 2))
    src = np.array([values[i] for i, j in sorted(pairs)])
    dst = np.array([values[j] for i, j in sorted(pairs)])
    corr = np.corrcoef(src, dst)[0, 1] if len(src) >= 4 and src.std() > 0 and dst.std() > 0 else 0.0
    return [len(sizes) / n, max(sizes) / n, entropy, corr]


def test_four_coincident_nodes_are_complete_and_have_known_features():
    result = _features([[1, 1]] * 4, values=[0, 0, 1, 1])
    # K4 has equal degree 3. Existing canonical-endpoint Pearson definition = 0.2.
    np.testing.assert_allclose(result[6:10], [0.25, 1.0, 1.0, 0.2], atol=1e-7)


@pytest.mark.parametrize("n,expected", [(0, [1, 0, 0, 0]), (1, [1, 1, 0, 0]), (2, [0.5, 1, 1, 0])])
def test_tiny_perimeter_branches(n, expected):
    result = _features([[1, 1]] * n)
    np.testing.assert_allclose(result[6:10], expected, atol=1e-7)


@pytest.mark.parametrize("position", ["first", "middle", "last", "absent"])
def test_controlled_ring_ties_exclude_identity_and_cap_three(monkeypatch, position):
    n = 8
    rows = np.array([[i, (i + 1) % n, (i + 2) % n, (i + 3) % n] for i in range(n)])
    rows[0] = {"first": [0, 1, 2, 3], "middle": [1, 0, 2, 3], "last": [1, 2, 3, 0], "absent": [1, 2, 3, 4]}[position]

    class ControlledTree:
        def __init__(self, points):
            assert points.shape == (n, 2)

        def query(self, points, k):
            assert k == 4
            return np.zeros(rows.shape), rows.copy()

    monkeypatch.setattr("scipy.spatial.cKDTree", ControlledTree)
    chosen = [[(i + 1) % n, (i + 2) % n, (i + 3) % n] for i in range(n)]
    pairs = {tuple(sorted((i, j))) for i, row in enumerate(chosen) for j in row}
    values = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_allclose(_features([[0, 0]] * n, values)[6:10], _edge_feature_oracle(n, pairs, values), atol=1e-7)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_untied_ring_matches_legacy_topology_oracle(seed):
    points = np.random.default_rng(seed).normal(size=(11, 2))
    _, indices = cKDTree(points).query(points, k=4)
    pairs = {tuple(sorted((i, j))) for i, row in enumerate(indices) for j in row[1:]}
    values = np.arange(11) % 3
    np.testing.assert_allclose(_features(points, values)[6:10], _edge_feature_oracle(11, pairs, values), atol=1e-7)


def test_nonperimeter_rows_do_not_enter_ring():
    points = [[0, 0], [10, 10], [0, 0], [11, 11], [0, 0], [0, 0]]
    result = _features(points, [0, 7, 0, 8, 1, 1], [1, 0, 1, 0, 1, 1])
    np.testing.assert_allclose(result[6:10], [0.25, 1, 1, 0.2], atol=1e-7)


def test_public_structure_extraction_accepts_duplicate_centroids_without_bias():
    data = ad.AnnData(np.ones((1, 1)))
    data.obsm["spatial"] = np.array([[1.0, 1.0]])
    data.uns["ecm_patches"] = pd.DataFrame({"x": [1.0] * 4, "y": [1.0] * 4, "feat_mean": [1.0] * 4})
    graph = nx.Graph()
    graph.add_edges_from([("cell_0", f"ecm_{j}", {"edge_type": "cell-ecm"}) for j in range(4)])
    data.uns["cell_ecm_graph"] = graph
    image = np.ones((3, 3), dtype=np.float32)
    result = extract_structure_ecm(
        data,
        structure_indices=np.array([0]),
        structure_uid="duplicate_fixture",
        ecm_channel_img=image,
        structure_mask=image,
        min_ecm_nodes=4,
        precomputed_sauvola_binary=image,
    )
    assert result.n_obs == 4
    np.testing.assert_array_equal(result.obs[["x", "y"]].to_numpy(), np.ones((4, 2)))
    np.testing.assert_allclose(result.uns["islet_graph_feat"][6:10], [0.25, 1, 1, 0], atol=1e-7)
