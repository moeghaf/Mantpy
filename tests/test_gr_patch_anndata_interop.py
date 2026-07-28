"""Focused tests for observation-native ECM patch graph interoperability."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import mantpy as mt


def _patch_anndata():
    image = (np.arange(144, dtype=np.uint16).reshape(12, 12) % 31) + 1
    adata = mt.pp.image_ecm_patches(
        image,
        channel_names=["CollIV"],
        patch_size=4,
        foreground_mask=np.ones((3, 3), dtype=bool),
    )
    adata.obsm["X_cnn"] = np.arange(36, dtype=np.float32).reshape(9, 4)
    return adata


def test_build_patch_graph_stores_anndata_graph_for_reuse():
    adata = _patch_anndata()

    graph = mt.gr.build_patch_graph(adata, k=2, key_added="ecm_knn", standardize_edges=False)

    assert isinstance(graph, mt.gr.PatchGraphResult)
    assert set(graph) == {"edge_index", "edge_attr", "n_edges", "edge_feature_names"}
    assert graph.n_nodes == adata.n_obs
    assert graph.graph_key == "ecm_knn"
    assert graph.connectivities_key == "ecm_knn_connectivities"
    assert graph.params["k"] == 2
    assert graph["edge_index"].shape == (2, 2 * graph["n_edges"])
    assert graph["edge_attr"].shape == (2 * graph["n_edges"], 3)
    assert graph["edge_feature_names"] == ("log_distance", "sin_angle2", "cos_angle2")
    assert "ecm_knn_connectivities" in adata.obsp
    connectivities = adata.obsp["ecm_knn_connectivities"]
    assert connectivities.shape == (adata.n_obs, adata.n_obs)
    assert (connectivities != connectivities.T).nnz == 0
    assert set(np.unique(connectivities.data)) == {1.0}
    assert type(adata.uns["ecm_knn"]) is dict
    np.testing.assert_array_equal(adata.uns["ecm_knn"]["edge_index"], graph["edge_index"])
    np.testing.assert_allclose(adata.uns["ecm_knn"]["edge_attr"], graph["edge_attr"])
    assert adata.uns["ecm_knn"]["connectivities_key"] == "ecm_knn_connectivities"
    assert adata.uns["ecm_knn"]["params"]["k"] == 2


def test_build_patch_graph_knn_union_is_invariant_to_row_order():
    coordinates = np.array([[0.0, 0.0], [100.0, 0.0], [101.0, 0.0]])

    original = mt.gr.build_patch_graph(coordinates, k=1)
    permutation = np.array([2, 1, 0])
    reordered = mt.gr.build_patch_graph(coordinates[permutation], k=1)

    def physical_edges(edge_index, row_to_original):
        directed = np.asarray(edge_index, dtype=int).T
        return {
            tuple(sorted((int(row_to_original[source]), int(row_to_original[target])))) for source, target in directed
        }

    identity = np.arange(len(coordinates))
    assert physical_edges(original["edge_index"], identity) == {(0, 1), (1, 2)}
    assert physical_edges(reordered["edge_index"], permutation) == {(0, 1), (1, 2)}


def test_explicit_foreground_requires_matching_pixel_coordinates():
    adata = _patch_anndata()

    with pytest.raises(ValueError, match="pixel_coords"):
        mt.gr.build_patch_graph(adata, k=2, foreground=np.ones((12, 12), dtype=bool))


HAS_PYG = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None


@pytest.mark.skipif(not HAS_PYG, reason="requires torch and torch-geometric (mantpy[gnn])")
def test_to_pyg_uses_direct_patch_graph_and_selected_representation():
    import torch

    adata = _patch_anndata()
    graph = mt.gr.build_patch_graph(adata, k=2, key_added="ecm_knn", standardize_edges=False)

    data = mt.gr.to_pyg(adata, graph_key="ecm_knn", node_feature_key="X_cnn")

    assert tuple(data.x.shape) == (adata.n_obs, 4)
    assert tuple(data.pos.shape) == (adata.n_obs, 2)
    assert tuple(data.edge_index.shape) == graph["edge_index"].shape
    assert tuple(data.edge_attr.shape) == graph["edge_attr"].shape
    assert torch.equal(data.x, torch.as_tensor(adata.obsm["X_cnn"]))
    assert torch.equal(data.edge_index, torch.as_tensor(graph["edge_index"]))
    assert torch.allclose(data.edge_attr, torch.as_tensor(graph["edge_attr"]))
    assert data.obs_names == list(adata.obs_names)
    assert data.node_feature_key == "X_cnn"


@pytest.mark.skipif(not HAS_PYG, reason="requires torch and torch-geometric (mantpy[gnn])")
def test_to_pyg_accepts_sparse_obsp_graph_and_defaults_to_x():
    import torch

    adata = _patch_anndata()

    data = mt.gr.to_pyg(adata, graph_key="grid_connectivities")

    assert tuple(data.x.shape) == adata.shape
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == adata.obsp["grid_connectivities"].nnz
    assert torch.allclose(data.x, torch.as_tensor(adata.X))
    assert data.node_feature_key == "X"


@pytest.mark.skipif(not HAS_PYG, reason="requires torch and torch-geometric (mantpy[gnn])")
def test_to_pyg_rejects_legacy_node_attributes_for_patch_graph():
    adata = _patch_anndata()

    with pytest.raises(ValueError, match="node_feature_keys"):
        mt.gr.to_pyg(adata, graph_key="grid_connectivities", node_feature_keys=["x", "y"])
