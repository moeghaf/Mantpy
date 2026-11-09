"""Tests for negative-control graphs derived from a stored patch graph."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import mantpy as mt


def _graph_anndata():
    image = (np.arange(24 * 24, dtype=np.uint16).reshape(24, 24) % 31) + 1
    adata = mt.pp.image_ecm_patches(
        image, channel_names=["ColIV"], patch_size=4, foreground_mask=np.ones((6, 6), dtype=bool)
    )
    adata.obsm["X_cnn"] = np.random.default_rng(0).normal(size=(adata.n_obs, 4)).astype(np.float32)
    mt.gr.build_patch_graph(adata, k=4)
    return adata


def test_shuffled_edges_permutes_destinations_and_keeps_the_source_graph():
    adata = _graph_anndata()
    edges = adata.uns["patch"]["edge_index"].copy()
    attributes = adata.uns["patch"]["edge_attr"].copy()

    control = mt.gr.patch_graph_control(adata, "shuffled_edges", random_state=7)

    permutation = np.random.default_rng(7).permutation(edges.shape[1])
    assert isinstance(control, mt.gr.PatchGraphResult)
    assert control.graph_key == "patch_shuffled_edges"
    np.testing.assert_array_equal(control["edge_index"][0], edges[0])
    np.testing.assert_array_equal(control["edge_index"][1], edges[1, permutation])
    np.testing.assert_array_equal(control["edge_attr"], attributes)
    np.testing.assert_array_equal(adata.uns["patch"]["edge_index"], edges)
    stored = adata.uns["patch_shuffled_edges"]
    np.testing.assert_array_equal(stored["edge_index"], control["edge_index"])
    assert stored["params"]["control"] == "shuffled_edges"
    assert stored["params"]["source_graph"] == "patch"
    assert stored["params"]["random_state"] == 7
    assert stored["params"]["k"] == adata.uns["patch"]["params"]["k"]
    assert list(stored["edge_feature_names"]) == list(adata.uns["patch"]["edge_feature_names"])
    assert adata.obsp["patch_shuffled_edges_connectivities"].shape == (adata.n_obs, adata.n_obs)
    assert not np.array_equal(control["edge_index"], edges)


def test_shuffled_edges_is_reproducible_and_seed_dependent():
    adata = _graph_anndata()

    first = mt.gr.patch_graph_control(adata, "shuffled_edges", random_state=1, key_added="a")
    again = mt.gr.patch_graph_control(adata, "shuffled_edges", random_state=1, key_added="b")
    other = mt.gr.patch_graph_control(adata, "shuffled_edges", random_state=2, key_added="c")

    np.testing.assert_array_equal(first["edge_index"], again["edge_index"])
    assert not np.array_equal(first["edge_index"], other["edge_index"])
    assert {"a", "b", "c"} <= set(adata.uns)
    assert first.graph_key == "a"


def test_no_edges_removes_every_edge():
    adata = _graph_anndata()
    n_features = adata.uns["patch"]["edge_attr"].shape[1]

    control = mt.gr.patch_graph_control(adata, "no_edges")

    assert control.graph_key == "patch_no_edges"
    assert control["edge_index"].shape == (2, 0)
    assert control["edge_attr"].shape == (0, n_features)
    assert control.n_directed_edges == 0
    assert adata.obsp["patch_no_edges_connectivities"].nnz == 0
    assert adata.uns["patch"]["edge_index"].shape[1] > 0
    assert "random_state" not in adata.uns["patch_no_edges"]["params"]


def test_invalid_control_or_missing_graph_is_rejected():
    adata = _graph_anndata()

    with pytest.raises(ValueError, match="shuffled_edges"):
        mt.gr.patch_graph_control(adata, "rewired")
    with pytest.raises(KeyError, match="build_patch_graph"):
        mt.gr.patch_graph_control(adata, "no_edges", graph_key="absent")


@pytest.mark.skipif(importlib.util.find_spec("torch_geometric") is None, reason="requires torch_geometric")
@pytest.mark.parametrize("control", ["shuffled_edges", "no_edges"])
def test_control_graph_converts_to_pyg(control):
    adata = _graph_anndata()

    result = mt.gr.patch_graph_control(adata, control, random_state=0)
    data = mt.gr.to_pyg(adata, graph_key=result.graph_key, node_feature_key="X_cnn", include_edge_features=True)

    assert tuple(data.x.shape) == (adata.n_obs, 4)
    assert tuple(data.edge_index.shape) == result["edge_index"].shape
    assert tuple(data.edge_attr.shape) == result["edge_attr"].shape
