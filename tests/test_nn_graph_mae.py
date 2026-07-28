"""Tests for mt.nn.GraphMAE (node-level masked-feature graph autoencoder)."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    from torch_geometric.data import Data

    HAS_GNN = True
except ImportError:
    HAS_GNN = False

pytestmark = pytest.mark.skipif(not HAS_GNN, reason="requires mantpy[gnn]")


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    n = 80
    x = torch.tensor(rng.standard_normal((n, 6)), dtype=torch.float32)
    ei = np.vstack([np.repeat(np.arange(n), 4), rng.integers(0, n, n * 4)])
    ei = np.hstack([ei, ei[::-1]])
    ea = torch.tensor(rng.standard_normal((ei.shape[1], 3)), dtype=torch.float32)
    return Data(x=x, edge_index=torch.tensor(ei, dtype=torch.long), edge_attr=ea)


def test_train_and_node_latent_shape(graph):
    from mantpy.nn import GraphMAE

    m = GraphMAE(hidden_dim=16, n_layers=2, encoder="gine").train(
        graph,
        max_epochs=3,
        rng=0,
        accelerator="cpu",
        enable_progress_bar=False,
    )
    h = m.get_node_latent()
    assert h.shape == (graph.x.shape[0], 16)
    assert np.isfinite(h).all()


def test_train_progress_reports_epoch_and_loss(graph, capsys):
    from mantpy.nn import GraphMAE

    GraphMAE(hidden_dim=8).train(
        graph,
        max_epochs=2,
        rng=0,
        accelerator="cpu",
        enable_progress_bar=True,
    )

    stderr = capsys.readouterr().err
    assert "GraphMAE" in stderr
    assert "loss=" in stderr


def test_reproducible_with_seed(graph):
    from mantpy.nn import GraphMAE

    a = GraphMAE(hidden_dim=8).train(graph, max_epochs=3, rng=1, accelerator="cpu").get_node_latent()
    b = GraphMAE(hidden_dim=8).train(graph, max_epochs=3, rng=1, accelerator="cpu").get_node_latent()
    np.testing.assert_allclose(a, b)


def test_node_only_encoder_runs_without_edges(graph):
    from mantpy.nn import GraphMAE

    g2 = Data(x=graph.x, edge_index=graph.edge_index)  # no edge_attr
    m = GraphMAE(encoder="sage", hidden_dim=8).train(g2, max_epochs=3, rng=0, accelerator="cpu")
    assert m.get_node_latent().shape == (graph.x.shape[0], 8)


def test_gine_without_edges_raises(graph):
    from mantpy.nn import GraphMAE

    with pytest.raises(ValueError):
        GraphMAE(encoder="gine").train(Data(x=graph.x, edge_index=graph.edge_index), max_epochs=1, accelerator="cpu")


def test_unknown_accelerator_raises(graph):
    from mantpy.nn import GraphMAE

    with pytest.raises(ValueError, match="Unknown accelerator"):
        GraphMAE().train(graph, max_epochs=1, rng=0, accelerator="tpu")


def test_invalid_mask_ratio_raises():
    from mantpy.nn import GraphMAE

    with pytest.raises(ValueError):
        GraphMAE(mask_ratio=1.5)


def test_get_node_latent_before_train_raises():
    from mantpy.nn import GraphMAE

    with pytest.raises(RuntimeError):
        GraphMAE().get_node_latent()


def test_repr_reports_fitted_state(graph):
    from mantpy.nn import GraphMAE

    model = GraphMAE(hidden_dim=8, encoder="gine")
    assert "status='unfitted'" in repr(model)
    model.train(graph, max_epochs=1, rng=0, accelerator="cpu")
    text = repr(model)
    assert "status='fitted'" in text
    assert "n_nodes=80" in text


def test_encode_graphmae_attaches_embedding_and_provenance():
    from anndata import AnnData

    from mantpy.nn import GraphMAE, encode_graphmae

    rng = np.random.default_rng(4)
    n_nodes = 24
    adata = AnnData(rng.normal(size=(n_nodes, 5)).astype(np.float32))
    destinations = np.roll(np.arange(n_nodes), -1)
    edge_index = np.vstack(
        [
            np.concatenate([np.arange(n_nodes), destinations]),
            np.concatenate([destinations, np.arange(n_nodes)]),
        ]
    )
    adata.uns["patch"] = {
        "edge_index": edge_index,
        "edge_attr": rng.normal(size=(edge_index.shape[1], 3)).astype(np.float32),
    }
    adata.obsm["X_cnn"] = rng.normal(size=(n_nodes, 7)).astype(np.float32)
    adata.obs["held_out_anatomy"] = np.resize(["crypt", "villus"], n_nodes)

    model = encode_graphmae(
        adata,
        graph_key="patch",
        node_feature_key="X_cnn",
        key_added="X_graphmae",
        hidden_dim=8,
        max_epochs=2,
        random_state=2,
        accelerator="cpu",
        enable_progress_bar=False,
    )

    assert isinstance(model, GraphMAE)
    assert adata.obsm["X_graphmae"].shape == (n_nodes, 8)
    assert np.isfinite(adata.obsm["X_graphmae"]).all()
    params = adata.uns["X_graphmae_params"]
    assert params["graph_key"] == "patch"
    assert params["node_feature_key"] == "X_cnn"
    assert params["random_state"] == 2
    assert params["model"]["hidden_dim"] == 8
    assert params["device"] == "cpu"
    assert params["n_nodes"] == n_nodes
    assert set(params["software"]) == {"mantpy", "torch", "torch_geometric"}
    assert all(isinstance(version, str) for version in params["software"].values())
    assert "held_out_anatomy" not in str(params)

    with pytest.raises(ValueError, match="overwrite=True"):
        encode_graphmae(adata, max_epochs=1, accelerator="cpu")
