"""Contracts for customizable training, independent units and fitted transforms."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data


class _LinearNodes(torch.nn.Module):
    def __init__(self, in_features=2, n_classes=2):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, n_classes)

    def forward(self, x, edge_index):
        return self.linear(x)


def _graphs():
    return [
        Data(x=torch.tensor([[0.0, 1.0], [1.0, 0.0]]), edge_index=torch.empty((2, 0), dtype=torch.long)),
        Data(x=torch.tensor([[1.0, 1.0], [0.0, 0.0], [2.0, 1.0]]), edge_index=torch.empty((2, 0), dtype=torch.long)),
    ]


def test_custom_patch_factory_raw_rows_scaler_and_safe_reload(tmp_path):
    pytest.importorskip("pytorch_lightning")
    from anndata import AnnData

    from mantpy.nn import PatchEncoder

    calls = []

    def backbone_factory(latent_dim):
        calls.append(torch.initial_seed())
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, latent_dim))

    patches = np.random.default_rng(2).random((9, 1, 4, 4), dtype=np.float32)
    encoder = PatchEncoder(
        latent_dim=3, backbone=backbone_factory, backbone_kwargs={"latent_dim": 3},
        projection=torch.nn.Identity, augment=lambda batch: batch,
    ).fit(
        patches, max_epochs=2, batch_size=4, drop_last=False, random_state=13,
        scheduler=None, optimizer_factory=torch.optim.SGD, lr=0.01,
        accelerator="cpu", enable_progress_bar=False, enable_model_summary=False,
    )
    assert calls == [13]
    assert encoder.optimizer_updates == 6
    assert len(encoder.history["train_loss"]) == 2
    raw = encoder.transform(patches, batch_size=4, standardize=False)
    with torch.no_grad():
        expected = encoder.model(torch.from_numpy(patches)).numpy()
    np.testing.assert_allclose(raw, expected, rtol=1e-6, atol=1e-6)
    adata = AnnData(np.zeros((len(patches), 0)))
    adata.obsm["image_patches"] = patches
    encoder.transform_adata(adata, standardize=False)
    np.testing.assert_allclose(adata.obsm["X_cnn"], raw, rtol=1e-6, atol=1e-6)
    assert adata.uns["X_cnn_params"]["standardization"] == "none"
    scaled = encoder.transform(patches)
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-5)
    saved = tmp_path / "custom_encoder"
    encoder.save(saved)
    with pytest.raises(ValueError, match="backbone"):
        PatchEncoder.load(saved)
    restored = PatchEncoder.load(saved, backbone=backbone_factory)
    np.testing.assert_allclose(restored.transform(patches), scaled, rtol=1e-6, atol=1e-6)


def test_patch_sampler_exhaustion_matches_two_epoch_manual_loop():
    pytest.importorskip("pytorch_lightning")
    from torch.utils.data import DataLoader

    from mantpy.nn import PatchEncoder
    from mantpy.nn._patch_encoder import _augment, _nt_xent

    def backbone():
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 3))

    patches = np.random.default_rng(7).random((8, 1, 4, 4), dtype=np.float32)
    torch.manual_seed(3)
    model = backbone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, weight_decay=0)
    loader = DataLoader(
        torch.from_numpy(patches), batch_size=8, shuffle=True,
        generator=torch.Generator().manual_seed(3),
    )
    for _ in range(2):
        for batch in loader:
            views = torch.cat([_augment(batch, (.7, 1.3, .1, .05)) for _ in range(2)])
            loss = _nt_xent(model(views), .2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    fit = PatchEncoder(latent_dim=3, backbone=backbone, projection=torch.nn.Identity).fit(
        patches, max_epochs=2, batch_size=8, random_state=3,
        optimizer_factory=torch.optim.SGD, lr=.01, weight_decay=0,
        scheduler=None, exhaust_sampler=True, accelerator="cpu",
        enable_progress_bar=False, enable_model_summary=False,
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(fit.model.state_dict()[key], value, rtol=0, atol=0)
    assert fit.optimizer_updates == 2


def test_roi_loss_weights_independent_rois_instead_of_nodes():
    from mantpy.nn import node_classification_loss

    logits = torch.tensor([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1, 1, 1])
    batches = torch.tensor([0, 1, 1, 1])
    graph_targets = torch.tensor([0, 1])
    bad = float(np.log1p(np.exp(2.0)))
    good = float(np.log1p(np.exp(-2.0)))
    node = node_classification_loss(logits, targets, batch_index=batches, graph_targets=graph_targets)
    roi = node_classification_loss(
        logits, targets, batch_index=batches, graph_targets=graph_targets, reduction="roi"
    )
    assert float(node) == pytest.approx((bad + 3 * good) / 4)
    assert float(roi) == pytest.approx((bad + good) / 2)
    weighted = node_classification_loss(
        logits, targets, batch_index=batches, graph_targets=graph_targets,
        reduction="roi", class_weights=torch.tensor([2.0, 0.5]),
    )
    assert float(weighted) == pytest.approx((2 * bad + 0.5 * good) / 2)


def test_single_fit_matches_explicit_roi_training_recipe():
    from torch_geometric.data import Batch

    from mantpy.nn import train_node_classifier

    graphs = _graphs()
    targets = np.array([0, 1])
    torch.manual_seed(4)
    model = _LinearNodes()
    all_x = torch.cat([g.x for g in graphs]).numpy()
    mean = torch.tensor(all_x.mean(axis=0))
    scale = torch.tensor(all_x.std(axis=0))
    batch = Batch.from_data_list(graphs)
    features = (batch.x - mean) / scale
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
    expected_losses = []
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(features, batch.edge_index)
        node_losses = torch.nn.functional.cross_entropy(
            logits, torch.tensor(targets)[batch.batch], reduction="none"
        )
        loss = torch.stack([node_losses[batch.batch == i].mean() for i in range(2)]).mean()
        expected_losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    fit = train_node_classifier(
        graphs, targets, model_factory=_LinearNodes, epochs=3, seed=4,
        loss_reduction="roi", class_weight=None,
    )
    np.testing.assert_array_equal(fit.history["train_loss"], expected_losses)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(fit.model.state_dict()[key], value, rtol=0, atol=0)
    assert fit.optimizer_updates == 3


def test_explicit_leave_one_group_out_retains_group_metadata():
    from sklearn.model_selection import LeaveOneGroupOut

    from mantpy.nn import NodeClassifier

    graphs = _graphs() * 2
    labels = np.array([0, 1, 0, 1])
    groups = np.array(["a", "b", "c", "d"])
    splits = list(LeaveOneGroupOut().split(np.zeros(4), labels, groups))
    classifier = NodeClassifier(
        graphs, labels=labels, model_factory=_LinearNodes, device="cpu"
    ).fit(
        splits=splits, groups=groups, seeds=[2, 3], epochs=2,
        loss_reduction="roi", reset_seed_per_fit=True,
    )
    assert classifier.cv_metrics()["group_ids"].tolist() == ["a", "b", "c", "d"]
    assert len(classifier.training_history) == 16
    assert classifier.split_assignments()["group_id"].tolist() == groups.tolist()


def test_linear_ig_returns_signed_completeness_and_node_ids():
    pytest.importorskip("captum")
    from mantpy.nn import node_integrated_gradients

    graph = _graphs()[0]
    graph.node_ids = ["p1", "p2"]
    model = _LinearNodes()
    with torch.no_grad():
        model.linear.weight[:] = torch.tensor([[0.5, -0.2], [2.0, 3.0]])
        model.linear.bias.zero_()
    result = node_integrated_gradients(
        model, graph, mean=torch.zeros(2), scale=torch.ones(2),
        target_class=1, n_steps=4, internal_batch_size=1, return_details=True,
    )
    expected = graph.x.numpy() * np.array([2.0, 3.0]) / 2
    np.testing.assert_allclose(result.signed, expected, atol=1e-6)
    np.testing.assert_allclose(result.convergence_delta, 0, atol=1e-6)
    assert result.node_ids == ("p1", "p2")


def test_graphmae_history_raw_output_and_fitted_transform():
    from mantpy.nn import GraphMAE

    rng = np.random.default_rng(4)
    graph = Data(
        x=torch.tensor(rng.normal(size=(30, 2)), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    events = []
    model = GraphMAE(hidden_dim=4, encoder="sage", norm="layernorm").train(
        graph, max_epochs=3, rng=2, accelerator="cpu",
        enable_progress_bar=False, callback=events.append,
    )
    assert model.optimizer_updates == 3
    assert len(model.history["train_loss"]) == len(events) == 3
    assert all(np.isfinite(model.history["train_loss"]))
    raw = model.get_node_latent(standardize_output=False)
    scaled = model.get_node_latent()
    expected = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-8)
    np.testing.assert_allclose(scaled, expected)
    shifted = Data(x=graph.x + 10, edge_index=graph.edge_index)
    own = model.get_node_latent(shifted, standardize_output=False)
    fitted = model.get_node_latent(
        shifted, feature_transform="fitted", standardize_output=False,
    )
    assert not np.allclose(own, fitted)
    model.history["train_loss"].clear()
    assert len(model.history["train_loss"]) == 3


def test_neighbor_estimator_is_cloned_and_features_keep_patch_ids():
    import networkx as nx
    import pandas as pd
    from anndata import AnnData
    from sklearn.ensemble import ExtraTreesClassifier

    from mantpy._constants import EDGE_TYPE_CE, NODE_TYPE_CELL, NODE_TYPE_ECM
    from mantpy.nn import NeighbourCompositionBaseline

    cells = AnnData(obs=pd.DataFrame({"cell_type": ["B", "T"]}, index=["c0", "c1"]))
    graph = nx.Graph()
    for index, label in enumerate(("B", "T")):
        graph.add_node(f"cell_{index}", node_type=NODE_TYPE_CELL, cell_type=label)
    labels = [0, 0, 1, 1]
    for index, label in enumerate(labels):
        graph.add_node(f"ecm_{index}", node_type=NODE_TYPE_ECM, ecm_cluster=label)
        graph.add_edge(f"cell_{label}", f"ecm_{index}", edge_type=EDGE_TYPE_CE)
    cells.uns["cell_ecm_graph"] = graph
    ecm = AnnData()
    ecm.uns["ecm_patches"] = pd.DataFrame({"ecm_cluster": labels}, index=["p0", "p1", "p2", "p3"])
    estimator = ExtraTreesClassifier(n_estimators=5, random_state=1)
    model = NeighbourCompositionBaseline(estimator=estimator).fit([(cells, ecm)])
    assert not hasattr(estimator, "classes_")
    features = model.transform(cells, ecm)
    assert features.index.tolist() == ["p0", "p1", "p2", "p3"]
    assert features.columns.tolist() == ["cell:B", "cell:T"]
    np.testing.assert_array_equal(features.to_numpy(), [[1, 0], [1, 0], [0, 1], [0, 1]])
    np.testing.assert_array_equal(model.score(cells, ecm)["denoised_cluster"], labels)
