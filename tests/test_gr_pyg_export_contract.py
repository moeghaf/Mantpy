"""Consumer checks for legacy graph exports and source identity."""

from __future__ import annotations

import io

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import mantpy as mt

torch = pytest.importorskip("torch")
NNConv = pytest.importorskip("torch_geometric.nn").NNConv


def _legacy_graph():
    graph = nx.Graph()
    # Node insertion order deliberately opposes several stored edge directions.
    for name, kind, position in [
        ("e49", "ecm", (8.0, 3.0)),
        ("c17", "cell", (1.0, 2.0)),
        ("e17", "ecm", (4.0, 1.0)),
        ("c49", "cell", (6.0, 5.0)),
    ]:
        graph.add_node(
            name,
            node_type=kind,
            x=position[0],
            y=position[1],
            source_id=f"original:{name}",
            ecm_features=[0.2] if kind == "ecm" else [],
        )
    for source, target in [("c49", "c17"), ("e17", "e49"), ("c17", "e49"), ("e17", "c49")]:
        delta = np.array([graph.nodes[target][a] - graph.nodes[source][a] for a in ("x", "y")])
        relation = "cell-ecm" if source[0] != target[0] else ("cell-cell" if source[0] == "c" else "ecm-ecm")
        graph.add_edge(
            source,
            target,
            edge_type=relation,
            feat_source=source,
            feat_target=target,
            feat_fwd=delta,
            feat_rev=-delta,
        )
    obj = ad.AnnData(X=np.empty((0, 0), dtype=np.float32))
    obj.uns["cell_ecm_graph"] = graph
    obj.uns["sample_id"] = "roi:source"
    return obj


@pytest.mark.parametrize("include_features", [True, False])
def test_heterogeneous_export_preserves_directed_edges_and_consumer_forward(include_features):
    obj = _legacy_graph()
    data = mt.gr.to_hetero_pyg(obj, include_edge_features=include_features)
    assert data.validate()
    assert set(data.edge_types) == {
        ("cell", "cell-cell", "cell"),
        ("ecm", "ecm-ecm", "ecm"),
        ("cell", "cell-ecm", "ecm"),
        ("ecm", "ecm-cell", "cell"),
    }
    actual_edges = set()
    for relation in data.edge_types:
        source, _, target = relation
        edges = data[relation].edge_index
        actual_edges.update((data[source].node_ids[u], data[target].node_ids[v]) for u, v in edges.T.tolist())
        if include_features:
            expected = data[target].pos[edges[1]] - data[source].pos[edges[0]]
            torch.testing.assert_close(data[relation].edge_attr, expected)
            width = data[source].x.shape[1]
            network = torch.nn.Linear(2, width * 4)
            conv = NNConv((width, data[target].x.shape[1]), 4, network)
            output = conv((data[source].x, data[target].x), edges, data[relation].edge_attr)
            assert output.shape == (data[target].num_nodes, 4)
            assert torch.isfinite(output).all()
        else:
            assert "edge_attr" not in data[relation]
    expected_edges = {orientation for u, v in obj.uns["cell_ecm_graph"].edges for orientation in ((u, v), (v, u))}
    assert actual_edges == expected_edges
    for kind in data.node_types:
        assert data[kind].source_ids == [f"original:{n}" for n in data[kind].node_ids]
    assert data.sample_id == "roi:source"


def test_heterogeneous_export_serialization_retains_identity_and_attributes():
    data = mt.gr.to_hetero_pyg(_legacy_graph())
    buffer = io.BytesIO()
    torch.save(data, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    for kind in data.node_types:
        assert restored[kind].node_ids == data[kind].node_ids
        assert restored[kind].source_ids == data[kind].source_ids
    for relation in data.edge_types:
        torch.testing.assert_close(restored[relation].edge_index, data[relation].edge_index)
        torch.testing.assert_close(restored[relation].edge_attr, data[relation].edge_attr)


def test_missing_edge_features_are_explicit_and_can_be_disabled():
    obj = _legacy_graph()
    graph = obj.uns["cell_ecm_graph"]
    del graph.edges["c17", "e49"]["feat_rev"]
    with pytest.raises(ValueError, match="both feat_fwd and feat_rev"):
        mt.gr.to_hetero_pyg(obj)
    assert mt.gr.to_hetero_pyg(obj, include_edge_features=False).validate()
    del graph.edges["c17", "e49"]["feat_fwd"]
    with pytest.raises(ValueError, match="missing or inconsistent"):
        mt.gr.to_hetero_pyg(obj)
    # Entirely feature-free relations are legitimate.
    del graph.edges["e17", "c49"]["feat_fwd"]
    del graph.edges["e17", "c49"]["feat_rev"]
    result = mt.gr.to_hetero_pyg(obj)
    assert "edge_attr" not in result["cell", "cell-ecm", "ecm"]
    assert "edge_attr" in result["cell", "cell-cell", "cell"]


def test_old_cross_edge_convention_is_independent_of_node_insertion_order():
    obj = _legacy_graph()
    graph = obj.uns["cell_ecm_graph"].edge_subgraph([("c17", "e49")]).copy()
    attributes = graph.edges["c17", "e49"]
    del attributes["feat_source"]
    del attributes["feat_target"]
    obj.uns["cell_ecm_graph"] = graph
    assert next(iter(graph.edges))[0] == "e49"
    for data in (mt.gr.to_pyg(obj), mt.gr.to_hetero_pyg(obj)):
        if hasattr(data, "edge_types"):
            store = data["cell", "cell-ecm", "ecm"]
            expected = data["ecm"].pos[store.edge_index[1]] - data["cell"].pos[store.edge_index[0]]
        else:
            store = data
            expected = data.pos[data.edge_index[1]] - data.pos[data.edge_index[0]]
        torch.testing.assert_close(store.edge_attr, expected)


def test_public_builders_preserve_source_ids_and_directional_features():
    obj = ad.AnnData(
        X=np.ones((3, 1), dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["A", "B", "A"]}, index=["cell:49", "cell:17", "cell:88"]),
    )
    obj.obsm["spatial"] = np.array([[1.0, 1.0], [5.0, 2.0], [2.0, 6.0]])
    obj.uns["ecm_patches"] = pd.DataFrame(
        {"x": [8.0, 3.0, 6.0], "y": [1.0, 4.0, 7.0], "ecm_cluster": [1, 2, 1], "feat_0": [0.2, 0.4, 0.8]},
        index=["patch:88", "patch:17", "patch:49"],
    )

    def delta(first, second, distance):
        return second - first

    mt.gr.build_cell_graph(obj, edge_method="delaunay", edge_features=[delta])
    mt.gr.build_ecm_graph(obj, edge_method="delaunay", edge_features=[delta])
    mt.gr.build_cell_ecm_graph(obj, k=2, edge_features=[delta])
    # Reordering graph nodes must not change the physical meaning of a feature.
    source = obj.uns["cell_ecm_graph"]
    reordered = nx.Graph()
    reordered.add_nodes_from(reversed(list(source.nodes(data=True))))
    reordered.add_edges_from(source.edges(data=True))
    obj.uns["cell_ecm_graph"] = reordered
    for data in (mt.gr.to_pyg(obj), mt.gr.to_hetero_pyg(obj)):
        if hasattr(data, "edge_types"):
            assert set(data["cell"].source_ids) == set(obj.obs_names)
            assert set(data["ecm"].source_ids) == set(obj.uns["ecm_patches"].index)
            for relation in data.edge_types:
                edges = data[relation].edge_index
                expected = data[relation[2]].pos[edges[1]] - data[relation[0]].pos[edges[0]]
                torch.testing.assert_close(data[relation].edge_attr, expected)
        else:
            expected = data.pos[data.edge_index[1]] - data.pos[data.edge_index[0]]
            torch.testing.assert_close(data.edge_attr, expected)
            assert set(data.source_ids) == set(obj.obs_names) | set(obj.uns["ecm_patches"].index)


def test_observation_native_graph_keeps_explicit_label_and_identity():
    obj = ad.AnnData(
        X=np.array([[1.0], [2.0]]),
        obs=pd.DataFrame({"condition": [1, 1]}, index=["patch:49", "patch:17"]),
    )
    obj.obsp["edges"] = sp.csr_matrix([[0, 1], [1, 0]])
    data = mt.gr.to_pyg(obj, graph_key="edges", label_key="condition")
    assert int(data.y) == 1
    assert data.node_ids == data.source_ids == ["patch:49", "patch:17"]
    assert data.graph_key == "edges"
