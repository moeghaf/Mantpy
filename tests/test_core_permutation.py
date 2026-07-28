"""Focused tests for deterministic permutation streams."""

from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pandas as pd

from mantpy._constants import NODE_TYPE_CELL, NODE_TYPE_ECM
from mantpy._core._permutation import _one_permutation, count_interactions, permutation_test


def _interaction_graph() -> nx.Graph:
    graph = nx.Graph()
    for index, cell_type in enumerate(("A", "A", "B", "B")):
        graph.add_node(f"cell-{index}", node_type=NODE_TYPE_CELL, cell_type=cell_type)
    for index, cluster in enumerate((0, 0, 1, 1, 2, 2)):
        graph.add_node(f"ecm-{index}", node_type=NODE_TYPE_ECM, ecm_cluster=cluster)
    graph.add_edges_from(
        [
            ("cell-0", "ecm-0"),
            ("cell-0", "ecm-1"),
            ("cell-1", "ecm-2"),
            ("cell-1", "ecm-4"),
            ("cell-2", "ecm-3"),
            ("cell-2", "ecm-5"),
            ("cell-3", "ecm-0"),
            ("cell-3", "ecm-5"),
        ]
    )
    return graph


def test_seeded_permutation_is_independent_of_global_numpy_state():
    graph = _interaction_graph()
    observed = count_interactions(graph)
    ecm_nodes = [node for node, data in graph.nodes(data=True) if data["node_type"] == NODE_TYPE_ECM]
    ecm_clusters = np.array([graph.nodes[node]["ecm_cluster"] for node in ecm_nodes])

    np.random.seed(1)
    first = _one_permutation(
        graph,
        ecm_nodes,
        ecm_clusters,
        observed.index,
        observed.columns,
        np.random.SeedSequence(42),
    )
    np.random.seed(999)
    second = _one_permutation(
        graph,
        ecm_nodes,
        ecm_clusters,
        observed.index,
        observed.columns,
        np.random.SeedSequence(42),
    )

    np.testing.assert_array_equal(first, second)
    assert all("_perm_cluster" not in graph.nodes[node] for node in ecm_nodes)


def test_seeded_permutation_test_is_independent_of_job_count():
    graph = _interaction_graph()

    serial = permutation_test(graph, n_iter=32, p=0.2, n_jobs=1, seed=7)
    parallel = permutation_test(graph, n_iter=32, p=0.2, n_jobs=2, seed=7)

    pd.testing.assert_frame_equal(serial, parallel)


def test_one_permutation_requires_an_explicit_seed_sequence():
    parameter = inspect.signature(_one_permutation).parameters["seed_seq"]

    assert parameter.default is inspect.Parameter.empty
