"""Representation adapters that preserve an already selected graph topology."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


def _matrix(value, shape, name):
    if value is None:
        return sp.csr_matrix(shape, dtype=np.float32)
    result = sp.csr_matrix(value, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {result.shape}.")
    result.sum_duplicates()
    result.sort_indices()
    if not np.isfinite(result.data).all() or (result.data < 0).any():
        raise ValueError(f"{name} must contain finite, non-negative values.")
    return result


def joint_graph_from_connectivities(
    cells: AnnData,
    ecm: AnnData,
    *,
    cell_ecm_connectivities,
    cell_connectivities=None,
    ecm_connectivities=None,
    cell_distances=None,
    ecm_distances=None,
    cell_ecm_distances=None,
    spatial_key: str = "spatial",
    graph_key: str = "cell_ecm_graph",
) -> AnnData:
    """Assemble a typed sparse joint graph from existing connectivity matrices.

    Cell, ECM and cross matrices have shapes C×C, E×E and C×E. Cross
    connections are undirected; within-compartment direction and weights are
    preserved. Distances are optional and zero distances remain valid edges,
    with edge presence determined exclusively by connectivities. Rows retain
    input order, with cells followed by ECM patches and original IDs recorded
    in ``obs['source_obs_name']``. Feature spaces are not padded or combined.
    """
    from mantpy.gr import _validated_spatial

    for label, data in (("cells", cells), ("ecm", ecm)):
        if not data.obs_names.is_unique:
            raise ValueError(f"{label}.obs_names must be unique.")
    nc, ne = cells.n_obs, ecm.n_obs
    cc = _matrix(cell_connectivities, (nc, nc), "cell_connectivities")
    ee = _matrix(ecm_connectivities, (ne, ne), "ecm_connectivities")
    ce = _matrix(cell_ecm_connectivities, (nc, ne), "cell_ecm_connectivities")
    cc.eliminate_zeros()
    ee.eliminate_zeros()
    ce.eliminate_zeros()
    cell_obs, ecm_obs = cells.obs.copy(), ecm.obs.copy()
    source_names = np.concatenate((cells.obs_names.astype(str), ecm.obs_names.astype(str)))
    cell_obs.index = pd.Index([f"cell:{name}" for name in cells.obs_names])
    ecm_obs.index = pd.Index([f"ecm:{name}" for name in ecm.obs_names])
    obs = pd.concat((cell_obs, ecm_obs), sort=False)
    for column in obs.columns:
        if isinstance(obs[column].dtype, pd.StringDtype):
            obs[column] = obs[column].astype(object)
    obs["source_obs_name"] = pd.Series(source_names, index=obs.index, dtype=object)
    obs["node_type"] = pd.Categorical(["cell"] * nc + ["ecm"] * ne, categories=["cell", "ecm"])
    result = AnnData(X=sp.csr_matrix((nc + ne, 0), dtype=np.float32), obs=obs)
    result.obsm[spatial_key] = np.vstack(
        (_validated_spatial(cells, spatial_key, "cells"), _validated_spatial(ecm, spatial_key, "ecm"))
    )
    if "spatial_um" in cells.obsm and "spatial_um" in ecm.obsm and spatial_key != "spatial_um":
        result.obsm["spatial_um"] = np.vstack((cells.obsm["spatial_um"], ecm.obsm["spatial_um"]))
    zcc, zee = sp.csr_matrix((nc, nc)), sp.csr_matrix((ne, ne))
    full = {
        "cell": sp.block_diag((cc, zee), format="csr"),
        "ecm": sp.block_diag((zcc, ee), format="csr"),
        "cell_ecm": sp.bmat([[zcc, ce], [ce.T, zee]], format="csr"),
    }
    full["joint"] = (full["cell"] + full["ecm"] + full["cell_ecm"]).tocsr()
    for name, matrix in full.items():
        result.obsp[f"{name}_connectivities"] = matrix
    distance_names = {}
    supplied = (
        ("cell", cell_distances, (nc, nc), cc),
        ("ecm", ecm_distances, (ne, ne), ee),
        ("cell_ecm", cell_ecm_distances, (nc, ne), ce),
    )
    for name, value, shape, adjacency in supplied:
        if value is None:
            continue
        distance = _matrix(value, shape, f"{name}_distances")
        # Reindex onto the adjacency structure; an unstored distance is zero.
        row, col = adjacency.nonzero()
        distance = sp.csr_matrix((np.asarray(distance[row, col]).ravel(), (row, col)), shape=shape)
        if name == "cell":
            layer = sp.block_diag((distance, zee), format="csr")
        elif name == "ecm":
            layer = sp.block_diag((zcc, distance), format="csr")
        else:
            layer = sp.bmat([[zcc, distance], [distance.T, zee]], format="csr")
        key = f"{name}_distances"
        result.obsp[key] = layer
        distance_names[name] = key
    result.uns[graph_key] = {
        "format": "mantpy_sparse_joint_v1",
        "node_type_key": "node_type",
        "spatial_key": spatial_key,
        "spatial_um_key": "spatial_um" if "spatial_um" in result.obsm else spatial_key,
        "connectivities": {name: f"{name}_connectivities" for name in full},
        "distances": distance_names,
        "counts": {"n_nodes": nc + ne, "n_cells": nc, "n_ecm": ne, "cell_ecm_edges": int(ce.nnz)},
        "params": {"topology": "supplied"},
    }
    return result


def to_networkx(
    adata: AnnData,
    *,
    graph_key: str = "joint_connectivities",
    legacy_ids: bool = False,
    directed: bool | None = None,
    spatial_key: str = "spatial",
) -> nx.Graph:
    """Export sparse connections and observation attributes without rebuilding edges.

    ``directed=None`` detects asymmetric weights. ``legacy_ids=True`` assigns
    ``cell_i`` / ``ecm_i`` compartment-local IDs for legacy Mantpy consumers;
    the observation ID and source ID remain node attributes.
    """
    if graph_key not in adata.obsp:
        metadata = adata.uns.get(graph_key, {})
        if not isinstance(metadata, dict):
            raise KeyError(f"obsp[{graph_key!r}] is missing.")
        spatial_key = metadata.get("spatial_key", spatial_key)
        graph_key = metadata.get("connectivities", {}).get("joint", graph_key)
    if graph_key not in adata.obsp:
        raise KeyError(f"obsp[{graph_key!r}] is missing.")
    adjacency = sp.csr_matrix(adata.obsp[graph_key])
    if directed is None:
        directed = bool((adjacency != adjacency.T).nnz)
    graph = nx.DiGraph() if directed else nx.Graph()
    ids = []
    counters = {}
    for i, (name, row) in enumerate(adata.obs.iterrows()):
        attrs = row.to_dict()
        role = str(attrs.get("node_type", "node"))
        index = counters.get(role, 0)
        counters[role] = index + 1
        node = f"{role}_{index}" if legacy_ids else name
        attrs["obs_name"] = name
        if spatial_key in adata.obsm:
            attrs["x"], attrs["y"] = map(float, adata.obsm[spatial_key][i, :2])
        graph.add_node(node, **attrs)
        ids.append(node)
    distance_key = graph_key.replace("connectivities", "distances")
    distances = adata.obsp.get(distance_key) if distance_key != graph_key else None
    matrix = adjacency if directed else sp.triu(adjacency, format="csr")
    for row, col, weight in zip(matrix.tocoo().row, matrix.tocoo().col, matrix.tocoo().data, strict=True):
        if weight == 0:
            continue
        attrs = {"weight": float(weight)}
        if distances is not None:
            attrs["distance"] = float(distances[row, col])
        graph.add_edge(ids[row], ids[col], **attrs)
    return graph


def extract_components_existing(
    adata: AnnData,
    *,
    target_cluster,
    cluster_attr: str = "ecm_cluster",
    pos_attrs: tuple[str, str] = ("x", "y"),
    min_nodes: int = 1,
    graph_key: str = "ecm_graph",
) -> list[dict]:
    """Extract cluster-induced components of the supplied graph.

    Accepts a legacy NetworkX graph in ``uns`` or sparse connections in
    ``obsp``. Directed graphs use weak components. No new edges are added.
    Every fragment is returned; ``kept`` records the minimum-size decision.
    """
    if not isinstance(min_nodes, (int, np.integer)) or isinstance(min_nodes, (bool, np.bool_)) or min_nodes < 1:
        raise ValueError("min_nodes must be a positive integer.")
    stored = adata.uns.get(graph_key)
    graph = stored if isinstance(stored, nx.Graph) else to_networkx(adata, graph_key=graph_key)
    x, y = pos_attrs
    selected = []
    for node, attrs in graph.nodes(data=True):
        if attrs.get(cluster_attr) != target_cluster:
            continue
        if x not in attrs or y not in attrs:
            raise KeyError(f"Node {node!r} is missing coordinates {pos_attrs}.")
        if not np.isfinite([attrs[x], attrs[y]]).all():
            raise ValueError(f"Node {node!r} has non-finite coordinates.")
        selected.append(node)
    induced = graph.subgraph(selected)
    components = nx.weakly_connected_components(induced) if graph.is_directed() else nx.connected_components(induced)
    order = {node: i for i, node in enumerate(selected)}
    records = []
    for component in components:
        ids = sorted(component, key=order.__getitem__)
        subgraph = induced.subgraph(ids).copy()
        coords = np.array([[graph.nodes[node][x], graph.nodes[node][y]] for node in ids], dtype=float)
        records.append(
            {
                "node_ids": ids,
                "coords": coords,
                "subgraph": subgraph,
                "n_nodes": len(ids),
                "n_edges": subgraph.number_of_edges(),
                "centroid": coords.mean(axis=0),
                "kept": len(ids) >= min_nodes,
            }
        )
    return records
