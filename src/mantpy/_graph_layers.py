"""Sparse views of an already built cell-ECM graph and distance kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from mantpy._constants import (
    CELL_ECM_GRAPH_KEY,
    ECM_PATCHES_KEY,
    NODE_TYPE_CELL,
    NODE_TYPE_ECM,
)

DistanceKernel = Literal["uniform", "inverse", "inverse_square", "gaussian"]


@dataclass(frozen=True)
class GraphLayers:
    """The three layers of a cell-ECM graph as row-aligned sparse matrices.

    ``cell`` (cells x cells) and ``ecm`` (patches x patches) are symmetric;
    ``cell_ecm`` is cells x patches. Each ``*_distances`` matrix has the same
    sparsity pattern as its connectivity matrix and holds edge lengths, or is
    ``None`` when the graph stores no lengths. A zero-length edge is a stored
    zero. Rows follow ``cell_ids`` and ``ecm_ids``.
    """

    cell: sp.csr_matrix
    ecm: sp.csr_matrix
    cell_ecm: sp.csr_matrix
    cell_distances: sp.csr_matrix | None
    ecm_distances: sp.csr_matrix | None
    cell_ecm_distances: sp.csr_matrix | None
    cell_ids: pd.Index
    ecm_ids: pd.Index


def _pair(rows, cols, values, shape, symmetric):
    rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
    values = np.asarray(values, dtype=float)
    if symmetric:
        rows, cols = np.concatenate((rows, cols)), np.concatenate((cols, rows))
        values = np.concatenate((values, values))
    # Explicit zeros (zero-length edges) are kept: no eliminate_zeros here.
    matrix = sp.coo_matrix((values, (rows, cols)), shape=shape).tocsr()
    matrix.sort_indices()
    return matrix


def _layers_from_networkx(adata, graph, connectivity):
    cells, patches = {}, {}
    source_ids = []
    for node, attrs in graph.nodes(data=True):
        kind = attrs.get("node_type")
        if kind == NODE_TYPE_CELL:
            cells[node] = len(cells)
        elif kind == NODE_TYPE_ECM:
            patches[node] = len(patches)
            source_ids.append(attrs.get("source_id"))
        else:
            raise ValueError(f"Graph node {node!r} has no cell or ECM node_type.")
    table = adata.uns.get(ECM_PATCHES_KEY)
    if cells and len(cells) != adata.n_obs:
        raise ValueError(
            f"The graph has {len(cells)} cell nodes but adata has {adata.n_obs} cells; rebuild the graph."
        )
    if table is not None and patches and len(patches) != len(table):
        raise ValueError(
            f"The graph has {len(patches)} ECM nodes but the patch table has {len(table)} rows; rebuild the graph."
        )
    cell_ids = adata.obs_names.copy() if cells else pd.Index([], dtype=object)
    if table is not None and patches:
        ecm_ids = pd.Index(table.index)
        if all(value is not None for value in source_ids) and list(map(str, ecm_ids)) != list(map(str, source_ids)):
            raise ValueError("ECM nodes do not match the patch table rows; rebuild the graph.")
    else:
        ecm_ids = pd.Index(
            source_ids if all(value is not None for value in source_ids) else range(len(patches))
        )
    buckets = {name: ([], [], [], []) for name in ("cell", "ecm", "cell_ecm")}
    has_distance = {name: True for name in buckets}
    for u, v, attrs in graph.edges(data=True):
        if u in cells and v in cells:
            name, row, col = "cell", cells[u], cells[v]
        elif u in patches and v in patches:
            name, row, col = "ecm", patches[u], patches[v]
        else:
            name = "cell_ecm"
            row, col = (cells[u], patches[v]) if u in cells else (cells[v], patches[u])
        rows, cols, weights, lengths = buckets[name]
        rows.append(row)
        cols.append(col)
        weights.append(1.0 if connectivity == "binary" else float(attrs.get("weight", 1.0)))
        if "distance" in attrs:
            lengths.append(float(attrs["distance"]))
        else:
            has_distance[name] = False
    shapes = {
        "cell": (len(cells), len(cells)),
        "ecm": (len(patches), len(patches)),
        "cell_ecm": (len(cells), len(patches)),
    }
    out = {}
    for name, (rows, cols, weights, lengths) in buckets.items():
        symmetric = name != "cell_ecm"
        out[name] = _pair(rows, cols, weights, shapes[name], symmetric)
        out[f"{name}_distances"] = (
            _pair(rows, cols, lengths, shapes[name], symmetric) if has_distance[name] else None
        )
    return GraphLayers(cell_ids=cell_ids, ecm_ids=ecm_ids, **out)


def _layers_from_sparse_joint(adata, metadata, connectivity):
    kind = adata.obs[metadata.get("node_type_key", "node_type")].astype(str).to_numpy()
    is_cell, is_ecm = np.flatnonzero(kind == NODE_TYPE_CELL), np.flatnonzero(kind == NODE_TYPE_ECM)
    names = adata.obs["source_obs_name"] if "source_obs_name" in adata.obs else pd.Series(adata.obs_names, index=adata.obs_names)
    slices = {"cell": (is_cell, is_cell), "ecm": (is_ecm, is_ecm), "cell_ecm": (is_cell, is_ecm)}
    out = {}
    for name, (rows, cols) in slices.items():
        matrix = sp.csr_matrix(adata.obsp[metadata["connectivities"][name]])[rows][:, cols].tocsr()
        matrix.sort_indices()
        if connectivity == "binary":
            matrix = matrix.copy()
            matrix.data[:] = 1.0
        out[name] = matrix.astype(float)
        key = metadata.get("distances", {}).get(name)
        if key is None:
            out[f"{name}_distances"] = None
        else:
            stored = sp.csr_matrix(adata.obsp[key])[rows][:, cols]
            row, col = matrix.nonzero()
            out[f"{name}_distances"] = _pair(
                row, col, np.asarray(stored[row, col]).ravel(), matrix.shape, False,
            )
    return GraphLayers(
        cell_ids=pd.Index(names.to_numpy()[is_cell]), ecm_ids=pd.Index(names.to_numpy()[is_ecm]), **out,
    )


def graph_layers(
    adata: AnnData,
    *,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    connectivity: Literal["binary", "weight"] = "binary",
) -> GraphLayers:
    """Return the layers of a built cell-ECM graph as sparse matrices.

    Reads the graph stored by :func:`mantpy.gr.ensure_cell_ecm_graph` /
    :func:`mantpy.gr.build_cell_ecm_graphs` (or a sparse joint graph from
    :func:`mantpy.gr.joint_graph_from_connectivities`) without rebuilding any
    edge. Cell rows follow ``adata.obs`` and patch columns follow
    ``adata.uns['ecm_patches']``, background patches included.

    Parameters
    ----------
    adata
        AnnData holding the graph in ``uns[graph_key]``.
    graph_key
        Key of the graph.
    connectivity
        ``"binary"`` stores 1.0 for every edge; ``"weight"`` stores the edge
        weight chosen when the graph was built.

    Returns
    -------
    :class:`GraphLayers` with connectivity and distance matrices. Combine a
    distance matrix with :func:`mantpy.gr.distance_weights` to weight edges
    for :func:`mantpy.tl.neighbourhood_features`.
    """
    if connectivity not in {"binary", "weight"}:
        raise ValueError("connectivity must be 'binary' or 'weight'.")
    if graph_key not in adata.uns:
        raise KeyError(f"adata.uns[{graph_key!r}] is missing; build the cell-ECM graph first.")
    graph = adata.uns[graph_key]
    if isinstance(graph, dict):
        if graph.get("format") != "mantpy_sparse_joint_v1":
            raise ValueError(f"adata.uns[{graph_key!r}] is not a recognised Mantpy graph.")
        return _layers_from_sparse_joint(adata, graph, connectivity)
    return _layers_from_networkx(adata, graph, connectivity)


def distance_weights(
    distances,
    *,
    kernel: DistanceKernel = "uniform",
    sigma: float | Literal["median"] = "median",
    min_distance: float | None = None,
):
    """Turn edge lengths into edge weights with a distance kernel.

    Parameters
    ----------
    distances
        Sparse matrix of edge lengths (every stored entry is an edge, stored
        zeros included) or a one-dimensional array of lengths.
    kernel
        ``"uniform"`` gives 1; ``"inverse"`` gives 1/d; ``"inverse_square"``
        gives 1/d^2; ``"gaussian"`` gives exp(-d^2 / (2 sigma^2)).
    sigma
        Gaussian width. ``"median"`` uses the median of the stored lengths of
        this matrix, so call the function once per layer and per sample.
    min_distance
        Lengths below this value are raised to it before the kernel and the
        median are evaluated. Required for an inverse kernel when a length
        is zero.

    Returns
    -------
    Float64 weights with the same sparsity pattern (CSR) or shape (array) as
    the input. The input is not modified.
    """
    sparse = sp.issparse(distances)
    if sparse:
        result = sp.csr_matrix(distances, dtype=np.float64, copy=True)
        d = result.data
    else:
        d = np.array(distances, dtype=np.float64)
        if d.ndim != 1:
            raise ValueError("distances must be a sparse matrix or a one-dimensional array.")
    if d.size and (not np.isfinite(d).all() or (d < 0).any()):
        raise ValueError("distances must be finite and non-negative.")
    if min_distance is not None:
        if not np.isfinite(min_distance) or min_distance < 0:
            raise ValueError("min_distance must be finite and non-negative.")
        d = np.maximum(d, min_distance)
    if kernel == "uniform":
        w = np.ones_like(d)
    elif kernel in {"inverse", "inverse_square"}:
        if d.size and (d == 0).any():
            raise ValueError("A zero length has no inverse weight; pass min_distance.")
        w = 1.0 / d if kernel == "inverse" else 1.0 / (d ** 2)
    elif kernel == "gaussian":
        if d.size:
            width = float(np.median(d)) if isinstance(sigma, str) and sigma == "median" else float(sigma)
            if not np.isfinite(width) or width <= 0:
                raise ValueError("sigma must be positive; pass a number when the median length is zero.")
            w = np.exp(-(d ** 2) / (2.0 * width ** 2))
        else:
            w = d.copy()
    else:
        raise ValueError("kernel must be 'uniform', 'inverse', 'inverse_square' or 'gaussian'.")
    if not sparse:
        return w
    result.data = w
    return result
