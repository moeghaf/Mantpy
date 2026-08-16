"""Graph construction functions for Mantpy."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from mantpy._constants import (
    CELL_ECM_GRAPH_KEY,
    CELL_GRAPH_KEY,
    CELLTYPE_COL,
    ECM_GRAPH_KEY,
    ECM_PATCHES_KEY,
    NODE_TYPE_CELL,
    NODE_TYPE_ECM,
    SPATIAL_KEY,
)
from mantpy._constants import (
    SQUIDPY_CONN_KEY as _SQUIDPY_CONN_KEY,
)
from mantpy._core._edges import (
    build_cell_nx_graph,
    build_ecm_nx_graph,
    delaunay_edges,
    grid_edges,
    knn_edges,
    merge_cell_ecm_graphs,
    radius_edges,
)
from mantpy._utils import log_params as _log_params

_log = logging.getLogger(__name__)

__all__ = [
    "EDGE_FEATURE_REGISTRY",
    "exp_neg_dist",
    "build_cell_ecm_graph",
    "build_cell_ecm_graphs",
    "build_cell_graph",
    "build_ecm_graph",
    "build_ecm_graphs",
    "build_graph",
    "build_patch_graph",
    "compose_cell_ecm_graph",
    "ensure_cell_ecm_graph",
    "extract_components_radius_reconnect",
    "JointGraphSummary",
    "GraphBuildResult",
    "ECMGraphBuildResult",
    "PatchGraphResult",
    "joint_graph_summary",
    "largest_component_radius_reconnect",
    "to_hetero_pyg",
    "to_pyg",
]

EdgeWeight = Literal["inv_distance", "distance", "uniform"] | Callable[[int, int, float], float]
EdgeFeatureFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]

#: Built-in edge feature extractors.
#: Each callable: ``(pos_i: [x,y], pos_j: [x,y], distance: float) -> 1-D float32``.
#:
#: Register custom extractors the same way::
#:
#:     from mantpy.gr import EDGE_FEATURE_REGISTRY
#:     EDGE_FEATURE_REGISTRY["sq_dist"] = lambda pi, pj, d: np.array([d**2], dtype=np.float32)
_EPS = 1e-8

EDGE_FEATURE_REGISTRY: dict[str, EdgeFeatureFn] = {
    "log_distance": lambda pi, pj, d: np.array([np.log1p(d)], dtype=np.float32),
    "norm_angle": lambda pi, pj, d: np.array(
        [(np.arctan2(pj[1] - pi[1], pj[0] - pi[0]) + np.pi) / (2.0 * np.pi)],
        dtype=np.float32,
    ),
    "distance": lambda pi, pj, d: np.array([float(d)], dtype=np.float32),
    # Direction-sensitive components (unit-vector-like; encoder sees anisotropy).
    "dx_norm": lambda pi, pj, d: np.array([(pj[0] - pi[0]) / (d + _EPS)], dtype=np.float32),
    "dy_norm": lambda pi, pj, d: np.array([(pj[1] - pi[1]) / (d + _EPS)], dtype=np.float32),
    # Periodic-safe angle encoding (avoids the 0/1 wrap of norm_angle).
    "sin_angle": lambda pi, pj, d: np.array([np.sin(np.arctan2(pj[1] - pi[1], pj[0] - pi[0]))], dtype=np.float32),
    "cos_angle": lambda pi, pj, d: np.array([np.cos(np.arctan2(pj[1] - pi[1], pj[0] - pi[0]))], dtype=np.float32),
    # Doubled-angle (orientation) encoding — invariant to edge direction (θ vs θ+π), so it encodes
    # the undirected orientation of a patch-patch link (used for the geometric ECM edge features).
    "sin_angle2": lambda pi, pj, d: np.array([np.sin(2 * np.arctan2(pj[1] - pi[1], pj[0] - pi[0]))], dtype=np.float32),
    "cos_angle2": lambda pi, pj, d: np.array([np.cos(2 * np.arctan2(pj[1] - pi[1], pj[0] - pi[0]))], dtype=np.float32),
    # Attention-style proximity kernel (the "gravity" form used in GAT edge weighting).
    "inv_distance": lambda pi, pj, d: np.array([1.0 / (d + _EPS)], dtype=np.float32),
    # Axis-aligned distance (sometimes preferred for grid/tile imaging data).
    "manhattan_distance": lambda pi, pj, d: np.array(
        [float(abs(pj[0] - pi[0]) + abs(pj[1] - pi[1]))], dtype=np.float32
    ),
}


def exp_neg_dist(sigma: float = 100.0) -> EdgeFeatureFn:
    """Factory for a Gaussian-kernel proximity edge feature.

    Returns a callable ``(pos_i, pos_j, distance) -> (1,) float32`` that emits
    ``exp(-distance / sigma)``.  Use when you want an attention-like edge
    weight with a configurable length scale:

        >>> from mantpy.gr import exp_neg_dist, EDGE_FEATURE_REGISTRY
        >>> EDGE_FEATURE_REGISTRY["exp_neg_100"] = exp_neg_dist(sigma=100.0)
        >>> build_ecm_graph(adata, edge_features=["exp_neg_100"])

    Parameters
    ----------
    sigma
        Length scale in pixels.  Smaller → sharper falloff.
    """
    sigma = float(sigma)

    def _fn(pi: np.ndarray, pj: np.ndarray, d: float) -> np.ndarray:
        return np.array([np.exp(-float(d) / sigma)], dtype=np.float32)

    _fn.__name__ = f"exp_neg_dist_sigma{sigma:g}"
    return _fn


def _edge_continuity(ys, xs, r, c, foreground, n_samples=6):
    """Fraction of samples along each edge segment that fall on foreground (tissue)."""
    h, w = foreground.shape
    t = np.linspace(0, 1, n_samples)[None, :]
    yy = np.clip(np.round(ys[r][:, None] + t * (ys[c] - ys[r])[:, None]).astype(int), 0, h - 1)
    xx = np.clip(np.round(xs[r][:, None] + t * (xs[c] - xs[r])[:, None]).astype(int), 0, w - 1)
    return foreground[yy, xx].mean(1)


def _edge_feature_matrix(pts, r, c, names):
    dx = pts[c, 0] - pts[r, 0]
    dy = pts[c, 1] - pts[r, 1]
    d = np.sqrt(dx * dx + dy * dy)
    th = np.arctan2(dy, dx)
    vec = {
        "log_distance": np.log1p(d),
        "distance": d,
        "sin_angle": np.sin(th),
        "cos_angle": np.cos(th),
        "sin_angle2": np.sin(2 * th),
        "cos_angle2": np.cos(2 * th),
        "dx_norm": dx / (d + _EPS),
        "dy_norm": dy / (d + _EPS),
        "inv_distance": 1.0 / (d + _EPS),
    }
    try:
        cols = [vec[nm] for nm in names]
    except KeyError as err:
        raise ValueError(f"unknown edge feature {err.args[0]!r}; build_patch_graph supports {sorted(vec)}.") from None
    return np.column_stack(cols).astype(np.float32)


class PatchGraphResult(dict[str, Any]):
    """Dict-compatible result from :func:`build_patch_graph`.

    The mapping deliberately contains only the four legacy keys. Lightweight
    graph metadata and shape summaries are exposed as properties so existing
    code that iterates, unpacks, or serialises the mapping is unaffected.
    """

    __slots__ = ("_connectivities_key", "_graph_key", "_n_nodes", "_params")

    def __init__(
        self,
        *,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        n_edges: int,
        edge_feature_names: Sequence[str],
        n_nodes: int,
        graph_key: str | None,
        connectivities_key: str | None,
        params: Mapping[str, Any],
    ) -> None:
        super().__init__(
            edge_index=edge_index,
            edge_attr=edge_attr,
            n_edges=int(n_edges),
            edge_feature_names=tuple(edge_feature_names),
        )
        self._n_nodes = int(n_nodes)
        self._graph_key = graph_key
        self._connectivities_key = connectivities_key
        self._params = dict(params)

    @property
    def edge_index(self) -> np.ndarray:
        """Directed edge indices with shape ``(2, n_directed_edges)``."""
        return self["edge_index"]

    @property
    def edge_attr(self) -> np.ndarray:
        """Directed edge features aligned to :attr:`edge_index`."""
        return self["edge_attr"]

    @property
    def n_edges(self) -> int:
        """Number of undirected edges."""
        return self["n_edges"]

    @property
    def edge_feature_names(self) -> tuple[str, ...]:
        """Edge-feature names in matrix-column order."""
        return self["edge_feature_names"]

    @property
    def n_nodes(self) -> int:
        """Number of graph nodes."""
        return self._n_nodes

    @property
    def n_directed_edges(self) -> int:
        """Number of directed edges, including both orientations."""
        return int(self.edge_index.shape[1])

    @property
    def n_edge_features(self) -> int:
        """Number of columns in :attr:`edge_attr`."""
        return int(self.edge_attr.shape[1])

    @property
    def graph_key(self) -> str | None:
        """AnnData ``uns`` key, or ``None`` for array input."""
        return self._graph_key

    @property
    def connectivities_key(self) -> str | None:
        """AnnData ``obsp`` key, or ``None`` for array input."""
        return self._connectivities_key

    @property
    def params(self) -> dict[str, Any]:
        """Parameters used to construct the graph."""
        return self._params

    def __repr__(self) -> str:
        graph = f", graph_key={self.graph_key!r}" if self.graph_key is not None else ""
        return (
            f"PatchGraphResult(n_nodes={self.n_nodes}, n_edges={self.n_edges}, "
            f"n_directed_edges={self.n_directed_edges}, n_edge_features={self.n_edge_features}{graph})"
        )


def build_patch_graph(
    coords: np.ndarray | AnnData,
    *,
    k: int = 40,
    pixel_coords: tuple[np.ndarray, np.ndarray] | None = None,
    foreground: np.ndarray | None = None,
    continuity_thresh: float = 0.5,
    continuity_samples: int = 6,
    edge_features: Sequence[str] = ("log_distance", "sin_angle2", "cos_angle2"),
    standardize_edges: bool = True,
    key_added: str = "patch",
) -> PatchGraphResult:
    """Spatial k-NN graph over image-patch centroids, pruning edges that cross background.

    Builds an undirected k-nearest-neighbour graph on patch centroids and, when a tissue
    ``foreground`` mask is given, drops edges whose straight segment runs mostly over
    background — keeping the graph on-tissue across air spaces or gaps. This is the spatial
    graph for a patch-level :class:`mantpy.nn.GraphMAE`: it returns a directed (both
    orientations) ``edge_index`` and matching geometric ``edge_attr`` ready to wrap in a PyG
    :class:`~torch_geometric.data.Data`.

    Parameters
    ----------
    coords
        Patch centroid coordinates, shape ``(n_patches, 2)``, or observation-
        native patch AnnData. For AnnData, coordinates are read from
        ``obsm['spatial']``; ``obs['grid_y']``/``obs['grid_x']`` and
        ``uns['image_ecm_patches']['patch_foreground']`` supply continuity
        pruning when explicit values are not passed.
    k
        Number of nearest neighbours.
    pixel_coords
        ``(ys, xs)`` integer pixel coordinates of each patch (each shape ``(n_patches,)``),
        indexing into ``foreground``. Required when ``foreground`` is given.
    foreground
        Boolean tissue mask, shape ``(H, W)``. Edges whose sampled segment is foreground in a
        fraction below ``continuity_thresh`` are pruned.
    continuity_thresh, continuity_samples
        Fraction-foreground cutoff and the number of samples along each edge segment.
    edge_features
        Edge-feature names; the geometric default ``("log_distance", "sin_angle2", "cos_angle2")``
        encodes link length and undirected orientation. The special name ``"continuity"`` appends
        each edge's fraction-on-tissue (requires ``foreground``); the 4-feature set
        ``("log_distance", "sin_angle2", "cos_angle2", "continuity")`` is the ``geom_cont`` variant.
    standardize_edges
        Z-score each edge-feature column across edges (matches the scale the encoder expects).
    key_added
        For AnnData input, store the sparse adjacency in
        ``obsp[f'{key_added}_connectivities']`` and the PyG-ready edge arrays
        and parameters in ``uns[key_added]``. Ignored for array input.

    Returns
    -------
    PatchGraphResult
        ``edge_index`` (``(2, n_directed)`` int64, both orientations), ``edge_attr``
        (``(n_directed, n_edge_features)`` float32), ``n_edges`` (undirected edge count),
        and ``edge_feature_names``. The result remains dict-compatible; graph
        metadata and concise shape summaries are available as properties.

    See Also
    --------
    build_ecm_graph : AnnData-based graph builder writing a NetworkX graph.
    mantpy.nn.GraphMAE : consumes the returned edges as a PyG ``Data``.
    """
    from sklearn.neighbors import NearestNeighbors

    adata = coords if isinstance(coords, AnnData) else None
    if adata is not None:
        if SPATIAL_KEY not in adata.obsm:
            raise KeyError(f"AnnData has no obsm[{SPATIAL_KEY!r}].")
        pts = np.asarray(adata.obsm[SPATIAL_KEY], dtype=float)
        if foreground is None:
            metadata = adata.uns.get("image_ecm_patches", {})
            if isinstance(metadata, Mapping) and "patch_foreground" in metadata:
                foreground = np.asarray(metadata["patch_foreground"], dtype=bool)
                if pixel_coords is None and {"grid_y", "grid_x"}.issubset(adata.obs.columns):
                    pixel_coords = (
                        adata.obs["grid_y"].to_numpy(dtype=int),
                        adata.obs["grid_x"].to_numpy(dtype=int),
                    )
    else:
        pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"coords must be (n_patches, 2); got shape {pts.shape}.")
    n = pts.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k={k!r}: expected 1 <= k < n_patches ({n}).")
    idx = NearestNeighbors(n_neighbors=k + 1).fit(pts).kneighbors(pts, return_distance=False)
    r = np.repeat(np.arange(n), k)
    c = idx[:, 1 : k + 1].ravel()
    # A kNN relation is directed: a high-index node may select a low-index
    # neighbour even when the reverse selection is absent. Canonicalise every
    # candidate pair before deduplicating so symmetrisation is invariant to
    # AnnData row order rather than silently dropping those one-way links.
    pairs = np.sort(np.column_stack([r, c]), axis=1)
    pairs = np.unique(pairs, axis=0)
    r, c = pairs[:, 0], pairs[:, 1]
    fg = ys = xs = None
    if foreground is not None:
        if pixel_coords is None:
            raise ValueError("pixel_coords=(ys, xs) is required when foreground is given.")
        ys, xs = (np.asarray(a) for a in pixel_coords)
        fg = np.asarray(foreground, dtype=bool)
        keep = _edge_continuity(ys, xs, r, c, fg, continuity_samples) >= continuity_thresh
        r, c = r[keep], c[keep]
    # "continuity" (per-edge fraction of the segment on tissue) is not a geometry-only feature, so
    # it is assembled here from the foreground mask rather than in _edge_feature_matrix.
    names = list(edge_features)
    if "continuity" in names:
        if fg is None:
            raise ValueError(
                "edge_features includes 'continuity' but no foreground mask was given; "
                "pass foreground=... and pixel_coords=(ys, xs)."
            )
        geom_names = [nm for nm in names if nm != "continuity"]
        geom = _edge_feature_matrix(pts, r, c, geom_names) if geom_names else np.empty((r.size, 0), dtype=np.float32)
        col = {nm: geom[:, i] for i, nm in enumerate(geom_names)}
        col["continuity"] = _edge_continuity(ys, xs, r, c, fg, continuity_samples).astype(np.float32)
        feats = np.column_stack([col[nm] for nm in names]).astype(np.float32)
    else:
        feats = _edge_feature_matrix(pts, r, c, names)
    if standardize_edges:
        feats = (feats - feats.mean(0)) / (feats.std(0) + _EPS)
    edge_index = np.hstack([np.vstack([r, c]), np.vstack([c, r])]).astype(np.int64)
    edge_attr = np.vstack([feats, feats]).astype(np.float32)
    params = {
        "k": int(k),
        "continuity_thresh": float(continuity_thresh),
        "continuity_samples": int(continuity_samples),
        "standardize_edges": bool(standardize_edges),
    }
    connectivities_key = f"{key_added}_connectivities" if adata is not None else None
    result = PatchGraphResult(
        edge_index=edge_index,
        edge_attr=edge_attr,
        n_edges=int(r.size),
        edge_feature_names=names,
        n_nodes=n,
        graph_key=key_added if adata is not None else None,
        connectivities_key=connectivities_key,
        params=params,
    )
    if adata is not None:
        connectivities = sp.csr_matrix(
            (np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])),
            shape=(n, n),
        )
        connectivities.sum_duplicates()
        connectivities.data[:] = 1.0
        adata.obsp[connectivities_key] = connectivities
        adata.uns[key_added] = {
            "connectivities_key": connectivities_key,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_feature_names": np.asarray(names, dtype=str),
            "params": params.copy(),
        }
    return result


def _require_dmax_knn_only(edge_method: str, dmax: float | None, name: str) -> None:
    """Guard: distance caps are a kNN-only knob.

    Delaunay and grid topologies are parameter-free by construction, so a
    non-``None`` distance cap with any non-``knn`` method is a user error rather
    than a silently-ignored argument.
    """
    if dmax is not None and edge_method != "knn":
        raise ValueError(
            f"{name}={dmax!r}: a distance cap is only used by edge_method='knn', "
            f"got edge_method={edge_method!r}. The delaunay, grid and radius "
            f"topologies do not take a distance cap — leave {name}=None (the default)."
        )


def build_graph(
    adata: AnnData,
    *,
    mode: Literal["cell", "ecm", "cell_ecm"] = "cell_ecm",
    edge_method: str | Mapping[str, str] | None = None,
    inplace: bool = True,
    **kwargs: Any,
) -> AnnData | None:
    """Build a compartment-specific spatial graph (unified entry point).

    Thin dispatcher over the typed builders that constructs one of the three
    graph layers from the cell--ECM model, selected by ``mode``:

    - ``"cell"``      → the cell--cell graph (:func:`build_cell_graph`),
    - ``"ecm"``       → the ECM--ECM graph (:func:`build_ecm_graph`),
    - ``"cell_ecm"``  → the bipartite cell--ECM graph
      (:func:`build_cell_ecm_graph`).

    All builder-specific keyword arguments (``k``, ``r``, ``Dmax*``,
    ``edge_weight``, ``edge_features``, ``grid_connectivity``,
    ``node_features``, ``key_added`` …) are forwarded through ``**kwargs`` to
    the selected builder, so this call is equivalent to invoking that builder
    directly. To build all three layers at once (each with its own rule), use
    :func:`ensure_cell_ecm_graph`.

    Parameters
    ----------
    adata
        Annotated data matrix with spatial coordinates in ``obsm`` and, for
        ``"cell"``/``"cell_ecm"`` modes, cell-type/ECM-patch node tables.
    mode
        Which graph layer to construct: ``"cell"``, ``"ecm"`` or
        ``"cell_ecm"``. Defaults to ``"cell_ecm"``. Note that ``"cell_ecm"``
        *merges* the cell and ECM layers, so both must already exist (build
        them first with ``mode="cell"`` and ``mode="ecm"``, or in one step
        via :func:`ensure_cell_ecm_graph`).
    edge_method
        Edge-construction rule. A string (e.g. ``"knn"``, ``"delaunay"``,
        ``"radius"``, ``"grid"``) is passed straight to the selected builder.
        A mapping keyed by ``mode`` (e.g.
        ``{"cell": "knn", "ecm": "delaunay"}``) lets a single per-interaction
        configuration be reused across calls — only the entry for ``mode`` is
        applied. ``None`` (default) keeps the selected builder's own default
        rule.
    inplace
        If ``True`` (default), write the graph into ``adata`` and return
        ``None``; if ``False``, return a modified copy.
    **kwargs
        Forwarded verbatim to the selected builder.

    Returns
    -------
    AnnData | None
        ``None`` when ``inplace=True``; otherwise the modified :class:`~anndata.AnnData`.

    See Also
    --------
    build_cell_graph, build_ecm_graph, build_cell_ecm_graph
        The per-compartment builders this dispatches to.
    ensure_cell_ecm_graph
        Build the cell, ECM and bipartite layers together in one call.

    Examples
    --------
    >>> import mantpy as mt  # doctest: +SKIP
    >>> mt.gr.build_graph(adata, mode="ecm", edge_method="delaunay")  # doctest: +SKIP
    """
    builders = {
        "cell": build_cell_graph,
        "ecm": build_ecm_graph,
        "cell_ecm": build_cell_ecm_graph,
    }
    if mode not in builders:
        raise ValueError(f"mode must be one of {sorted(builders)!r}, got {mode!r}.")
    if edge_method is not None:
        resolved = edge_method.get(mode) if isinstance(edge_method, Mapping) else edge_method
        if resolved is not None:
            kwargs["edge_method"] = resolved
    return builders[mode](adata, inplace=inplace, **kwargs)


def build_cell_graph(
    adata: AnnData,
    *,
    Dmax_CC: float | None = None,
    k: int = 5,
    r: float | None = None,
    edge_method: Literal["knn", "delaunay", "radius"] = "knn",
    edge_weight: EdgeWeight = "inv_distance",
    edge_features: list[str | Callable] | None = None,
    node_features: list[str] | None = None,
    connectivity_key: str | None = None,
    key_added: str = CELL_GRAPH_KEY,
    inplace: bool = True,
) -> AnnData | None:
    """Build a cell-cell spatial proximity graph.

    Parameters
    ----------
    adata
        AnnData with ``adata.obsm['spatial']``.
    connectivity_key
        Reuse an adjacency that already exists in ``adata.obsp`` instead of
        building one — e.g. ``"spatial_connectivities"`` after
        ``sq.gr.spatial_neighbors``. The graph is adopted as-is, so
        ``edge_method``, ``k``, ``r`` and ``Dmax_CC`` are ignored; edge
        distances are recomputed from ``obsm['spatial']``. Use this when the
        cell graph is squidpy's job and mantpy's only job is the matrix.
    Dmax_CC
        Maximum cell-cell distance (pixels) for edge pruning.  ``None`` = no
        pruning (default).  **Only applies to** ``edge_method='knn'`` — Delaunay
        is parameter-free, so passing a non-``None`` ``Dmax_CC`` with any other
        method raises ``ValueError``.
    k
        Number of nearest neighbours (used when ``edge_method='knn'``).
    r
        Radius threshold (used when ``edge_method='radius'``).
    edge_method
        ``"knn"`` / ``"delaunay"`` / ``"radius"``.
    edge_weight
        How to weight edges.  One of ``"inv_distance"`` (default), ``"distance"``,
        ``"uniform"``, or a callable ``(i, j, distance) -> float``.
    edge_features
        Optional list of edge feature extractors.  Each item is a string key from
        :data:`EDGE_FEATURE_REGISTRY` (``"log_distance"``, ``"norm_angle"``,
        ``"distance"``) or a callable ``(pos_i, pos_j, distance) -> 1-D float32``.
        When provided, each edge stores ``feat_fwd`` and ``feat_rev`` arrays that
        ``to_pyg`` exports as ``edge_attr``.  ``None`` = no edge features (default).
    node_features
        Optional list of ``adata.obs`` column names to embed as node attributes.
    key_added
        Key under which to store the sparse adjacency in ``adata.obsp``.
    inplace
        Modify ``adata`` in place, or return a modified copy.

    Returns
    -------
    Depending on ``inplace``:

    - If ``inplace=True`` (default): updates ``adata`` in place with

      - ``adata.obsp[key_added]`` — Sparse (N_cells, N_cells) CSR adjacency matrix.
      - ``adata.uns[key_added + "_nx"]`` — NetworkX Graph with full node/edge attributes.

      and returns ``None``.

    - If ``inplace=False``: returns a copy of ``adata`` with the above fields populated.
    """
    if SPATIAL_KEY not in adata.obsm:
        raise ValueError(f"Key '{SPATIAL_KEY}' not found in adata.obsm. Run `mt.read_imc(...)` first.")
    if connectivity_key is None:
        _require_dmax_knn_only(edge_method, Dmax_CC, "Dmax_CC")

    if not inplace:
        adata = adata.copy()

    coords = adata.obsm[SPATIAL_KEY].astype(float)
    _ct_default = adata.obs.get(CELLTYPE_COL)
    cell_types = _ct_default.values if _ct_default is not None else np.full(len(adata), "unknown")

    if connectivity_key is None:
        edges = _build_edges(coords, edge_method, k=k, Dmax=Dmax_CC, r=r)
    else:
        edges = _edges_from_obsp(adata, connectivity_key, coords)

    n = len(coords)
    rows, cols, data = [], [], []
    for i, j, d in edges:
        w = _compute_weight(edge_weight, i, j, d)
        rows += [i, j]
        cols += [j, i]
        data += [w, w]

    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    adata.obsp[key_added] = adj
    _publish_spatial_connectivities(adata, adj, adopted_from=connectivity_key)

    extra_attrs: dict[str, np.ndarray] = {}
    if node_features:
        for feat in node_features:
            if feat in adata.obs.columns:
                extra_attrs[feat] = adata.obs[feat].values

    weight_fn = lambda i, j, d: _compute_weight(edge_weight, i, j, d)
    edge_feat_fn = (
        _make_edge_feat_fn(_resolve_edge_features(edge_features), coords) if edge_features is not None else None
    )
    nx_graph = build_cell_nx_graph(
        coords,
        cell_types,
        edges,
        extra_node_attrs=extra_attrs or None,
        weight_fn=weight_fn,
        edge_feat_fn=edge_feat_fn,
    )
    adata.uns[key_added + "_nx"] = nx_graph

    _log_params(
        adata,
        "gr",
        {
            "build_cell_graph": {
                "Dmax_CC": Dmax_CC,
                "k": k,
                "edge_method": edge_method,
                "edge_weight": edge_weight if isinstance(edge_weight, str) else "custom",
                "edge_features": [f if isinstance(f, str) else f.__name__ for f in edge_features]
                if edge_features
                else None,
                "n_edges": len(edges),
            }
        },
    )

    if not inplace:
        return adata
    return None


def build_ecm_graph(
    adata: AnnData,
    *,
    k: int = 5,
    r: float | None = None,
    Dmax: float | None = None,
    edge_method: Literal["knn", "delaunay", "radius", "grid"] = "knn",
    grid_connectivity: int = 8,
    edge_weight: EdgeWeight = "inv_distance",
    edge_features: list[str | Callable] | None = None,
    key_added: str = ECM_GRAPH_KEY,
    inplace: bool = True,
) -> AnnData | None:
    """Build an ECM-ECM spatial proximity graph from patch centroids.

    Parameters
    ----------
    adata
        AnnData with ``adata.uns['ecm_patches']`` (run
        :func:`~mantpy.pp.extract_ecm_patches` first).
    k
        Number of nearest neighbours (``edge_method='knn'``).
    r
        Radius threshold (``edge_method='radius'``).
    Dmax
        Maximum distance for edge pruning.  ``None`` = no pruning (default).
        **Only applies to** ``edge_method='knn'`` — Delaunay and grid topologies
        are parameter-free, so a non-``None`` ``Dmax`` with any other method
        raises ``ValueError``.
    edge_method
        Edge construction strategy. ``'knn'`` / ``'delaunay'`` / ``'radius'`` /
        ``'grid'``. Use ``'grid'`` for ECM patches on a regular lattice (the
        output of :func:`~mantpy.pp.extract_ecm_patches`): it connects each
        patch to its lattice neighbours with no distance threshold to tune.
    grid_connectivity
        Lattice stencil when ``edge_method='grid'``: ``8`` (Moore — orthogonal +
        diagonal, default) or ``4`` (von Neumann — orthogonal only).
    edge_weight
        How to weight edges.  One of ``"inv_distance"``, ``"distance"``,
        ``"uniform"``, or a callable ``(i, j, distance) -> float``.
    edge_features
        Optional list of edge feature extractors.  String keys from
        :data:`EDGE_FEATURE_REGISTRY` or callables
        ``(pos_i, pos_j, distance) -> 1-D float32``.  ``None`` = no edge features.
        Example: ``edge_features=["log_distance", "norm_angle"]``
    key_added
        Key under which to store the NetworkX graph in ``adata.uns``.
    inplace
        Modify ``adata`` in place, or return a modified copy.

    Returns
    -------
    Depending on ``inplace``:

    - If ``inplace=True`` (default): updates ``adata`` in place with

      - ``adata.uns[key_added]`` — NetworkX Graph of the ECM patch network.

      and returns ``None``.

    - If ``inplace=False``: returns a copy of ``adata`` with the above field populated.
    """
    if ECM_PATCHES_KEY not in adata.uns:
        raise ValueError(
            f"Key '{ECM_PATCHES_KEY}' not found in adata.uns. Run `mt.pp.extract_ecm_patches(adata, img)` first."
        )
    _require_dmax_knn_only(edge_method, Dmax, "Dmax")

    if not inplace:
        adata = adata.copy()

    patch_df = adata.uns[ECM_PATCHES_KEY]
    coords = patch_df[["x", "y"]].values.astype(float)

    edges = _build_edges(coords, edge_method, k=k, Dmax=Dmax, r=r, grid_connectivity=grid_connectivity)

    weight_fn = lambda i, j, d: _compute_weight(edge_weight, i, j, d)
    edge_feat_fn = (
        _make_edge_feat_fn(_resolve_edge_features(edge_features), coords) if edge_features is not None else None
    )
    G = build_ecm_nx_graph(patch_df, edges, weight_fn=weight_fn, edge_feat_fn=edge_feat_fn)
    adata.uns[key_added] = G

    _log_params(
        adata,
        "gr",
        {
            "build_ecm_graph": {
                "k": k,
                "edge_method": edge_method,
                "grid_connectivity": grid_connectivity if edge_method == "grid" else None,
                "edge_weight": edge_weight if isinstance(edge_weight, str) else "custom",
                "edge_features": [f if isinstance(f, str) else f.__name__ for f in edge_features]
                if edge_features
                else None,
                "n_ecm_nodes": len(patch_df),
                "n_edges": len(edges),
            }
        },
    )

    if not inplace:
        return adata
    return None


@dataclass(frozen=True)
class ECMGraphBuildResult:
    """Compact cohort summary plus the reusable ECM graph recipe."""

    rebuild_kwargs: dict[str, Any]
    samples: tuple[str, ...]
    node_counts: tuple[int, ...]
    edge_counts: tuple[int, ...]

    @property
    def n_nodes(self) -> int:
        """Total ECM nodes across the cohort."""
        return int(sum(self.node_counts))

    @property
    def n_edges(self) -> int:
        """Total undirected ECM edges across the cohort."""
        return int(sum(self.edge_counts))

    def __repr__(self) -> str:
        return "\n".join(
            [
                "ECM graph cohort",
                f"  samples       {len(self.samples)}",
                f"  ECM nodes     {self.n_nodes:,}",
                f"  ECM--ECM      {self.n_edges:,} undirected edges",
                "",
                "rebuild_kwargs stores the reusable Mantpy graph recipe.",
            ]
        )


def build_ecm_graphs(
    adatas: Mapping[str, AnnData],
    *,
    k: int = 5,
    r: float | None = None,
    Dmax: float | None = None,
    edge_method: Literal["knn", "delaunay", "radius", "grid"] = "knn",
    grid_connectivity: int = 8,
    edge_weight: EdgeWeight = "inv_distance",
    edge_features: list[str | Callable] | None = None,
    key_added: str = ECM_GRAPH_KEY,
) -> ECMGraphBuildResult:
    """Build the same ECM--ECM graph for every sample in a cohort."""
    if not adatas:
        raise ValueError("adatas is empty; expected at least one ECM sample.")
    recipe: dict[str, Any] = {
        "k": k,
        "r": r,
        "Dmax": Dmax,
        "edge_method": edge_method,
        "grid_connectivity": grid_connectivity,
        "edge_weight": edge_weight,
        "edge_features": edge_features,
        "key_added": key_added,
    }
    samples: list[str] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    for sample, adata in adatas.items():
        build_ecm_graph(adata, **recipe)
        graph = adata.uns[key_added]
        samples.append(str(sample))
        node_counts.append(int(graph.number_of_nodes()))
        edge_counts.append(int(graph.number_of_edges()))
    return ECMGraphBuildResult(
        rebuild_kwargs=recipe,
        samples=tuple(samples),
        node_counts=tuple(node_counts),
        edge_counts=tuple(edge_counts),
    )


def build_cell_ecm_graph(
    adata: AnnData,
    *,
    edge_method: Literal["knn", "delaunay"] = "delaunay",
    Dmax_CE: float | None = None,
    k: int = 5,
    edge_weight: EdgeWeight = "inv_distance",
    edge_features: list[str | Callable] | None = None,
    key_added: str = CELL_ECM_GRAPH_KEY,
    inplace: bool = True,
) -> AnnData | None:
    """Build a graph combining cell and ECM nodes.

    Works with whatever combination of cell graph / ECM graph is present:

    - **Both** present → full heterogeneous merge with cell-ECM cross edges.
    - **Only ECM graph** present → ECM-only graph stored at ``key_added``.
    - **Only cell graph** present → cell-only graph stored at ``key_added``.
    - **Neither** → raises ``ValueError``.

    Parameters
    ----------
    adata
        AnnData with at least one of ``adata.uns['cell_graph_nx']`` or
        ``adata.uns['ecm_graph']``.
    edge_method
        How to build cross edges when both cell and ECM graphs are present.

        - ``"delaunay"`` (default) — Delaunay triangulation on the combined
          point set, keeping only simplex edges that cross the cell/ECM
          boundary. Parameter-free; adapts to local density.
        - ``"knn"`` — K nearest ECM patches per cell, with optional
          distance cap ``Dmax_CE``.
    Dmax_CE
        Maximum cell-ECM distance for cross edges (pixels).
        ``None`` = no pruning (default).  **Only applies to**
        ``edge_method="knn"`` — Delaunay cross edges are parameter-free, so a
        non-``None`` ``Dmax_CE`` with ``edge_method="delaunay"`` raises
        ``ValueError``.  Only relevant when both graphs are present.
    k
        Maximum number of ECM neighbours per cell for cross edges.
        Only used when ``edge_method="knn"``.
    edge_weight
        How to weight cross edges.
    edge_features
        Optional edge feature extractors for cross edges.  Same format as in
        :func:`build_ecm_graph`.  ``None`` = no edge features on cross edges.
    key_added
        Key under which to store the unified NetworkX graph in ``adata.uns``.
    inplace
        Modify ``adata`` in place, or return a modified copy.

    Returns
    -------
    Depending on ``inplace``:

    - If ``inplace=True`` (default): updates ``adata`` in place with

      - ``adata.uns[key_added]`` — NetworkX Graph with ``node_type`` and
        ``edge_type`` attributes on every node and edge.

      and returns ``None``.

    - If ``inplace=False``: returns a copy of ``adata`` with the above field populated.
    """
    _require_dmax_knn_only(edge_method, Dmax_CE, "Dmax_CE")

    cell_graph_key = CELL_GRAPH_KEY + "_nx"
    has_cell = cell_graph_key in adata.uns
    has_ecm = ECM_GRAPH_KEY in adata.uns

    if not has_cell and not has_ecm:
        raise ValueError(
            "Neither a cell graph nor an ECM graph was found in adata.uns. "
            "Run `mt.gr.build_cell_graph` and/or `mt.gr.build_ecm_graph` first."
        )

    if not inplace:
        adata = adata.copy()

    if has_cell and has_ecm:
        cell_graph = adata.uns[cell_graph_key]
        ecm_graph = adata.uns[ECM_GRAPH_KEY]
        cell_coords = adata.obsm[SPATIAL_KEY].astype(float)
        ecm_coords = adata.uns[ECM_PATCHES_KEY][["x", "y"]].values.astype(float)

        weight_fn = lambda i, j, d: _compute_weight(edge_weight, i, j, d)
        # Cross-edge feat fn. Called by merge_cell_ecm_graphs as
        # (ci, ei, d[, reverse=True]) — ci is always cell index, ei always ecm.
        # reverse=True computes the ecm → cell orientation (swap pos_i / pos_j).
        cross_edge_feat_fn: Callable | None = None
        if edge_features is not None:
            fns = _resolve_edge_features(edge_features)

            def cross_edge_feat_fn(ci, ei, d, reverse=False, _c=cell_coords, _e=ecm_coords, _fns=fns):
                if reverse:
                    pos_i, pos_j = _e[ei], _c[ci]
                else:
                    pos_i, pos_j = _c[ci], _e[ei]
                parts = [f(pos_i, pos_j, d) for f in _fns]
                return np.concatenate(parts).astype(np.float32)

        G = merge_cell_ecm_graphs(
            cell_graph,
            ecm_graph,
            cell_coords,
            ecm_coords,
            Dmax_CE=Dmax_CE,
            k=k,
            weight_fn=weight_fn,
            edge_feat_fn=cross_edge_feat_fn,
            edge_method=edge_method,
        )
        mode = "cell+ecm"
    elif has_ecm:
        warnings.warn(
            "No cell graph found — building an ECM-only combined graph; cross edges will "
            "not be added. Run mt.gr.build_cell_graph first for a joint cell–ECM graph.",
            UserWarning,
            stacklevel=2,
        )
        G = adata.uns[ECM_GRAPH_KEY].copy()
        mode = "ecm-only"
    else:
        warnings.warn(
            "No ECM graph found — building a cell-only combined graph; cross edges will "
            "not be added. Run mt.gr.build_ecm_graph first for a joint cell–ECM graph.",
            UserWarning,
            stacklevel=2,
        )
        G = adata.uns[cell_graph_key].copy()
        mode = "cell-only"

    adata.uns[key_added] = G

    n_cell_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == NODE_TYPE_CELL)
    n_ecm_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == NODE_TYPE_ECM)
    n_ce_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "cell-ecm")

    _log_params(
        adata,
        "gr",
        {
            "build_cell_ecm_graph": {
                "mode": mode,
                "edge_method": edge_method,
                "Dmax_CE": Dmax_CE,
                "k": k,
                "edge_weight": edge_weight if isinstance(edge_weight, str) else "custom",
                "n_cell_nodes": n_cell_nodes,
                "n_ecm_nodes": n_ecm_nodes,
                "n_cell_ecm_edges": n_ce_edges,
            }
        },
    )

    if not inplace:
        return adata
    return None


def ensure_cell_ecm_graph(
    adata: AnnData,
    *,
    cell_k: int = 5,
    cell_Dmax: float | None = None,
    cell_edge_method: Literal["knn", "delaunay", "radius"] = "knn",
    cell_connectivity_key: str | None = None,
    ecm_k: int = 5,
    ecm_Dmax: float | None = None,
    ecm_edge_method: Literal["knn", "delaunay", "radius", "grid"] = "knn",
    ecm_grid_connectivity: int = 8,
    cell_ecm_k: int = 5,
    cell_ecm_Dmax: float | None = None,
    cell_ecm_edge_method: Literal["knn", "delaunay"] = "knn",
    rebuild: bool = True,
    edge_features: list[str | Callable] | None = None,
) -> None:
    """One-shot rebuild of all three graphs (cell, ECM, cell-ECM) on an AnnData.

    This is the convenience entry point most ML workflows want: pristine →
    artefact-injected, train → held-out, original → cropped, etc.
    Replacing only ``ecm_patches`` and then calling
    :func:`build_cell_ecm_graph` on its own would silently reuse the
    stale ECM-graph in ``adata.uns``; this helper pops the three graph
    keys first and re-runs all three builders with a single, consistent
    parameter dict.

    Parameters
    ----------
    adata
        Cell-side AnnData. Must carry ``adata.obsm['spatial']`` and
        ``adata.uns['ecm_patches']``.
    cell_k, cell_Dmax, cell_edge_method
        Forwarded to :func:`build_cell_graph` (``k``, ``Dmax_CC``,
        ``edge_method``).
    cell_connectivity_key
        Forwarded to :func:`build_cell_graph`. Set to
        ``"spatial_connectivities"`` to adopt a cell graph squidpy already
        built rather than rebuilding one; ``cell_k``/``cell_Dmax``/
        ``cell_edge_method`` are then ignored.
    ecm_k, ecm_Dmax, ecm_edge_method, ecm_grid_connectivity
        Forwarded to :func:`build_ecm_graph` (``k``, ``Dmax``,
        ``edge_method``, ``grid_connectivity``).
    cell_ecm_k, cell_ecm_Dmax, cell_ecm_edge_method
        Forwarded to :func:`build_cell_ecm_graph` (``k``, ``Dmax_CE``,
        ``edge_method``).
    rebuild
        When ``True`` (default), drops any pre-existing graphs first so
        the rebuild is guaranteed fresh. When ``False``, only builds the
        graphs that are missing — useful for the "just complete the
        layer cake" case (e.g. the user already built a custom cell
        graph and only wants the ECM and cross-edge layers added).
    edge_features
        Optional list passed identically to all three builders. See
        :data:`EDGE_FEATURE_REGISTRY` for valid string keys.

    Returns
    -------
    None
        Mutates ``adata`` in place. Use ``adata = adata.copy(); ensure_…``
        if you need an unmodified original.

    Notes
    -----
    Each ``*_Dmax`` distance cap is a **kNN-only** knob; the parameter-free
    Delaunay and grid topologies reject a non-``None`` cap with ``ValueError``.

    Examples
    --------
    >>> import mantpy as mt  # doctest: +SKIP
    >>> # Held-out ROI for an LOO fold: swap in artefact patches, then
    >>> # rebuild every graph in one call. Parameter-free topology:
    >>> # Delaunay cell + cell-ECM graphs, 8-connected ECM patch grid.
    >>> cell_target = cells_by_sample[held_out].copy()
    >>> cell_target.uns["ecm_patches"] = artefact_pool[held_out].uns["ecm_patches"]
    >>> mt.gr.ensure_cell_ecm_graph(
    ...     cell_target,
    ...     cell_edge_method="delaunay",
    ...     ecm_edge_method="grid",  # 8-connected lattice
    ...     cell_ecm_edge_method="delaunay",
    ... )
    """
    if SPATIAL_KEY not in adata.obsm:
        raise ValueError(f"Key '{SPATIAL_KEY}' not found in adata.obsm. Run `mt.read_imc(...)` first.")
    if ECM_PATCHES_KEY not in adata.uns:
        raise ValueError(
            f"Key '{ECM_PATCHES_KEY}' not found in adata.uns. Run `mt.pp.extract_ecm_patches(adata, img)` first."
        )
    # Validate every distance cap up front, before any graph keys are popped,
    # so a misuse never leaves the AnnData half-rebuilt.
    _require_dmax_knn_only(cell_edge_method, cell_Dmax, "cell_Dmax")
    _require_dmax_knn_only(ecm_edge_method, ecm_Dmax, "ecm_Dmax")
    _require_dmax_knn_only(cell_ecm_edge_method, cell_ecm_Dmax, "cell_ecm_Dmax")

    if rebuild:
        # Pop only the keys we know our own builders own. Leaves any
        # custom adata.uns content untouched.
        for key in (CELL_GRAPH_KEY + "_nx", ECM_GRAPH_KEY, CELL_ECM_GRAPH_KEY):
            adata.uns.pop(key, None)
        # The cell adjacency lives in obsp.
        if CELL_GRAPH_KEY in adata.obsp:
            del adata.obsp[CELL_GRAPH_KEY]

    has_cell = (CELL_GRAPH_KEY + "_nx") in adata.uns
    has_ecm = ECM_GRAPH_KEY in adata.uns
    has_ce = CELL_ECM_GRAPH_KEY in adata.uns

    if not has_cell:
        build_cell_graph(
            adata,
            k=cell_k,
            Dmax_CC=cell_Dmax,
            edge_method=cell_edge_method,
            edge_features=edge_features,
            connectivity_key=cell_connectivity_key,
        )
    if not has_ecm:
        build_ecm_graph(
            adata,
            k=ecm_k,
            Dmax=ecm_Dmax,
            edge_method=ecm_edge_method,
            grid_connectivity=ecm_grid_connectivity,
            edge_features=edge_features,
        )
    if not has_ce:
        build_cell_ecm_graph(
            adata,
            k=cell_ecm_k,
            Dmax_CE=cell_ecm_Dmax,
            edge_method=cell_ecm_edge_method,
            edge_features=edge_features,
        )


@dataclass(frozen=True)
class GraphBuildResult:
    """Reusable graph recipe and compact summaries for a cohort build."""

    rebuild_kwargs: dict[str, Any]
    summaries: tuple[JointGraphSummary, ...]

    def __repr__(self) -> str:
        cell_edges = sum(summary.cell_cell_edges for summary in self.summaries)
        ecm_edges = sum(summary.ecm_ecm_edges for summary in self.summaries)
        cross_edges = sum(summary.cell_ecm_edges for summary in self.summaries)
        return "\n".join(
            [
                "Joint cell--ECM graph cohort",
                f"  samples       {len(self.summaries)}",
                f"  cell--cell    {cell_edges:,} edges",
                f"  ECM--ECM      {ecm_edges:,} edges",
                f"  cell--ECM     {cross_edges:,} edges",
                "",
                "rebuild_kwargs stores the reusable Mantpy graph recipe.",
            ]
        )


def build_cell_ecm_graphs(
    adatas: Mapping[str, AnnData],
    *,
    cell_graph: Literal["squidpy", "mantpy"] = "squidpy",
    cell_k: int = 5,
    cell_radius: float | tuple[float, float] | None = None,
    cell_Dmax: float | None = None,
    cell_edge_method: Literal["knn", "delaunay", "radius"] = "knn",
    squidpy_kwargs: Mapping[str, Any] | None = None,
    ecm_k: int = 5,
    ecm_Dmax: float | None = None,
    ecm_edge_method: Literal["knn", "delaunay", "radius", "grid"] = "knn",
    ecm_grid_connectivity: int = 8,
    cell_ecm_k: int = 5,
    cell_ecm_Dmax: float | None = None,
    cell_ecm_edge_method: Literal["knn", "delaunay"] = "knn",
    edge_features: list[str | Callable] | None = None,
    suppress_squidpy_logs: bool = True,
) -> GraphBuildResult:
    """Build the complete cell--ECM graph for every sample in a cohort.

    The default keeps the scverse division of responsibilities explicit:
    Squidpy builds each cell--cell spatial graph, then Mantpy adopts that
    adjacency and adds the ECM--ECM and cell--ECM layers. Set
    ``cell_graph='mantpy'`` only when a Squidpy cell graph is not wanted.

    Parameters are the cohort equivalents of :func:`ensure_cell_ecm_graph`.
    ``squidpy_kwargs`` is forwarded to ``squidpy.gr.spatial_neighbors``;
    Mantpy supplies ``coord_type='generic'``, ``n_neighs=cell_k``, and the
    optional ``cell_radius`` directly. No post-hoc adjacency pruning occurs.
    ``suppress_squidpy_logs=True`` hides Squidpy's one INFO line per sample
    while preserving warnings and errors.

    Returns
    -------
    GraphBuildResult
        Compact cohort summaries plus the exact keyword recipe accepted by
        :func:`ensure_cell_ecm_graph` for copied or perturbed samples.

    Examples
    --------
    >>> graph_result = mt.gr.build_cell_ecm_graphs(  # doctest: +SKIP
    ...     cells_by_sample,
    ...     cell_graph="squidpy",
    ...     cell_k=5,
    ...     cell_radius=(0, 15),
    ...     ecm_edge_method="grid",
    ...     cell_ecm_Dmax=15,
    ... )
    """
    if not adatas:
        raise ValueError("adatas is empty; expected at least one sample.")
    if cell_graph not in {"squidpy", "mantpy"}:
        raise ValueError(f"cell_graph={cell_graph!r}; expected 'squidpy' or 'mantpy'.")
    if cell_graph == "mantpy" and cell_radius is not None:
        raise ValueError("cell_radius is a Squidpy setting; leave it None with cell_graph='mantpy'.")
    if cell_graph == "squidpy" and cell_Dmax is not None:
        raise ValueError(
            "cell_Dmax is a Mantpy kNN setting. With Squidpy, pass cell_radius "
            "so the radius is applied by squidpy.gr.spatial_neighbors."
        )

    recipe: dict[str, Any] = {
        "cell_k": cell_k,
        "cell_Dmax": cell_Dmax,
        "cell_edge_method": cell_edge_method,
        "ecm_k": ecm_k,
        "ecm_Dmax": ecm_Dmax,
        "ecm_edge_method": ecm_edge_method,
        "ecm_grid_connectivity": ecm_grid_connectivity,
        "cell_ecm_k": cell_ecm_k,
        "cell_ecm_Dmax": cell_ecm_Dmax,
        "cell_ecm_edge_method": cell_ecm_edge_method,
        "edge_features": edge_features,
    }

    if cell_graph == "squidpy":
        import squidpy as sq

        sq_kwargs: dict[str, Any] = {"coord_type": "generic", "n_neighs": cell_k}
        if cell_radius is not None:
            sq_kwargs["radius"] = cell_radius
        sq_kwargs.update(dict(squidpy_kwargs or {}))
        connectivity_key = "spatial_connectivities"
        spatialdata_logger = logging.getLogger("spatialdata._logging")
        previous_level = spatialdata_logger.level
        if suppress_squidpy_logs:
            spatialdata_logger.setLevel(logging.WARNING)
        try:
            for sample, adata in adatas.items():
                if SPATIAL_KEY not in adata.obsm:
                    raise ValueError(f"Sample {sample!r} has no adata.obsm[{SPATIAL_KEY!r}].")
                sq.gr.spatial_neighbors(adata, **sq_kwargs)
                if connectivity_key not in adata.obsp:
                    raise KeyError(f"Squidpy did not create adata.obsp[{connectivity_key!r}] for sample {sample!r}.")
        finally:
            if suppress_squidpy_logs:
                spatialdata_logger.setLevel(previous_level)
        recipe["cell_connectivity_key"] = connectivity_key
        recipe["cell_Dmax"] = None

    for adata in adatas.values():
        ensure_cell_ecm_graph(adata, rebuild=True, **recipe)
    summaries = tuple(joint_graph_summary(adata, sample=sample) for sample, adata in adatas.items())
    return GraphBuildResult(rebuild_kwargs=recipe, summaries=summaries)


@dataclass(frozen=True)
class JointGraphSummary:
    """Counts for the three node/edge types in a joint graph."""

    sample: str | None
    n_nodes: int
    n_cells: int
    n_ecm: int
    cell_cell_edges: int
    ecm_ecm_edges: int
    cell_ecm_edges: int

    def __repr__(self) -> str:
        label = "Joint graph" if self.sample is None else f"Joint graph: {self.sample}"
        return "\n".join(
            [
                label,
                f"  nodes          {self.n_nodes:,} ({self.n_cells:,} cells + {self.n_ecm:,} ECM patches)",
                f"  cell-cell      {self.cell_cell_edges:,} edges",
                f"  ECM-ECM        {self.ecm_ecm_edges:,} edges",
                f"  cell-ECM       {self.cell_ecm_edges:,} edges",
            ]
        )


def joint_graph_summary(
    adata: AnnData,
    *,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    sample: str | None = None,
) -> JointGraphSummary:
    """Return a compact, notebook-friendly summary of a joint graph."""
    if graph_key not in adata.uns:
        raise KeyError(f"adata.uns[{graph_key!r}] is missing; build the joint graph first.")
    graph = adata.uns[graph_key]
    if _is_sparse_joint_graph(adata, graph):
        node_type = adata.obs["node_type"].astype(str).to_numpy()
        n_cells = int(np.count_nonzero(node_type == NODE_TYPE_CELL))
        n_ecm = int(np.count_nonzero(node_type == NODE_TYPE_ECM))
        keys = graph.get("connectivities", {})
        cell_key = str(keys.get("cell", "cell_connectivities"))
        ecm_key = str(keys.get("ecm", "ecm_connectivities"))
        cross_key = str(keys.get("cell_ecm", "cell_ecm_connectivities"))
        cell_mask = node_type == NODE_TYPE_CELL
        ecm_mask = node_type == NODE_TYPE_ECM
        return JointGraphSummary(
            sample=sample,
            n_nodes=adata.n_obs,
            n_cells=n_cells,
            n_ecm=n_ecm,
            cell_cell_edges=_undirected_edge_count(adata.obsp[cell_key]),
            ecm_ecm_edges=_undirected_edge_count(adata.obsp[ecm_key]),
            cell_ecm_edges=int(adata.obsp[cross_key][cell_mask][:, ecm_mask].nnz),
        )
    node_types = [attrs.get("node_type") for _, attrs in graph.nodes(data=True)]
    edge_types = [attrs.get("edge_type") for _, _, attrs in graph.edges(data=True)]
    return JointGraphSummary(
        sample=sample,
        n_nodes=graph.number_of_nodes(),
        n_cells=node_types.count(NODE_TYPE_CELL),
        n_ecm=node_types.count(NODE_TYPE_ECM),
        cell_cell_edges=edge_types.count("cell-cell"),
        ecm_ecm_edges=edge_types.count("ecm-ecm"),
        cell_ecm_edges=edge_types.count("cell-ecm"),
    )


def compose_cell_ecm_graph(
    cells: AnnData,
    ecm: AnnData,
    *,
    cell_k: int = 8,
    ecm_grid_connectivity: Literal[4, 8] = 8,
    cell_ecm_radius: float = 48.0,
    pixel_size_um: float | tuple[float, float] | None = None,
    spatial_key: str = SPATIAL_KEY,
    spatial_um_key: str = "spatial_um",
    graph_key: str = CELL_ECM_GRAPH_KEY,
    n_jobs: int | None = None,
) -> AnnData:
    """Compose cell and ECM observations into a serialisable sparse graph.

    This is the whole-section counterpart of the legacy NetworkX graph
    builders. Cells and ECM patches become rows of one observation-native
    :class:`~anndata.AnnData`; graph layers are stored as CSR matrices rather
    than Python graph objects, so the result can be written directly to H5AD.

    Parameters
    ----------
    cells, ecm
        Observation-native cell and ECM objects. Both require pixel centroids
        in ``obsm[spatial_key]``. Physical centroids are read from
        ``obsm[spatial_um_key]`` or derived using ``pixel_size_um``.
    cell_k
        Number of nearest cell neighbours. The directed kNN relation is
        symmetrised by union.
    ecm_grid_connectivity
        Four- or eight-connected ECM patch lattice. Explicit ``grid_y`` /
        ``grid_x`` (or ``grid_row`` / ``grid_col``) columns are preferred;
        otherwise regular spacing is inferred from the ECM centroids.
    cell_ecm_radius
        Physical radius in micrometres for bipartite cell--ECM edges.
    pixel_size_um
        Scalar or ``(x, y)`` pixel scale used only when physical coordinates
        are absent from either input.
    spatial_key, spatial_um_key
        Pixel- and micrometre-coordinate slots.
    graph_key
        Provenance key in ``uns``. Connectivity slot names remain the public
        sparse schema: ``cell_connectivities``, ``ecm_connectivities``,
        ``cell_ecm_connectivities`` and ``joint_connectivities``.
    n_jobs
        Parallel workers used by the cell kNN query. ``None`` uses the
        scikit-learn default.

    Returns
    -------
    AnnData
        Rows are cells followed by ECM patches. ``obs['node_type']`` records
        the compartment; pixel and physical coordinates are in ``obsm`` and
        all four graph layers are sparse matrices in ``obsp``.

    Notes
    -----
    Input feature matrices are deliberately not padded into a shared dense
    matrix: cell and ECM feature spaces can differ and may be very large.
    Add a common representation to ``obsm`` when one is needed downstream,
    or use spatial coordinates (the default) with :func:`to_hetero_pyg`.
    """
    if cells.n_obs == 0 or ecm.n_obs == 0:
        raise ValueError("cells and ecm must each contain at least one observation.")
    if not isinstance(cell_k, int) or not 1 <= cell_k < cells.n_obs:
        raise ValueError(f"cell_k must satisfy 1 <= cell_k < n_cells ({cells.n_obs}); got {cell_k!r}.")
    if ecm_grid_connectivity not in (4, 8):
        raise ValueError("ecm_grid_connectivity must be 4 or 8.")
    if not np.isfinite(cell_ecm_radius) or cell_ecm_radius <= 0:
        raise ValueError("cell_ecm_radius must be a positive finite distance in micrometres.")

    cell_px = _validated_spatial(cells, spatial_key, "cells")
    ecm_px = _validated_spatial(ecm, spatial_key, "ecm")
    cell_um = _physical_spatial(cells, cell_px, spatial_um_key, pixel_size_um, "cells")
    ecm_um = _physical_spatial(ecm, ecm_px, spatial_um_key, pixel_size_um, "ecm")

    cell_adj = _symmetric_knn_connectivity(cell_um, cell_k, n_jobs=n_jobs)
    ecm_adj = _ecm_grid_connectivity(ecm, ecm_um, ecm_grid_connectivity)
    cross_rect = _radius_bipartite_connectivity(cell_um, ecm_um, float(cell_ecm_radius))

    n_cells, n_ecm = cells.n_obs, ecm.n_obs
    empty_cc = sp.csr_matrix((n_cells, n_cells), dtype=np.float32)
    empty_ee = sp.csr_matrix((n_ecm, n_ecm), dtype=np.float32)
    cell_full = sp.block_diag((cell_adj, empty_ee), format="csr")
    ecm_full = sp.block_diag((empty_cc, ecm_adj), format="csr")
    cross_full = sp.bmat([[None, cross_rect], [cross_rect.T, None]], format="csr", dtype=np.float32)
    joint_full = _binary_csr(cell_full + ecm_full + cross_full)

    cell_obs = cells.obs.copy()
    ecm_obs = ecm.obs.copy()
    cell_source_names = np.asarray(cells.obs_names.astype(str), dtype=object)
    ecm_source_names = np.asarray(ecm.obs_names.astype(str), dtype=object)
    cell_obs.index = pd.Index([f"cell:{name}" for name in cell_source_names], dtype=object)
    ecm_obs.index = pd.Index([f"ecm:{name}" for name in ecm_source_names], dtype=object)
    obs = pd.concat([cell_obs, ecm_obs], axis=0, sort=False)
    # Pandas 3 may infer Arrow-backed strings. Explicit object arrays keep the
    # graph writable by both current AnnData and older readers encountered in
    # long-lived reproducibility environments.
    for column in obs.columns:
        if isinstance(obs[column].dtype, pd.StringDtype):
            obs[column] = obs[column].astype(object)
    obs["source_obs_name"] = pd.Series(
        np.concatenate([cell_source_names, ecm_source_names]).astype(object),
        index=obs.index,
        dtype=object,
    )
    obs["node_type"] = pd.Categorical(
        np.asarray([NODE_TYPE_CELL] * n_cells + [NODE_TYPE_ECM] * n_ecm, dtype=object),
        categories=np.asarray([NODE_TYPE_CELL, NODE_TYPE_ECM], dtype=object),
    )

    joint = AnnData(X=sp.csr_matrix((n_cells + n_ecm, 0), dtype=np.float32), obs=obs)
    joint.obsm[spatial_key] = np.vstack([cell_px, ecm_px]).astype(np.float32, copy=False)
    joint.obsm[spatial_um_key] = np.vstack([cell_um, ecm_um]).astype(np.float32, copy=False)
    joint.obsp["cell_connectivities"] = cell_full
    joint.obsp["ecm_connectivities"] = ecm_full
    joint.obsp["cell_ecm_connectivities"] = cross_full
    joint.obsp["joint_connectivities"] = joint_full

    counts = {
        "n_nodes": int(joint.n_obs),
        "n_cells": int(n_cells),
        "n_ecm": int(n_ecm),
        "cell_cell_edges": _undirected_edge_count(cell_full),
        "ecm_ecm_edges": _undirected_edge_count(ecm_full),
        "cell_ecm_edges": int(cross_rect.nnz),
    }
    joint.uns[graph_key] = {
        "format": "mantpy_sparse_joint_v1",
        "node_type_key": "node_type",
        "spatial_key": spatial_key,
        "spatial_um_key": spatial_um_key,
        "connectivities": {
            "cell": "cell_connectivities",
            "ecm": "ecm_connectivities",
            "cell_ecm": "cell_ecm_connectivities",
            "joint": "joint_connectivities",
        },
        "counts": counts,
        "params": {
            "cell_k": int(cell_k),
            "ecm_grid_connectivity": int(ecm_grid_connectivity),
            "cell_ecm_radius_um": float(cell_ecm_radius),
            "cell_graph": "symmetric_knn_union",
            "ecm_graph": "grid",
            "cell_ecm_graph": "radius",
        },
    }
    _log_params(joint, "gr", {"compose_cell_ecm_graph": {**counts, **joint.uns[graph_key]["params"]}})
    return joint


def _validated_spatial(adata: AnnData, key: str, label: str) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"{label} has no obsm[{key!r}] pixel coordinates.")
    coords = np.asarray(adata.obsm[key], dtype=float)
    if coords.shape != (adata.n_obs, 2):
        raise ValueError(f"{label}.obsm[{key!r}] must have shape ({adata.n_obs}, 2); got {coords.shape}.")
    if not np.isfinite(coords).all():
        raise ValueError(f"{label}.obsm[{key!r}] contains non-finite coordinates.")
    return coords


def _physical_spatial(
    adata: AnnData,
    pixels: np.ndarray,
    key: str,
    pixel_size_um: float | tuple[float, float] | None,
    label: str,
) -> np.ndarray:
    if key in adata.obsm:
        return _validated_spatial(adata, key, label)
    if pixel_size_um is None:
        raise KeyError(
            f"{label} has no obsm[{key!r}]. Supply pixel_size_um to derive physical coordinates from pixels."
        )
    scale = np.asarray(pixel_size_um, dtype=float)
    if scale.ndim == 0:
        scale = np.repeat(scale, 2)
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("pixel_size_um must be a positive scalar or an (x, y) pair.")
    return pixels * scale[None, :]


def _binary_csr(matrix: sp.spmatrix, *, clear_diagonal: bool = True) -> sp.csr_matrix:
    result = matrix.tocsr().astype(np.float32, copy=False)
    result.sum_duplicates()
    if clear_diagonal:
        result.setdiag(0)
    result.eliminate_zeros()
    result.data[:] = 1.0
    result.sort_indices()
    return result


def _symmetric_knn_connectivity(coords: np.ndarray, k: int, *, n_jobs: int | None) -> sp.csr_matrix:
    from sklearn.neighbors import kneighbors_graph

    directed = kneighbors_graph(
        coords,
        n_neighbors=k,
        mode="connectivity",
        include_self=False,
        n_jobs=n_jobs,
    ).tocsr()
    return _binary_csr(directed.maximum(directed.T))


def _ecm_grid_connectivity(adata: AnnData, coords: np.ndarray, connectivity: int) -> sp.csr_matrix:
    grid_columns = next(
        (
            pair
            for pair in (("grid_y", "grid_x"), ("grid_row", "grid_col"), ("row", "col"))
            if set(pair) <= set(adata.obs)
        ),
        None,
    )
    if grid_columns is None:
        edges = grid_edges(coords, connectivity=connectivity)
    else:
        grid = adata.obs[list(grid_columns)].to_numpy(dtype=int)
        if np.unique(grid, axis=0).shape[0] != adata.n_obs:
            raise ValueError(f"ECM grid columns {grid_columns!r} contain duplicate locations.")
        lookup = {tuple(location): index for index, location in enumerate(grid)}
        offsets = [(0, 1), (1, 0)]
        if connectivity == 8:
            offsets.extend([(1, 1), (1, -1)])
        edges = []
        for index, (row, col) in enumerate(grid):
            for dr, dc in offsets:
                neighbour = lookup.get((int(row + dr), int(col + dc)))
                if neighbour is not None:
                    edges.append((index, neighbour, float(np.linalg.norm(coords[index] - coords[neighbour]))))
    if not edges:
        return sp.csr_matrix((adata.n_obs, adata.n_obs), dtype=np.float32)
    rows = np.fromiter((edge[0] for edge in edges), dtype=np.int64)
    cols = np.fromiter((edge[1] for edge in edges), dtype=np.int64)
    both_rows = np.concatenate([rows, cols])
    both_cols = np.concatenate([cols, rows])
    return _binary_csr(
        sp.csr_matrix(
            (np.ones(both_rows.size, dtype=np.float32), (both_rows, both_cols)),
            shape=(adata.n_obs, adata.n_obs),
        )
    )


def _radius_bipartite_connectivity(
    cell_coords: np.ndarray,
    ecm_coords: np.ndarray,
    radius: float,
) -> sp.csr_matrix:
    from scipy.spatial import cKDTree

    distances = cKDTree(cell_coords).sparse_distance_matrix(
        cKDTree(ecm_coords),
        max_distance=radius,
        output_type="coo_matrix",
    )
    return _binary_csr(
        sp.csr_matrix(
            (np.ones(distances.nnz, dtype=np.float32), (distances.row, distances.col)),
            shape=(cell_coords.shape[0], ecm_coords.shape[0]),
        ),
        clear_diagonal=False,
    )


def _is_sparse_joint_graph(adata: AnnData, graph: Any) -> bool:
    return (
        isinstance(graph, Mapping)
        and graph.get("format") == "mantpy_sparse_joint_v1"
        and "node_type" in adata.obs
        and "joint_connectivities" in adata.obsp
    )


def _undirected_edge_count(matrix: sp.spmatrix) -> int:
    return int(sp.triu(matrix, k=1, format="csr").nnz)


def extract_components_radius_reconnect(
    adata: AnnData,
    *,
    target_cluster: int | str,
    radius: float,
    cluster_attr: str = "ecm_cluster",
    pos_attrs: tuple[str, str] = ("x", "y"),
    min_nodes: int = 1,
    graph_key: str = ECM_GRAPH_KEY,
) -> list[dict]:
    """Extract radius-reconnected connected components of one cluster.

    Given a spatial graph in ``adata.uns[graph_key]`` whose nodes carry a
    cluster label and 2-D coordinates, pulls out every node belonging to
    ``target_cluster``, rebuilds edges among *those nodes only* using a
    Euclidean radius rule (any two points within ``radius`` get an edge),
    then returns one record per resulting connected component.

    This is the canonical "lesion" / "domain" / "patch-island" extraction:
    you keep the spatial proximity rule but ignore the original graph's
    edge structure (which may have been Delaunay / kNN with different
    connectivity properties).  Use ``min_nodes`` to filter out
    singletons / noise.

    Parameters
    ----------
    adata
        AnnData containing a NetworkX graph at ``adata.uns[graph_key]``.
    target_cluster
        Cluster label to extract (e.g. integer cluster id from Leiden).
        Nodes are matched via the node attribute ``cluster_attr``.
    radius
        Maximum Euclidean distance between two target-cluster points to
        place an edge.  Typical choice: ``4 × patch_size_px`` for IMC
        patch grids.
    cluster_attr
        Node-attribute name that carries the cluster label.
        Default ``"ecm_cluster"`` matches Mantpy's ECM-clustering output.
    pos_attrs
        Pair of node-attribute names that carry the (x, y) coordinates.
        Default ``("x", "y")`` matches :func:`build_ecm_graph`.
    min_nodes
        Minimum component size for ``kept=True``.  Components below this
        threshold are still returned (so users can audit / visualise the
        rejected fragments) but flagged ``kept=False``.
    graph_key
        Key under which the NetworkX graph is stored in ``adata.uns``.

    Returns
    -------
    list of dict
        One record per connected component, each with:

        - ``"node_ids"``  : list of original graph node IDs in this component
        - ``"coords"``    : ``(n, 2)`` ndarray of (x, y) positions
        - ``"subgraph"``  : NetworkX subgraph of the radius-graph induced on
          this component (edge weights = Euclidean distance)
        - ``"n_nodes"``   : component size
        - ``"n_edges"``   : number of radius-edges inside this component
        - ``"centroid"``  : ``(x, y)`` ndarray of the component centroid
        - ``"kept"``      : ``bool``, ``True`` iff ``n_nodes >= min_nodes``

    Examples
    --------
    Extract every cluster-5 lesion of radius 40 px in a single ROI::

        lesions = mt.gr.extract_components_radius_reconnect(
            adata,
            target_cluster=5,
            radius=40,
            min_nodes=5,
        )
        kept = [c for c in lesions if c["kept"]]
        print(f"{len(kept)} lesions, sizes {[c['n_nodes'] for c in kept]}")

    The returned subgraphs are ready to feed straight into network-stats
    functions::

        for c in kept:
            stats = mt.tl.lesion_topology_stats(c["subgraph"])

    Notes
    -----
    Implementation uses ``scipy.spatial.cKDTree.query_pairs`` for an O(N log N)
    radius query, then ``networkx.connected_components`` on the resulting
    radius-graph.  No edges from the input graph are inherited — only
    coordinates and the cluster label.
    """
    from scipy.spatial import cKDTree

    if graph_key not in adata.uns:
        raise ValueError(
            f"Key '{graph_key}' not found in adata.uns. "
            "Run a graph-build function (e.g. `mt.gr.build_ecm_graph`) first."
        )
    G: nx.Graph = adata.uns[graph_key]
    x_attr, y_attr = pos_attrs

    # Pull the target-cluster nodes and their coordinates.
    node_ids: list = []
    coords_list: list[tuple[float, float]] = []
    for v, attrs in G.nodes(data=True):
        if attrs.get(cluster_attr) != target_cluster:
            continue
        if x_attr not in attrs or y_attr not in attrs:
            continue
        node_ids.append(v)
        coords_list.append((float(attrs[x_attr]), float(attrs[y_attr])))

    if not node_ids:
        return []
    coords = np.asarray(coords_list, dtype=float)

    # Build the radius graph among target nodes only.
    G_rad = nx.Graph()
    G_rad.add_nodes_from(range(len(node_ids)))
    if len(node_ids) >= 2 and radius > 0:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=float(radius), output_type="ndarray")
        if len(pairs):
            for i, j in pairs:
                d = float(np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]))
                G_rad.add_edge(int(i), int(j), weight=d)

    components = list(nx.connected_components(G_rad))
    records: list[dict] = []
    for comp in components:
        idx = sorted(comp)
        sub = G_rad.subgraph(idx).copy()
        sub_coords = coords[idx]
        records.append(
            {
                "node_ids": [node_ids[i] for i in idx],
                "coords": sub_coords,
                "subgraph": sub,
                "n_nodes": len(idx),
                "n_edges": sub.number_of_edges(),
                "centroid": sub_coords.mean(axis=0) if len(idx) else np.array([0.0, 0.0]),
                "kept": len(idx) >= min_nodes,
            }
        )
    return records


def largest_component_radius_reconnect(
    adata: AnnData,
    *,
    target_cluster: int | str,
    radius: float,
    cluster_attr: str = "ecm_cluster",
    pos_attrs: tuple[str, str] = ("x", "y"),
    graph_key: str = ECM_GRAPH_KEY,
) -> dict | None:
    """Largest radius-reconnected component of one cluster (single-record convenience).

    Convenience wrapper around :func:`extract_components_radius_reconnect`
    for the very common "pick the biggest lesion / domain of cluster X
    in this ROI" pattern.  Calls the underlying full extraction with
    ``min_nodes=1`` (so the result is never silently filtered) and
    returns the record with the largest ``n_nodes``, or ``None`` if
    ``target_cluster`` has no nodes in the graph.

    Tie-break matches :func:`extract_components_radius_reconnect`'s own
    iteration order (which is the deterministic NetworkX
    ``connected_components`` order on the radius-graph), so calling this
    function and then picking ``max(records, key=lambda r: r["n_nodes"])``
    on the full record list yields the same record bit-for-bit.

    Parameters
    ----------
    adata
        AnnData containing a NetworkX graph at ``adata.uns[graph_key]``.
    target_cluster, radius, cluster_attr, pos_attrs, graph_key
        Forwarded to :func:`extract_components_radius_reconnect` — see
        that function's docstring for semantics.

    Returns
    -------
    dict | None
        A single component record (the same dict shape produced by
        :func:`extract_components_radius_reconnect`), or ``None`` when
        ``target_cluster`` has no nodes in the graph (a singleton
        cluster still returns a 1-node, 0-edge record).

    See Also
    --------
    :func:`extract_components_radius_reconnect` : full per-component
        extraction; use it when you need every lesion, not just the biggest.

    Examples
    --------
    Pick the biggest ECM-5 lesion in one ROI::

        rec = mt.gr.largest_component_radius_reconnect(
            adata,
            target_cluster=5,
            radius=40,
        )
        if rec is not None:
            coords = rec["coords"]  # (n, 2)
            sub = rec["subgraph"]  # nx.Graph of the radius-graph induced on this lesion
    """
    records = extract_components_radius_reconnect(
        adata,
        target_cluster=target_cluster,
        radius=radius,
        cluster_attr=cluster_attr,
        pos_attrs=pos_attrs,
        min_nodes=1,
        graph_key=graph_key,
    )
    if not records:
        return None
    # CPython max() returns the first-seen on ties, which keeps the result
    # deterministic given extract_components_radius_reconnect's iteration order
    # (deterministic nx.connected_components on the radius-graph).
    return max(records, key=lambda r: r["n_nodes"])


def to_pyg(
    adata: AnnData,
    *,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    node_feature_key: str | None = None,
    node_feature_keys: list[str] | None = None,
    include_edge_features: bool = True,
    label_key: str | None = None,
):
    """Convert a Mantpy graph to a PyTorch Geometric ``Data`` object.

    Exports whatever features are stored on nodes and edges — no hard-coded
    assumptions. For observation-native patch graphs, ``x`` is the selected
    ``X``/``obsm`` representation and coordinates are stored separately in
    ``pos``. For legacy NetworkX graphs, node features are built from:

    1. ``[x, y]`` — always included (spatial position)
    2. ``ecm_features`` list — appended if present on the node
    3. ``topo_degree``, ``topo_clustering``, ``topo_betweenness``,
       ``topo_pagerank`` — appended if present

    Edge features (``edge_attr``) are included only when edges carry ``feat_fwd``
    / ``feat_rev`` attributes, which are added by passing ``edge_features=...``
    to the graph builder.  Both directed orientations are emitted.

    Parameters
    ----------
    adata
        AnnData containing the graph.
    graph_key
        Key of the NetworkX graph in ``adata.uns``. Patch graphs produced by
        :func:`build_patch_graph` are also supported, as are sparse matrices in
        ``obsp``.
    node_feature_key
        For observation-native patch graphs, ``None``/``"X"`` uses
        ``adata.X``; otherwise read the named matrix from ``adata.obsm``.
    node_feature_keys
        If provided, use exactly these node attribute keys as feature columns
        (legacy / manual override).  ``None`` = auto-detect (recommended).
    include_edge_features
        Whether to include ``edge_attr`` when edge features are present.
    label_key
        Column in ``adata.obs`` holding a single per-graph class label, attached
        as ``data.y``. The column must hold one distinct value. When omitted,
        ``adata.uns["label"]`` is used if present, otherwise ``data.y`` is not
        set. Note that ``obs["y"]`` is the y-centroid written by
        :func:`mantpy.io.read_imc` and is never treated as a label.

    Returns
    -------
    torch_geometric.data.Data

    Notes
    -----
    For heterogeneous cell+ECM graphs use :func:`to_hetero_pyg`.
    Requires the ``[gnn]`` optional dependency: ``pip install mantpy[gnn]``.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as e:
        raise ImportError("PyTorch Geometric is required for `to_pyg`. Install with: pip install mantpy[gnn]") from e

    if node_feature_key is not None and node_feature_keys is not None:
        raise ValueError(
            "Pass node_feature_key for patch AnnData or node_feature_keys for legacy NetworkX nodes, not both."
        )

    graph_object = adata.uns.get(graph_key)
    direct_graph: Mapping[str, Any] | None = None
    if isinstance(graph_object, Mapping) and "edge_index" in graph_object:
        direct_graph = graph_object
    elif graph_key in adata.obsp:
        matrix = adata.obsp[graph_key].tocoo()
        direct_graph = {"edge_index": np.vstack((matrix.row, matrix.col)).astype(np.int64)}

    if direct_graph is not None:
        if node_feature_keys is not None:
            raise ValueError(
                "node_feature_keys selects NetworkX node attributes and cannot be used with an "
                "observation-native patch graph; use node_feature_key='X' or an obsm key."
            )
        if node_feature_key in (None, "X"):
            matrix = adata.X
        elif node_feature_key in adata.obsm:
            matrix = adata.obsm[node_feature_key]
        else:
            raise KeyError(f"Node representation {node_feature_key!r} not found in adata.obsm.")
        if sp.issparse(matrix):
            matrix = matrix.toarray()
        x = torch.tensor(np.asarray(matrix, dtype=np.float32), dtype=torch.float32)
        edge_index = torch.tensor(np.asarray(direct_graph["edge_index"], dtype=np.int64), dtype=torch.long)
        position = adata.obsm.get(SPATIAL_KEY)
        if position is None:
            position = np.zeros((adata.n_obs, 2), dtype=np.float32)
        data = Data(x=x, edge_index=edge_index, pos=torch.tensor(np.asarray(position), dtype=torch.float32))
        if include_edge_features and "edge_attr" in direct_graph:
            data.edge_attr = torch.tensor(np.asarray(direct_graph["edge_attr"], dtype=np.float32))
        data.obs_names = list(map(str, adata.obs_names))
        data.node_feature_key = node_feature_key or "X"
        return data

    if graph_key not in adata.uns:
        raise ValueError(f"Key '{graph_key}' not found in adata.uns or adata.obsp. Run the graph-build function first.")

    G: nx.Graph = adata.uns[graph_key]
    nodes_list = list(G.nodes(data=True))
    node_idx = {n: i for i, (n, _) in enumerate(nodes_list)}

    # ---- node features -------------------------------------------------------
    if node_feature_keys is not None:
        rows = [[float(d.get(k, 0.0)) for k in node_feature_keys] for _, d in nodes_list]
    else:
        # Spatial/geometric node attributes from legacy NetworkX inputs.
        _obs_attr_keys = ("sin_theta", "cos_theta", "radial_dist", "sauvola_pos")
        rows = []
        for _, d in nodes_list:
            row: list[float] = [d.get("x", 0.0), d.get("y", 0.0)]
            row.extend(float(v) for v in d.get("ecm_features", []))
            for attr in _obs_attr_keys:
                if attr in d:
                    row.append(float(d[attr]))
            for attr in ("topo_degree", "topo_clustering", "topo_betweenness", "topo_pagerank"):
                if attr in d:
                    row.append(float(d[attr]))
            rows.append(row)

    # Pad to uniform length in case of mixed node types
    max_len = max((len(r) for r in rows), default=0)
    x_data = [[*r, *([0.0] * (max_len - len(r)))] for r in rows]
    x = torch.tensor(x_data, dtype=torch.float32) if x_data else torch.empty((0, max_len), dtype=torch.float32)

    pos = torch.tensor(
        [[d.get("x", 0.0), d.get("y", 0.0)] for _, d in nodes_list],
        dtype=torch.float32,
    )

    # ---- edges ---------------------------------------------------------------
    src_list, dst_list = [], []
    feat_fwd_list: list[np.ndarray] = []
    feat_rev_list: list[np.ndarray] = []
    has_edge_feat = False

    for u, v, edata in G.edges(data=True):
        ui, vi = node_idx[u], node_idx[v]
        src_list += [ui, vi]
        dst_list += [vi, ui]
        if include_edge_features and "feat_fwd" in edata:
            feat_fwd_list.append(np.asarray(edata["feat_fwd"], dtype=np.float32))
            feat_rev_list.append(np.asarray(edata["feat_rev"], dtype=np.float32))
            has_edge_feat = True

    ei = torch.tensor([src_list, dst_list], dtype=torch.long) if src_list else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=ei, pos=pos)

    if include_edge_features and has_edge_feat and len(feat_fwd_list) == len(src_list) // 2:
        combined = []
        for f, r in zip(feat_fwd_list, feat_rev_list, strict=False):
            combined.append(f)
            combined.append(r)
        data.edge_attr = torch.tensor(np.stack(combined), dtype=torch.float32)

    # ---- optional islet-classifier fields (only populated when present) ------
    if "islet_graph_feat" in adata.uns:
        gf = np.asarray(adata.uns["islet_graph_feat"], dtype=np.float32)
        data.graph_feat = torch.tensor(gf.reshape(1, -1), dtype=torch.float32)

    # `obs["y"]` is deliberately NOT consulted here: Y_COL is "y", so read_imc
    # writes the y-centroid there and every graph would otherwise be labelled
    # with a truncated coordinate. Labels must be named explicitly.
    if label_key is not None:
        if label_key not in adata.obs.columns:
            raise KeyError(f"obs[{label_key!r}] is required when `label_key` is given.")
        labels = pd.unique(adata.obs[label_key])
        if len(labels) != 1:
            raise ValueError(
                f"obs[{label_key!r}] must hold a single per-graph label; "
                f"found {len(labels)} distinct values."
            )
        data.y = torch.tensor(int(labels[0]), dtype=torch.long)
    elif "label" in adata.uns:
        data.y = torch.tensor(int(adata.uns["label"]), dtype=torch.long)

    return data


def to_hetero_pyg(
    adata: AnnData,
    *,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    node_feature_key: str | Mapping[str, str] | None = None,
    include_edge_features: bool = True,
):
    """Convert a cell+ECM graph to a PyTorch Geometric ``HeteroData`` object.

    Separates nodes into ``'cell'`` and ``'ecm'`` types with their own feature
    tensors, and builds separate ``edge_index`` (and ``edge_attr`` when present)
    for each edge type:
    ``('cell', 'cell-cell', 'cell')``, ``('ecm', 'ecm-ecm', 'ecm')``, and
    ``('cell', 'cell-ecm', 'ecm')``.

    Node features follow the same auto-detection logic as :func:`to_pyg`:
    ``[x, y]`` + ``ecm_features`` (ECM nodes only) + topology attrs if present.

    Parameters
    ----------
    node_feature_key
        Feature representation for sparse observation-native graphs. ``None``
        uses physical positions; a string selects the same ``X``, ``obsm`` or
        numeric ``obs`` representation for both compartments; a mapping such
        as ``{"cell": "X_cell", "ecm": "X_ecm"}`` selects them separately.
        Legacy NetworkX export retains its historical automatic node features.
    include_edge_features
        Whether to include ``edge_attr`` tensors when edge features are stored.

    Requires the ``[gnn]`` optional dependency: ``pip install mantpy[gnn]``.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as e:
        raise ImportError(
            "PyTorch Geometric is required for `to_hetero_pyg`. Install with: pip install mantpy[gnn]"
        ) from e

    if graph_key not in adata.uns:
        raise ValueError(f"Key '{graph_key}' not found in adata.uns. Run `mt.gr.build_cell_ecm_graph(adata)` first.")

    graph_object = adata.uns[graph_key]
    if _is_sparse_joint_graph(adata, graph_object):
        return _sparse_joint_to_hetero_pyg(
            adata,
            graph_object,
            node_feature_key=node_feature_key,
            HeteroData=HeteroData,
            torch=torch,
        )
    if node_feature_key is not None:
        raise ValueError("node_feature_key is only supported for sparse observation-native joint graphs.")

    G: nx.Graph = graph_object

    cell_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("node_type") == NODE_TYPE_CELL]
    ecm_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("node_type") == NODE_TYPE_ECM]

    cell_idx = {n: i for i, (n, _) in enumerate(cell_nodes)}
    ecm_idx = {n: i for i, (n, _) in enumerate(ecm_nodes)}

    def _node_feats(nodes: list) -> torch.Tensor:
        rows = []
        for _, d in nodes:
            row: list[float] = [d.get("x", 0.0), d.get("y", 0.0)]
            row.extend(float(v) for v in d.get("ecm_features", []))
            for attr in ("topo_degree", "topo_clustering", "topo_betweenness", "topo_pagerank"):
                if attr in d:
                    row.append(float(d[attr]))
            rows.append(row)
        if not rows:
            return torch.empty((0, 2), dtype=torch.float32)
        max_len = max(len(r) for r in rows)
        padded = [[*r, *([0.0] * (max_len - len(r)))] for r in rows]
        return torch.tensor(padded, dtype=torch.float32)

    data = HeteroData()
    data["cell"].x = _node_feats(cell_nodes)
    data["ecm"].x = _node_feats(ecm_nodes)

    cc_src, cc_dst, cc_feat_fwd, cc_feat_rev = [], [], [], []
    ee_src, ee_dst, ee_feat_fwd, ee_feat_rev = [], [], [], []
    ce_src, ce_dst, ce_feat_fwd, ce_feat_rev = [], [], [], []

    for u, v, edata in G.edges(data=True):
        etype = edata.get("edge_type", "")
        f_fwd = np.asarray(edata["feat_fwd"], dtype=np.float32) if "feat_fwd" in edata else None
        f_rev = np.asarray(edata["feat_rev"], dtype=np.float32) if "feat_rev" in edata else None

        if etype == "cell-cell":
            if u in cell_idx and v in cell_idx:
                cc_src.append(cell_idx[u])
                cc_dst.append(cell_idx[v])
                if f_fwd is not None:
                    cc_feat_fwd.append(f_fwd)
                    cc_feat_rev.append(f_rev)
        elif etype == "ecm-ecm":
            if u in ecm_idx and v in ecm_idx:
                ee_src.append(ecm_idx[u])
                ee_dst.append(ecm_idx[v])
                if f_fwd is not None:
                    ee_feat_fwd.append(f_fwd)
                    ee_feat_rev.append(f_rev)
        elif etype == "cell-ecm":
            c_node, e_node = (u, v) if u in cell_idx else (v, u)
            if c_node in cell_idx and e_node in ecm_idx:
                ce_src.append(cell_idx[c_node])
                ce_dst.append(ecm_idx[e_node])
                if f_fwd is not None:
                    ce_feat_fwd.append(f_fwd)
                    ce_feat_rev.append(f_rev)

    def _ei(src: list, dst: list) -> torch.Tensor:
        if not src:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    def _ea(fwd: list, rev: list) -> torch.Tensor | None:
        if not include_edge_features or not fwd:
            return None
        combined = []
        for f, r in zip(fwd, rev, strict=False):
            combined.append(f)
            combined.append(r)
        return torch.tensor(np.stack(combined), dtype=torch.float32)

    data["cell", "cell-cell", "cell"].edge_index = _ei(cc_src, cc_dst)
    data["ecm", "ecm-ecm", "ecm"].edge_index = _ei(ee_src, ee_dst)
    data["cell", "cell-ecm", "ecm"].edge_index = _ei(ce_src, ce_dst)

    for rel, fwd, rev in [
        (("cell", "cell-cell", "cell"), cc_feat_fwd, cc_feat_rev),
        (("ecm", "ecm-ecm", "ecm"), ee_feat_fwd, ee_feat_rev),
        (("cell", "cell-ecm", "ecm"), ce_feat_fwd, ce_feat_rev),
    ]:
        ea = _ea(fwd, rev)
        if ea is not None:
            data[rel].edge_attr = ea

    return data


def _sparse_joint_to_hetero_pyg(
    adata: AnnData,
    graph: Mapping[str, Any],
    *,
    node_feature_key: str | Mapping[str, str] | None,
    HeteroData: Any,
    torch: Any,
):
    """Export the serialisable sparse joint schema without densifying edges."""
    node_type = adata.obs["node_type"].astype(str).to_numpy()
    cell_mask = node_type == NODE_TYPE_CELL
    ecm_mask = node_type == NODE_TYPE_ECM
    spatial_um_key = str(graph.get("spatial_um_key", "spatial_um"))
    position = np.asarray(adata.obsm[spatial_um_key], dtype=np.float32)

    def _selected_key(kind: str) -> str | None:
        if isinstance(node_feature_key, Mapping):
            if kind not in node_feature_key:
                raise KeyError(f"node_feature_key mapping has no {kind!r} entry.")
            return str(node_feature_key[kind])
        return node_feature_key

    def _features(kind: str, mask: np.ndarray) -> tuple[Any, str]:
        key = _selected_key(kind)
        if key is None:
            matrix = position[mask]
            resolved = spatial_um_key
        elif key == "X":
            matrix = adata.X[mask]
            resolved = "X"
        elif key in adata.obsm:
            matrix = adata.obsm[key][mask]
            resolved = key
        elif key in adata.obs:
            matrix = adata.obs.loc[mask, key].to_numpy(dtype=np.float32)[:, None]
            resolved = f"obs:{key}"
        else:
            raise KeyError(f"Node representation {key!r} was not found in joint.X, obsm or obs.")
        if sp.issparse(matrix):
            matrix = matrix.toarray()
        array = np.asarray(matrix, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        return torch.as_tensor(array, dtype=torch.float32), resolved

    keys = graph["connectivities"]

    def _edge_index(key: str, source_mask: np.ndarray, target_mask: np.ndarray):
        matrix = adata.obsp[key][source_mask][:, target_mask].tocoo()
        if matrix.nnz == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.as_tensor(np.vstack([matrix.row, matrix.col]), dtype=torch.long)

    cell_x, cell_feature_key = _features(NODE_TYPE_CELL, cell_mask)
    ecm_x, ecm_feature_key = _features(NODE_TYPE_ECM, ecm_mask)
    data = HeteroData()
    data[NODE_TYPE_CELL].x = cell_x
    data[NODE_TYPE_CELL].pos = torch.as_tensor(position[cell_mask], dtype=torch.float32)
    data[NODE_TYPE_CELL].obs_names = list(map(str, adata.obs_names[cell_mask]))
    data[NODE_TYPE_CELL].node_feature_key = cell_feature_key
    data[NODE_TYPE_ECM].x = ecm_x
    data[NODE_TYPE_ECM].pos = torch.as_tensor(position[ecm_mask], dtype=torch.float32)
    data[NODE_TYPE_ECM].obs_names = list(map(str, adata.obs_names[ecm_mask]))
    data[NODE_TYPE_ECM].node_feature_key = ecm_feature_key

    cc = (NODE_TYPE_CELL, "cell-cell", NODE_TYPE_CELL)
    ee = (NODE_TYPE_ECM, "ecm-ecm", NODE_TYPE_ECM)
    ce = (NODE_TYPE_CELL, "cell-ecm", NODE_TYPE_ECM)
    ec = (NODE_TYPE_ECM, "ecm-cell", NODE_TYPE_CELL)
    data[cc].edge_index = _edge_index(str(keys["cell"]), cell_mask, cell_mask)
    data[ee].edge_index = _edge_index(str(keys["ecm"]), ecm_mask, ecm_mask)
    data[ce].edge_index = _edge_index(str(keys["cell_ecm"]), cell_mask, ecm_mask)
    data[ec].edge_index = _edge_index(str(keys["cell_ecm"]), ecm_mask, cell_mask)
    data.graph_format = str(graph["format"])
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_edges(
    coords: np.ndarray,
    method: Literal["knn", "delaunay", "radius", "grid"],
    *,
    k: int = 5,
    Dmax: float | None = None,
    r: float | None = None,
    grid_connectivity: int = 8,
):
    if method == "knn":
        return knn_edges(coords, k=k, Dmax=Dmax)
    if method == "delaunay":
        # Parameter-free: Delaunay topology has no distance cap (Dmax is
        # knn-only and is validated away before reaching here).
        return delaunay_edges(coords)
    if method == "radius":
        if r is None:
            raise ValueError("edge_method='radius' requires the `r` parameter.")
        return radius_edges(coords, r=r)
    if method == "grid":
        # Parameter-free lattice stencil for regular ECM patch grids.
        return grid_edges(coords, connectivity=grid_connectivity)
    raise ValueError(f"Unknown edge_method '{method}'. Choose 'knn', 'delaunay', 'radius', or 'grid'.")


def _compute_weight(
    edge_weight: EdgeWeight,
    i: int,
    j: int,
    d: float,
) -> float:
    if edge_weight == "inv_distance":
        return 1.0 / d if d > 0 else 0.0
    if edge_weight == "distance":
        return d
    if edge_weight == "uniform":
        return 1.0
    return float(edge_weight(i, j, d))


def _resolve_edge_features(
    edge_features: list[str | Callable] | None,
) -> list[EdgeFeatureFn]:
    """Resolve string aliases and callables to a list of EdgeFeatureFn."""
    if edge_features is None:
        return []
    resolved: list[EdgeFeatureFn] = []
    for f in edge_features:
        if callable(f):
            resolved.append(f)
        elif isinstance(f, str):
            if f not in EDGE_FEATURE_REGISTRY:
                raise ValueError(
                    f"Unknown edge feature '{f}'. "
                    f"Available: {list(EDGE_FEATURE_REGISTRY)}. "
                    "Pass a callable for custom edge features."
                )
            resolved.append(EDGE_FEATURE_REGISTRY[f])
        else:
            raise TypeError(f"edge_features items must be str or callable, got {type(f)}.")
    return resolved


def _make_edge_feat_fn(
    fns: list[EdgeFeatureFn],
    coords: np.ndarray,
) -> Callable[[int, int, float], np.ndarray]:
    """Return a function (i, j, d) -> 1-D float32 array using coords for positions."""

    def fn(i: int, j: int, d: float) -> np.ndarray:
        parts = [f(coords[i], coords[j], d) for f in fns]
        return np.concatenate(parts).astype(np.float32)

    return fn


# ---------------------------------------------------------------------------
# Squidpy graph interoperability.
# ---------------------------------------------------------------------------


def _edges_from_obsp(adata: AnnData, connectivity_key: str, coords: np.ndarray) -> list[tuple[int, int, float]]:
    """Adopt an existing ``obsp`` adjacency as mantpy's edge list.

    Lets squidpy (or anything else) own the cell graph while mantpy adds the
    matrix on top, rather than building a second, silently different one.

    Distances are recomputed from ``obsm['spatial']`` rather than read from a
    companion ``*_distances`` matrix: the stored values are whatever the builder
    chose to put there (squidpy's ``coord_type='grid'`` writes all-ones, for
    instance), while mantpy's edge weights are meant to be true euclidean.
    """
    if connectivity_key not in adata.obsp:
        raise ValueError(
            f"connectivity_key='{connectivity_key}' not found in adata.obsp "
            f"(available: {sorted(adata.obsp.keys()) or 'none'}). "
            "Build it first, e.g. sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6)."
        )
    A = sp.csr_matrix(adata.obsp[connectivity_key])
    # squidpy's kNN adjacency is directed — symmetrise before taking the upper
    # triangle or roughly half the neighbours are silently dropped.
    A = A.maximum(A.T)
    coo = sp.triu(A, k=1).tocoo()
    if coo.nnz == 0:
        raise ValueError(f"adata.obsp['{connectivity_key}'] has no edges.")
    d = np.linalg.norm(coords[coo.row] - coords[coo.col], axis=1)
    return list(zip(coo.row.tolist(), coo.col.tolist(), d.tolist(), strict=True))


def _publish_spatial_connectivities(adata: AnnData, adj: sp.spmatrix, *, adopted_from: str | None = None) -> None:
    """Also expose a freshly built graph under squidpy's canonical key.

    ``sq.gr.*`` default to ``obsp['spatial_connectivities']``
    (``Key.obsp.spatial_conn``), so a graph stored only under mantpy's own key
    is invisible to squidpy — and ``sq.gr.spatial_neighbors`` will happily build
    a *second*, different one beside it with no warning. Publishing here is a
    pointer copy, not a matrix copy.

    If the slot already holds a different graph we overwrite it — the caller
    explicitly asked mantpy to build one — but we say so, because silently
    replacing a graph the user built with squidpy is exactly the failure this
    function exists to prevent.

    ``adopted_from`` names the key the edges were just taken from. Writing back
    to that same key is a no-op in substance — mantpy stores the symmetrised,
    distance-weighted form of the graph it was handed, so the matrices differ
    numerically while describing the same topology — and must not warn.
    """
    if adopted_from == _SQUIDPY_CONN_KEY:
        adata.obsp[_SQUIDPY_CONN_KEY] = adj
        return

    existing = adata.obsp.get(_SQUIDPY_CONN_KEY)
    if existing is not None and existing is not adj:
        same = existing.shape == adj.shape and (existing != adj).nnz == 0
        if not same:
            warnings.warn(
                f"Replacing an existing adata.obsp['{_SQUIDPY_CONN_KEY}'] "
                f"({existing.nnz} stored edges) with the graph mantpy just built "
                f"({adj.nnz}). If that graph came from sq.gr.spatial_neighbors, "
                "the two disagree and downstream squidpy results will now use "
                "mantpy's. Pass a different key_added to keep both.",
                UserWarning,
                stacklevel=3,
            )
    adata.obsp[_SQUIDPY_CONN_KEY] = adj
