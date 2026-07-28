"""Edge construction algorithms for spatial graphs.

Internal module — not part of the public API.

All edge-builder functions return a list of (i, j, distance) tuples so callers
can decide how to weight or threshold them.
"""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx
import numpy as np

EdgeList = list[tuple[int, int, float]]
WeightFn = Callable[[int, int, float], float]


def knn_edges(
    coords: np.ndarray,
    k: int,
    Dmax: float | None = None,
) -> EdgeList:
    """K-nearest-neighbour edges.

    Parameters
    ----------
    coords
        (N, 2) array of (x, y) coordinates.
    k
        Number of neighbours per node.
    Dmax
        Maximum allowed distance.  Edges longer than this are pruned.

    Returns
    -------
    List of (i, j, distance) tuples (undirected — each edge appears once).
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(coords)
    if n < 2:
        return []

    k_actual = min(k + 1, n)  # +1 because NearestNeighbors includes self
    nn = NearestNeighbors(n_neighbors=k_actual, algorithm="ball_tree")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    edges: EdgeList = []
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        for j_pos in range(1, len(indices[i])):
            j = int(indices[i][j_pos])
            d = float(distances[i][j_pos])
            if Dmax is not None and d > Dmax:
                continue
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append((i, j, d))
    return edges


def delaunay_edges(
    coords: np.ndarray,
    Dmax: float | None = None,
) -> EdgeList:
    """Delaunay triangulation edges.

    Parameters
    ----------
    coords
        (N, 2) array of (x, y) coordinates.
    Dmax
        Maximum allowed distance.

    Returns
    -------
    List of (i, j, distance) tuples.
    """
    from scipy.spatial import Delaunay

    n = len(coords)
    if n < 3:
        return knn_edges(coords, k=min(2, n - 1), Dmax=Dmax)

    tri = Delaunay(coords)
    edges: EdgeList = []
    seen: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            i, j = int(simplex[a]), int(simplex[b])
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if Dmax is not None and d > Dmax:
                continue
            edges.append((i, j, d))
    return edges


def radius_edges(
    coords: np.ndarray,
    r: float,
) -> EdgeList:
    """Radius-based edges: connect all pairs within distance ``r``.

    Parameters
    ----------
    coords
        (N, 2) array of (x, y) coordinates.
    r
        Radius threshold.

    Returns
    -------
    List of (i, j, distance) tuples.
    """
    from sklearn.neighbors import BallTree

    n = len(coords)
    if n < 2:
        return []

    tree = BallTree(coords)
    indices = tree.query_radius(coords, r=r)
    edges: EdgeList = []
    seen: set[tuple[int, int]] = set()
    for i, neighbours in enumerate(indices):
        for j in neighbours:
            j = int(j)
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            edges.append((i, j, d))
    return edges


def grid_edges(
    coords: np.ndarray,
    connectivity: int = 8,
    spacing: float | None = None,
) -> EdgeList:
    """Lattice-connectivity edges for points sampled on a regular grid.

    Designed for ECM patch centroids, which
    :func:`~mantpy.pp.extract_ecm_patches` places on a regular, non-overlapping
    grid. Each node connects to its immediate lattice neighbours — either the
    4-connected von Neumann neighbourhood (orthogonal only) or the 8-connected
    Moore neighbourhood (orthogonal + diagonal). The grid spacing is inferred
    from the data, so the construction is **parameter-free**: there is no
    distance threshold to tune or defend.

    Unlike ``knn_edges(coords, k=8)``, this never produces long edges that reach
    across gaps in the lattice (e.g. where background patches were dropped) — a
    node on the edge of a hole simply ends up with fewer neighbours. It is the
    clean equivalent of "k=8 KNN then cap diagonals", but expressed directly as
    a grid stencil.

    Parameters
    ----------
    coords
        (N, 2) array of (x, y) coordinates on a regular grid.
    connectivity
        ``8`` — Moore neighbourhood (4 orthogonal + 4 diagonal, default).
        ``4`` — von Neumann neighbourhood (orthogonal only).
    spacing
        Grid spacing. ``None`` (default) infers it as the median
        nearest-neighbour distance, which equals the lattice step for a regular
        grid and is robust to a few duplicate/short distances.

    Returns
    -------
    List of (i, j, distance) tuples (undirected — each edge appears once).
    """
    from sklearn.neighbors import BallTree

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}.")

    n = len(coords)
    if n < 2:
        return []

    tree = BallTree(coords)
    if spacing is None:
        nn_dist, _ = tree.query(coords, k=2)
        step = nn_dist[:, 1]
        step = step[step > 0]
        if len(step) == 0:
            return []
        spacing = float(np.median(step))

    # Pick a query radius strictly between the farthest in-stencil neighbour and
    # the nearest out-of-stencil one, so only true lattice neighbours connect:
    #   4-conn: orthogonal (=spacing)        < r < diagonal (=spacing*sqrt2)
    #   8-conn: diagonal (=spacing*sqrt2)    < r < 2-step orthogonal (=2*spacing)
    sqrt2 = 2.0**0.5
    radius = spacing * (2.0 + sqrt2) / 2.0 if connectivity == 8 else spacing * (1.0 + sqrt2) / 2.0

    indices = tree.query_radius(coords, r=radius)
    edges: EdgeList = []
    seen: set[tuple[int, int]] = set()
    for i, neighbours in enumerate(indices):
        for j in neighbours:
            j = int(j)
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            edges.append((i, j, d))
    return edges


def build_cell_nx_graph(
    coords: np.ndarray,
    cell_types: np.ndarray,
    edges: EdgeList,
    extra_node_attrs: dict[str, np.ndarray] | None = None,
    weight_fn: WeightFn | None = None,
    edge_feat_fn: WeightFn | None = None,
) -> nx.Graph:
    """Build a NetworkX graph from cell coordinates and edges.

    Parameters
    ----------
    coords
        (N, 2) array of (x, y) coordinates.
    cell_types
        (N,) array of cell-type strings.
    edges
        List of (i, j, distance) tuples.
    extra_node_attrs
        Optional dict of ``{attr_name: array_of_length_N}`` for additional
        node attributes (e.g. marker expression).
    weight_fn
        Callable ``(i, j, distance) -> float`` for edge weight.
        Defaults to ``1/distance``.
    edge_feat_fn
        Optional callable ``(i, j, distance) -> 1-D float32 array`` for edge
        feature vectors.  When provided, each edge stores ``feat_fwd`` (i→j)
        and ``feat_rev`` (j→i) attributes, which ``to_pyg`` exports as
        ``edge_attr``.

    Returns
    -------
    nx.Graph with node attributes: ``node_type``, ``x``, ``y``, ``cell_type``,
    and edge attributes: ``distance``, ``weight``, ``edge_type``.
    """
    from mantpy._constants import EDGE_TYPE_CC, NODE_TYPE_CELL

    if weight_fn is None:
        weight_fn = lambda i, j, d: 1.0 / d if d > 0 else float("inf")

    G = nx.Graph()
    for i in range(len(coords)):
        attrs = {
            "node_type": NODE_TYPE_CELL,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "cell_type": str(cell_types[i]),
        }
        if extra_node_attrs:
            for k, v in extra_node_attrs.items():
                attrs[k] = v[i]
        G.add_node(f"cell_{i}", **attrs)

    for i, j, d in edges:
        w = weight_fn(i, j, d)
        eattrs: dict = {"distance": d, "weight": w, "edge_type": EDGE_TYPE_CC}
        if edge_feat_fn is not None:
            eattrs["feat_fwd"] = edge_feat_fn(i, j, d)
            eattrs["feat_rev"] = edge_feat_fn(j, i, d)
        G.add_edge(f"cell_{i}", f"cell_{j}", **eattrs)

    return G


def build_ecm_nx_graph(
    ecm_patches_df,
    edges: EdgeList,
    extra_node_attrs: dict[str, list] | None = None,
    weight_fn: WeightFn | None = None,
    edge_feat_fn: WeightFn | None = None,
) -> nx.Graph:
    """Build a NetworkX graph from ECM patch nodes and edges.

    Parameters
    ----------
    ecm_patches_df
        DataFrame with ``x``, ``y``, ``ecm_cluster``, and feature columns.
    edges
        List of (i, j, distance) tuples.
    extra_node_attrs
        Optional additional node attributes.
    weight_fn
        Callable ``(i, j, distance) -> float`` for edge weight.
        Defaults to ``1/distance``.
    edge_feat_fn
        Optional callable ``(i, j, distance) -> 1-D float32 array``.
        When provided, stores ``feat_fwd`` (i→j) and ``feat_rev`` (j→i) on each
        edge, exported as ``edge_attr`` by ``to_pyg``.

    Returns
    -------
    nx.Graph with node attributes: ``node_type``, ``x``, ``y``,
    ``ecm_cluster``, ``ecm_features``, and any extra attrs.
    """
    from mantpy._constants import EDGE_TYPE_EE, NODE_TYPE_ECM

    if weight_fn is None:
        weight_fn = lambda i, j, d: 1.0 / d if d > 0 else float("inf")

    feat_cols = [c for c in ecm_patches_df.columns if c.startswith("feat_")]
    G = nx.Graph()

    for i, row in enumerate(ecm_patches_df.itertuples(index=False)):
        features = [getattr(row, c) for c in feat_cols]
        attrs = {
            "node_type": NODE_TYPE_ECM,
            "x": float(row.x),
            "y": float(row.y),
            "ecm_cluster": int(row.ecm_cluster),
            "ecm_features": features,
        }
        if extra_node_attrs:
            for k, v in extra_node_attrs.items():
                attrs[k] = v[i]
        G.add_node(f"ecm_{i}", **attrs)

    for i, j, d in edges:
        w = weight_fn(i, j, d)
        eattrs: dict = {"distance": d, "weight": w, "edge_type": EDGE_TYPE_EE}
        if edge_feat_fn is not None:
            eattrs["feat_fwd"] = edge_feat_fn(i, j, d)
            eattrs["feat_rev"] = edge_feat_fn(j, i, d)
        G.add_edge(f"ecm_{i}", f"ecm_{j}", **eattrs)

    return G


def merge_cell_ecm_graphs(
    cell_graph: nx.Graph,
    ecm_graph: nx.Graph,
    cell_coords: np.ndarray,
    ecm_coords: np.ndarray,
    Dmax_CE: float | None,
    k: int,
    weight_fn: WeightFn | None = None,
    edge_feat_fn=None,
    edge_method: str = "knn",
) -> nx.Graph:
    """Merge cell and ECM graphs, adding cross edges within ``Dmax_CE``.

    Parameters
    ----------
    cell_graph
        NetworkX graph of cell nodes.
    ecm_graph
        NetworkX graph of ECM nodes.
    cell_coords
        (N_cells, 2) coordinates.
    ecm_coords
        (N_ecm, 2) coordinates.
    Dmax_CE
        Maximum cell-ECM distance to add a cross edge.  ``None`` = no limit.
    k
        Maximum number of ECM neighbours per cell.
    weight_fn
        Callable ``(i, j, distance) -> float`` for cross-edge weight.
    edge_feat_fn
        Optional callable ``(ci, ei, distance, *, reverse=False) -> 1-D
        float32 array`` for cross-edge features. ``ci`` indexes
        ``cell_coords`` and ``ei`` indexes ``ecm_coords``; the callable is
        invoked twice per cross edge with ``reverse=False`` for the
        cell→ECM orientation and ``reverse=True`` for ECM→cell. Custom
        callables MUST accept the ``reverse`` keyword — passing one that
        doesn't will raise ``TypeError`` (mantpy's own
        :func:`mantpy.gr.build_cell_ecm_graph` already wraps user-supplied
        edge features into a closure that supports it).

    Returns
    -------
    nx.Graph with all cell, ECM, and cross edges.
    """
    from mantpy._constants import EDGE_TYPE_CE

    if weight_fn is None:
        weight_fn = lambda i, j, d: 1.0 / d if d > 0 else float("inf")

    G = nx.Graph()
    G.add_nodes_from(cell_graph.nodes(data=True))
    G.add_edges_from(cell_graph.edges(data=True))
    G.add_nodes_from(ecm_graph.nodes(data=True))
    G.add_edges_from(ecm_graph.edges(data=True))

    if len(cell_coords) == 0 or len(ecm_coords) == 0:
        return G

    if edge_method == "delaunay":
        cross_edges = delaunay_edges_bipartite(cell_coords, ecm_coords, Dmax=Dmax_CE)
    elif edge_method == "knn":
        cross_edges = knn_edges_bipartite(cell_coords, ecm_coords, k=k, Dmax=Dmax_CE)
    else:
        raise ValueError(f"edge_method must be 'knn' or 'delaunay', got {edge_method!r}")

    # Node identity is recovered by name prefix, so the incoming graphs must use
    # mantpy's own naming. A graph built elsewhere (squidpy, or any networkx
    # default) has integer node names, which match neither prefix — see below.
    cell_nodes = [n for n in G.nodes if str(n).startswith("cell_")]
    ecm_nodes = [n for n in G.nodes if str(n).startswith("ecm_")]

    # Cross-edges are indexed into these lists positionally. If a list is short,
    # the index is unresolvable: previously we minted an attribute-less
    # `cell_{ci}`/`ecm_{ei}` placeholder, which silently routed every cross-edge
    # onto phantom nodes and left the real ones with none — leaving downstream
    # callers to fail far away with "No cell-ECM edges with signal clusters
    # found." Fail here instead, where the cause is visible.
    for name, nodes, coords in (
        ("cell", cell_nodes, cell_coords),
        ("ecm", ecm_nodes, ecm_coords),
    ):
        if len(nodes) < len(coords):
            raise ValueError(
                f"merge_cell_ecm_graphs found {len(nodes)} '{name}_*' nodes but was given "
                f"{len(coords)} {name} coordinates. Node names must be '{name}_<i>' matching "
                f"the row order of {name}_coords; graphs built outside mantpy (e.g. by "
                "sq.gr.spatial_neighbors) use integer names and must be relabelled first — "
                f"e.g. nx.relabel_nodes(g, {{i: f'{name}_{{i}}' for i in range(g.number_of_nodes())}})."
            )

    for ci, ei, d in cross_edges:
        cn = cell_nodes[ci]
        en = ecm_nodes[ei]
        w = weight_fn(ci, ei, d)
        eattrs: dict = {"distance": d, "weight": w, "edge_type": EDGE_TYPE_CE}
        if edge_feat_fn is not None:
            # Cross-edge conventions:
            #   forward  = cell → ecm  (edge_feat_fn expects ci = cell idx, ei = ecm idx)
            #   reverse  = ecm  → cell (pass reverse=True so the closure swaps which
            #                           coords array it indexes into)
            # Custom callables passed in must accept reverse=False/True;
            # the previous "swap indices" fallback silently returned wrong
            # values whenever Na != Nb (cell coords and ecm coords differ
            # in length, which is the normal case).
            eattrs["feat_fwd"] = edge_feat_fn(ci, ei, d, reverse=False)
            eattrs["feat_rev"] = edge_feat_fn(ci, ei, d, reverse=True)
        G.add_edge(cn, en, **eattrs)

    return G


def knn_edges_bipartite(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    k: int,
    Dmax: float | None = None,
) -> list[tuple[int, int, float]]:
    """K-nearest neighbours from set A to set B (directed A→B)."""
    from sklearn.neighbors import NearestNeighbors

    if len(coords_a) == 0 or len(coords_b) == 0:
        return []

    k_actual = min(k, len(coords_b))
    nn = NearestNeighbors(n_neighbors=k_actual, algorithm="ball_tree")
    nn.fit(coords_b)
    distances, indices = nn.kneighbors(coords_a)

    edges = []
    for i in range(len(coords_a)):
        for pos in range(k_actual):
            j = int(indices[i][pos])
            d = float(distances[i][pos])
            if Dmax is not None and d > Dmax:
                continue
            edges.append((i, j, d))
    return edges


def delaunay_edges_bipartite(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    Dmax: float | None = None,
) -> list[tuple[int, int, float]]:
    """Bipartite Delaunay edges between two point sets.

    Runs Delaunay triangulation on the stacked point cloud
    ``[coords_a; coords_b]`` and keeps only simplex edges that cross the
    A/B boundary. Returns ``(i_a, i_b, distance)`` tuples indexed into
    the two input arrays.

    Parameters
    ----------
    coords_a
        ``(Na, 2)`` coordinates for set A.
    coords_b
        ``(Nb, 2)`` coordinates for set B.
    Dmax
        Optional distance cap. Edges longer than this are pruned.
    """
    from scipy.spatial import Delaunay, QhullError

    na = len(coords_a)
    nb = len(coords_b)
    if na == 0 or nb == 0:
        return []
    if na + nb < 3:
        return knn_edges_bipartite(coords_a, coords_b, k=min(1, nb), Dmax=Dmax)

    stacked = np.vstack([coords_a, coords_b])
    try:
        tri = Delaunay(stacked)
    except QhullError:
        return knn_edges_bipartite(coords_a, coords_b, k=min(5, nb), Dmax=Dmax)

    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int, float]] = []
    for simplex in tri.simplices:
        pts = [int(p) for p in simplex]
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            p, q = pts[a], pts[b]
            # one endpoint in A (< na), the other in B (>= na)
            if (p < na) == (q < na):
                continue
            if p < na:
                ia, ib = p, q - na
            else:
                ia, ib = q, p - na
            key = (ia, ib)
            if key in seen:
                continue
            seen.add(key)
            d = float(np.linalg.norm(coords_a[ia] - coords_b[ib]))
            if Dmax is not None and d > Dmax:
                continue
            edges.append((ia, ib, d))
    return edges
