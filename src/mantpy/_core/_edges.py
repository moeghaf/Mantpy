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

    Each point selects its ``min(k, N - 1)`` nearest *other* points and the
    selections are merged into an undirected edge list: an edge is kept when
    either endpoint selected the other, so a node's final degree can exceed
    ``k``.  The query point is excluded by observation identity, not by its
    position in the returned neighbourhood.  Distinct points with identical
    coordinates therefore remain valid neighbours of each other (at distance
    0), and no self-loop is produced.  A selection whose distance exceeds
    ``Dmax`` is dropped; a distance equal to ``Dmax`` is kept.

    Candidates tied at the same distance are taken in the order returned by
    scikit-learn's ball-tree query, as before.  Relative to earlier releases,
    selection changes only for coincident points: the self-match no longer
    occupies one of the ``k`` slots.  scikit-learn's ``kneighbors()`` without
    a query array removes each point's own index wherever it appears; when
    more than ``k`` coincident points crowd the point itself out of the
    returned list, the first returned point is dropped instead.

    Parameters
    ----------
    coords
        (N, 2) array of (x, y) coordinates.
    k
        Number of neighbours selected per node (capped at ``N - 1``).
    Dmax
        Maximum allowed distance.  Edges longer than this are pruned.

    Returns
    -------
    List of (i, j, distance) tuples (undirected — each edge appears once).

    .. versionchanged:: unreleased
        Self-exclusion is now by identity.  Earlier releases skipped the first
        returned neighbour, which for duplicate coordinates could be another
        point at distance 0; the self-match was then kept as a self-loop
        ``(i, i, 0.0)`` while that coincident neighbour was dropped.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(coords)
    if n < 2:
        return []
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k!r}.")

    k_eff = min(k, n - 1)
    if k_eff == 0:
        return []
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree")
    nn.fit(coords)
    # Without a query array, kneighbors() returns neighbours of the fitted
    # points and excludes each point's own index by identity.
    distances, indices = nn.kneighbors(n_neighbors=k_eff)

    edges: EdgeList = []
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        for j_pos in range(len(indices[i])):
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
    *,
    jitter: float | None = None,
    seed: int = 0,
) -> EdgeList:
    """Delaunay triangulation edges.

    Parameters
    ----------
    coords
        (N, 2) array of (x, y) coordinates.
    Dmax
        Maximum allowed distance.
    jitter
        Optional tie-break perturbation. When given, the triangulation is
        computed on ``jitter_coords(coords, jitter, seed)``, a copy of
        ``coords`` plus seeded Gaussian noise with standard deviation
        ``jitter``, so that co-circular points (for example a regular patch
        lattice, where every square has two valid diagonals) are triangulated
        reproducibly. Edge lengths and the ``Dmax`` test always use the
        unperturbed ``coords``. ``None`` (default) triangulates ``coords`` as
        given.
    seed
        Seed of :func:`numpy.random.default_rng` used when ``jitter`` is given.

    Returns
    -------
    List of (i, j, distance) tuples.
    """
    from scipy.spatial import Delaunay

    n = len(coords)
    if n < 3:
        return knn_edges(coords, k=min(2, n - 1), Dmax=Dmax)

    tri = Delaunay(coords if jitter is None else jitter_coords(coords, jitter, seed))
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
            eattrs["feat_source"] = f"cell_{i}"
            eattrs["feat_target"] = f"cell_{j}"
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
            "source_id": str(ecm_patches_df.index[i]),
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
            eattrs["feat_source"] = f"ecm_{i}"
            eattrs["feat_target"] = f"ecm_{j}"
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
    cross_edges: EdgeList | None = None,
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
    cross_edges
        Optional precomputed ``(cell_index, ecm_index, distance)`` cross edges
        (any rule). When given they are added as they are and ``edge_method``,
        ``k`` and ``Dmax_CE`` are not used. ``None`` (default) builds them with
        ``edge_method``.

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

    if cross_edges is not None:
        cross_edges = list(cross_edges)
    elif edge_method == "delaunay":
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
            eattrs["feat_source"] = cn
            eattrs["feat_target"] = en
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
    *,
    jitter: float | None = None,
    seed: int = 0,
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
    jitter, seed
        Optional tie-break perturbation of the stacked points before
        triangulating, as in :func:`delaunay_edges`; lengths and ``Dmax`` use
        the unperturbed coordinates. ``None`` (default) triangulates the
        points as given.
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
        tri = Delaunay(stacked if jitter is None else jitter_coords(stacked, jitter, seed))
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


def jitter_coords(coords: np.ndarray, jitter: float, seed: int = 0) -> np.ndarray:
    """Return a seeded, Gaussian-perturbed copy of ``coords``.

    The copy is ``coords + numpy.random.default_rng(seed).normal(scale=jitter,
    size=coords.shape)``. It is used only to choose a triangulation of tied
    (co-circular) points, never to measure distances.
    """
    coords = np.asarray(coords, dtype=float)
    rng = np.random.default_rng(seed)
    return coords + rng.normal(scale=jitter, size=coords.shape)


def knn_edges_kdtree(
    coords: np.ndarray,
    k: int,
    Dmax: float | None = None,
) -> EdgeList:
    """K-nearest-neighbour edges found with :class:`scipy.spatial.cKDTree`.

    Same contract as :func:`knn_edges`: each point selects its ``min(k, N - 1)``
    nearest other points, an edge is kept when either endpoint selected the
    other, a selection farther than ``Dmax`` is dropped (a distance equal to
    ``Dmax`` is kept), and the query point is excluded by identity, so
    coincident distinct points remain neighbours at distance 0.

    The two searches return the same neighbours unless candidates are tied at
    the k-th distance, as on a regular patch lattice; each search then keeps
    the tied candidates it reaches first. Use this backend to reproduce graphs
    built with ``cKDTree.query``. Distances are those returned by the tree.
    """
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return []
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k!r}.")
    k_eff = min(int(k), n - 1)
    if k_eff == 0:
        return []
    distances, indices = cKDTree(coords).query(coords, k=k_eff + 1)
    distances = np.asarray(distances).reshape(n, -1)
    indices = np.asarray(indices).reshape(n, -1)

    edges: EdgeList = []
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        own = np.flatnonzero(indices[i] == i)
        # When more than k coincident points crowd the query point out of its
        # own neighbourhood, drop the first returned point instead.
        skip = int(own[0]) if own.size else 0
        for pos in range(indices.shape[1]):
            if pos == skip:
                continue
            j = int(indices[i, pos])
            d = float(distances[i, pos])
            if Dmax is not None and d > Dmax:
                continue
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append((i, j, d))
    return edges


def knn_edges_bipartite_kdtree(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    k: int,
    Dmax: float | None = None,
) -> list[tuple[int, int, float]]:
    """K nearest neighbours from set A to set B with :class:`scipy.spatial.cKDTree`.

    Same contract as :func:`knn_edges_bipartite` (directed A -> B, ``Dmax``
    inclusive); see :func:`knn_edges_kdtree` for when the two searches differ.
    """
    from scipy.spatial import cKDTree

    if len(coords_a) == 0 or len(coords_b) == 0:
        return []
    k_actual = min(int(k), len(coords_b))
    if k_actual <= 0:
        return []
    coords_a = np.asarray(coords_a, dtype=float)
    distances, indices = cKDTree(np.asarray(coords_b, dtype=float)).query(coords_a, k=k_actual)
    distances = np.asarray(distances).reshape(len(coords_a), -1)
    indices = np.asarray(indices).reshape(len(coords_a), -1)

    edges = []
    for i in range(len(coords_a)):
        for pos in range(k_actual):
            d = float(distances[i, pos])
            if Dmax is not None and d > Dmax:
                continue
            edges.append((i, int(indices[i, pos]), d))
    return edges


def radius_edges_bipartite(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    r: float,
) -> list[tuple[int, int, float]]:
    """All pairs from set A to set B at distance ``<= r`` (directed A -> B).

    The bipartite counterpart of :func:`radius_edges`: a pair at exactly ``r``
    is kept, and a point of A that coincides with a point of B is joined to it
    at distance 0.
    """
    from sklearn.neighbors import BallTree

    if len(coords_a) == 0 or len(coords_b) == 0:
        return []
    coords_a = np.asarray(coords_a, dtype=float)
    coords_b = np.asarray(coords_b, dtype=float)
    neighbours = BallTree(coords_b).query_radius(coords_a, r=r)
    edges = []
    for i, js in enumerate(neighbours):
        for j in np.sort(js):
            j = int(j)
            edges.append((i, j, float(np.linalg.norm(coords_a[i] - coords_b[j]))))
    return edges


def containing_ring_edges(
    cell_coords: np.ndarray,
    ecm_coords: np.ndarray,
    patch_size: float,
) -> list[tuple[int, int, float]]:
    """Join each cell to the patch that contains it and that patch's 8 lattice neighbours.

    ECM patches must lie on the regular lattice of
    :func:`mantpy.pp.extract_ecm_patches`: patch ``(row, col)`` covers the
    pixels ``[col * s, (col + 1) * s) x [row * s, (row + 1) * s)`` and is
    centred at ``((col + 0.5) * s, (row + 0.5) * s)`` for ``s = patch_size``.
    A cell at ``(x, y)`` lies in patch ``(floor(y / s), floor(x / s))`` and is
    joined to every patch present in ``ecm_coords`` among that patch and its
    eight neighbours (a 3 x 3 block, at most nine edges per cell), whether or
    not the containing patch itself is present. Edge length is the Euclidean
    distance from the cell to the patch centre (0 for a cell at a centre).

    Returns ``(cell_index, ecm_index, distance)`` tuples ordered by cell, then
    by row offset and column offset.
    """
    s = float(patch_size)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"patch_size must be a positive, finite number, got {patch_size!r}.")
    cell_coords = np.asarray(cell_coords, dtype=float).reshape(-1, 2)
    ecm_coords = np.asarray(ecm_coords, dtype=float).reshape(-1, 2)
    if len(cell_coords) == 0 or len(ecm_coords) == 0:
        return []

    index = np.rint(ecm_coords / s - 0.5)
    if not np.allclose((index + 0.5) * s, ecm_coords, rtol=0.0, atol=1e-6 * s):
        raise ValueError(
            "edge_method='containing_ring' needs ECM patch centres on the regular patch lattice "
            f"((index + 0.5) * patch_size, patch_size={s:g}); these centres are not. Pass the "
            "patch_size used to extract the patches."
        )
    col, row = index[:, 0].astype(np.int64), index[:, 1].astype(np.int64)
    r0, c0 = int(row.min()) - 1, int(col.min()) - 1
    lookup = np.full((int(row.max()) - r0 + 2, int(col.max()) - c0 + 2), -1, dtype=np.int64)
    if np.unique(np.column_stack([row, col]), axis=0).shape[0] != len(ecm_coords):
        raise ValueError("edge_method='containing_ring': two ECM patches share a lattice position (duplicate patches).")
    lookup[row - r0, col - c0] = np.arange(len(ecm_coords))

    cell_row = np.floor(cell_coords[:, 1] / s).astype(np.int64)
    cell_col = np.floor(cell_coords[:, 0] / s).astype(np.int64)
    cells, patches, order = [], [], []
    for offset, (dr, dc) in enumerate((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
        rr, cc = cell_row + dr - r0, cell_col + dc - c0
        inside = (rr >= 0) & (rr < lookup.shape[0]) & (cc >= 0) & (cc < lookup.shape[1])
        hit = np.full(len(cell_coords), -1, dtype=np.int64)
        hit[inside] = lookup[rr[inside], cc[inside]]
        present = np.flatnonzero(hit >= 0)
        cells.append(present)
        patches.append(hit[present])
        order.append(np.full(present.size, offset))
    ci, ei, off = np.concatenate(cells), np.concatenate(patches), np.concatenate(order)
    sort = np.lexsort((off, ci))
    ci, ei = ci[sort], ei[sort]
    d = np.linalg.norm(cell_coords[ci] - ecm_coords[ei], axis=1)
    return [(int(i), int(j), float(dd)) for i, j, dd in zip(ci, ei, d, strict=True)]
