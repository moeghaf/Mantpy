"""Opt-in graph rules for regular patch lattices and SciPy-built graphs.

The new rules are checked against an independent oracle: a minimal SciPy
re-implementation of the graph code of the mouse-lung analysis
(``mantpy_reproducibility``, ``notebooks/cell_ecm_lung/lung_analysis/graphs.py``).
It uses ``cKDTree`` for kNN (distance cap inclusive) and radius queries, Delaunay
on points jittered by ``default_rng(0).normal(scale=1e-6)`` with an optional cap on
the unjittered length, index-lattice adjacency, and the containing-patch plus
eight-neighbour cell--ECM rule. It shares no code with Mantpy.

Documented convention differences, tested explicitly:

* the oracle drops zero-length edges between distinct coincident points; Mantpy keeps
  them at distance 0, as its existing radius rule (and the kNN identity fix) do;
* the oracle stores 1e-6 for a cell exactly at a patch centre in the containing-ring
  rule; Mantpy stores the true distance 0.0;
* the oracle excludes a kNN query point by position (first returned neighbour);
  Mantpy excludes it by identity, which differs only for coincident points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.spatial import Delaunay, cKDTree

import mantpy as mt
from mantpy._core._edges import (
    containing_ring_edges,
    delaunay_edges,
    delaunay_edges_bipartite,
    jitter_coords,
    knn_edges_bipartite_kdtree,
    knn_edges_kdtree,
    radius_edges_bipartite,
)

S = 10.0  # patch side (pixels)


# --------------------------------------------------------------------------- inputs


def _lattice(nx: int = 9, ny: int = 8, holes=((2, 3), (5, 5), (0, 7), (6, 1), (7, 2))):
    """Patch centres ``(g + 0.5) * S`` on an ``nx`` x ``ny`` lattice with holes; returns (xy, grid_rc)."""
    grid = np.array([(gy, gx) for gy in range(ny) for gx in range(nx) if (gx, gy) not in set(holes)], dtype=int)
    xy = np.column_stack([(grid[:, 1] + 0.5) * S, (grid[:, 0] + 0.5) * S]).astype(float)
    return xy, grid


def _cells(n: int = 70, seed: int = 3, hi: float = 90.0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-4.0, hi, size=(n, 2))


def _special_cells(patches: np.ndarray) -> np.ndarray:
    """Cells on patch boundaries, at patch centres and at exact distances 15 / 20 from a centre."""
    c0 = patches[10]
    return np.array(
        [
            [20.0, 33.3],  # on a vertical patch boundary
            [41.7, 50.0],  # on a horizontal patch boundary
            [30.0, 40.0],  # on a lattice corner
            patches[4],  # exactly at a patch centre
            patches[17],
            c0 + [9.0, 12.0],  # exactly 15 from c0
            c0 + [12.0, 16.0],  # exactly 20 from c0
            [40.0, 20.0],  # on a lattice corner: four centres tie
            [25.0, 50.0],  # on a horizontal boundary: two centres tie
        ]
    )


def _adata(cells: np.ndarray, patches: np.ndarray) -> AnnData:
    adata = AnnData(obs=pd.DataFrame({"cell_type": ["T"] * len(cells)}, index=[f"c{i}" for i in range(len(cells))]))
    adata.obsm["spatial"] = np.asarray(cells, float)
    adata.uns["ecm_patches"] = pd.DataFrame(
        {
            "x": patches[:, 0],
            "y": patches[:, 1],
            "ecm_cluster": np.zeros(len(patches), int),
            "feat_0": np.ones(len(patches)),
        }
    )
    return adata


# --------------------------------------------------------------------------- oracle


def _o_sym(xy, i, j) -> dict:
    out = {}
    for a, b in zip(np.asarray(i, int), np.asarray(j, int), strict=True):
        if a == b:
            continue
        d = float(np.linalg.norm(xy[a] - xy[b]))
        if d > 0:  # coincident points: zero-length edge dropped
            out[(min(a, b), max(a, b))] = d
    return out


def o_knn(xy, k, dmax=None) -> dict:
    n = len(xy)
    d, j = cKDTree(xy).query(xy, k=min(k + 1, n))
    d, j = d[:, 1:], j[:, 1:]
    i = np.repeat(np.arange(n), j.shape[1])
    j, d = j.ravel(), d.ravel()
    keep = np.isfinite(d) & (d <= dmax if dmax is not None else True)
    return _o_sym(xy, i[keep], j[keep])


def o_knn_identity(xy, k, dmax=None) -> dict:
    """kNN with identity self-exclusion and zero-length edges kept (Mantpy's documented semantics)."""
    n = len(xy)
    k = min(k, n - 1)
    d, j = cKDTree(xy).query(xy, k=k + 1)
    out = {}
    for a in range(n):
        own = np.flatnonzero(j[a] == a)
        drop = own[0] if own.size else 0
        for pos in range(k + 1):
            if pos == drop or (dmax is not None and d[a, pos] > dmax):
                continue
            b = int(j[a, pos])
            out[(min(a, b), max(a, b))] = float(np.linalg.norm(xy[a] - xy[b]))
    return out


def o_radius(xy, r) -> dict:
    pairs = cKDTree(xy).query_pairs(r=r, output_type="ndarray")
    return _o_sym(xy, pairs[:, 0], pairs[:, 1]) if len(pairs) else {}


def _o_jitter(xy):
    return xy + np.random.default_rng(0).normal(scale=1e-6, size=xy.shape)


def o_delaunay(xy, max_len=None) -> dict:
    s = Delaunay(_o_jitter(xy)).simplices
    i = np.concatenate([s[:, 0], s[:, 1], s[:, 2]])
    j = np.concatenate([s[:, 1], s[:, 2], s[:, 0]])
    if max_len is not None:
        keep = np.linalg.norm(xy[i] - xy[j], axis=1) <= max_len
        i, j = i[keep], j[keep]
    return _o_sym(xy, i, j)


def o_cross(cells, patches, rule, *, k=5, dmax=None, grid=None) -> dict:
    out = {}
    if rule == "knn":
        kk = min(k, len(patches))
        d, j = cKDTree(patches).query(cells, k=kk)
        d, j = np.asarray(d).reshape(len(cells), -1), np.asarray(j).reshape(len(cells), -1)
        for a in range(len(cells)):
            for dd, b in zip(d[a], j[a], strict=True):
                if np.isfinite(dd) and dd > 0 and (dmax is None or dd <= dmax):
                    out[(a, int(b))] = float(dd)
    elif rule == "radius":
        for a, lst in enumerate(cKDTree(patches).query_ball_point(cells, r=dmax)):
            for b in lst:
                dd = float(np.linalg.norm(cells[a] - patches[b]))
                if dd > 0:
                    out[(a, int(b))] = dd
    elif rule == "delaunay":
        pts, n_c = np.vstack([cells, patches]), len(cells)
        s = Delaunay(_o_jitter(pts)).simplices
        a = np.concatenate([s[:, 0], s[:, 1], s[:, 2]])
        b = np.concatenate([s[:, 1], s[:, 2], s[:, 0]])
        if dmax is not None:
            keep = np.linalg.norm(pts[a] - pts[b], axis=1) <= dmax
            a, b = a[keep], b[keep]
        for p, q in zip(a, b, strict=True):
            if (p < n_c) == (q < n_c):
                continue
            c, e = (p, q - n_c) if p < n_c else (q, p - n_c)
            dd = float(np.linalg.norm(cells[c] - patches[e]))
            if dd > 0:
                out[(int(c), int(e))] = dd
    elif rule == "containing_ring":
        lookup = {(int(r), int(c)): m for m, (r, c) in enumerate(grid)}
        for a, (x, y) in enumerate(cells):
            cr, cc = int(np.floor(y / S)), int(np.floor(x / S))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    m = lookup.get((cr + dr, cc + dc))
                    if m is not None:
                        out[(a, m)] = max(float(np.linalg.norm(cells[a] - patches[m])), 1e-6)
    else:  # pragma: no cover
        raise ValueError(rule)
    return out


# --------------------------------------------------------------------------- helpers


def _sym(edges) -> dict:
    out = {}
    for i, j, d in edges:
        assert i != j, "self-loop"
        key = (min(i, j), max(i, j))
        assert key not in out, "duplicate edge"
        out[key] = float(d)
    return out


def _cross(edges) -> dict:
    out = {}
    for i, j, d in edges:
        assert (i, j) not in out, "duplicate edge"
        out[(int(i), int(j))] = float(d)
    return out


def _assert_same(got: dict, want: dict) -> None:
    assert set(got) == set(want), (sorted(set(got) - set(want))[:5], sorted(set(want) - set(got))[:5])
    keys = sorted(want)
    np.testing.assert_allclose([got[k] for k in keys], [want[k] for k in keys], rtol=1e-12, atol=0)


def _positive(d: dict) -> dict:
    return {k: v for k, v in d.items() if v > 0}


def _nx(adata, key, prefix_a, prefix_b=None, edge_type=None) -> dict:
    out = {}
    for u, v, d in adata.uns[key].edges(data=True):
        if edge_type is not None and d.get("edge_type") != edge_type:
            continue
        if prefix_b is None:
            i, j = int(str(u).split("_")[1]), int(str(v).split("_")[1])
            out[(min(i, j), max(i, j))] = float(d["distance"])
        else:
            u, v = (u, v) if str(u).startswith(prefix_a) else (v, u)
            out[(int(str(u).split("_")[1]), int(str(v).split("_")[1]))] = float(d["distance"])
    return out


# --------------------------------------------------------------------------- jitter / Delaunay


def test_jitter_coords_is_the_seeded_gaussian_copy():
    xy, _ = _lattice()
    before = xy.copy()
    got = jitter_coords(xy, 1e-6, 0)
    np.testing.assert_array_equal(got, xy + np.random.default_rng(0).normal(scale=1e-6, size=xy.shape))
    np.testing.assert_array_equal(xy, before)
    assert not np.array_equal(got, jitter_coords(xy, 1e-6, 1))


@pytest.mark.parametrize("points", ["random", "lattice"])
def test_delaunay_without_jitter_is_the_default(points):
    xy = _cells(60) if points == "random" else _lattice()[0]
    assert delaunay_edges(xy) == delaunay_edges(xy, None, jitter=None, seed=0)
    a, b = _cells(30), _lattice()[0]
    assert delaunay_edges_bipartite(a, b) == delaunay_edges_bipartite(a, b, None, jitter=None, seed=0)


@pytest.mark.parametrize("cap", [None, 10.0, 14.2, 25.0])
def test_jittered_delaunay_on_lattice_matches_oracle(cap):
    xy, _ = _lattice()
    _assert_same(_sym(delaunay_edges(xy, Dmax=cap, jitter=1e-6, seed=0)), o_delaunay(xy, cap))


def test_delaunay_cap_is_inclusive_and_uses_unjittered_lengths():
    xy, _ = _lattice()
    got = _sym(delaunay_edges(xy, Dmax=10.0, jitter=1e-6, seed=0))
    assert got and all(d == 10.0 for d in got.values())  # orthogonal lattice steps kept, diagonals dropped
    for (i, j), d in _sym(delaunay_edges(xy, jitter=1e-6, seed=0)).items():
        assert d == float(np.linalg.norm(xy[i] - xy[j]))


@pytest.mark.parametrize("cap", [None, 20.0])
def test_jittered_delaunay_on_random_points_matches_oracle(cap):
    xy = _cells(80, seed=11)
    _assert_same(_sym(delaunay_edges(xy, Dmax=cap, jitter=1e-6, seed=0)), o_delaunay(xy, cap))


def test_jittered_delaunay_coincident_points_match_oracle_up_to_zero_length_edges():
    xy = np.vstack([_cells(40, seed=5), _cells(40, seed=5)[:6]])  # six coincident pairs
    got = _sym(delaunay_edges(xy, jitter=1e-6, seed=0))
    _assert_same(_positive(got), o_delaunay(xy))
    assert {k for k, v in got.items() if v == 0} <= {(i, 40 + i) for i in range(6)}


@pytest.mark.parametrize("cap", [None, 30.0, 60.0])
@pytest.mark.parametrize("patches_on", ["lattice", "random"])
def test_jittered_bipartite_delaunay_matches_oracle(cap, patches_on):
    patches = _lattice()[0] if patches_on == "lattice" else _cells(50, seed=9)
    cells = np.vstack([_cells(60), _special_cells(_lattice()[0])])
    got = _cross(delaunay_edges_bipartite(cells, patches, Dmax=cap, jitter=1e-6, seed=0))
    _assert_same(_positive(got), o_cross(cells, patches, "delaunay", dmax=cap))


# --------------------------------------------------------------------------- SciPy kNN backend


@pytest.mark.parametrize(("k", "dmax"), [(4, None), (7, None), (7, 20.0), (11, 27.9), (11, None)])
def test_kdtree_knn_matches_oracle_on_lattice_ties(k, dmax):
    xy, _ = _lattice()
    _assert_same(_sym(knn_edges_kdtree(xy, k, Dmax=dmax)), o_knn(xy, k, dmax))


@pytest.mark.parametrize(("k", "dmax"), [(3, None), (10, 25.0)])
def test_kdtree_knn_matches_oracle_on_random_points(k, dmax):
    xy = _cells(90, seed=2)
    _assert_same(_sym(knn_edges_kdtree(xy, k, Dmax=dmax)), o_knn(xy, k, dmax))


def test_kdtree_knn_dmax_is_inclusive():
    xy, _ = _lattice()
    got = _sym(knn_edges_kdtree(xy, 8, Dmax=10.0))
    assert got and set(got.values()) == {10.0}


def test_kdtree_knn_coincident_points_exclude_self_by_identity():
    xy = np.vstack([_cells(30, seed=4), _cells(30, seed=4)[:5], _cells(30, seed=4)[:2]])
    got = _sym(knn_edges_kdtree(xy, 3))
    _assert_same(got, o_knn_identity(xy, 3))
    for i in range(5):  # every coincident partner is a neighbour at distance 0
        assert got[(i, 30 + i)] == 0.0


def test_kdtree_knn_small_inputs():
    assert knn_edges_kdtree(np.zeros((0, 2)), 3) == []
    assert knn_edges_kdtree(np.zeros((1, 2)), 3) == []
    assert knn_edges_kdtree(np.array([[0.0, 0.0], [3.0, 4.0]]), 5) == [(0, 1, 5.0)]
    assert knn_edges_kdtree(np.array([[0.0, 0.0], [3.0, 4.0]]), 0) == []
    with pytest.raises(ValueError, match="non-negative"):
        knn_edges_kdtree(np.array([[0.0, 0.0], [3.0, 4.0]]), -1)


@pytest.mark.parametrize(("k", "dmax"), [(1, None), (4, None), (5, 15.0), (8, None)])
def test_kdtree_bipartite_knn_matches_oracle(k, dmax):
    patches, _ = _lattice()
    cells = np.vstack([_cells(60), _special_cells(patches)[[0, 1, 2, 5, 6, 7, 8]]])  # no cell at a centre
    _assert_same(
        _cross(knn_edges_bipartite_kdtree(cells, patches, k, Dmax=dmax)), o_cross(cells, patches, "knn", k=k, dmax=dmax)
    )


# --------------------------------------------------------------------------- radius and containing ring


@pytest.mark.parametrize("r", [10.0, 15.0, 20.0, 30.0])
def test_radius_bipartite_matches_oracle_including_boundary(r):
    patches, _ = _lattice()
    cells = np.vstack([_cells(60), _special_cells(patches)])
    got = _cross(radius_edges_bipartite(cells, patches, r))
    _assert_same(_positive(got), o_cross(cells, patches, "radius", dmax=r))
    if r in (15.0, 20.0):
        assert got[(65 if r == 15.0 else 66, 10)] == r  # exact boundary distance is kept
    assert {k for k, v in got.items() if v == 0} == {(63, 4), (64, 17)}  # cells at a patch centre keep d = 0


def test_radius_bipartite_random_points_match_oracle():
    cells, patches = _cells(50, seed=21), _cells(40, seed=22)
    _assert_same(_cross(radius_edges_bipartite(cells, patches, 12.0)), o_cross(cells, patches, "radius", dmax=12.0))


def test_containing_ring_matches_oracle():
    patches, grid = _lattice()
    cells = np.vstack([_cells(80, hi=100.0), _special_cells(patches), [[-15.0, 3.0], [88.0, 82.0], [94.0, 77.0]]])
    got = _cross(containing_ring_edges(cells, patches, S))
    want = o_cross(cells, patches, "containing_ring", grid=grid)
    assert set(got) == set(want)
    for key, d in got.items():  # oracle floors a zero distance at 1e-6; Mantpy keeps the true 0.0
        assert d == (0.0 if want[key] == 1e-6 else want[key])
    per_cell = np.bincount([i for i, _ in got], minlength=len(cells))
    assert per_cell.max() <= 9 and per_cell[len(cells) - 3] == 0  # far outside the lattice: no edge


def test_containing_ring_only_uses_present_patches():
    patches, grid = _lattice()
    keep = np.arange(len(patches)) % 3 != 0  # e.g. background removed from the node set
    cells = _cells(50)
    got = _cross(containing_ring_edges(cells, patches[keep], S))
    want = o_cross(cells, patches[keep], "containing_ring", grid=grid[keep])
    assert set(got) == set(want)


def test_containing_ring_rejects_off_lattice_or_duplicate_patches():
    patches, _ = _lattice()
    with pytest.raises(ValueError, match="lattice"):
        containing_ring_edges(_cells(5), patches + 1.0, S)
    with pytest.raises(ValueError, match="duplicate"):
        containing_ring_edges(_cells(5), np.vstack([patches, patches[:1]]), S)
    with pytest.raises(ValueError, match="patch_size"):
        containing_ring_edges(_cells(5), patches, 0.0)
    assert containing_ring_edges(np.zeros((0, 2)), patches, S) == []
    assert containing_ring_edges(_cells(5), np.zeros((0, 2)), S) == []


# --------------------------------------------------------------------------- public API


def test_build_cell_graph_scipy_knn_and_capped_jittered_delaunay():
    patches, _ = _lattice()
    cells = _cells(80)
    a = _adata(cells, patches)
    mt.gr.build_cell_graph(a, k=6, Dmax_CC=18.0, knn_backend="scipy")
    _assert_same(_nx(a, "cell_graph_nx", "cell"), o_knn(cells, 6, 18.0))
    log = a.uns["mantpy"]["gr"]["build_cell_graph"]
    assert log["knn_backend"] == "scipy" and "delaunay_jitter" not in log
    b = _adata(cells, patches)
    mt.gr.build_cell_graph(b, edge_method="delaunay", delaunay_max_length=20.0, delaunay_jitter=1e-6)
    _assert_same(_nx(b, "cell_graph_nx", "cell"), o_delaunay(cells, 20.0))
    log = b.uns["mantpy"]["gr"]["build_cell_graph"]
    assert log["delaunay_max_length"] == 20.0 and log["delaunay_jitter"] == 1e-6 and log["delaunay_seed"] == 0
    c = _adata(cells, patches)
    mt.gr.build_cell_graph(c, k=5)
    assert not {"knn_backend", "delaunay_max_length", "delaunay_jitter", "delaunay_seed"} & set(
        c.uns["mantpy"]["gr"]["build_cell_graph"]
    )


@pytest.mark.parametrize(
    ("kwargs", "oracle"),
    [
        ({"edge_method": "knn", "k": 11, "Dmax": 27.9, "knn_backend": "scipy"}, lambda xy: o_knn(xy, 11, 27.9)),
        ({"edge_method": "knn", "k": 7, "knn_backend": "scipy"}, lambda xy: o_knn(xy, 7)),
        (
            {"edge_method": "delaunay", "delaunay_max_length": 31.4, "delaunay_jitter": 1e-6},
            lambda xy: o_delaunay(xy, 31.4),
        ),
        ({"edge_method": "radius", "r": 10.2}, lambda xy: o_radius(xy, 10.2)),
    ],
)
def test_build_ecm_graph_new_options_match_oracle(kwargs, oracle):
    patches, _ = _lattice()
    a = _adata(_cells(10), patches)
    mt.gr.build_ecm_graph(a, **kwargs)
    _assert_same(_nx(a, "ecm_graph", "ecm"), oracle(patches))


@pytest.mark.parametrize(
    ("kwargs", "rule", "okw"),
    [
        ({"edge_method": "radius", "r": 20.0}, "radius", {"dmax": 20.0}),
        ({"edge_method": "containing_ring", "patch_size": S}, "containing_ring", {}),
        ({"edge_method": "containing_ring"}, "containing_ring", {}),  # patch size inferred
        ({"edge_method": "delaunay", "delaunay_max_length": 60.0, "delaunay_jitter": 1e-6}, "delaunay", {"dmax": 60.0}),
        ({"edge_method": "knn", "k": 8, "knn_backend": "scipy"}, "knn", {"k": 8}),
    ],
)
def test_build_cell_ecm_graph_new_rules_match_oracle(kwargs, rule, okw):
    patches, grid = _lattice()
    cells = np.vstack([_cells(60), _special_cells(patches)[[0, 1, 2, 5, 6, 7, 8]]])
    a = _adata(cells, patches)
    mt.gr.build_cell_graph(a, k=3)
    mt.gr.build_ecm_graph(a, edge_method="grid")
    mt.gr.build_cell_ecm_graph(a, **kwargs)
    got = _nx(a, "cell_ecm_graph", "cell", "ecm", edge_type="cell-ecm")
    want = o_cross(cells, patches, rule, grid=grid, **okw)
    if rule == "containing_ring":
        assert set(got) == set(want)
    else:
        _assert_same(got, want)
    log = a.uns["mantpy"]["gr"]["build_cell_ecm_graph"]
    assert log["edge_method"] == kwargs["edge_method"] and log["n_cell_ecm_edges"] == len(got)
    if rule == "radius":
        assert log["r"] == 20.0
    if rule == "containing_ring":
        assert log["patch_size"] == S


def test_ensure_cell_ecm_graph_forwards_every_new_option():
    patches, grid = _lattice()
    cells = _cells(70)
    a = _adata(cells, patches)
    mt.gr.ensure_cell_ecm_graph(
        a,
        cell_edge_method="radius",
        cell_r=12.0,
        ecm_edge_method="delaunay",
        ecm_delaunay_max_length=25.0,
        cell_ecm_edge_method="containing_ring",
        cell_ecm_patch_size=S,
        delaunay_jitter=1e-6,
    )
    _assert_same(_nx(a, "cell_graph_nx", "cell"), o_radius(cells, 12.0))
    _assert_same(_nx(a, "ecm_graph", "ecm"), o_delaunay(patches, 25.0))
    assert set(_nx(a, "cell_ecm_graph", "cell", "ecm", edge_type="cell-ecm")) == set(
        o_cross(cells, patches, "containing_ring", grid=grid)
    )
    b = _adata(cells, patches)
    mt.gr.ensure_cell_ecm_graph(
        b,
        cell_edge_method="delaunay",
        cell_delaunay_max_length=20.0,
        ecm_edge_method="knn",
        ecm_k=7,
        ecm_Dmax=20.0,
        cell_ecm_edge_method="radius",
        cell_ecm_r=15.0,
        knn_backend="scipy",
        delaunay_jitter=1e-6,
    )
    _assert_same(_nx(b, "cell_graph_nx", "cell"), o_delaunay(cells, 20.0))
    _assert_same(_nx(b, "ecm_graph", "ecm"), o_knn(patches, 7, 20.0))
    _assert_same(
        _nx(b, "cell_ecm_graph", "cell", "ecm", edge_type="cell-ecm"), o_cross(cells, patches, "radius", dmax=15.0)
    )


def test_ensure_validates_new_options_before_popping_graphs():
    a = _adata(_cells(30), _lattice()[0])
    mt.gr.ensure_cell_ecm_graph(a, ecm_edge_method="grid")
    with pytest.raises(ValueError, match="knn_backend"):
        mt.gr.ensure_cell_ecm_graph(
            a, cell_edge_method="delaunay", ecm_edge_method="grid", cell_ecm_edge_method="delaunay", knn_backend="scipy"
        )
    with pytest.raises(ValueError, match="delaunay_jitter"):
        mt.gr.ensure_cell_ecm_graph(a, ecm_edge_method="grid", delaunay_jitter=1e-6)
    with pytest.raises(ValueError, match="cell_ecm_delaunay_max_length"):
        mt.gr.ensure_cell_ecm_graph(a, ecm_edge_method="grid", cell_ecm_delaunay_max_length=30.0)
    assert "ecm_graph" in a.uns and "cell_ecm_graph" in a.uns


def test_cohort_builder_builds_a_mantpy_radius_cell_graph():
    patches, _ = _lattice()
    cells = _cells(60)
    cohort = {"s1": _adata(cells, patches), "s2": _adata(cells[::-1].copy(), patches)}
    result = mt.gr.build_cell_ecm_graphs(
        cohort,
        cell_graph="mantpy",
        cell_edge_method="radius",
        cell_radius=12.0,
        ecm_edge_method="grid",
        cell_ecm_Dmax=15.0,
    )
    _assert_same(_nx(cohort["s1"], "cell_graph_nx", "cell"), o_radius(cells, 12.0))
    assert result.rebuild_kwargs["cell_r"] == 12.0
    copy = _adata(cells, patches)
    mt.gr.ensure_cell_ecm_graph(copy, **result.rebuild_kwargs)
    assert _nx(copy, "cell_graph_nx", "cell") == _nx(cohort["s1"], "cell_graph_nx", "cell")
    same = {"s": _adata(cells, patches)}
    mt.gr.build_cell_ecm_graphs(same, cell_graph="mantpy", cell_edge_method="radius", cell_radius=(0, 12.0))
    assert _nx(same["s"], "cell_graph_nx", "cell") == _nx(cohort["s1"], "cell_graph_nx", "cell")
    with pytest.raises(ValueError, match="cell_radius"):
        mt.gr.build_cell_ecm_graphs(
            {"s": _adata(cells, patches)}, cell_graph="mantpy", cell_edge_method="radius", cell_radius=(2, 12.0)
        )
    with pytest.raises(ValueError, match="cell_radius"):
        mt.gr.build_cell_ecm_graphs(
            {"s": _adata(cells, patches)}, cell_graph="mantpy", cell_edge_method="knn", cell_radius=12.0
        )


def test_cohort_builder_forwards_new_options_and_keeps_default_recipe():
    patches, grid = _lattice()
    cells = _cells(60)
    cohort = {"s": _adata(cells, patches)}
    result = mt.gr.build_cell_ecm_graphs(
        cohort,
        cell_graph="mantpy",
        cell_edge_method="knn",
        cell_k=5,
        cell_Dmax=30.0,
        ecm_edge_method="radius",
        ecm_r=10.5,
        cell_ecm_edge_method="containing_ring",
        cell_ecm_patch_size=S,
        knn_backend="scipy",
    )
    _assert_same(_nx(cohort["s"], "cell_graph_nx", "cell"), o_knn(cells, 5, 30.0))
    _assert_same(_nx(cohort["s"], "ecm_graph", "ecm"), o_radius(patches, 10.5))
    assert {"ecm_r", "cell_ecm_patch_size", "knn_backend"} <= set(result.rebuild_kwargs)
    default = mt.gr.build_cell_ecm_graphs({"s": _adata(cells, patches)}, cell_graph="mantpy")
    assert set(default.rebuild_kwargs) == {
        "cell_k",
        "cell_Dmax",
        "cell_edge_method",
        "ecm_k",
        "ecm_Dmax",
        "ecm_edge_method",
        "ecm_grid_connectivity",
        "cell_ecm_k",
        "cell_ecm_Dmax",
        "cell_ecm_edge_method",
        "edge_features",
    }


def test_build_ecm_graphs_records_new_options_only_when_set():
    patches, _ = _lattice()
    res = mt.gr.build_ecm_graphs(
        {"s": _adata(_cells(5), patches)}, edge_method="delaunay", delaunay_jitter=1e-6, delaunay_max_length=25.0
    )
    assert res.rebuild_kwargs["delaunay_jitter"] == 1e-6 and res.rebuild_kwargs["delaunay_max_length"] == 25.0
    plain = mt.gr.build_ecm_graphs({"s": _adata(_cells(5), patches)}, edge_method="delaunay")
    assert set(plain.rebuild_kwargs) == {
        "k",
        "r",
        "Dmax",
        "edge_method",
        "grid_connectivity",
        "edge_weight",
        "edge_features",
        "key_added",
    }


# --------------------------------------------------------------------------- validation


def _ready():
    a = _adata(_cells(20), _lattice()[0])
    mt.gr.build_cell_graph(a, k=3)
    mt.gr.build_ecm_graph(a, k=3)
    return a


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda a: mt.gr.build_cell_graph(a, edge_method="knn", delaunay_max_length=10.0),
            "delaunay_max_length.*delaunay",
        ),
        (
            lambda a: mt.gr.build_cell_graph(a, edge_method="radius", r=5.0, delaunay_jitter=1e-6),
            "delaunay_jitter.*delaunay",
        ),
        (lambda a: mt.gr.build_cell_graph(a, edge_method="delaunay", delaunay_max_length=0.0), "positive"),
        (lambda a: mt.gr.build_cell_graph(a, edge_method="delaunay", delaunay_max_length=float("nan")), "positive"),
        (lambda a: mt.gr.build_cell_graph(a, edge_method="delaunay", delaunay_jitter=float("inf")), "positive"),
        (lambda a: mt.gr.build_cell_graph(a, edge_method="delaunay", delaunay_jitter=True), "positive"),
        (
            lambda a: mt.gr.build_cell_graph(a, edge_method="delaunay", delaunay_jitter=1e-6, delaunay_seed=-1),
            "delaunay_seed",
        ),
        (
            lambda a: mt.gr.build_cell_graph(a, edge_method="delaunay", delaunay_jitter=1e-6, delaunay_seed=1.5),
            "delaunay_seed",
        ),
        (
            lambda a: mt.gr.build_cell_graph(a, edge_method="delaunay", delaunay_seed=3),
            "delaunay_seed.*delaunay_jitter",
        ),
        (lambda a: mt.gr.build_cell_graph(a, knn_backend="kd_tree"), "knn_backend"),
        (lambda a: mt.gr.build_cell_graph(a, edge_method="delaunay", knn_backend="scipy"), "knn_backend.*knn"),
        (lambda a: mt.gr.build_ecm_graph(a, edge_method="grid", delaunay_max_length=10.0), "delaunay_max_length"),
        (lambda a: mt.gr.build_ecm_graph(a, edge_method="grid", knn_backend="scipy"), "knn_backend"),
        (lambda a: mt.gr.build_cell_ecm_graph(a, edge_method="radius"), "requires the `r` parameter"),
        (lambda a: mt.gr.build_cell_ecm_graph(a, edge_method="radius", r=-1.0), "positive"),
        (lambda a: mt.gr.build_cell_ecm_graph(a, edge_method="radius", r=10.0, Dmax_CE=5.0), "Dmax_CE.*knn"),
        (lambda a: mt.gr.build_cell_ecm_graph(a, edge_method="containing_ring", Dmax_CE=5.0), "Dmax_CE.*knn"),
        (lambda a: mt.gr.build_cell_ecm_graph(a, edge_method="containing_ring", patch_size=-2.0), "patch_size"),
        (lambda a: mt.gr.build_cell_ecm_graph(a, edge_method="knn", delaunay_jitter=1e-6), "delaunay_jitter.*delaunay"),
        (
            lambda a: mt.gr.build_cell_ecm_graph(a, edge_method="radius", r=10.0, knn_backend="scipy"),
            "knn_backend.*knn",
        ),
        (
            lambda a: mt.gr.build_cell_ecm_graph(a, edge_method="grid"),
            "'knn', 'delaunay', 'radius' or 'containing_ring'",
        ),
    ],
)
def test_new_options_are_validated(call, match):
    with pytest.raises(ValueError, match=match):
        call(_ready())


def test_cohort_rejects_cell_delaunay_cap_with_squidpy():
    a = _adata(_cells(20), _lattice()[0])
    with pytest.raises(ValueError, match="cell_delaunay_max_length"):
        mt.gr.build_cell_ecm_graphs(
            {"s": a}, cell_graph="squidpy", cell_edge_method="delaunay", cell_delaunay_max_length=20.0
        )


def test_existing_distance_cap_guards_are_unchanged():
    a = _ready()
    with pytest.raises(ValueError, match="Dmax_CC.*knn"):
        mt.gr.build_cell_graph(a, edge_method="delaunay", Dmax_CC=15.0)
    with pytest.raises(ValueError, match="Dmax_CE.*knn"):
        mt.gr.build_cell_ecm_graph(a, edge_method="delaunay", Dmax_CE=15.0)
    with pytest.raises(ValueError, match="Dmax.*knn"):
        mt.gr.build_ecm_graph(a, edge_method="grid", Dmax=14.0)
