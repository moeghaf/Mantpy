"""Existing graph rules still give exactly the edges of Mantpy 1.0.1.

``data/gr_defaults_v1_0_1.json`` was written by Mantpy 1.0.1 (tag v1.0.1, commit
2a9d829) from the inputs built by :func:`_inputs`, calling only the 1.0.1 API. Every
case records the edges, edge lengths and stored weights of each layer and the
``uns['mantpy']['gr']`` provenance log, so new opt-in options must leave all of them
unchanged when they are not used.

Inputs avoid ties on purpose: kNN, Delaunay and radius layers use untied random
coordinates and the grid layers an exact lattice, so the expectations do not depend
on how a neighbour search orders equidistant candidates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import mantpy as mt

GOLDEN = Path(__file__).parent / "data" / "gr_defaults_v1_0_1.json"
_HOLES = {(2, 3), (5, 5), (0, 6), (6, 1)}


def _inputs(kind: str) -> AnnData:
    """40 random cells; 49 ECM patches either random (``"random"``) or on a 7 x 7 lattice of side 10."""
    rng = np.random.default_rng(20261025)
    cells = rng.uniform(0.0, 70.0, size=(40, 2))
    types = rng.choice(np.array(["A", "B", "C"]), size=40)
    if kind == "random":
        patches = rng.uniform(0.0, 70.0, size=(45, 2))
    elif kind == "lattice":
        grid = np.array([(gx, gy) for gy in range(7) for gx in range(7) if (gx, gy) not in _HOLES], dtype=float)
        patches = (grid + 0.5) * 10.0
    else:  # pragma: no cover - test helper misuse
        raise ValueError(kind)
    feats = np.random.default_rng(7).normal(size=(len(patches), 3))
    clusters = np.random.default_rng(8).integers(0, 3, size=len(patches))
    adata = AnnData(obs=pd.DataFrame({"cell_type": types}, index=[f"c{i}" for i in range(len(cells))]))
    adata.obsm["spatial"] = cells
    adata.uns["ecm_patches"] = pd.DataFrame(
        {
            "x": patches[:, 0],
            "y": patches[:, 1],
            "ecm_cluster": clusters,
            **{f"feat_{i}": feats[:, i] for i in range(3)},
        }
    )
    return adata


def _cell_layers(adata: AnnData, method: str, **kw) -> None:
    mt.gr.build_cell_graph(adata, edge_method=method, **kw)


def _both_then_cross(adata: AnnData, **kw) -> None:
    mt.gr.build_cell_graph(adata, k=3)
    mt.gr.build_ecm_graph(adata, k=3)
    mt.gr.build_cell_ecm_graph(adata, **kw)


CASES = {
    "cell_knn5": ("random", lambda a: mt.gr.build_cell_graph(a, k=5)),
    "cell_knn4_dmax12": ("random", lambda a: mt.gr.build_cell_graph(a, k=4, Dmax_CC=12.0)),
    "cell_delaunay": ("random", lambda a: mt.gr.build_cell_graph(a, edge_method="delaunay")),
    "cell_radius11": ("random", lambda a: mt.gr.build_cell_graph(a, edge_method="radius", r=11.0)),
    "cell_knn3_uniform": ("random", lambda a: mt.gr.build_cell_graph(a, k=3, edge_weight="uniform")),
    "ecm_knn5": ("random", lambda a: mt.gr.build_ecm_graph(a, k=5)),
    "ecm_knn6_dmax14": ("random", lambda a: mt.gr.build_ecm_graph(a, k=6, Dmax=14.0)),
    "ecm_delaunay": ("random", lambda a: mt.gr.build_ecm_graph(a, edge_method="delaunay")),
    "ecm_radius13": ("random", lambda a: mt.gr.build_ecm_graph(a, edge_method="radius", r=13.0)),
    "ecm_grid8": ("lattice", lambda a: mt.gr.build_ecm_graph(a, edge_method="grid")),
    "ecm_grid4": ("lattice", lambda a: mt.gr.build_ecm_graph(a, edge_method="grid", grid_connectivity=4)),
    "cross_default_delaunay": ("random", lambda a: _both_then_cross(a)),
    "cross_knn3": ("random", lambda a: _both_then_cross(a, edge_method="knn", k=3)),
    "cross_knn4_dmax9": ("random", lambda a: _both_then_cross(a, edge_method="knn", k=4, Dmax_CE=9.0)),
    "cross_lattice_knn4": ("lattice", lambda a: _both_then_cross(a, edge_method="knn", k=4)),
    "ensure_default": ("random", lambda a: mt.gr.ensure_cell_ecm_graph(a)),
    "ensure_delaunay_delaunay": (
        "random",
        lambda a: mt.gr.ensure_cell_ecm_graph(
            a, cell_edge_method="delaunay", ecm_edge_method="delaunay", cell_ecm_edge_method="delaunay"
        ),
    ),
    "ensure_lattice_grid": (
        "lattice",
        lambda a: mt.gr.ensure_cell_ecm_graph(
            a,
            cell_edge_method="knn",
            cell_k=4,
            cell_Dmax=15.0,
            ecm_edge_method="grid",
            ecm_grid_connectivity=4,
            cell_ecm_edge_method="knn",
            cell_ecm_k=4,
            cell_ecm_Dmax=14.0,
        ),
    ),
}


def _digest(rows) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _layer(rows: list[tuple]) -> dict:
    """Edge count, exact edge-list digest and a digest of lengths/weights to 10 significant digits.

    Rounding to 10 digits keeps the value digest independent of last-bit differences
    between platforms; any real change in a length or weight still alters it.
    """
    rows = sorted(rows)
    return {
        "n": len(rows),
        "edges": _digest([[r[0], r[1]] for r in rows]),
        "values": _digest([[format(v, ".10g") for v in r[2:]] for r in rows]),
    }


def _nx_layer(graph, edge_type: str | None = None) -> dict:
    rows = []
    for u, v, d in graph.edges(data=True):
        if edge_type is not None and d.get("edge_type") != edge_type:
            continue
        a, b = sorted((str(u), str(v)))
        rows.append((a, b, float(d["distance"]), float(d["weight"])))
    return _layer(rows)


def _snapshot(adata: AnnData) -> dict:
    out: dict = {}
    if "cell_graph" in adata.obsp:
        m = adata.obsp["cell_graph"].tocoo()
        out["cell_obsp"] = _layer([(int(i), int(j), float(w)) for i, j, w in zip(m.row, m.col, m.data, strict=True)])
    if "cell_graph_nx" in adata.uns:
        out["cell_nx"] = _nx_layer(adata.uns["cell_graph_nx"])
    if "ecm_graph" in adata.uns:
        out["ecm_nx"] = _nx_layer(adata.uns["ecm_graph"])
    if "cell_ecm_graph" in adata.uns:
        g = adata.uns["cell_ecm_graph"]
        out["cross_nx"] = _nx_layer(g, "cell-ecm")
        out["joint_counts"] = [g.number_of_nodes(), g.number_of_edges()]
    out["log"] = json.loads(json.dumps(adata.uns.get("mantpy", {}).get("gr", {}), default=str))
    return out


def _cohort_snapshot() -> dict:
    adata = _inputs("random")
    result = mt.gr.build_cell_ecm_graphs(
        {"s": adata}, cell_graph="mantpy", cell_k=4, ecm_edge_method="delaunay", cell_ecm_edge_method="delaunay"
    )
    snap = _snapshot(adata)
    snap["rebuild_kwargs"] = json.loads(json.dumps(result.rebuild_kwargs, default=str))
    s = result.summaries[0]
    snap["summary"] = [s.n_nodes, s.n_cells, s.n_ecm, s.cell_cell_edges, s.ecm_ecm_edges, s.cell_ecm_edges]
    return snap


def compute_all() -> dict:
    """Snapshots of every case (used to write the golden file with Mantpy 1.0.1)."""
    out = {}
    for name, (kind, call) in CASES.items():
        adata = _inputs(kind)
        call(adata)
        out[name] = _snapshot(adata)
    out["cohort_mantpy"] = _cohort_snapshot()
    return out


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", [*CASES, "cohort_mantpy"])
def test_existing_rules_match_mantpy_1_0_1(golden, name):
    if name == "cohort_mantpy":
        got = _cohort_snapshot()
    else:
        kind, call = CASES[name]
        adata = _inputs(kind)
        call(adata)
        got = _snapshot(adata)
    assert got == golden[name]
