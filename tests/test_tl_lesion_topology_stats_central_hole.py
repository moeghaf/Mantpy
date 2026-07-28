"""Tests for the new ``central_hole`` metric group on lesion_topology_stats."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("gudhi")

import mantpy as mt


def _doughnut(r_in: float = 10.0, r_out: float = 30.0, n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0)
    a1 = rng.uniform(0, 2 * np.pi, n)
    inner = np.column_stack([r_in * np.cos(a1), r_in * np.sin(a1)])
    a2 = rng.uniform(0, 2 * np.pi, n)
    outer = np.column_stack([r_out * np.cos(a2), r_out * np.sin(a2)])
    return np.vstack([inner, outer])


def _toy_subgraph(n: int):
    """Minimal connected subgraph with n nodes — only used so basic/degree work."""
    G = nx.cycle_graph(n)
    return G


def test_default_metrics_omit_central_hole():
    """metrics=None must NOT compute central_hole (opt-in only)."""
    coords = _doughnut()
    G = _toy_subgraph(len(coords))
    stats = mt.tl.lesion_topology_stats(G, coords=coords)
    assert "central_hole_area" not in stats
    assert "n_nodes" in stats  # basic group is still in default


def test_central_hole_metric_computes_area():
    coords = _doughnut()
    G = _toy_subgraph(len(coords))
    stats = mt.tl.lesion_topology_stats(
        G,
        coords=coords,
        metrics=["basic", "central_hole"],
        central_hole_radius=40.0,
    )
    assert "central_hole_area" in stats
    assert stats["central_hole_area"] >= 0.0
    # Doughnut should produce a non-trivial interior void.
    assert stats["central_hole_area"] > 0.0


def test_central_hole_requires_coords():
    G = _toy_subgraph(10)
    with pytest.raises(ValueError, match="coords"):
        mt.tl.lesion_topology_stats(G, metrics=["central_hole"], central_hole_radius=10.0)


def test_central_hole_requires_radius():
    coords = _doughnut()
    G = _toy_subgraph(len(coords))
    with pytest.raises(ValueError, match="central_hole_radius"):
        mt.tl.lesion_topology_stats(G, coords=coords, metrics=["central_hole"])


def test_unknown_metric_still_rejected():
    coords = _doughnut()
    G = _toy_subgraph(len(coords))
    with pytest.raises(ValueError, match="Unknown"):
        mt.tl.lesion_topology_stats(G, coords=coords, metrics=["bogus"])


def test_df_forwards_central_hole_radius():
    """lesion_topology_stats_df with metrics=['central_hole', ...] adds the column."""
    coords = _doughnut()
    G = _toy_subgraph(len(coords))
    records_by_sample = {
        "sample_A": [
            {"coords": coords, "subgraph": G, "kept": True, "n_nodes": len(coords)},
        ],
    }
    df = mt.tl.lesion_topology_stats_df(
        records_by_sample,
        metrics=["basic", "central_hole"],
        central_hole_radius=40.0,
        only_kept=True,
    )
    assert "central_hole_area" in df.columns
    assert df["central_hole_area"].iloc[0] > 0.0


def test_df_central_hole_matches_post_hoc_loop():
    """The new in-call central-hole column must match the post-hoc loop the
    notebook used to do (the friction-removal contract)."""
    coords_a = _doughnut(10, 30, 40)
    coords_b = _doughnut(15, 45, 30)
    records_by_sample = {
        "A": [{"coords": coords_a, "subgraph": _toy_subgraph(len(coords_a)), "kept": True, "n_nodes": len(coords_a)}],
        "B": [{"coords": coords_b, "subgraph": _toy_subgraph(len(coords_b)), "kept": True, "n_nodes": len(coords_b)}],
    }

    # New one-shot computation
    df_new = mt.tl.lesion_topology_stats_df(
        records_by_sample,
        metrics=["basic", "central_hole"],
        central_hole_radius=40.0,
        only_kept=True,
    )

    # Old post-hoc loop, exactly as the schistosoma notebook used to do
    df_old = mt.tl.lesion_topology_stats_df(
        records_by_sample,
        metrics=["basic"],
        only_kept=True,
    )
    hole_areas = []
    for _, row in df_old.iterrows():
        rec = records_by_sample[row["sample_id"]][int(row["lesion_id"])]
        area, *_ = mt.tl.lesion_central_void(rec["coords"], radius=40.0)
        hole_areas.append(area)
    df_old["central_hole_area"] = hole_areas

    # Compare bit-for-bit on the new column
    np.testing.assert_array_equal(
        df_new["central_hole_area"].values,
        df_old["central_hole_area"].values,
    )
