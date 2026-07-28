"""Tests for mt.gr.build_patch_graph + the doubled-angle edge features."""

from __future__ import annotations

import numpy as np
import pytest

import mantpy as mt
from mantpy.gr import EDGE_FEATURE_REGISTRY


def _grid_two_bands():
    """Two sparse patch rows (y=0 and y=10) separated by a background gap (y 1-9), 3 patches each."""
    ys = np.array([0, 0, 0, 10, 10, 10])
    xs = np.array([0, 1, 2, 0, 1, 2])
    coords = np.column_stack([xs, ys]).astype(float)
    fg = np.zeros((11, 3), bool)
    fg[[0, 10], :] = True  # tissue only on the two rows; y=1..9 is background
    return coords, ys, xs, fg


class TestBuildPatchGraph:
    def test_result_keeps_legacy_mapping_and_exposes_shape_properties(self):
        coords, *_ = _grid_two_bands()

        graph = mt.gr.build_patch_graph(coords, k=4)

        assert isinstance(graph, dict)
        assert isinstance(graph, mt.gr.PatchGraphResult)
        assert set(graph) == {"edge_index", "edge_attr", "n_edges", "edge_feature_names"}
        assert graph.n_nodes == len(coords)
        assert graph.n_directed_edges == graph["edge_index"].shape[1]
        assert graph.n_edge_features == graph["edge_attr"].shape[1]
        assert graph.graph_key is None
        assert graph.connectivities_key is None
        assert graph.params["k"] == 4
        assert repr(graph).startswith("PatchGraphResult(n_nodes=6, n_edges=")
        assert "array(" not in repr(graph)

    def test_shapes_and_both_orientations(self):
        coords, ys, xs, _ = _grid_two_bands()
        g = mt.gr.build_patch_graph(coords, k=4)
        assert g["edge_index"].shape == (2, 2 * g["n_edges"])
        assert g["edge_attr"].shape == (2 * g["n_edges"], 3)
        # every undirected edge appears in both directions
        fwd = set(map(tuple, g["edge_index"].T[: g["n_edges"]]))
        rev = {(b, a) for a, b in fwd}
        assert rev == set(map(tuple, g["edge_index"].T[g["n_edges"] :]))

    def test_foreground_prunes_cross_background_edges(self):
        coords, ys, xs, fg = _grid_two_bands()
        without = mt.gr.build_patch_graph(coords, k=4, standardize_edges=False)["n_edges"]
        withfg = mt.gr.build_patch_graph(coords, k=4, pixel_coords=(ys, xs), foreground=fg, standardize_edges=False)[
            "n_edges"
        ]
        assert withfg < without

    def test_default_edge_features_are_geometric(self):
        coords, *_ = _grid_two_bands()
        g = mt.gr.build_patch_graph(coords, k=4)
        assert g["edge_attr"].shape[1] == 3  # log_distance, sin_angle2, cos_angle2

    def test_custom_edge_features(self):
        coords, *_ = _grid_two_bands()
        g = mt.gr.build_patch_graph(coords, k=4, edge_features=["distance"])
        assert g["edge_attr"].shape[1] == 1

    def test_unknown_edge_feature_raises(self):
        coords, *_ = _grid_two_bands()
        with pytest.raises(ValueError, match="unknown edge feature"):
            mt.gr.build_patch_graph(coords, k=4, edge_features=["bogus"])

    def test_standardize_edges_zero_mean_unit_std(self):
        coords, *_ = _grid_two_bands()
        ea = mt.gr.build_patch_graph(coords, k=4, standardize_edges=True)["edge_attr"]
        assert np.allclose(ea.mean(0), 0.0, atol=1e-4)
        assert np.allclose(ea.std(0), 1.0, atol=1e-2)

    def test_foreground_requires_pixel_coords(self):
        coords, ys, xs, fg = _grid_two_bands()
        with pytest.raises(ValueError, match="pixel_coords"):
            mt.gr.build_patch_graph(coords, k=4, foreground=fg)

    def test_k_out_of_range_raises(self):
        coords, *_ = _grid_two_bands()
        with pytest.raises(ValueError, match="expected 1 <= k"):
            mt.gr.build_patch_graph(coords, k=coords.shape[0])


class TestDoubledAngleEdgeFeatures:
    def test_registry_has_doubled_angle(self):
        assert "sin_angle2" in EDGE_FEATURE_REGISTRY
        assert "cos_angle2" in EDGE_FEATURE_REGISTRY

    def test_doubled_angle_values(self):
        pi = np.array([0.0, 0.0])
        pj = np.array([1.0, 1.0])  # theta = pi/4 -> 2*theta = pi/2
        d = float(np.hypot(1.0, 1.0))
        assert EDGE_FEATURE_REGISTRY["sin_angle2"](pi, pj, d)[0] == pytest.approx(1.0, abs=1e-6)
        assert EDGE_FEATURE_REGISTRY["cos_angle2"](pi, pj, d)[0] == pytest.approx(0.0, abs=1e-6)

    def test_doubled_angle_direction_invariant(self):
        pi = np.array([0.0, 0.0])
        pj = np.array([2.0, 1.0])
        d = float(np.hypot(2.0, 1.0))
        fwd = EDGE_FEATURE_REGISTRY["sin_angle2"](pi, pj, d)[0]
        rev = EDGE_FEATURE_REGISTRY["sin_angle2"](pj, pi, d)[0]
        assert fwd == pytest.approx(rev, abs=1e-6)
