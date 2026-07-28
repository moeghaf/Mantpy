"""Tests for mantpy.gr."""

from __future__ import annotations

import importlib.util

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

import mantpy as mt
from mantpy._constants import (
    CELL_ECM_GRAPH_KEY,
    CELL_GRAPH_KEY,
    ECM_GRAPH_KEY,
    ECM_PATCHES_KEY,
    EDGE_TYPE_CC,
    EDGE_TYPE_CE,
    EDGE_TYPE_EE,
    MANTPY_UNS_KEY,
    NODE_TYPE_CELL,
    NODE_TYPE_ECM,
)

_HAS_PYG = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None


class TestBuildCellGraph:
    def test_writes_obsp(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        assert CELL_GRAPH_KEY in adata_with_patches.obsp

    def test_adj_is_sparse(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        assert sp.issparse(adata_with_patches.obsp[CELL_GRAPH_KEY])

    def test_adj_shape(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        n = adata_with_patches.n_obs
        assert adata_with_patches.obsp[CELL_GRAPH_KEY].shape == (n, n)

    def test_adj_symmetric(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        A = adata_with_patches.obsp[CELL_GRAPH_KEY]
        diff = (A - A.T).data
        assert len(diff) == 0 or np.allclose(diff, 0, atol=1e-5)

    def test_nx_graph_stored(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        assert CELL_GRAPH_KEY + "_nx" in adata_with_patches.uns
        G = adata_with_patches.uns[CELL_GRAPH_KEY + "_nx"]
        assert isinstance(G, nx.Graph)

    def test_nx_node_types(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        G = adata_with_patches.uns[CELL_GRAPH_KEY + "_nx"]
        node_types = set(nx.get_node_attributes(G, "node_type").values())
        assert node_types == {NODE_TYPE_CELL}

    def test_edge_type_attr(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        G = adata_with_patches.uns[CELL_GRAPH_KEY + "_nx"]
        for _, _, d in G.edges(data=True):
            assert d["edge_type"] == EDGE_TYPE_CC

    def test_dmax_none_no_pruning(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3, Dmax_CC=None)
        G = adata_with_patches.uns[CELL_GRAPH_KEY + "_nx"]
        assert G.number_of_edges() > 0

    def test_dmax_prunes_edges(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, Dmax_CC=1.0, k=10)
        G = adata_with_patches.uns[CELL_GRAPH_KEY + "_nx"]
        for _, _, d in G.edges(data=True):
            assert d["distance"] <= 1.0 + 1e-9

    def test_edge_weight_distance(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3, edge_weight="distance")
        G = adata_with_patches.uns[CELL_GRAPH_KEY + "_nx"]
        for _, _, d in G.edges(data=True):
            assert d["weight"] == pytest.approx(d["distance"])

    def test_edge_weight_uniform(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3, edge_weight="uniform")
        G = adata_with_patches.uns[CELL_GRAPH_KEY + "_nx"]
        for _, _, d in G.edges(data=True):
            assert d["weight"] == 1.0

    def test_edge_weight_custom_callable(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3, edge_weight=lambda i, j, d: d * 2)
        G = adata_with_patches.uns[CELL_GRAPH_KEY + "_nx"]
        for _, _, d in G.edges(data=True):
            assert d["weight"] == pytest.approx(d["distance"] * 2)

    def test_delaunay_method(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, edge_method="delaunay")
        assert CELL_GRAPH_KEY in adata_with_patches.obsp

    def test_radius_method(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, edge_method="radius", r=100.0)
        assert CELL_GRAPH_KEY in adata_with_patches.obsp

    def test_inplace_false(self, adata_with_patches):
        result = mt.gr.build_cell_graph(adata_with_patches, k=3, inplace=False)
        assert result is not None
        assert CELL_GRAPH_KEY not in adata_with_patches.obsp

    def test_params_logged(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        assert "build_cell_graph" in adata_with_patches.uns[MANTPY_UNS_KEY]["gr"]

    def test_missing_spatial_raises(self, adata_basic):
        del adata_basic.obsm["spatial"]
        with pytest.raises(ValueError, match="spatial"):
            mt.gr.build_cell_graph(adata_basic)


class TestBuildEcmGraph:
    def test_writes_uns_graph(self, adata_with_patches):
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        assert ECM_GRAPH_KEY in adata_with_patches.uns
        assert isinstance(adata_with_patches.uns[ECM_GRAPH_KEY], nx.Graph)

    def test_ecm_node_count(self, adata_with_patches):
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        G = adata_with_patches.uns[ECM_GRAPH_KEY]
        n_ecm = len(adata_with_patches.uns[ECM_PATCHES_KEY])
        assert G.number_of_nodes() == n_ecm

    def test_node_types(self, adata_with_patches):
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        G = adata_with_patches.uns[ECM_GRAPH_KEY]
        node_types = set(nx.get_node_attributes(G, "node_type").values())
        assert node_types == {NODE_TYPE_ECM}

    def test_ecm_cluster_attr(self, adata_with_patches):
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        G = adata_with_patches.uns[ECM_GRAPH_KEY]
        for _, d in G.nodes(data=True):
            assert "ecm_cluster" in d

    def test_ecm_features_attr(self, adata_with_patches):
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        G = adata_with_patches.uns[ECM_GRAPH_KEY]
        for _, d in G.nodes(data=True):
            assert "ecm_features" in d

    def test_missing_patches_raises(self, adata_basic):
        with pytest.raises(ValueError, match="ecm_patches"):
            mt.gr.build_ecm_graph(adata_basic)

    def test_params_logged(self, adata_with_patches):
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        assert "build_ecm_graph" in adata_with_patches.uns[MANTPY_UNS_KEY]["gr"]

    def test_grid_method_builds(self, adata_with_patches):
        mt.gr.build_ecm_graph(adata_with_patches, edge_method="grid")
        G = adata_with_patches.uns[ECM_GRAPH_KEY]
        assert isinstance(G, nx.Graph)
        assert G.number_of_edges() > 0

    def test_grid_connectivity_logged(self, adata_with_patches):
        mt.gr.build_ecm_graph(adata_with_patches, edge_method="grid", grid_connectivity=4)
        log = adata_with_patches.uns[MANTPY_UNS_KEY]["gr"]["build_ecm_graph"]
        assert log["grid_connectivity"] == 4


class TestDmaxKnnOnly:
    """Distance caps are a kNN-only knob; non-kNN methods must reject them."""

    def test_cell_graph_delaunay_with_dmax_raises(self, adata_with_patches):
        with pytest.raises(ValueError, match="Dmax_CC.*knn"):
            mt.gr.build_cell_graph(adata_with_patches, edge_method="delaunay", Dmax_CC=15.0)

    def test_ecm_graph_grid_with_dmax_raises(self, adata_with_patches):
        with pytest.raises(ValueError, match="Dmax.*knn"):
            mt.gr.build_ecm_graph(adata_with_patches, edge_method="grid", Dmax=14.0)

    def test_cell_ecm_graph_delaunay_with_dmax_raises(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        with pytest.raises(ValueError, match="Dmax_CE.*knn"):
            mt.gr.build_cell_ecm_graph(adata_with_patches, edge_method="delaunay", Dmax_CE=15.0)

    def test_ensure_validates_before_popping_graphs(self, adata_with_patches):
        # Build a valid graph first, then a misuse must raise WITHOUT wiping it.
        mt.gr.ensure_cell_ecm_graph(
            adata_with_patches, cell_edge_method="delaunay", ecm_edge_method="grid", cell_ecm_edge_method="delaunay"
        )
        assert ECM_GRAPH_KEY in adata_with_patches.uns
        with pytest.raises(ValueError, match="ecm_Dmax.*knn"):
            mt.gr.ensure_cell_ecm_graph(adata_with_patches, ecm_edge_method="grid", ecm_Dmax=14.0)
        # Graph from the first (valid) call survives the rejected rebuild.
        assert ECM_GRAPH_KEY in adata_with_patches.uns

    def test_knn_with_dmax_still_allowed(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, edge_method="knn", Dmax_CC=15.0, k=3)
        assert CELL_GRAPH_KEY in adata_with_patches.obsp


class TestGridEdges:
    """grid_edges on a synthetic regular lattice."""

    @staticmethod
    def _square_grid(n=4, step=10.0):
        xs, ys = np.meshgrid(np.arange(n) * step, np.arange(n) * step)
        return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)

    def test_moore_8_interior_degree(self):
        from mantpy._core._edges import grid_edges

        coords = self._square_grid(n=4, step=10.0)
        edges = grid_edges(coords, connectivity=8)
        G = nx.Graph()
        G.add_nodes_from(range(len(coords)))
        G.add_edges_from((i, j) for i, j, _ in edges)
        # Interior node of a 4x4 grid (e.g. index for (1,1)) has all 8 neighbours.
        interior = [v for v in G.nodes if G.degree(v) == 8]
        assert len(interior) == 4  # the 2x2 interior of a 4x4 grid

    def test_von_neumann_4_interior_degree(self):
        from mantpy._core._edges import grid_edges

        coords = self._square_grid(n=4, step=10.0)
        edges = grid_edges(coords, connectivity=4)
        G = nx.Graph()
        G.add_nodes_from(range(len(coords)))
        G.add_edges_from((i, j) for i, j, _ in edges)
        interior = [v for v in G.nodes if G.degree(v) == 4]
        assert len(interior) == 4

    def test_no_long_edges_across_hole(self):
        from mantpy._core._edges import grid_edges

        coords = self._square_grid(n=5, step=10.0)
        # Drop the centre node to make a hole; no edge should bridge the gap.
        centre = np.array([20.0, 20.0])
        coords = coords[~np.all(coords == centre, axis=1)]
        edges = grid_edges(coords, connectivity=8)
        max_d = max(d for _, _, d in edges)
        assert max_d <= 10.0 * 2**0.5 + 1e-6  # never exceeds the diagonal step

    def test_invalid_connectivity_raises(self):
        from mantpy._core._edges import grid_edges

        with pytest.raises(ValueError, match="connectivity"):
            grid_edges(self._square_grid(), connectivity=6)

    def test_single_node_returns_empty(self):
        from mantpy._core._edges import grid_edges

        assert grid_edges(np.array([[0.0, 0.0]]), connectivity=8) == []

    def test_two_nodes_returns_one_edge(self):
        from mantpy._core._edges import grid_edges

        edges = grid_edges(np.array([[0.0, 0.0], [10.0, 0.0]]), connectivity=8)
        assert len(edges) == 1
        assert edges[0][2] == pytest.approx(10.0)


class TestBuildCellEcmGraph:
    def test_writes_unified_graph(self, adata_with_graphs):
        assert CELL_ECM_GRAPH_KEY in adata_with_graphs.uns
        assert isinstance(adata_with_graphs.uns[CELL_ECM_GRAPH_KEY], nx.Graph)

    def test_has_both_node_types(self, adata_with_graphs):
        G = adata_with_graphs.uns[CELL_ECM_GRAPH_KEY]
        node_types = set(nx.get_node_attributes(G, "node_type").values())
        assert NODE_TYPE_CELL in node_types
        assert NODE_TYPE_ECM in node_types

    def test_has_all_edge_types(self, adata_with_graphs):
        G = adata_with_graphs.uns[CELL_ECM_GRAPH_KEY]
        edge_types = set(nx.get_edge_attributes(G, "edge_type").values())
        assert EDGE_TYPE_CC in edge_types
        assert EDGE_TYPE_EE in edge_types

    def test_cross_edges_within_dmax(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        # Dmax_CE is a kNN-only knob, so the cross edges must be built with knn.
        mt.gr.build_cell_ecm_graph(adata_with_patches, edge_method="knn", Dmax_CE=20.0, k=3)
        G = adata_with_patches.uns[CELL_ECM_GRAPH_KEY]
        for _, _, d in G.edges(data=True):
            if d.get("edge_type") == EDGE_TYPE_CE:
                assert d["distance"] <= 20.0 + 1e-9

    def test_dmax_ce_none_allows_all(self, adata_with_patches):
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        mt.gr.build_cell_ecm_graph(adata_with_patches, Dmax_CE=None, k=3)
        G = adata_with_patches.uns[CELL_ECM_GRAPH_KEY]
        ce_edges = [e for e in G.edges(data=True) if e[2].get("edge_type") == EDGE_TYPE_CE]
        assert len(ce_edges) > 0

    def test_ecm_only_mode(self, adata_with_patches):
        """build_cell_ecm_graph with only ECM graph → ECM-only graph, no error."""
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        mt.gr.build_cell_ecm_graph(adata_with_patches)
        G = adata_with_patches.uns[CELL_ECM_GRAPH_KEY]
        node_types = set(nx.get_node_attributes(G, "node_type").values())
        assert node_types == {NODE_TYPE_ECM}

    def test_ecm_only_mode_logged(self, adata_with_patches):
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        mt.gr.build_cell_ecm_graph(adata_with_patches)
        log = adata_with_patches.uns[MANTPY_UNS_KEY]["gr"]["build_cell_ecm_graph"]
        assert log["mode"] == "ecm-only"

    def test_cell_only_mode(self, adata_with_patches):
        """build_cell_ecm_graph with only cell graph → cell-only graph, no error."""
        mt.gr.build_cell_graph(adata_with_patches, k=3)
        mt.gr.build_cell_ecm_graph(adata_with_patches)
        G = adata_with_patches.uns[CELL_ECM_GRAPH_KEY]
        node_types = set(nx.get_node_attributes(G, "node_type").values())
        assert node_types == {NODE_TYPE_CELL}

    def test_neither_raises(self, adata_with_patches):
        with pytest.raises(ValueError, match="Neither"):
            mt.gr.build_cell_ecm_graph(adata_with_patches)

    def test_params_logged(self, adata_with_graphs):
        assert "build_cell_ecm_graph" in adata_with_graphs.uns[MANTPY_UNS_KEY]["gr"]


@pytest.mark.skipif(not _HAS_PYG, reason="requires mantpy[gnn]")
class TestToPyg:
    def test_missing_graph_raises(self, adata_with_graphs):
        with pytest.raises(ValueError, match="bad_key"):
            mt.gr.to_pyg(adata_with_graphs, graph_key="bad_key")

    def test_to_pyg_attaches_graph_feat_and_y(self, adata_with_graphs):
        """graph_feat + y are attached when uns/obs fields are present."""
        adata = adata_with_graphs.copy()
        adata.uns["islet_graph_feat"] = np.ones(11, dtype=np.float32)
        adata.obs["islet_label"] = 0  # binary label
        data = mt.gr.to_pyg(adata, graph_key=ECM_GRAPH_KEY, label_key="islet_label")
        assert hasattr(data, "graph_feat")
        assert data.graph_feat.shape == (1, 11)
        assert hasattr(data, "y")
        assert int(data.y.item()) == 0

    def test_to_pyg_ignores_y_centroid_column(self, adata_with_graphs):
        """obs['y'] is the y-centroid from read_imc, never a class label."""
        adata = adata_with_graphs.copy()
        adata.obs["y"] = np.linspace(10.5, 99.5, adata.n_obs)
        data = mt.gr.to_pyg(adata, graph_key=ECM_GRAPH_KEY)
        assert not hasattr(data, "y") or data.y is None

    def test_to_pyg_rejects_non_constant_label(self, adata_with_graphs):
        adata = adata_with_graphs.copy()
        adata.obs["islet_label"] = np.arange(adata.n_obs)
        with pytest.raises(ValueError, match="single per-graph label"):
            mt.gr.to_pyg(adata, graph_key=ECM_GRAPH_KEY, label_key="islet_label")

    def test_to_pyg_rejects_missing_label_key(self, adata_with_graphs):
        with pytest.raises(KeyError, match="absent_label"):
            mt.gr.to_pyg(
                adata_with_graphs, graph_key=ECM_GRAPH_KEY, label_key="absent_label"
            )

    def test_to_pyg_works_without_optional_fields(self, adata_with_graphs):
        """to_pyg works without islet_graph_feat — backward compat."""
        adata = adata_with_graphs.copy()
        adata.uns.pop("islet_graph_feat", None)
        data = mt.gr.to_pyg(adata, graph_key=ECM_GRAPH_KEY)
        assert not hasattr(data, "graph_feat")
        assert data.x.shape[1] > 0  # features intact


class TestEdgeFeatures:
    def test_edge_feat_stored_on_graph(self, adata_with_patches):
        mt.gr.build_ecm_graph(
            adata_with_patches,
            k=3,
            edge_features=["log_distance", "norm_angle"],
        )
        G = adata_with_patches.uns[ECM_GRAPH_KEY]
        for _, _, d in G.edges(data=True):
            assert "feat_fwd" in d
            assert "feat_rev" in d
            assert len(d["feat_fwd"]) == 2  # log_dist + angle

    def test_edge_feat_shape_correct(self, adata_with_patches):
        mt.gr.build_ecm_graph(
            adata_with_patches,
            k=3,
            edge_features=["log_distance"],
        )
        G = adata_with_patches.uns[ECM_GRAPH_KEY]
        for _, _, d in G.edges(data=True):
            assert len(d["feat_fwd"]) == 1

    def test_custom_edge_callable(self, adata_with_patches):
        def sq_dist(pi, pj, d):
            return np.array([d**2], dtype=np.float32)

        mt.gr.build_ecm_graph(adata_with_patches, k=3, edge_features=[sq_dist])
        G = adata_with_patches.uns[ECM_GRAPH_KEY]
        for _, _, edata in G.edges(data=True):
            assert edata["feat_fwd"][0] == pytest.approx(edata["distance"] ** 2)

    def test_unknown_edge_feat_raises(self, adata_with_patches):
        with pytest.raises(ValueError, match="Unknown edge feature"):
            mt.gr.build_ecm_graph(adata_with_patches, k=3, edge_features=["bogus"])

    def test_no_edge_features_by_default(self, adata_with_patches):
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        G = adata_with_patches.uns[ECM_GRAPH_KEY]
        for _, _, d in G.edges(data=True):
            assert "feat_fwd" not in d


class TestNewEdgeFeatures:
    """Tests for the geometric edge features added in the 2025 refresh."""

    @pytest.mark.parametrize(
        "name",
        [
            "dx_norm",
            "dy_norm",
            "sin_angle",
            "cos_angle",
            "inv_distance",
            "manhattan_distance",
        ],
    )
    def test_returns_1d_float32(self, name):
        from mantpy.gr import EDGE_FEATURE_REGISTRY

        fn = EDGE_FEATURE_REGISTRY[name]
        out = fn(np.array([0.0, 0.0]), np.array([3.0, 4.0]), 5.0)
        assert out.shape == (1,)
        assert out.dtype == np.float32

    def test_dx_dy_are_unit_vector_components(self):
        from mantpy.gr import EDGE_FEATURE_REGISTRY

        dx = EDGE_FEATURE_REGISTRY["dx_norm"](np.array([0.0, 0.0]), np.array([3.0, 4.0]), 5.0)
        dy = EDGE_FEATURE_REGISTRY["dy_norm"](np.array([0.0, 0.0]), np.array([3.0, 4.0]), 5.0)
        assert dx[0] == pytest.approx(0.6, abs=1e-6)
        assert dy[0] == pytest.approx(0.8, abs=1e-6)
        assert dx[0] ** 2 + dy[0] ** 2 == pytest.approx(1.0, abs=1e-6)

    def test_sin_cos_angle_consistent(self):
        from mantpy.gr import EDGE_FEATURE_REGISTRY

        s = EDGE_FEATURE_REGISTRY["sin_angle"](np.array([0.0, 0.0]), np.array([3.0, 4.0]), 5.0)
        c = EDGE_FEATURE_REGISTRY["cos_angle"](np.array([0.0, 0.0]), np.array([3.0, 4.0]), 5.0)
        assert s[0] ** 2 + c[0] ** 2 == pytest.approx(1.0, abs=1e-5)

    def test_inv_distance_value(self):
        from mantpy.gr import EDGE_FEATURE_REGISTRY

        out = EDGE_FEATURE_REGISTRY["inv_distance"](np.array([0.0, 0.0]), np.array([3.0, 4.0]), 5.0)
        assert out[0] == pytest.approx(0.2, abs=1e-6)

    def test_manhattan_distance_value(self):
        from mantpy.gr import EDGE_FEATURE_REGISTRY

        out = EDGE_FEATURE_REGISTRY["manhattan_distance"](
            np.array([0.0, 0.0]),
            np.array([3.0, 4.0]),
            5.0,
        )
        assert out[0] == pytest.approx(7.0)

    def test_exp_neg_dist_factory(self):
        from mantpy.gr import exp_neg_dist

        fn = exp_neg_dist(sigma=10.0)
        out = fn(np.array([0.0, 0.0]), np.array([3.0, 4.0]), 5.0)
        assert out.shape == (1,)
        assert out[0] == pytest.approx(float(np.exp(-0.5)), abs=1e-6)

    def test_new_features_flow_through_build_ecm_graph(self, adata_with_patches):
        mt.gr.build_ecm_graph(
            adata_with_patches,
            k=3,
            edge_features=["log_distance", "sin_angle", "cos_angle", "inv_distance"],
        )
        G = adata_with_patches.uns[ECM_GRAPH_KEY]
        for _, _, d in G.edges(data=True):
            assert "feat_fwd" in d
            assert len(d["feat_fwd"]) == 4


class TestToPygGeneralExport:
    def test_xy_only_no_edge_feat(self, adata_with_patches):
        pytest.importorskip("torch_geometric")
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        data = mt.gr.to_pyg(adata_with_patches, graph_key=ECM_GRAPH_KEY)
        assert data.x.shape[1] >= 2  # at least x, y
        assert not hasattr(data, "edge_attr") or data.edge_attr is None

    def test_includes_ecm_features(self, adata_with_patches):
        pytest.importorskip("torch_geometric")
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        data = mt.gr.to_pyg(adata_with_patches, graph_key=ECM_GRAPH_KEY)
        n_node_feats_expected = 2 + len(
            [c for c in adata_with_patches.uns[ECM_PATCHES_KEY].columns if c.startswith("feat_")]
        )
        assert data.x.shape[1] == n_node_feats_expected

    def test_topology_in_node_feats(self, adata_with_patches):
        pytest.importorskip("torch_geometric")
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        n_before_topo = mt.gr.to_pyg(adata_with_patches, graph_key=ECM_GRAPH_KEY).x.shape[1]
        topology_attrs = (
            "topo_degree",
            "topo_clustering",
            "topo_betweenness",
            "topo_pagerank",
        )
        graph = adata_with_patches.uns[ECM_GRAPH_KEY]
        for node in graph.nodes:
            for value, attr in enumerate(topology_attrs, start=1):
                graph.nodes[node][attr] = float(value)
        n_after_topo = mt.gr.to_pyg(adata_with_patches, graph_key=ECM_GRAPH_KEY).x.shape[1]
        assert n_after_topo == n_before_topo + len(topology_attrs)

    def test_edge_attr_present_when_edge_features_set(self, adata_with_patches):
        pytest.importorskip("torch_geometric")
        mt.gr.build_ecm_graph(
            adata_with_patches,
            k=3,
            edge_features=["log_distance", "norm_angle"],
        )
        data = mt.gr.to_pyg(adata_with_patches, graph_key=ECM_GRAPH_KEY)
        assert data.edge_attr is not None
        assert data.edge_attr.shape[1] == 2
        assert data.edge_attr.shape[0] == data.edge_index.shape[1]


class TestMantpyDataset:
    def test_create_and_getitem(self, adata_with_patches):
        ds = mt.MantpyDataset({"s1": adata_with_patches})
        assert ds["s1"] is adata_with_patches

    def test_len_and_iter(self, adata_with_patches):
        ds = mt.MantpyDataset({"s1": adata_with_patches, "s2": adata_with_patches})
        assert len(ds) == 2
        assert set(ds) == {"s1", "s2"}

    def test_apply_modifies_adatas(self, adata_with_patches):
        ds = mt.MantpyDataset({"s1": adata_with_patches})
        ds.apply(lambda a: mt.gr.build_ecm_graph(a, k=3))
        assert ECM_GRAPH_KEY in adata_with_patches.uns

    def test_graphs_returns_dict(self, adata_with_patches):
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        ds = mt.MantpyDataset({"s1": adata_with_patches})
        gs = ds.graphs(key=ECM_GRAPH_KEY)
        assert "s1" in gs
        assert isinstance(gs["s1"], nx.Graph)

    def test_graphs_skips_missing(self, adata_with_patches):
        ds = mt.MantpyDataset({"s1": adata_with_patches})
        assert ds.graphs(key=ECM_GRAPH_KEY) == {}

    def test_pyg_dataset(self, adata_with_patches):
        pytest.importorskip("torch_geometric")
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        ds = mt.MantpyDataset({"s1": adata_with_patches})
        pyg_list = ds.pyg_dataset(graph_key=ECM_GRAPH_KEY)
        assert len(pyg_list) == 1
        assert pyg_list[0].sample_id == "s1"

    def test_repr(self, adata_with_patches):
        ds = mt.MantpyDataset({"s1": adata_with_patches})
        assert "1 ROIs" in repr(ds)


class TestCrossEdgeFeatBugFix:
    """Custom edge_feat_fn callables passed to merge_cell_ecm_graphs must
    accept reverse=False/True. The previous fallback silently swapped
    (ci, ei) indices, which produced wrong values when Na != Nb (the
    normal case). Now the call raises TypeError so users see the issue.
    """

    def test_callable_without_reverse_kwarg_raises(self):
        from mantpy._core._edges import merge_cell_ecm_graphs

        cell_g = nx.Graph()
        cell_g.add_node("cell_0", node_type=NODE_TYPE_CELL, x=0.0, y=0.0)
        cell_g.add_node("cell_1", node_type=NODE_TYPE_CELL, x=1.0, y=0.0)
        ecm_g = nx.Graph()
        ecm_g.add_node("ecm_0", node_type=NODE_TYPE_ECM, x=0.5, y=0.5)
        ecm_g.add_node("ecm_1", node_type=NODE_TYPE_ECM, x=1.5, y=0.5)
        ecm_g.add_node("ecm_2", node_type=NODE_TYPE_ECM, x=2.5, y=0.5)

        cell_coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
        ecm_coords = np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]], dtype=float)

        def bad_fn(ci, ei, d):  # missing reverse kwarg
            return np.array([float(d)], dtype=np.float32)

        with pytest.raises(TypeError):
            merge_cell_ecm_graphs(
                cell_g,
                ecm_g,
                cell_coords,
                ecm_coords,
                Dmax_CE=None,
                k=2,
                edge_feat_fn=bad_fn,
                edge_method="knn",
            )

    def test_callable_with_reverse_kwarg_succeeds(self):
        from mantpy._core._edges import merge_cell_ecm_graphs

        cell_g = nx.Graph()
        cell_g.add_node("cell_0", node_type=NODE_TYPE_CELL, x=0.0, y=0.0)
        ecm_g = nx.Graph()
        ecm_g.add_node("ecm_0", node_type=NODE_TYPE_ECM, x=0.5, y=0.5)
        ecm_g.add_node("ecm_1", node_type=NODE_TYPE_ECM, x=1.5, y=0.5)

        cell_coords = np.array([[0.0, 0.0]], dtype=float)
        ecm_coords = np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float)

        def good_fn(ci, ei, d, reverse=False):
            return np.array([float(d), 1.0 if reverse else 0.0], dtype=np.float32)

        merged = merge_cell_ecm_graphs(
            cell_g,
            ecm_g,
            cell_coords,
            ecm_coords,
            Dmax_CE=None,
            k=2,
            edge_feat_fn=good_fn,
            edge_method="knn",
        )
        # Cross edges should carry both fwd (reverse=False) and rev (reverse=True).
        ce_edges = [(u, v, d) for u, v, d in merged.edges(data=True) if d.get("edge_type") == EDGE_TYPE_CE]
        assert ce_edges
        u, v, d = ce_edges[0]
        assert d["feat_fwd"][1] == 0.0  # forward orientation
        assert d["feat_rev"][1] == 1.0  # reverse orientation
