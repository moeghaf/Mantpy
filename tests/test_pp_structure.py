"""Tests for mt.pp.annotate_structure, split_structures, and extract_structure_ecm."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mantpy.pp import annotate_structure, extract_structure_ecm, split_structures


@pytest.fixture
def clustered_adata():
    """AnnData with 3 clusters, 2 ROIs, spatial coords, and marker expression.

    Cluster '0' has high marker expression (the 'structure').
    Clusters '1' and '2' are background.
    """
    rng = np.random.default_rng(42)
    n = 200

    # Spatial: two ROIs, cluster 0 cells form two tight blobs in ROI 'A'
    coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
    # Make first 50 cells (cluster 0) cluster tightly in two groups
    coords[:25] = rng.normal(loc=[100, 100], scale=10, size=(25, 2))
    coords[25:50] = rng.normal(loc=[300, 300], scale=10, size=(25, 2))

    # Expression: 5 markers, first 2 are 'structure markers' (high in cluster 0)
    X = rng.random((n, 5)).astype(np.float32) * 0.3
    X[:50, 0] = rng.uniform(0.8, 1.0, 50)  # marker_A high in cluster 0
    X[:50, 1] = rng.uniform(0.7, 1.0, 50)  # marker_B high in cluster 0

    obs = pd.DataFrame(
        {
            "leiden": pd.Categorical(["0"] * 50 + ["1"] * 75 + ["2"] * 75),
            "roi_id": ["roi_A"] * 100 + ["roi_B"] * 100,
        }
    )

    var = pd.DataFrame(index=["Nd145_marker_A", "Sm147_marker_B", "Gd160_C", "Er170_D", "Yb176_E"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords
    return adata


class TestAnnotateStructure:
    def test_basic_annotation(self, clustered_adata):
        annotate_structure(clustered_adata, name="target", markers=["marker_A", "marker_B"])
        assert "structure" in clustered_adata.obs.columns
        cats = clustered_adata.obs["structure"].cat.categories.tolist()
        assert "target" in cats
        assert "other" in cats

    def test_correct_cluster_labelled(self, clustered_adata):
        annotate_structure(clustered_adata, name="target", markers=["marker_A", "marker_B"])
        # Cluster 0 (first 50 cells) should be labelled 'target'
        target_mask = clustered_adata.obs["structure"] == "target"
        # Most of cluster 0 should be target (some may be filtered by spatial)
        cluster0_target = target_mask.values[:50].sum()
        assert cluster0_target > 30  # at least 60% survive spatial filter

    def test_other_clusters_not_labelled(self, clustered_adata):
        annotate_structure(clustered_adata, name="target", markers=["marker_A", "marker_B"])
        # Cells 50+ (clusters 1, 2) should all be "other"
        other_cells = clustered_adata.obs["structure"].values[50:]
        assert all(c == "other" for c in other_cells)

    def test_spatial_filter_removes_isolated_cells(self):
        """Isolated cells (far from others) should be demoted to 'other'."""
        rng = np.random.default_rng(99)
        n = 28
        X = np.zeros((n, 3), dtype=np.float32)
        X[:, 0] = 1.0  # all cells express the marker

        coords = np.zeros((n, 2), dtype=np.float32)
        # 25 cells tightly clustered
        coords[:25] = rng.normal(loc=[50, 50], scale=5, size=(25, 2))
        # 3 isolated cells far away (below min_size=5 so they get demoted)
        coords[25:] = rng.normal(loc=[500, 500], scale=2, size=(3, 2))

        obs = pd.DataFrame({"leiden": pd.Categorical(["0"] * n)})
        var = pd.DataFrame(index=["marker_X", "marker_Y", "marker_Z"])
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.obsm["spatial"] = coords

        annotate_structure(adata, name="blob", markers=["marker_X"], roi_key=None)
        # The 3 isolated cells should be "other" (below min_size=5)
        assert all(adata.obs["structure"].values[25:] == "other")
        # The 25 clustered cells should be "blob"
        assert all(adata.obs["structure"].values[:25] == "blob")

    def test_no_matching_markers_raises(self, clustered_adata):
        with pytest.raises(ValueError, match="None of the markers"):
            annotate_structure(clustered_adata, name="x", markers=["NONEXISTENT_MARKER"])

    def test_custom_key_added(self, clustered_adata):
        annotate_structure(
            clustered_adata,
            name="islet",
            markers=["marker_A"],
            key_added="compartment",
        )
        assert "compartment" in clustered_adata.obs.columns

    def test_inplace_false_returns_copy(self, clustered_adata):
        result = annotate_structure(clustered_adata, name="x", markers=["marker_A"], inplace=False)
        assert result is not None
        assert "structure" in result.obs.columns
        assert "structure" not in clustered_adata.obs.columns

    def test_n_top_clusters(self, clustered_adata):
        annotate_structure(
            clustered_adata,
            name="target",
            markers=["marker_A"],
            n_top_clusters=2,
        )
        target_count = (clustered_adata.obs["structure"] == "target").sum()
        # With 2 top clusters, more cells should be labelled
        assert target_count > 50


class TestSplitStructures:
    def test_basic_split(self, clustered_adata):
        annotate_structure(
            clustered_adata,
            name="target",
            markers=["marker_A", "marker_B"],
            roi_key=None,
        )
        results = split_structures(clustered_adata, structure_value="target", roi_key=None)
        # Should find 2 spatial components (two tight blobs in the fixture)
        assert len(results) == 2

    def test_uid_format(self, clustered_adata):
        annotate_structure(
            clustered_adata,
            name="target",
            markers=["marker_A", "marker_B"],
            roi_key=None,
        )
        results = split_structures(clustered_adata, structure_value="target", roi_key=None)
        for uid, idx in results:
            assert "target" in uid
            assert isinstance(idx, np.ndarray)
            assert idx.dtype in (np.int64, np.int32, np.intp)

    def test_indices_are_valid(self, clustered_adata):
        annotate_structure(
            clustered_adata,
            name="target",
            markers=["marker_A", "marker_B"],
            roi_key=None,
        )
        results = split_structures(clustered_adata, structure_value="target", roi_key=None)
        all_idx = np.concatenate([idx for _, idx in results])
        assert all_idx.max() < len(clustered_adata)
        assert all_idx.min() >= 0
        # No duplicates
        assert len(np.unique(all_idx)) == len(all_idx)

    def test_min_cells_filter(self, clustered_adata):
        annotate_structure(
            clustered_adata,
            name="target",
            markers=["marker_A", "marker_B"],
            roi_key=None,
        )
        results = split_structures(
            clustered_adata,
            structure_value="target",
            min_cells=100,
            roi_key=None,
        )
        # Both blobs have ~25 cells, so min_cells=100 should filter all
        assert len(results) == 0

    def test_missing_key_raises(self, clustered_adata):
        with pytest.raises(KeyError, match="not found"):
            split_structures(clustered_adata, structure_value="x")

    def test_per_roi_split(self, clustered_adata):
        # Move second blob to roi_B so each ROI has one blob
        clustered_adata.obs["roi_id"].values[25:50] = "roi_B"
        annotate_structure(
            clustered_adata,
            name="target",
            markers=["marker_A", "marker_B"],
        )
        results = split_structures(clustered_adata, structure_value="target")
        # Should find components in both ROIs
        rois_found = {uid.split("_target")[0] for uid, _ in results}
        assert len(rois_found) >= 1

    def test_auto_detect_structure_value(self, clustered_adata):
        annotate_structure(
            clustered_adata,
            name="myblob",
            markers=["marker_A"],
            roi_key=None,
        )
        results = split_structures(clustered_adata, roi_key=None)
        # Should auto-detect "myblob" as the structure value
        for uid, _ in results:
            assert "myblob" in uid

    def test_key_added_writes_obs_column(self, clustered_adata):
        annotate_structure(
            clustered_adata,
            name="target",
            markers=["marker_A", "marker_B"],
            roi_key=None,
        )
        results = split_structures(
            clustered_adata,
            structure_value="target",
            roi_key=None,
            key_added="structure_id",
        )
        assert "structure_id" in clustered_adata.obs.columns
        # Categorical with one level per retained component
        col = clustered_adata.obs["structure_id"]
        assert hasattr(col, "cat")
        assert set(col.cat.categories) == {uid for uid, _ in results}
        # Each tagged cell maps back to its component
        for uid, idx in results:
            assert (col.iloc[idx] == uid).all()

    def test_key_added_default_does_not_write(self, clustered_adata):
        # Backward-compat: default key_added=None means no obs column written
        annotate_structure(
            clustered_adata,
            name="target",
            markers=["marker_A"],
            roi_key=None,
        )
        split_structures(
            clustered_adata,
            structure_value="target",
            roi_key=None,
        )
        assert "structure_id" not in clustered_adata.obs.columns


class TestAnnotateStructureVerbose:
    def test_verbose_off_silent(self, clustered_adata, capsys):
        annotate_structure(
            clustered_adata,
            name="silent",
            markers=["marker_A"],
            roi_key=None,
        )
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_on_prints_summary(self, clustered_adata, capsys):
        annotate_structure(
            clustered_adata,
            name="loud",
            markers=["marker_A"],
            roi_key=None,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert "matched markers" in out
        assert "loud" in out
        # The final summary mentions cells labelled / final / demoted
        assert "final" in out
        assert "demoted" in out


# -----------------------------------------------------------------------------
# extract_structure_ecm
# -----------------------------------------------------------------------------


@pytest.fixture
def roi_with_cell_ecm_graph():
    """ROI AnnData with a cell-ECM graph built, one tight cell cluster."""
    import networkx as nx

    # 20 cells clustered at (100, 100)
    rng = np.random.default_rng(0)
    cell_xy = rng.normal(loc=[100, 100], scale=5, size=(20, 2)).astype(np.float32)
    # Some scattered background cells
    bg = rng.uniform(0, 500, (30, 2)).astype(np.float32)
    coords = np.vstack([cell_xy, bg])

    X = np.zeros((50, 3), dtype=np.float32)
    X[:20, 0] = 1.0  # structure cells have marker
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(50)]),
        var=pd.DataFrame(index=["marker_A", "marker_B", "marker_C"]),
    )
    adata.obsm["spatial"] = coords

    # 10 ECM patches: 6 near the cluster, 4 far away
    ecm_xy = np.array(
        [
            [100, 100],
            [105, 95],
            [95, 110],
            [110, 105],
            [92, 100],
            [108, 92],
            [400, 400],
            [410, 405],
            [395, 395],
            [405, 400],
        ],
        dtype=np.float32,
    )
    patches_df = pd.DataFrame(
        {
            "x": ecm_xy[:, 0],
            "y": ecm_xy[:, 1],
            "ecm_cluster": np.zeros(10, dtype=int),
            "feat_mean": rng.uniform(0, 1, 10),
            "feat_std": rng.uniform(0, 1, 10),
        }
    )
    adata.uns["ecm_patches"] = patches_df

    # Build a minimal cell-ECM graph: connect the 6 near patches to all 20
    # structure cells via cell-ecm edges.
    G = nx.Graph()
    for i in range(50):
        G.add_node(f"cell_{i}", node_type="cell")
    for j in range(10):
        G.add_node(f"ecm_{j}", node_type="ecm")
    for i in range(20):
        for j in range(6):
            G.add_edge(f"cell_{i}", f"ecm_{j}", edge_type="cell-ecm", distance=1.0)
    adata.uns["cell_ecm_graph"] = G
    return adata


class TestExtractStructureEcm:
    def test_returns_adata_with_expected_columns(self, roi_with_cell_ecm_graph):
        img = np.ones((512, 512), dtype=np.float32) * 10.0
        out = extract_structure_ecm(
            roi_with_cell_ecm_graph,
            structure_indices=np.arange(20),
            structure_uid="test_islet01",
            ecm_channel_img=img,
        )
        assert out is not None
        # The 6 near patches should all be selected
        assert out.n_obs == 6
        for col in ("x", "y", "sin_theta", "cos_theta", "radial_dist", "sauvola_pos"):
            assert col in out.obs.columns

    def test_polar_coords_are_centered_on_structure(self, roi_with_cell_ecm_graph):
        img = np.ones((512, 512), dtype=np.float32)
        out = extract_structure_ecm(
            roi_with_cell_ecm_graph,
            structure_indices=np.arange(20),
            structure_uid="test_islet01",
            ecm_channel_img=img,
        )
        assert out is not None
        # sin^2 + cos^2 == 1 for every patch
        s = out.obs["sin_theta"].values
        c = out.obs["cos_theta"].values
        np.testing.assert_allclose(s * s + c * c, 1.0, atol=1e-5)
        # Radial dist should be small (patches are near cluster centroid ~100,100)
        assert float(out.obs["radial_dist"].max()) < 30

    def test_returns_none_when_no_edges(self, roi_with_cell_ecm_graph):
        img = np.ones((512, 512), dtype=np.float32)
        # Structure is the background cells (indices 20..49), which have no edges
        out = extract_structure_ecm(
            roi_with_cell_ecm_graph,
            structure_indices=np.arange(20, 50),
            structure_uid="none_islet01",
            ecm_channel_img=img,
            min_ecm_nodes=5,
        )
        assert out is None

    def test_graph_feat_attached(self, roi_with_cell_ecm_graph):
        img = np.ones((512, 512), dtype=np.float32)
        out = extract_structure_ecm(
            roi_with_cell_ecm_graph,
            structure_indices=np.arange(20),
            structure_uid="test_islet01",
            ecm_channel_img=img,
        )
        assert out is not None
        assert "islet_graph_feat" in out.uns
        assert np.asarray(out.uns["islet_graph_feat"]).shape == (11,)


# -----------------------------------------------------------------------------
# build_cell_ecm_graph with edge_method="delaunay"
# -----------------------------------------------------------------------------


def test_build_cell_ecm_graph_delaunay():
    """Delaunay mode should produce a bipartite graph without needing k."""
    import mantpy as mt

    rng = np.random.default_rng(1)
    n_cells = 30
    cell_xy = rng.uniform(0, 100, (n_cells, 2)).astype(np.float32)
    X = rng.random((n_cells, 3)).astype(np.float32)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame({"is_ecm": [False, False, True]}, index=["mA", "mB", "collagen"]),
    )
    adata.obsm["spatial"] = cell_xy

    img = rng.uniform(0.5, 1.0, (3, 100, 100)).astype(np.float32)
    mt.pp.extract_ecm_patches(
        adata,
        img=img,
        patch_size=10,
        ecm_K=None,
        features=["mean", "std"],
        min_signal_fraction=0.01,
    )
    mt.gr.build_cell_graph(adata, edge_method="delaunay")
    mt.gr.build_ecm_graph(adata, edge_method="delaunay")
    mt.gr.build_cell_ecm_graph(adata, edge_method="delaunay")

    G = adata.uns["cell_ecm_graph"]
    ce_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "cell-ecm"]
    # Some cell-ECM edges should exist
    assert len(ce_edges) > 0
    # And each cross edge has one cell and one ecm endpoint
    for u, v in ce_edges:
        types = sorted([str(u).split("_")[0], str(v).split("_")[0]])
        assert types == ["cell", "ecm"]
