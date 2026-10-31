"""Scientific and representation contracts for the notebook adapters."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData, read_h5ad
from sklearn.preprocessing import StandardScaler

import mantpy as mt


def _observations(names, coords, **columns):
    data = AnnData(X=np.empty((len(names), 0)), obs=pd.DataFrame(columns, index=names))
    data.obsm["spatial"] = np.asarray(coords, dtype=float)
    return data


def test_joint_adapter_preserves_ids_weights_directions_and_zero_distance(tmp_path):
    cells = _observations(["a", "b"], [[0, 0], [2, 0]], cell_type=["A", "B"])
    ecm = _observations(["a", "c"], [[0, 0], [4, 0]], ecm_cluster=[0, 1])
    joint = mt.gr.joint_graph_from_connectivities(
        cells,
        ecm,
        cell_connectivities=sp.csr_matrix([[0, 3], [0, 0]]),
        ecm_connectivities=sp.csr_matrix([[0, 7], [7, 0]]),
        cell_ecm_connectivities=sp.csr_matrix([[5, 0], [0, 11]]),
        cell_ecm_distances=sp.csr_matrix([[0, 0], [0, 2]]),
    )
    assert joint.obs_names.tolist() == ["cell:a", "cell:b", "ecm:a", "ecm:c"]
    assert joint.obs["source_obs_name"].tolist() == ["a", "b", "a", "c"]
    assert joint.obsp["joint_connectivities"][0, 1] == 3
    assert joint.obsp["joint_connectivities"][1, 0] == 0
    assert joint.obsp["joint_connectivities"][0, 2] == 5
    assert joint.obsp["cell_ecm_distances"][0, 2] == 0
    legacy = mt.gr.to_networkx(joint, graph_key="cell_ecm_connectivities", legacy_ids=True)
    assert legacy["cell_0"]["ecm_0"] == {"weight": 5, "distance": 0}
    assert legacy.nodes["ecm_0"]["source_obs_name"] == "a"
    assert mt.gr.to_networkx(joint).is_directed()
    path = tmp_path / "joint.h5ad"
    joint.write_h5ad(path)
    restored = read_h5ad(path)
    assert (restored.obsp["joint_connectivities"] != joint.obsp["joint_connectivities"]).nnz == 0
    assert restored.uns["cell_ecm_graph"]["format"] == "mantpy_sparse_joint_v1"


def test_existing_components_use_induced_edges_and_original_ids():
    graph = nx.DiGraph()
    for node, cluster, x in [("a", 5, 0), ("b", 0, 1), ("c", 5, 2), ("d", 5, 3)]:
        graph.add_node(node, ecm_cluster=cluster, x=x, y=0)
    graph.add_edges_from([("a", "b"), ("b", "c"), ("d", "c")])
    data = AnnData()
    data.uns["ecm_graph"] = graph
    records = mt.gr.extract_components_existing(data, target_cluster=5, min_nodes=2)
    assert [item["node_ids"] for item in records] == [["a"], ["c", "d"]]
    assert [item["kept"] for item in records] == [False, True]
    assert records[1]["subgraph"].has_edge("d", "c")
    assert records[1]["centroid"].tolist() == [2.5, 0]


def test_existing_sparse_components_align_obs_ids_and_coordinates():
    data = _observations(["x", "z", "q"], [[9, 0], [20, 0], [25, 0]], ecm_cluster=[5, 5, 0])
    data.obsp["ecm_graph"] = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    records = mt.gr.extract_components_existing(data, target_cluster=5)
    assert records[0]["node_ids"] == ["x", "z"]
    np.testing.assert_array_equal(records[0]["coords"], [[9, 0], [20, 0]])


def test_image_normalizer_preserves_pooled_math_and_held_out_statistics():
    images = {"a": np.arange(24, dtype=np.float32).reshape(2, 3, 4), "b": np.ones((2, 2, 3))}
    normalizer = mt.pp.ImageNormalizer(chunk_size=5, channel_names=["A", "B"]).fit(images)
    reference = StandardScaler()
    for image in images.values():
        pixels = image.reshape(2, -1).T
        for start in range(0, len(pixels), 5):
            reference.partial_fit(pixels[start : start + 5])
    held_out = np.full((2, 4, 3), 50, dtype=np.float32)
    mean = normalizer.scaler_.mean_.copy()
    expected = reference.transform(held_out.reshape(2, -1).T).astype(np.float32).T.reshape(held_out.shape)
    np.testing.assert_array_equal(normalizer.transform(held_out, channel_names=["A", "B"]), expected)
    np.testing.assert_array_equal(normalizer.scaler_.mean_, mean)
    with pytest.raises(ValueError, match="channel order"):
        normalizer.transform(held_out, channel_names=["B", "A"])


def test_percentile_normalizer_uses_float32_pool_and_only_selected_bounds():
    images = [np.arange(24).reshape(2, 3, 4), np.full((2, 2, 4), 3)]
    normalizer = mt.pp.ImageNormalizer(
        "pooled_arcsinh_percentile", cofactor=5, percentile=90, channel_indices=[1]
    ).fit(images)
    pooled = np.concatenate([np.arcsinh(np.asarray(image[1], dtype=np.float32).ravel() / 5) for image in images])
    bound = np.asarray([np.percentile(pooled, 90)], dtype=np.float32)[0]
    assert normalizer.bounds_[0] == bound
    actual = normalizer.transform(images[0])
    expected = np.arcsinh(images[0].astype(np.float32) / 5)
    expected[1] = np.clip(expected[1] / bound, 0, 1)
    np.testing.assert_array_equal(actual, expected)


def test_patch_map_preserves_full_lattice_id_colors_and_overlay_extent():
    data = _observations(["p1", "p2"], [[15, 15], [25, 15]], grid_y=[1, 1], grid_x=[1, 2], ecm_cluster=[0, 1])
    data.uns["image_ecm_patches"] = {"patch_grid_shape": np.array([4, 5]), "patch_size": 10}
    values = pd.Series([1, 0], index=["p2", "p1"])
    ax = mt.pl.patch_map(data, values, palette={0: "red", 1: "blue"})
    canvas = np.asarray(ax.images[-1].get_array())
    assert canvas.shape == (4, 5, 3)
    np.testing.assert_array_equal(canvas[1, 1], [1, 0, 0])
    np.testing.assert_array_equal(canvas[1, 2], [0, 0, 1])
    np.testing.assert_array_equal(canvas[3, 4], [1, 1, 1])
    assert tuple(ax.images[-1].get_extent()) == (-0.5, 4.5, 3.5, -0.5)
    plt.close(ax.figure)
    ax = mt.pl.patch_map(data, "ecm_cluster", palette={0: "red", 1: "blue"}, extent=(0, 50, 40, 0))
    assert tuple(ax.images[-1].get_extent()) == (0, 50, 40, 0)
    plt.close(ax.figure)


def test_tiles_only_skip_descriptor_backend(monkeypatch):
    import mantpy.pp as pp

    def reject_descriptors(*args, **kwargs):
        raise AssertionError("Descriptor backend should not run for tiles-only extraction.")

    monkeypatch.setattr(pp, "_squidpy_image_features", reject_descriptors)
    image = np.arange(1, 65, dtype=np.float32).reshape(1, 8, 8)
    data = mt.pp.image_ecm_patches(
        image,
        patch_size=4,
        feature_families=(),
        pixel_thresholds=[0],
        patch_thresholds=[0],
    )
    assert data.shape == (4, 0)
    assert data.obsm["image_patches"].shape == (4, 1, 4, 4)
    assert data.uns["image_ecm_patches"]["feature_backend"] == "none"


def test_log_euclidean_distance_keeps_existing_log1p_contract():
    from mantpy.gr import _edge_feature_matrix

    points = np.array([[0, 0], [3, 4], [0, 0]], dtype=float)
    actual = _edge_feature_matrix(points, np.array([0, 0]), np.array([1, 2]), ["log_euclidean_distance"])
    np.testing.assert_array_equal(actual[:, 0], np.asarray([np.log(5), np.log(1e-9)], dtype=np.float32))
    assert mt.gr.EDGE_FEATURE_REGISTRY["log_distance"](points[0], points[1], 5)[0] == np.float32(np.log1p(5))


@pytest.mark.parametrize("coincident", [False, True])
def test_patch_graph_directed_float64_standardization_matches_geometric_recipe(coincident):
    points = np.array([[0, 0], [3, 4], [8, 1], [9, 7]], dtype=float)
    if coincident:
        points[1] = points[0]
    result = mt.gr.build_patch_graph(
        points, k=2,
        edge_features=["log_euclidean_distance", "sin_angle2", "cos_angle2"],
        edge_feature_precision="float64",
        standardization_scope="directed",
        standardization_eps=0,
    )
    source, target = result.edge_index
    displacement = points[target] - points[source]
    distance = np.sqrt(np.sum(displacement ** 2, axis=1))
    angle = np.arctan2(displacement[:, 1], displacement[:, 0])
    features = np.column_stack([
        np.log(np.maximum(distance, 1e-9)), np.sin(2 * angle), np.cos(2 * angle),
    ])
    deviation = features.std(axis=0)
    deviation[deviation == 0] = 1
    expected = ((features - features.mean(axis=0)) / deviation).astype(np.float32)
    np.testing.assert_array_equal(result.edge_attr, expected)
    assert result.params["standardization_scope"] == "directed"
    default = mt.gr.build_patch_graph(points, k=2)
    assert "standardization_scope" not in default.params


@pytest.mark.parametrize("options", [
    {"edge_feature_precision": "float16"},
    {"standardization_scope": "both"},
    {"standardization_eps": -1},
    {"standardization_eps": np.nan},
])
def test_patch_graph_rejects_invalid_standardization_options(options):
    with pytest.raises(ValueError):
        mt.gr.build_patch_graph([[0, 0], [1, 0]], k=1, **options)


def test_supplied_joint_graph_exports_to_both_pyg_representations():
    pytest.importorskip("torch_geometric")
    cells = _observations(["c"], [[0, 0]], cell_type=["A"])
    ecm = _observations(["p"], [[7, 0]], ecm_cluster=[0])
    joint = mt.gr.joint_graph_from_connectivities(
        cells, ecm, cell_ecm_connectivities=sp.csr_matrix([[2.]]),
    )
    homogeneous = mt.gr.to_pyg(joint)
    assert homogeneous.node_ids == ["cell:c", "ecm:p"]
    assert homogeneous.edge_index.tolist() == [[0, 1], [1, 0]]
    np.testing.assert_array_equal(homogeneous.pos.numpy(), [[0, 0], [7, 0]])
    heterogeneous = mt.gr.to_hetero_pyg(joint)
    assert heterogeneous["cell"].source_ids == ["c"]
    assert heterogeneous["ecm"].source_ids == ["p"]
    np.testing.assert_array_equal(heterogeneous["ecm"].pos.numpy(), [[7, 0]])
    assert joint.uns["cell_ecm_graph"]["spatial_um_key"] == "spatial"


def test_to_networkx_does_not_invent_geometry_from_connectivity():
    data = _observations(["a", "b"], [[0, 0], [30, 0]], ecm_cluster=[0, 0])
    data.obsp["ecm_graph"] = sp.csr_matrix([[0, .25], [.25, 0]])
    graph = mt.gr.to_networkx(data, graph_key="ecm_graph")
    assert graph["a"]["b"] == {"weight": .25}


def test_patch_map_nullable_background_palette_and_lower_crop():
    table = pd.DataFrame(
        {"grid_y": [0, 0], "grid_x": [0, 1], "cluster_id": pd.array([0, pd.NA], dtype="Int64")},
        index=["p0", "p1"],
    )
    ax = mt.pl.patch_map(
        table, "cluster_id", shape=(10, 10), palette={0: "red"},
        origin="lower", crop=True, extent=(0, 100, 0, 100),
    )
    canvas = np.asarray(ax.images[-1].get_array())
    np.testing.assert_array_equal(canvas[0, 0], [1, 0, 0])
    np.testing.assert_array_equal(canvas[0, 1], [1, 1, 1])
    assert ax.images[-1].get_extent() == [0., 40., 0., 30.]
    plt.close(ax.figure)


def test_image_normalizer_indices_and_serialized_fit():
    import pickle

    image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    for indices in ([1.9], [True], [0, 0], [2]):
        with pytest.raises(ValueError, match="channel"):
            mt.pp.ImageNormalizer("pooled_arcsinh_percentile", channel_indices=indices).fit([image])
    normalizer = mt.pp.ImageNormalizer(
        "pooled_arcsinh_percentile", cofactor=5, channel_indices=[np.int64(1)],
    ).fit([image])
    restored = pickle.loads(pickle.dumps(normalizer))
    np.testing.assert_array_equal(restored.transform(image), normalizer.transform(image))
    np.testing.assert_array_equal(restored.provenance_["bounds"], normalizer.provenance_["bounds"])


def test_adapters_empty_observations_and_missing_coordinates():
    cells = _observations([], np.empty((0, 2)), cell_type=[])
    ecm = _observations([], np.empty((0, 2)), ecm_cluster=[])
    joint = mt.gr.joint_graph_from_connectivities(
        cells, ecm, cell_ecm_connectivities=sp.csr_matrix((0, 0)),
    )
    assert joint.n_obs == 0
    assert joint.obsp["joint_connectivities"].shape == (0, 0)
    assert mt.gr.to_networkx(joint).number_of_nodes() == 0
    del cells.obsm["spatial"]
    with pytest.raises(KeyError, match="coordinates"):
        mt.gr.joint_graph_from_connectivities(
            cells, ecm, cell_ecm_connectivities=sp.csr_matrix((0, 0)),
        )
