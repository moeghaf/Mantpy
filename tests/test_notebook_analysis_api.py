"""Scientific contracts for reusable descriptive notebook calculations."""

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import mantpy as mt


def labels_adata(labels, *, native=True):
    a = ad.AnnData(np.zeros((len(labels), 0)))
    frame = pd.DataFrame({"ecm_cluster": labels}, index=a.obs_names)
    if native:
        a.obs["ecm_cluster"] = frame.ecm_cluster
    else:
        a.uns["ecm_patches"] = frame
    return a


def graph_adata(labels, edges=(), *, xy=None, ids=None):
    a = ad.AnnData(np.zeros((0, 0)))
    graph = nx.Graph()
    ids = list(range(len(labels))) if ids is None else ids
    xy = [(i, 0) for i in range(len(labels))] if xy is None else xy
    for node, label, point in zip(ids, labels, xy, strict=True):
        graph.add_node(node, ecm_cluster=label, x=point[0], y=point[1])
    graph.add_edges_from(edges)
    a.uns["ecm_graph"] = graph
    return a


def test_composition_mouse_pooling_differs_from_roi_means():
    cohort = {
        "small": labels_adata(["A"]),
        "large": labels_adata(["B"] * 9),
        "other": labels_adata(["A"]),
    }
    meta = pd.DataFrame(
        {"group": ["g"] * 3, "mouse": ["m1", "m1", "m2"]},
        index=list(cohort),
    )
    pooled = mt.tl.compute_cluster_composition(
        cohort, sample_meta=meta, unit_col="mouse", return_details=True,
        categories=["B", "A", "absent"],
    )
    np.testing.assert_array_equal(pooled.per_unit_counts.loc["m1"], [9, 1, 0])
    np.testing.assert_allclose(pooled.per_unit.loc["m1"], [.9, .1, 0])
    np.testing.assert_allclose(pooled.per_group_mean.loc["g"], [.45, .55, 0])
    equal_roi = mt.tl.compute_cluster_composition(
        cohort, sample_meta=meta, unit_col="mouse",
        unit_aggregation="mean_roi_fraction", return_details=True,
        categories=["B", "A", "absent"],
    )
    np.testing.assert_allclose(equal_roi.per_group_mean.loc["g"], [.25, .75, 0])
    # The default keeps the original two-item result and equal-ROI mean.
    default = mt.tl.compute_cluster_composition(cohort, sample_meta=meta)
    assert isinstance(default, tuple) and len(default) == 2
    np.testing.assert_allclose(default.per_group_mean.loc["g"], [2 / 3, 1 / 3])


def test_composition_background_empty_native_and_legacy():
    cohort = {
        "native": labels_adata(["signal", "background"]),
        "legacy": labels_adata(["signal", "signal"], native=False),
        "empty": labels_adata([], native=False),
    }
    meta = pd.DataFrame({"group": ["g"] * 3}, index=list(cohort))
    result = mt.tl.compute_cluster_composition(
        cohort, sample_meta=meta, categories=["background", "signal", "missing"],
        background_labels=["background"], return_details=True,
    )
    assert result.cluster_cols == ["signal", "missing"]
    np.testing.assert_array_equal(result.counts.to_numpy(), [[1, 0], [2, 0], [0, 0]])
    np.testing.assert_allclose(result.per_sample.loc["empty"], 0)
    # Legacy tables remain authoritative unless obs is explicitly selected.
    cohort["legacy"].obs["ecm_cluster"] = ["different", "different"]
    legacy = mt.tl.compute_cluster_composition(cohort, sample_meta=meta)
    assert "different" not in legacy.cluster_cols
    native = mt.tl.compute_cluster_composition(
        {"legacy": cohort["legacy"]}, sample_meta=meta.loc[["legacy"]], label_source="obs",
    )
    assert native.cluster_cols == ["different"]


def test_composition_rejects_units_spanning_groups():
    meta = pd.DataFrame({"group": ["A", "B"], "mouse": ["m", "m"]}, index=["a", "b"])
    with pytest.raises(ValueError, match="more than one group"):
        mt.tl.compute_cluster_composition(
            {"a": labels_adata([0]), "b": labels_adata([0])},
            sample_meta=meta, unit_col="mouse",
        )


def test_cooccurrence_ordered_counts_abundance_and_missing_categories():
    a = graph_adata(
        ["A", "A", "B", "background", "A"],
        [(0, 1), (1, 2), (2, 2), (2, 3)],
    )
    result = mt.tl.cluster_cooccurrence(
        {"roi": a}, categories=["B", "A", "absent"],
        return_details=True, missing="nan",
    )
    np.testing.assert_array_equal(result.observed, [[1, 1, 0], [1, 2, 0], [0, 0, 0]])
    np.testing.assert_array_equal(result.node_counts, [1, 3, 0])
    expected = 5 * np.outer([.25, .75, 0], [.25, .75, 0])
    np.testing.assert_allclose(result.expected, expected)
    assert np.isnan(result.enrichment[2]).all()
    assert np.isnan(result.enrichment[:, 2]).all()
    assert result.categories == ("B", "A", "absent")
    assert set(result.per_sample) == {"roi"}


def test_cooccurrence_pools_within_mouse_then_equal_mouse_mean():
    cohort = {
        "r1": graph_adata([0, 0, 1], [(0, 1)]),
        "r2": graph_adata([0, 1, 1], [(0, 1), (1, 2)]),
        "r3": graph_adata([0, 1], [(0, 1)]),
    }
    meta = pd.DataFrame({"mouse": ["m1", "m1", "m2"]}, index=list(cohort))
    result = mt.tl.cluster_cooccurrence(
        cohort, n_clusters=2, sample_meta=meta, unit_col="mouse",
        missing="nan", return_details=True,
    )
    pooled_m1 = mt.tl.cluster_cooccurrence([cohort["r1"], cohort["r2"]], n_clusters=2)
    single_m2 = mt.tl.cluster_cooccurrence([cohort["r3"]], n_clusters=2)
    np.testing.assert_allclose(result.per_unit["m1"].enrichment, pooled_m1)
    np.testing.assert_allclose(result.enrichment, (pooled_m1 + single_m2) / 2)
    np.testing.assert_allclose(
        result.observed, sum(r.observed for r in result.per_sample.values()),
    )


def test_cooccurrence_empty_missing_graph_and_legacy_result():
    a = graph_adata([0, 1], [(0, 1)])
    missing = ad.AnnData(np.zeros((0, 0)))
    old = mt.tl.cluster_cooccurrence([a, missing], n_clusters=3)
    assert isinstance(old, np.ndarray)
    assert np.isfinite(old).all() and not old[2].any()
    detail = mt.tl.cluster_cooccurrence(
        {"a": a, "missing": missing}, n_clusters=3,
        missing="nan", return_details=True,
    )
    np.testing.assert_allclose(detail.enrichment[:2, :2], old[:2, :2])
    assert np.isnan(detail.per_sample["missing"].enrichment).all()
    empty = mt.tl.cluster_cooccurrence({}, categories=["x"], missing="nan")
    assert empty.shape == (1, 1) and np.isnan(empty).all()


def test_cooccurrence_native_graph_and_numeric_legacy_coercion():
    native = labels_adata(["A", "A", "B"])
    native.obsp["spatial"] = sp.csr_matrix([[0, 2, 0], [2, 0, 5], [0, 5, 0]])
    detail = mt.tl.cluster_cooccurrence(
        [native], graph_key="spatial", categories=["A", "B"], return_details=True,
    )
    np.testing.assert_array_equal(detail.observed, [[2, 1], [1, 0]])
    numeric = graph_adata(["0", "0", "1"], [(0, 1), (1, 2)])
    np.testing.assert_array_equal(
        mt.tl.cluster_cooccurrence([numeric], n_clusters=2), detail.enrichment,
    )
    native.obsp["spatial"] = sp.csr_matrix([[0, 2, 0], [0, 0, 5], [0, 5, 0]])
    with pytest.raises(ValueError, match="symmetric"):
        mt.tl.cluster_cooccurrence([native], graph_key="spatial", categories=["A", "B"])


def test_cooccurrence_mean_roi_enrichment_is_explicit():
    cohort = [
        graph_adata([0, 0, 1], [(0, 1)]),
        graph_adata([0, 1], [(0, 1)]),
    ]
    expected = np.mean([
        mt.tl.cluster_cooccurrence([a], n_clusters=2) for a in cohort
    ], axis=0)
    got = mt.tl.cluster_cooccurrence(
        cohort, n_clusters=2, unit_aggregation="mean_roi_enrichment",
    )
    np.testing.assert_allclose(got, expected)


def test_neighbourhood_ids_category_order_weights_and_isolates():
    adjacency = sp.csr_matrix([[0, 2, 3], [0, 0, 0], [1, 0, 0]])
    ids = ["p0", "p1", "p2"]
    # Shuffle Series rows; alignment is by the explicit stable IDs.
    labels = pd.Series(["B", "A", "B"], index=["p2", "p0", "p1"])
    counts = mt.tl.neighbourhood_features(
        adjacency, labels, categories=["B", "A", "missing"],
        target_ids=ids, source_ids=ids, normalization="counts",
    )
    assert counts.index.tolist() == ids
    assert counts.columns.tolist() == ["B", "A", "missing"]
    np.testing.assert_array_equal(counts, [[5, 0, 0], [0, 0, 0], [0, 1, 0]])
    fractions = mt.tl.neighbourhood_features(
        adjacency, labels, categories=["B", "A"],
        target_ids=ids, source_ids=ids,
    )
    np.testing.assert_array_equal(fractions, [[1, 0], [0, 0], [0, 1]])
    logged = mt.tl.neighbourhood_features(
        adjacency, labels, categories=["B", "A"],
        target_ids=ids, source_ids=ids, normalization="log1p_counts",
    )
    np.testing.assert_allclose(logged, np.log1p(counts.iloc[:, :2]))


def test_directed_two_hop_return_paths_excluded():
    adjacency = sp.csr_matrix([[0, 2, 0], [3, 0, 5], [0, 7, 0]])
    result = mt.tl.neighbourhood_features(
        adjacency, ["red", "blue", "red"], categories=["red", "blue"],
        normalization="counts", hops=2,
    )
    np.testing.assert_array_equal(result, [[10, 0], [0, 0], [21, 0]])
    with_self = mt.tl.neighbourhood_features(
        adjacency, ["red", "blue", "red"], categories=["red", "blue"],
        normalization="counts", hops=2, exclude_self=False,
    )
    np.testing.assert_array_equal(with_self, [[16, 0], [0, 41], [56, 0]])


def test_bipartite_two_hop_and_continuous_neighbor_means():
    incidence = sp.csr_matrix([[2, 1, 0], [0, 3, 4]])
    result = mt.tl.neighbourhood_features(
        incidence, ["a", "b"], categories=["a", "b"],
        target_ids=["p0", "p1"], source_ids=["p0", "p1"],
        normalization="counts", hops=2, via=incidence.T,
    )
    np.testing.assert_array_equal(result, [[0, 3], [3, 0]])
    feature_frame = pd.DataFrame({"marker": [10., 4., 1.]}, index=["c2", "c1", "c0"])
    mean = mt.tl.neighbourhood_features(
        incidence, values=feature_frame, target_ids=["p0", "p1"],
        source_ids=["c0", "c1", "c2"], normalization="mean",
    )
    np.testing.assert_allclose(mean.marker, [2., 52 / 7])


def test_rectangular_neighborhood_preserves_cross_edges_without_ids():
    result = mt.tl.neighbourhood_features(
        [[2, 0, 1], [0, 4, 0]], ["a", "b", "a"],
        categories=["a", "b"], normalization="counts",
    )
    np.testing.assert_array_equal(result, [[3, 0], [0, 4]])


def test_neighborhood_rejects_bad_ids_weights_and_shapes():
    with pytest.raises(ValueError, match="non-negative"):
        mt.tl.neighbourhood_features([[-1]], ["a"])
    with pytest.raises(ValueError, match="unique"):
        mt.tl.neighbourhood_features(np.eye(2), ["a", "b"], target_ids=["x", "x"])
    with pytest.raises(ValueError, match="columns"):
        mt.tl.neighbourhood_features(np.ones((2, 3)), ["a", "b"], hops=2)
    with pytest.raises(ValueError, match="requested ID"):
        mt.tl.neighbourhood_features(
            np.eye(2), pd.Series(["a", "b"], index=["x", "y"]), source_ids=["x", "z"],
        )


def test_contact_marker_contrast_unique_patch_and_edge_weighting():
    incidence = sp.csr_matrix([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
    features = pd.DataFrame(
        {"marker": [30., 20., 10., 0.]}, index=["p3", "p2", "p1", "p0"],
    )
    kwargs = {
        "cell_labels": ["AEC", "AEC", "other"], "cell_type": "AEC",
        "patch_labels": pd.Series(["wall"] * 4, index=["p0", "p1", "p2", "p3"]),
        "cluster": "wall", "patch_ids": ["p0", "p1", "p2", "p3"],
    }
    unique = mt.tl.contact_marker_contrast(incidence, features, **kwargs)
    edges = mt.tl.contact_marker_contrast(incidence, features, weighting="edge", **kwargs)
    assert unique.n_contacted == 2 and unique.n_uncontacted == 2
    assert unique.contrast.marker == -20.
    np.testing.assert_allclose(edges.contrast.marker, 10 / 3 - 25)
    np.testing.assert_array_equal(unique.contact_mask, [True, True, False, False])
    np.testing.assert_array_equal(edges.weights, [2, 1, 0, 0])


def test_contact_empty_arms_and_no_hidden_normalization():
    values = pd.DataFrame({"marker": [100., 400.]}, index=["p0", "p1"])
    all_contact = mt.tl.contact_marker_contrast([[5, 9]], values)
    assert all_contact.contacted_mean.marker == 250.
    assert np.isnan(all_contact.contrast.marker)
    no_contact = mt.tl.contact_marker_contrast([[0, 0]], values)
    assert np.isnan(no_contact.contrast.marker)
    assert no_contact.uncontacted_mean.marker == 250.
    empty = mt.tl.contact_marker_contrast(sp.csr_matrix((0, 0)), pd.DataFrame(columns=["m"]))
    assert empty.n_contacted == 0 and np.isnan(empty.contrast.m)


def test_contact_input_dtype_retains_original_array_and_dataframe_means():
    rng = np.random.default_rng(2)
    values = (rng.normal(size=(111, 3)) * 300).astype(np.float32)
    contact = np.arange(len(values)) % 3 == 0
    adjacency = sp.csr_matrix(contact[None])
    expected = values[contact].mean(axis=0) - values[~contact].mean(axis=0)
    result = mt.tl.contact_marker_contrast(
        adjacency, values, accumulation_dtype="input",
    )
    np.testing.assert_array_equal(result.contrast, expected)
    assert result.contrast.dtype == np.float32
    frame = pd.DataFrame(values)
    frame_expected = frame.loc[contact].mean() - frame.loc[~contact].mean()
    pandas_result = mt.tl.contact_marker_contrast(
        adjacency, frame, accumulation_dtype="input",
    )
    pd.testing.assert_series_equal(pandas_result.contrast, frame_expected)
    assert mt.tl.contact_marker_contrast(adjacency, values).contrast.dtype == np.float64


def test_contact_cell_labels_align_by_explicit_ids():
    labels = pd.Series(["other", "AEC"], index=["c1", "c0"])
    result = mt.tl.contact_marker_contrast(
        np.eye(2), [[10.], [50.]],
        cell_labels=labels, cell_ids=["c0", "c1"], cell_type="AEC",
    )
    assert result.contact_mask.tolist() == [True, False]
    assert result.contrast.iloc[0] == -40
    with pytest.raises(ValueError, match="requested ID"):
        mt.tl.contact_marker_contrast(np.eye(2), [[10.], [50.]], cell_labels=labels, cell_type="AEC")


def test_contact_enrichment_details_share_original_per_roi_expectation():
    def roi(clusters, edges):
        a = ad.AnnData(np.zeros((0, 0)))
        graph = nx.Graph()
        graph.add_node("a", node_type="cell", cell_type="AEC")
        graph.add_node("b", node_type="cell", cell_type="other")
        for node, cluster in enumerate(clusters):
            graph.add_node(node, node_type="ecm", ecm_cluster=cluster)
        graph.add_edges_from(edges)
        a.uns["cell_ecm_graph"] = graph
        return a

    cohort = {
        "r1": roi([0, 0, 1], [("a", 0), ("a", 1), ("b", 2)]),
        "r2": roi([0, 1], [("b", 0), ("a", 1)]),
    }
    details = mt.tl.cell_ecm_enrichment(
        cohort, "AEC", K_ecm=3, n_perm=0, inplace=False, return_details=True,
    )
    np.testing.assert_allclose(details.pooled.n_obs, [2, 1, 0])
    np.testing.assert_allclose(details.pooled.n_exp, [4 / 3 + .5, 2 / 3 + .5, 0])
    assert details.per_roi.roi.tolist() == ["r1"] * 3 + ["r2"] * 3
    assert details.per_sample is details.per_roi
    assert all("cell_ecm_enrichment" not in a.uns for a in cohort.values())
    default = mt.tl.cell_ecm_enrichment(
        cohort, "AEC", K_ecm=3, n_perm=0, inplace=False,
    )
    pd.testing.assert_frame_equal(default, details.pooled)
    matrix = mt.tl.cell_ecm_enrichment_matrix(
        cohort, K_ecm=3, n_perm=0, return_details=True, eps=0,
    )
    aec_roi = matrix.per_roi.loc[matrix.per_roi.cell_type == "AEC"]
    assert np.isneginf(aec_roi.loc[
        (aec_roi.roi == "r1") & (aec_roi.cluster == 1), "log2_enr",
    ]).all()
    assert np.isnan(matrix.pooled.loc[matrix.pooled.cluster == 2, "log2_enr"]).all()
    assert set(matrix.per_roi.cell_type) == {"AEC", "other"}
    np.testing.assert_allclose(
        matrix.pooled.loc[matrix.pooled.cell_type == "AEC", "n_exp"],
        details.pooled.n_exp,
    )


def test_component_radius_summary_keeps_original_ids_and_empty_units():
    cohort = {
        "r1": graph_adata(["a", "a", "a"], [(0, 2)], xy=[(0, 0), (1, 0), (9, 0)]),
        "r2": graph_adata(["b"]),
    }
    meta = pd.DataFrame({"mouse": ["m1", "m2"]}, index=list(cohort))
    result = mt.tl.component_summary(
        cohort, target_cluster="a", mode="radius", radius=2, min_nodes=2,
        sample_meta=meta, unit_col="mouse",
    )
    assert result.components.n_nodes.tolist() == [2, 1]
    assert result.components.kept.tolist() == [True, False]
    assert result.membership.node_id.tolist() == [0, 1, 2]
    assert result.membership.degree.tolist() == [1, 1, 0]
    assert result.per_unit.loc["m1", "median_nodes"] == 2
    assert result.per_unit.loc["m1", "median_mean_degree"] == 1
    assert result.per_unit.loc["m2", "n_components"] == 0
    assert np.isnan(result.per_unit.loc["m2", "median_nodes"])


def test_component_existing_differs_from_radius():
    a = graph_adata(["a", "a", "a"], [("p0", "p2")], ids=["p0", "p1", "p2"])
    inherited = mt.tl.component_summary(a, target_cluster="a")
    reconnected = mt.tl.component_summary(a, target_cluster="a", mode="radius", radius=1.1)
    assert sorted(inherited.components.n_nodes) == [1, 2]
    assert reconnected.components.n_nodes.tolist() == [3]
    assert set(inherited.membership.node_id) == {"p0", "p1", "p2"}


def test_component_unit_pooling_and_empty_all_target():
    cohort = {
        "small": graph_adata(["a"]),
        "large": graph_adata(["a"] * 3, [(0, 1), (1, 2)]),
        "empty": graph_adata(["b"]),
    }
    meta = pd.DataFrame({"mouse": ["m", "m", "empty"]}, index=list(cohort))
    pool = mt.tl.component_summary(
        cohort, target_cluster="a", sample_meta=meta, unit_col="mouse",
    )
    assert pool.per_unit.loc["m", "median_nodes"] == 2
    assert pool.per_unit.loc["m", "n_components"] == 2
    empty = mt.tl.component_summary(cohort, target_cluster="absent", sample_meta=meta)
    assert empty.components.empty and empty.membership.empty
    assert empty.per_sample.n_components.tolist() == [0, 0, 0]
    assert empty.per_sample.median_nodes.isna().all()
    absent_cohort = mt.tl.component_summary({}, target_cluster="a")
    assert absent_cohort.components.empty and absent_cohort.membership.empty
    assert absent_cohort.per_sample.columns.tolist() == [
        "n_components", "median_nodes", "median_mean_degree",
    ]
    assert absent_cohort.per_unit.columns.tolist() == absent_cohort.per_sample.columns.tolist()


def test_reconstruction_raw_minmax_background_and_confidence():
    observed = np.array(["A", "A", "background"])
    proba = np.array([[.1, .9], [.7, .3], [.01, .99]])
    raw = mt.tl.reconstruct_labels(
        observed, proba, categories=["A", "longer-label-B"],
        threshold=1., score="raw_neglogp",
    )
    assert raw.labels.tolist() == ["longer-label-B", "A", "background"]
    np.testing.assert_allclose(raw.anomaly_scores, [-np.log(.1), -np.log(.7), 0])
    scaled = mt.tl.reconstruct_labels(
        observed, proba, categories=["A", "longer-label-B"], threshold=.5,
    )
    np.testing.assert_allclose(scaled.anomaly_scores, [1, 0, 0])
    low_conf = mt.tl.reconstruct_labels(
        observed, proba, categories=["A", "longer-label-B"], threshold=.5,
        min_confidence=.95,
    )
    assert not low_conf.relabelled.any()


def test_reconstruction_spatial_gate_other_candidates_not_all_patches():
    observed = [0, 0, 0, 0]
    proba = np.array([[.1, .9], [.2, .8], [.8, .2], [.1, .9]])
    xy = np.array([[0., 0], [1, 0], [0, 1], [9, 0]])
    result = mt.tl.reconstruct_labels(
        observed, proba, threshold=.5, score="one_minus_p",
        coordinates=xy, spatial_radius=2, min_neighbours=1,
    )
    assert result.relabelled.tolist() == [True, True, False, False]
    np.testing.assert_array_equal(result.labels, [1, 1, 0, 0])


def test_reconstruction_probability_alignment_calibration_and_empty_signal():
    observed = pd.Series(["A", "B"], index=["p0", "p1"])
    proba = pd.DataFrame([[.2, .8], [.7, .3]], index=["p1", "p0"], columns=["A", "B"])
    result = mt.tl.reconstruct_labels(
        observed, proba, threshold=0., temperature=2.,
    )
    expected = np.sqrt(proba.loc[observed.index].to_numpy())
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result.probabilities, expected)
    none = mt.tl.reconstruct_labels(
        ["background"], [[.8, .2]], threshold=0., categories=["A", "B"],
    )
    assert none.labels.tolist() == ["background"] and not none.relabelled.any()
    empty = mt.tl.reconstruct_labels([], np.empty((0, 2)), threshold=0.)
    assert empty.labels.size == 0 and empty.probabilities.shape == (0, 2)


def test_reconstruction_signal_and_coordinates_align_to_observed_ids():
    observed = pd.Series([0, 0, 0], index=["p0", "p1", "p2"])
    proba = pd.DataFrame(
        [[.1, .9], [.2, .8], [.1, .9]],
        index=observed.index, columns=[0, 1],
    )
    signal = pd.Series([False, True, True], index=["p2", "p0", "p1"])
    xy = pd.DataFrame([[1, 0], [50, 0], [0, 0]], index=["p1", "p2", "p0"])
    result = mt.tl.reconstruct_labels(
        observed, proba, signal=signal, coordinates=xy,
        threshold=.5, score="one_minus_p", spatial_radius=2., min_neighbours=1,
    )
    assert result.labels.tolist() == [1, 1, 0]
    explicit = mt.tl.reconstruct_labels(
        observed.to_numpy(), proba.to_numpy(), ids=observed.index,
        signal=signal, coordinates=xy,
        threshold=.5, score="one_minus_p", spatial_radius=2., min_neighbours=1,
    )
    np.testing.assert_array_equal(explicit.labels, result.labels)


def test_reconstruction_validates_probabilities_and_gate():
    with pytest.raises(ValueError, match="sum to one"):
        mt.tl.reconstruct_labels([0], [[.2, .2]], threshold=0.)
    with pytest.raises(ValueError, match="positive spatial_radius"):
        mt.tl.reconstruct_labels([0], [[.1, .9]], threshold=0., min_neighbours=1)
    with pytest.raises(ValueError, match="coordinates"):
        mt.tl.reconstruct_labels(
            [0], [[.1, .9]], threshold=0., spatial_radius=2., min_neighbours=1,
        )
