"""Contracts for the patch long table, the composition contrast and the membership correlation."""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantpy as mt
from mantpy._analysis import ClusterCompositionDetails

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _patch_adata(labels, *, n_columns: int = 4, patch_size: int = 10) -> ad.AnnData:
    """Carrier with a lattice patch table: float centres, int32 labels and two feature columns."""
    labels = np.asarray(labels)
    position = np.arange(len(labels))
    carrier = ad.AnnData(X=np.empty((0, 0), dtype=np.float32))
    carrier.uns["ecm_patches"] = pd.DataFrame(
        {
            "x": (position % n_columns) * patch_size + patch_size / 2,
            "y": (position // n_columns) * patch_size + patch_size / 2,
            "ecm_cluster": labels.astype(np.int32),
            "feat_0": np.linspace(0.0, 1.0, len(labels)),
            "feat_1": np.linspace(1.0, 2.0, len(labels)),
        }
    )
    return carrier


def _image_cohort() -> dict[str, ad.AnnData]:
    """Two ROIs through the public extraction path, then labelled with some background patches."""
    rng = np.random.default_rng(0)
    images = {
        "roi_b": rng.random((3, 20, 30), dtype=np.float32),
        "roi_a": rng.random((3, 15, 25), dtype=np.float32),
    }
    panel = pd.DataFrame({"name": ["c1", "c2", "c3"], "keep": [1, 1, 1], "ecm": [1, 1, 1]})
    cohort = mt.pp.ecm_patches_from_images(
        images, panel, normalize="none", patch_size=5, features=["mean"], ecm_K=None, background_quantile=0.0
    )
    for adata in cohort.values():
        table = adata.uns["ecm_patches"].copy()
        labels = rng.integers(0, 3, size=len(table)).astype(np.int32)
        labels[rng.random(len(table)) < 0.3] = -1
        table["ecm_cluster"] = labels
        adata.uns["ecm_patches"] = table
    return cohort


def _details(per_unit: pd.DataFrame, groups) -> ClusterCompositionDetails:
    """Hand-made composition result; only ``per_unit`` and ``unit_meta`` feed the contrast."""
    unit_meta = pd.DataFrame({"group": list(groups)}, index=per_unit.index)
    return ClusterCompositionDetails(
        per_sample=per_unit,
        per_group_mean=per_unit.groupby(unit_meta["group"].to_numpy()).mean(),
        counts=per_unit,
        per_unit=per_unit,
        per_unit_counts=per_unit,
        unit_meta=unit_meta,
        unit_aggregation="pool_counts",
    )


# ---------------------------------------------------------------------------
# A3. mt.tl.ecm_patch_table
# ---------------------------------------------------------------------------


def test_ecm_patch_table_is_exported():
    assert "ecm_patch_table" in mt.tl.__all__
    assert "cluster_membership_correlation" in mt.tl.__all__


def test_ecm_patch_table_node_ids_match_graph_nodes():
    cohort = _image_cohort()
    for adata in cohort.values():
        mt.gr.build_ecm_graph(adata, edge_method="delaunay")

    table = mt.tl.ecm_patch_table(cohort)
    signal = mt.tl.ecm_patch_table(cohort, drop_background=True)

    every_node = {
        (sample, node, attrs["x"], attrs["y"], attrs["ecm_cluster"])
        for sample, adata in cohort.items()
        for node, attrs in adata.uns["ecm_graph"].nodes(data=True)
    }
    key = ["sample_id", "node_id", "x", "y", "ecm_cluster"]
    assert set(table[key].itertuples(index=False, name=None)) == every_node
    assert len(table) == len(every_node) == sum(len(adata.uns["ecm_patches"]) for adata in cohort.values())

    signal_nodes = {node for node in every_node if node[4] >= 0}
    assert 0 < len(signal_nodes) < len(every_node)
    assert set(signal[key].itertuples(index=False, name=None)) == signal_nodes
    assert len(signal) == len(signal_nodes)
    # Node order inside each sample is the graph's own node order.
    for sample, adata in cohort.items():
        assert table.loc[table.sample_id == sample, "node_id"].tolist() == list(adata.uns["ecm_graph"].nodes)


def test_ecm_patch_table_rows_match_one_hot_cluster_graphs():
    pytest.importorskip("torch_geometric")
    cohort = _image_cohort()
    for adata in cohort.values():
        mt.gr.build_ecm_graph(adata, edge_method="grid")
    meta = pd.DataFrame({"condition": ["Infected", "Naive"]}, index=pd.Index(["roi_a", "roi_b"], name="sample_id"))

    learning = mt.nn.build_ohe_cluster_graphs(cohort, meta, group_col="condition")
    signal = mt.tl.ecm_patch_table(cohort, sample_meta=meta, drop_background=True)

    assert learning.roi_ids == ["roi_a", "roi_b"]
    assert signal.sample_id.drop_duplicates().tolist() == learning.roi_ids
    for graph in learning.graphs:
        rows = signal[signal.sample_id == graph.roi_id]
        assert rows.node_id.tolist() == graph.node_ids
        np.testing.assert_array_equal(rows[["x", "y"]].to_numpy(np.float32), graph.pos.numpy())
        np.testing.assert_array_equal(rows.ecm_cluster.to_numpy(), graph.cluster.numpy())
        assert set(rows.condition) == {meta.loc[graph.roi_id, "condition"]}


def test_ecm_patch_table_columns_dtypes_and_feature_columns():
    cohort = {"s1": _patch_adata([0, -1, 1, 1, 0]), "s2": _patch_adata([2, 2, -1])}

    table = mt.tl.ecm_patch_table(cohort)

    assert table.columns.tolist() == ["sample_id", "node_id", "x", "y", "ecm_cluster"]
    assert isinstance(table.index, pd.RangeIndex) and len(table) == 8
    assert table.sample_id.tolist() == ["s1"] * 5 + ["s2"] * 3
    assert table.node_id.tolist() == [f"ecm_{i}" for i in range(5)] + [f"ecm_{i}" for i in range(3)]
    assert table.x.dtype == np.float64 and table.y.dtype == np.float64
    assert table.ecm_cluster.dtype == np.int32
    np.testing.assert_array_equal(table.x[:5], cohort["s1"].uns["ecm_patches"].x)
    # The stored tables are read, not changed.
    assert cohort["s1"].uns["ecm_patches"].columns.tolist() == ["x", "y", "ecm_cluster", "feat_0", "feat_1"]

    other_prefix = mt.tl.ecm_patch_table(cohort, feature_prefix="ecm_")
    assert other_prefix.columns.tolist() == ["sample_id", "node_id", "x", "y", "feat_0", "feat_1"]


def test_ecm_patch_table_ignores_the_stored_index():
    carrier = _patch_adata([0, 1, -1, 1])
    carrier.uns["ecm_patches"].index = [10, 11, 12, 13]

    table = mt.tl.ecm_patch_table({"s": carrier}, drop_background=True)

    assert table.node_id.tolist() == ["ecm_0", "ecm_1", "ecm_3"]


def test_ecm_patch_table_drop_background_keeps_original_ids():
    cohort = {"s1": _patch_adata([-1, 0, -1, 2, 1, -1]), "s2": _patch_adata([1, -1, 0])}

    signal = mt.tl.ecm_patch_table(cohort, drop_background=True)

    assert isinstance(signal.index, pd.RangeIndex)
    assert signal.sample_id.tolist() == ["s1", "s1", "s1", "s2", "s2"]
    assert signal.node_id.tolist() == ["ecm_1", "ecm_3", "ecm_4", "ecm_0", "ecm_2"]
    assert signal.ecm_cluster.tolist() == [0, 2, 1, 1, 0]
    assert signal.ecm_cluster.dtype == np.int32
    complete = mt.tl.ecm_patch_table(cohort)
    pd.testing.assert_frame_equal(signal, complete[complete.ecm_cluster >= 0].reset_index(drop=True))


def test_ecm_patch_table_drop_background_drops_missing_labels_and_other_columns():
    carrier = _patch_adata([0, 0, 0, 0])
    table = carrier.uns["ecm_patches"]
    table["ecm_cluster"] = [0.0, np.nan, -1.0, 2.0]
    table["domain"] = pd.array([1, pd.NA, -1, 0], dtype="Int64")
    table["state"] = ["fibrotic", None, -1, "normal"]

    assert mt.tl.ecm_patch_table({"s": carrier}, drop_background=True).node_id.tolist() == ["ecm_0", "ecm_3"]
    assert mt.tl.ecm_patch_table({"s": carrier}, cluster_col="domain", drop_background=True).node_id.tolist() == [
        "ecm_0",
        "ecm_3",
    ]
    # Non-numeric labels follow compute_cluster_composition: only missing and negative numbers are background.
    assert mt.tl.ecm_patch_table({"s": carrier}, cluster_col="state", drop_background=True).node_id.tolist() == [
        "ecm_0",
        "ecm_3",
    ]


def test_ecm_patch_table_meta_join_and_order():
    cohort = {"s1": _patch_adata([0, 1]), "s2": _patch_adata([1]), "s3": _patch_adata([2, -1, 0])}
    meta = pd.DataFrame(
        {"mouse": ["m2", "m1"], "group": pd.Categorical(["KO", "WT"]), "age": [8, 12]},
        index=pd.Index(["s3", "s1"], name="sample_id"),
    )

    table = mt.tl.ecm_patch_table(cohort, sample_meta=meta)

    assert table.columns.tolist() == ["sample_id", "node_id", "x", "y", "ecm_cluster", "mouse", "group", "age"]
    assert table.sample_id.tolist() == ["s3", "s3", "s3", "s1", "s1"]  # sample_meta order; s2 is left out
    assert table.mouse.tolist() == ["m2"] * 3 + ["m1"] * 2
    assert table.group.tolist() == ["KO"] * 3 + ["WT"] * 2
    assert table.age.tolist() == [8, 8, 8, 12, 12]
    assert table.age.dtype == meta.age.dtype and table.group.dtype == meta.group.dtype
    assert isinstance(table.index, pd.RangeIndex)

    without_meta = mt.tl.ecm_patch_table(cohort)
    assert without_meta.sample_id.drop_duplicates().tolist() == ["s1", "s2", "s3"]  # adatas order


def test_ecm_patch_table_equals_notebook_long_table():
    """Cell 13 of the liver notebook, written by hand, against the replacement call."""
    cohort = {"r2": _patch_adata([1, -1, 0, 2, -1, 1, 0]), "r1": _patch_adata([-1, -1, 3, 0])}
    meta = pd.DataFrame(
        {"mouse": ["mouse_02", "mouse_01"], "group": ["Infected WT", "Naive WT"], "other": [1, 2]},
        index=pd.Index(["r2", "r1"], name="roi"),
    )
    signal = {roi: adata.uns["ecm_patches"]["ecm_cluster"].to_numpy() >= 0 for roi, adata in cohort.items()}
    draft = pd.concat(
        [
            pd.DataFrame(
                {
                    "roi": roi,
                    "mouse": meta.loc[roi, "mouse"],
                    "group": meta.loc[roi, "group"],
                    "node_id": [f"ecm_{row}" for row in range(len(signal[roi]))],
                    "x": adata.uns["ecm_patches"]["x"].to_numpy(int),
                    "y": adata.uns["ecm_patches"]["y"].to_numpy(int),
                    "cluster": adata.uns["ecm_patches"]["ecm_cluster"].to_numpy(int),
                }
            )[signal[roi]]
            for roi, adata in cohort.items()
        ],
        ignore_index=True,
    )

    patches = (
        mt.tl.ecm_patch_table(cohort, sample_meta=meta[["mouse", "group"]], drop_background=True)
        .rename(columns={"sample_id": "roi", "ecm_cluster": "cluster"})
        .astype({"x": int, "y": int, "cluster": int})
    )

    pd.testing.assert_frame_equal(patches[draft.columns.tolist()], draft)


def test_ecm_patch_table_errors():
    cohort = {"s1": _patch_adata([0, 1]), "s2": _patch_adata([1, -1])}
    index = pd.Index(["s1", "s2"], name="sample_id")

    with pytest.raises(ValueError, match="adatas is empty; expected at least one ECM sample."):
        mt.tl.ecm_patch_table({})
    with pytest.raises(ValueError, match=r"Sample 's1' has no uns\['patches'\]; run mt.pp.extract_ecm_patches first."):
        mt.tl.ecm_patch_table(cohort, patches_key="patches")
    with pytest.raises(ValueError, match="sample_meta must have unique sample IDs."):
        mt.tl.ecm_patch_table(cohort, sample_meta=pd.DataFrame({"group": ["a", "b"]}, index=["s1", "s1"]))
    with pytest.raises(KeyError, match="sample_meta contains samples absent from adatas."):
        mt.tl.ecm_patch_table(cohort, sample_meta=pd.DataFrame({"group": ["a"]}, index=["s9"]))
    with pytest.raises(ValueError, match=r"sample_meta columns \['x'\] collide with patch columns."):
        mt.tl.ecm_patch_table(cohort, sample_meta=pd.DataFrame({"x": [1, 2], "group": ["a", "b"]}, index=index))
    with pytest.raises(ValueError, match=r"sample_meta columns \['node_id'\] collide with patch columns."):
        mt.tl.ecm_patch_table(cohort, sample_meta=pd.DataFrame({"node_id": [1, 2]}, index=index))
    with pytest.raises(KeyError, match="Sample 's1' has no 'domain' column."):
        mt.tl.ecm_patch_table(cohort, cluster_col="domain", drop_background=True)
    # The label column is only needed for filtering.
    assert len(mt.tl.ecm_patch_table(cohort, cluster_col="domain")) == 4

    clashing = _patch_adata([0, 1])
    clashing.uns["ecm_patches"]["node_id"] = ["a", "b"]
    with pytest.raises(ValueError, match=r"Sample 's': patch columns \['node_id'\] collide"):
        mt.tl.ecm_patch_table({"s": clashing})


# ---------------------------------------------------------------------------
# A4. ClusterCompositionDetails.contrast
# ---------------------------------------------------------------------------


def _mouse_cohort():
    """Two case mice with two ROIs each and two control mice with one ROI each."""
    from collections import Counter

    roi_labels = {
        "a": [0, 0, 1, -1],  # mouse m1 pools a and b: 3 x cluster 0, 4 x cluster 1
        "b": [0, 1, 1, 1],
        "c": [0, 2],  # mouse m2 pools c and d: 1, 1, 2
        "d": [1, 2],
        "e": [0, 0, 0, 1],  # mouse m3: 3, 1, 0
        "f": [0, 1],  # mouse m4: 1, 1, 0
    }
    assert Counter(roi_labels["a"] + roi_labels["b"]) == {0: 3, 1: 4, -1: 1}
    cohort = {roi: _patch_adata(labels) for roi, labels in roi_labels.items()}
    meta = pd.DataFrame(
        {
            "mouse": ["m1", "m1", "m2", "m2", "m3", "m4"],
            "group": ["Infected KO", "Infected KO", "Infected WT", "Infected WT", "Naive KO", "Naive WT"],
        },
        index=pd.Index(list(roi_labels), name="sample_id"),
    )
    return cohort, meta


def _mouse_composition():
    cohort, meta = _mouse_cohort()
    return mt.tl.compute_cluster_composition(
        cohort,
        sample_meta=meta,
        group_col="group",
        label_source="patches",
        drop_background=True,
        unit_col="mouse",
        unit_aggregation="pool_counts",
        return_details=True,
    )


def test_contrast_values_match_hand_computation():
    composition = _mouse_composition()

    table = composition.contrast(
        case=["Infected KO", "Infected WT"], control=["Naive KO", "Naive WT"], pseudocount=1e-3, min_fraction=0.01
    )

    assert table.columns.tolist() == [
        "cluster",
        "mean_case",
        "mean_control",
        "difference",
        "log2_ratio",
        "min_case_fraction",
        "present",
        "n_case_units",
        "n_control_units",
    ]
    assert isinstance(table.index, pd.RangeIndex)
    assert table.cluster.tolist() == [0, 1, 2]
    # Mouse fractions: m1 = (3/7, 4/7, 0), m2 = (1/4, 1/4, 1/2); m3 = (3/4, 1/4, 0), m4 = (1/2, 1/2, 0).
    mean_case = np.array([(3 / 7 + 1 / 4) / 2, (4 / 7 + 1 / 4) / 2, (0 + 1 / 2) / 2])
    mean_control = np.array([(3 / 4 + 1 / 2) / 2, (1 / 4 + 1 / 2) / 2, 0.0])
    np.testing.assert_allclose(table.mean_case, mean_case, rtol=0, atol=1e-15)
    np.testing.assert_allclose(table.mean_control, mean_control, rtol=0, atol=1e-15)
    np.testing.assert_allclose(table.difference, mean_case - mean_control, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        table.log2_ratio, np.log2((mean_case + 0.001) / (mean_control + 0.001)), rtol=0, atol=1e-12
    )
    # Exact ratios 1191/2191, 1441/1316 and 251, from the fractions 19/56, 23/56, 1/4 against 5/8, 3/8, 0.
    np.testing.assert_allclose(table.log2_ratio, [-0.879416071126, 0.130910846440, 7.971543553951], rtol=0, atol=1e-11)
    np.testing.assert_allclose(table.min_case_fraction, [1 / 4, 1 / 4, 0.0], rtol=0, atol=1e-15)
    assert table.present.tolist() == [True, True, False]
    assert table.present.dtype == bool
    assert table.n_case_units.tolist() == [2, 2, 2]
    assert table.n_control_units.tolist() == [2, 2, 2]


def test_contrast_weights_units_equally_not_rois_or_patches():
    composition = _mouse_composition()

    table = composition.contrast(case=["Infected KO", "Infected WT"], control=["Naive KO", "Naive WT"])

    unit_mean = (3 / 7 + 1 / 4) / 2
    roi_mean = (2 / 3 + 1 / 4 + 1 / 2 + 0) / 4
    patch_pooled = 4 / 11
    assert table.mean_case[0] == pytest.approx(unit_mean, abs=1e-15)
    assert abs(unit_mean - roi_mean) > 0.01 and abs(unit_mean - patch_pooled) > 0.01


def test_contrast_min_fraction_excludes_cluster_then_top_enriched_picks_next():
    composition = _mouse_composition()
    groups = {"case": ["Infected KO", "Infected WT"], "control": ["Naive KO", "Naive WT"]}

    # Cluster 2 has by far the largest ratio but is absent from mouse m1.
    without_threshold = composition.contrast(**groups)
    assert without_threshold.present.tolist() == [True, True, True]
    assert mt.tl.top_enriched_cluster(without_threshold, log2_col="log2_ratio", sig_col="present") == 2

    with_threshold = composition.contrast(**groups, min_fraction=0.01)
    assert with_threshold.present.tolist() == [True, True, False]
    assert mt.tl.top_enriched_cluster(with_threshold, log2_col="log2_ratio", sig_col="present") == 1

    # The threshold is inclusive: both case mice hold exactly 1/4 of clusters 0 and 1.
    assert composition.contrast(**groups, min_fraction=0.25).present.tolist() == [True, True, False]
    assert composition.contrast(**groups, min_fraction=0.26).present.tolist() == [False, False, False]


def test_contrast_without_present_cluster_raises_in_selector():
    composition = _mouse_composition()

    table = composition.contrast(["Infected KO", "Infected WT"], ["Naive KO", "Naive WT"], min_fraction=1.0)

    assert not table.present.any()
    with pytest.raises(ValueError, match="no rows with positive log2 enrichment"):
        mt.tl.top_enriched_cluster(table, log2_col="log2_ratio", sig_col="present")


def test_contrast_tie_selects_first_cluster():
    per_unit = pd.DataFrame(
        {3: [0.1, 0.1, 0.4, 0.4], 1: [0.3, 0.5, 0.1, 0.1], 2: [0.3, 0.5, 0.1, 0.1], 0: [0.3, -0.1, 0.4, 0.4]},
        index=pd.Index(["u1", "u2", "u3", "u4"], name="mouse"),
    )
    details = _details(per_unit, ["case", "case", "control", "control"])

    table = details.contrast("case", "control")

    assert table.cluster.tolist() == [3, 1, 2, 0]  # per_unit column order
    assert table.log2_ratio[1] == table.log2_ratio[2] == table.log2_ratio.max()
    assert mt.tl.top_enriched_cluster(table, log2_col="log2_ratio", sig_col="present") == 1
    assert table.present.tolist() == [True, True, True, False]


def test_contrast_accepts_one_label_or_several():
    per_unit = pd.DataFrame({0: [0.2, 0.4, 0.6, 0.8], 1: [0.8, 0.6, 0.4, 0.2]}, index=list("abcd"))
    details = _details(per_unit, ["x", "y", "z", "z"])

    single = details.contrast("z", "x")
    pd.testing.assert_frame_equal(single, details.contrast(["z"], ("x",)))
    assert single.n_case_units.tolist() == [2, 2] and single.n_control_units.tolist() == [1, 1]
    np.testing.assert_allclose(single.mean_case, [0.7, 0.3])
    np.testing.assert_allclose(single.mean_control, [0.2, 0.8])
    # Group "y" is in neither side and is ignored.
    np.testing.assert_allclose(details.contrast(["x", "y"], "z").mean_case, [0.3, 0.7])


@pytest.mark.parametrize("seed", range(5))
def test_contrast_equals_draft_pandas_expression(seed):
    """Cell 15 of the liver notebook, written by hand, against the method (bit-identical)."""
    rng = np.random.default_rng(seed)
    per_mouse = pd.DataFrame(
        rng.dirichlet(np.full(6, 0.4), size=8),
        index=pd.Index([f"mouse_{i:02d}" for i in range(1, 9)], name="mouse"),
        columns=pd.Index(range(6), name="cluster"),
    )
    group_names = ["Naive WT", "Infected WT", "Naive KO", "Infected KO"]
    details = _details(per_mouse, [group_names[i % 4] for i in rng.permutation(8)])
    mouse_group = details.unit_meta["group"]

    infected_mice = per_mouse[mouse_group.str.startswith("Infected").to_numpy()]
    naive_mice = per_mouse[mouse_group.str.startswith("Naive").to_numpy()]
    log2_ratio = np.log2((infected_mice.mean() + 1e-3) / (naive_mice.mean() + 1e-3))
    present = (infected_mice >= 0.01).all()

    table = details.contrast(
        case=["Infected KO", "Infected WT"], control=["Naive KO", "Naive WT"], pseudocount=1e-3, min_fraction=0.01
    )

    np.testing.assert_array_equal(table.cluster, per_mouse.columns)
    np.testing.assert_array_equal(table.mean_case, infected_mice.mean())
    np.testing.assert_array_equal(table.mean_control, naive_mice.mean())
    np.testing.assert_array_equal(table.difference, infected_mice.mean() - naive_mice.mean())
    np.testing.assert_array_equal(table.log2_ratio, log2_ratio)
    np.testing.assert_array_equal(table.present, present)
    assert table.n_case_units.eq(4).all() and table.n_control_units.eq(4).all()
    assert present.any() and (log2_ratio[present] > 0).any()
    assert mt.tl.top_enriched_cluster(table, log2_col="log2_ratio", sig_col="present") == int(
        log2_ratio[present].idxmax()
    )


def test_contrast_errors():
    composition = _mouse_composition()

    with pytest.raises(KeyError, match=r"Groups \['Infected'\] are not in unit_meta; available: \['Infected KO', "):
        composition.contrast("Infected", "Naive KO")
    with pytest.raises(KeyError, match=r"Groups \['Naive'\] are not in unit_meta"):
        composition.contrast("Infected KO", ["Naive KO", "Naive"])
    with pytest.raises(ValueError, match="case and control groups overlap."):
        composition.contrast(["Infected KO", "Naive KO"], ["Naive KO", "Naive WT"])
    with pytest.raises(ValueError, match="case has no units."):
        composition.contrast([], "Naive KO")
    with pytest.raises(ValueError, match="control has no units."):
        composition.contrast("Infected KO", [])
    for pseudocount in (0.0, -1e-3, np.nan, np.inf):
        with pytest.raises(ValueError, match="pseudocount must be finite and positive."):
            composition.contrast("Infected KO", "Naive KO", pseudocount=pseudocount)
    for min_fraction in (-0.01, 1.01, np.nan):
        with pytest.raises(ValueError, match=r"min_fraction must lie in \[0, 1\]."):
            composition.contrast("Infected KO", "Naive KO", min_fraction=min_fraction)


def test_contrast_leaves_the_result_and_legacy_tuple_unchanged():
    cohort, meta = _mouse_cohort()
    composition = _mouse_composition()
    before = composition.per_unit.copy()

    composition.contrast("Infected KO", "Naive KO")

    pd.testing.assert_frame_equal(composition.per_unit, before)
    legacy = mt.tl.compute_cluster_composition(cohort, sample_meta=meta)
    assert not hasattr(legacy, "contrast")
    assert legacy._fields == ("per_sample", "per_group_mean")


# ---------------------------------------------------------------------------
# A7. mt.tl.cluster_membership_correlation
# ---------------------------------------------------------------------------


def _draft_localisation(patches: pd.DataFrame, n_clusters: int, *, min_cluster_nodes: int = 2) -> pd.DataFrame:
    """The nested loop of cell 31 of the liver notebook."""
    rows = []
    for roi, here in patches.groupby("roi", sort=True):
        for k in range(n_clusters):
            member = (here.cluster == k).to_numpy(float)
            defined = min_cluster_nodes <= member.sum() < len(member) and here.abs_ig.std() > 0
            rows.append(
                {
                    "roi": roi,
                    "cluster": k,
                    "n_nodes": len(member),
                    "n_cluster_nodes": int(member.sum()),
                    "pearson_r": np.corrcoef(here.abs_ig, member)[0, 1] if defined else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_cluster_membership_correlation_matches_corrcoef():
    table = pd.DataFrame(
        {
            "sample_id": ["s2"] * 6 + ["s1"] * 5,
            "cluster": [0, 0, 1, 1, 2, 2, 1, 1, 0, 0, 0],
            "score": [0.9, 0.7, 0.2, 0.1, 0.4, 0.6, 5.0, 4.0, 1.0, 2.0, 0.0],
        }
    )

    result = mt.tl.cluster_membership_correlation(table, score_col="score")

    assert result.columns.tolist() == ["sample_id", "cluster", "n_nodes", "n_cluster_nodes", "pearson_r"]
    assert isinstance(result.index, pd.RangeIndex)
    assert result.sample_id.tolist() == ["s1"] * 3 + ["s2"] * 3  # sorted samples
    assert result.cluster.tolist() == [0, 1, 2, 0, 1, 2]  # inferred, ascending
    assert result.n_nodes.tolist() == [5, 5, 5, 6, 6, 6]
    assert result.n_cluster_nodes.tolist() == [3, 2, 0, 2, 2, 2]
    assert result.n_nodes.dtype == np.int64 and result.n_cluster_nodes.dtype == np.int64
    assert result.pearson_r.dtype == np.float64

    for row in result.itertuples():
        here = table[table.sample_id == row.sample_id]
        member = (here.cluster == row.cluster).to_numpy(float)
        if row.n_cluster_nodes == 0:
            assert np.isnan(row.pearson_r)
        else:
            assert row.pearson_r == np.corrcoef(here.score.to_numpy(), member)[0, 1]
    # In s1 the two cluster-1 nodes carry the highest scores.
    s1 = result[result.sample_id == "s1"].set_index("cluster").pearson_r
    assert s1[1] > 0.9 and s1[0] == pytest.approx(-s1[1])


def test_cluster_membership_correlation_undefined_cases_are_nan():
    table = pd.DataFrame(
        {
            "sample_id": ["one"] * 4 + ["all"] * 3 + ["flat"] * 4 + ["solo"],
            "cluster": [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            "score": [1.0, 2.0, 3.0, 5.0, 1.0, 2.0, 3.0, 0.3, 0.3, 0.3, 0.3, 4.0],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no division warnings from undefined correlations
        result = mt.tl.cluster_membership_correlation(table, score_col="score", categories=[0, 1, 2])
    by_key = result.set_index(["sample_id", "cluster"])

    assert len(result) == 12  # every category for every sample, absent ones included
    # n_k = 1: below min_cluster_nodes.
    assert by_key.loc[("one", 0), "n_cluster_nodes"] == 1 and np.isnan(by_key.loc[("one", 0), "pearson_r"])
    assert by_key.loc[("one", 1), "n_cluster_nodes"] == 3 and np.isfinite(by_key.loc[("one", 1), "pearson_r"])
    # n_k = n: the cluster holds every node.
    assert by_key.loc[("all", 1), "n_cluster_nodes"] == 3 == by_key.loc[("all", 1), "n_nodes"]
    assert np.isnan(by_key.loc[("all", 1), "pearson_r"])
    # Constant score.
    assert by_key.loc[("flat", 0), "n_cluster_nodes"] == 2 and by_key.loc["flat", "pearson_r"].isna().all()
    # Absent category: row present, zero nodes.
    assert (by_key.xs(2, level="cluster").n_cluster_nodes == 0).all()
    assert by_key.xs(2, level="cluster").pearson_r.isna().all()
    # One node in the sample.
    assert by_key.loc["solo", "pearson_r"].isna().all()

    with_singletons = mt.tl.cluster_membership_correlation(
        table, score_col="score", categories=[0, 1, 2], min_cluster_nodes=1
    ).set_index(["sample_id", "cluster"])
    here = table[table.sample_id == "one"]
    assert with_singletons.loc[("one", 0), "pearson_r"] == np.corrcoef(here.score, here.cluster == 0)[0, 1]
    stricter = mt.tl.cluster_membership_correlation(table, score_col="score", min_cluster_nodes=4)
    assert stricter.pearson_r.isna().all()


def test_cluster_membership_correlation_constant_score_with_inexact_mean_is_nan():
    table = pd.DataFrame({"sample_id": "s", "cluster": [0, 0, 0, 1, 1, 1, 1], "score": [0.3] * 7})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = mt.tl.cluster_membership_correlation(table, score_col="score")

    assert result.pearson_r.isna().all()


def test_cluster_membership_correlation_labels_outside_categories_count_as_nodes():
    table = pd.DataFrame(
        {
            "sample_id": ["s"] * 6,
            "cluster": [0, 0, 1, 1, -1, -1],
            "score": [3.0, 2.5, 1.0, 0.5, 0.1, 0.2],
        }
    )

    result = mt.tl.cluster_membership_correlation(table, score_col="score", categories=range(2))

    assert result.cluster.tolist() == [0, 1]
    assert result.n_nodes.tolist() == [6, 6]
    assert result.pearson_r[0] == np.corrcoef(table.score, table.cluster == 0)[0, 1]


@pytest.mark.parametrize("seed", range(3))
def test_cluster_membership_correlation_equals_draft_loop(seed):
    rng = np.random.default_rng(seed)
    sizes = {"roi_c": 40, "roi_a": 25, "roi_b": 3}
    patches = pd.DataFrame(
        {
            "roi": np.repeat(list(sizes), list(sizes.values())),
            "cluster": rng.choice(6, size=sum(sizes.values()), p=[0.4, 0.3, 0.15, 0.1, 0.04, 0.01]),
            "abs_ig": np.abs(rng.normal(size=sum(sizes.values()))).astype(np.float32),
        }
    ).sample(frac=1.0, random_state=seed)  # row order must not matter for the grouping

    result = mt.tl.cluster_membership_correlation(
        patches, score_col="abs_ig", sample_col="roi", categories=range(6), min_cluster_nodes=2
    )

    draft = _draft_localisation(patches, 6)
    assert result.pearson_r.isna().any() and result.pearson_r.notna().any()
    pd.testing.assert_frame_equal(result, draft, check_exact=True)


def test_cluster_membership_correlation_string_labels_and_column_names():
    table = pd.DataFrame(
        {
            "roi": ["r"] * 6,
            "domain": ["stroma", "stroma", "lesion", "lesion", "lesion", "stroma"],
            "importance": [0.1, 0.2, 0.9, 0.8, 0.7, 0.3],
        }
    )

    result = mt.tl.cluster_membership_correlation(
        table, score_col="importance", cluster_col="domain", sample_col="roi", categories=["lesion", "stroma", "void"]
    )

    assert result.columns.tolist() == ["roi", "cluster", "n_nodes", "n_cluster_nodes", "pearson_r"]
    assert result.cluster.tolist() == ["lesion", "stroma", "void"]
    assert result.pearson_r[0] > 0.9 and result.pearson_r[1] == pytest.approx(-result.pearson_r[0])
    assert np.isnan(result.pearson_r[2])


def test_cluster_membership_correlation_errors():
    table = pd.DataFrame({"sample_id": ["s"] * 4, "cluster": [0, 0, 1, 1], "score": [1.0, 2.0, 3.0, 4.0]})

    with pytest.raises(KeyError, match=r"table is missing columns \['importance'\]."):
        mt.tl.cluster_membership_correlation(table, score_col="importance")
    with pytest.raises(KeyError, match=r"table is missing columns \['domain', 'roi'\]."):
        mt.tl.cluster_membership_correlation(table, score_col="score", cluster_col="domain", sample_col="roi")
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="score contains NaN or infinite values."):
            mt.tl.cluster_membership_correlation(table.assign(score=[1.0, bad, 3.0, 4.0]), score_col="score")
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="min_cluster_nodes must be a positive integer."):
            mt.tl.cluster_membership_correlation(table, score_col="score", min_cluster_nodes=bad)
    with pytest.raises(ValueError, match="categories must contain unique, non-missing labels."):
        mt.tl.cluster_membership_correlation(table, score_col="score", categories=[0, 0, 1])

    empty = mt.tl.cluster_membership_correlation(table.iloc[:0], score_col="score")
    assert empty.columns.tolist() == ["sample_id", "cluster", "n_nodes", "n_cluster_nodes", "pearson_r"]
    assert empty.empty
