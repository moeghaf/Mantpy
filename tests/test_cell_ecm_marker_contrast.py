"""Tests for ``mt.tl.cell_ecm_marker_contrast``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import mantpy as mt

FEATURES = ["feat_0", "feat_1", "feat_2"]
MARKERS = ["Fibronectin", "Col-I", "HABP"]


def _toy(seed: int = 0) -> AnnData:
    rng = np.random.default_rng(seed)
    adata = AnnData(X=np.zeros((30, 1), dtype=np.float32))
    adata.obs["cell_type"] = rng.choice(["A", "B", "C"], size=30)
    adata.obsm["spatial"] = rng.uniform(0, 60, size=(30, 2))
    gx, gy = np.meshgrid(np.arange(5, 60, 10), np.arange(5, 60, 10))
    patches = pd.DataFrame({"x": gx.ravel().astype(float), "y": gy.ravel().astype(float)})
    patches["ecm_cluster"] = rng.integers(-1, 3, size=len(patches))
    for name in FEATURES:
        patches[name] = rng.normal(size=len(patches)).astype(np.float32)
    patches.index = patches.index.astype(str)
    adata.uns["ecm_patches"] = patches
    mt.gr.ensure_cell_ecm_graph(
        adata, cell_edge_method="radius", cell_r=15,
        ecm_edge_method="grid", ecm_grid_connectivity=8,
        cell_ecm_edge_method="radius", cell_ecm_r=9,
    )
    return adata


def _cohort() -> dict[str, AnnData]:
    return {f"roi_{i}": _toy(i) for i in range(3)}


def _one_sample(adata, cluster=None, **options):
    patches = adata.uns["ecm_patches"]
    return mt.tl.contact_marker_contrast(
        mt.gr.graph_layers(adata).cell_ecm, patches[FEATURES],
        cell_labels=adata.obs["cell_type"].astype(str).to_numpy(), cell_type="A",
        patch_labels=None if cluster is None else patches["ecm_cluster"].to_numpy(), cluster=cluster,
        **options,
    )


@pytest.mark.parametrize("cluster", [None, 0, 1])
def test_each_sample_equals_contact_marker_contrast(cluster):
    cohort = _cohort()
    result = mt.tl.cell_ecm_marker_contrast(cohort, "A", cluster=cluster, ecm_names=MARKERS)
    assert list(result.per_sample.index) == list(cohort)
    assert list(result.per_sample.columns) == MARKERS
    for sample, adata in cohort.items():
        expected = _one_sample(adata, cluster)
        np.testing.assert_array_equal(result.per_sample.loc[sample].to_numpy(), expected.contrast.to_numpy())
        assert result.counts.loc[sample, "n_contacted"] == expected.n_contacted
        assert result.counts.loc[sample, "n_uncontacted"] == expected.n_uncontacted
        assert expected.n_contacted > 0 and expected.n_uncontacted > 0


def test_the_mean_counts_every_sample_once():
    result = mt.tl.cell_ecm_marker_contrast(_cohort(), "A", cluster=1, ecm_names=MARKERS)
    pd.testing.assert_series_equal(result.mean, result.per_sample.mean().rename("contrast"))
    assert result.counts["n_contacted"].nunique() > 1  # the samples differ in size, the mean does not weight them


def test_contacted_marks_the_compared_patches_with_a_contact():
    cohort = _cohort()
    result = mt.tl.cell_ecm_marker_contrast(cohort, "A", cluster=1)
    for sample, adata in cohort.items():
        patches = adata.uns["ecm_patches"]
        mask = result.contacted[sample]
        assert mask.index.equals(patches.index)
        assert mask.dtype == bool
        assert int(mask.sum()) == result.counts.loc[sample, "n_contacted"]
        assert (patches.loc[mask, "ecm_cluster"] == 1).all()
        is_a = (adata.obs["cell_type"] == "A").to_numpy()
        touched = np.asarray(mt.gr.graph_layers(adata).cell_ecm[is_a].sum(axis=0)).ravel() > 0
        np.testing.assert_array_equal(mask.to_numpy(), touched & (patches["ecm_cluster"] == 1).to_numpy())

    everywhere = mt.tl.cell_ecm_marker_contrast(cohort, "A")
    for sample in cohort:
        assert int(everywhere.contacted[sample].sum()) == everywhere.counts.loc[sample, "n_contacted"]
        assert everywhere.contacted[sample].sum() >= result.contacted[sample].sum()


def test_feature_columns_are_named_by_ecm_names():
    cohort = _cohort()
    unnamed = mt.tl.cell_ecm_marker_contrast(cohort, "A", cluster=1)
    assert list(unnamed.per_sample.columns) == FEATURES
    first_two = mt.tl.cell_ecm_marker_contrast(cohort, "A", cluster=1, ecm_names=MARKERS[:2])
    assert list(first_two.per_sample.columns) == MARKERS[:2]
    np.testing.assert_array_equal(first_two.per_sample.to_numpy(), unnamed.per_sample[FEATURES[:2]].to_numpy())
    with pytest.raises(ValueError, match="ecm_names"):
        mt.tl.cell_ecm_marker_contrast(cohort, "A", ecm_names=[*MARKERS, "one too many"])


def test_weighting_and_dtype_are_passed_on():
    cohort = _cohort()
    result = mt.tl.cell_ecm_marker_contrast(cohort, "A", cluster=1, weighting="edge", accumulation_dtype="input")
    for sample, adata in cohort.items():
        expected = _one_sample(adata, 1, weighting="edge", accumulation_dtype="input")
        np.testing.assert_array_equal(result.per_sample.loc[sample].to_numpy(), expected.contrast.to_numpy())
    unique = mt.tl.cell_ecm_marker_contrast(cohort, "A", cluster=1)
    assert not np.allclose(result.per_sample.to_numpy(), unique.per_sample.to_numpy())


def test_one_anndata_and_a_sequence_are_accepted():
    cohort = _cohort()
    named = mt.tl.cell_ecm_marker_contrast(cohort, "A", cluster=1)
    single = mt.tl.cell_ecm_marker_contrast(cohort["roi_0"], "A", cluster=1)
    assert list(single.per_sample.index) == ["sample"]
    np.testing.assert_array_equal(single.per_sample.to_numpy()[0], named.per_sample.loc["roi_0"].to_numpy())
    listed = mt.tl.cell_ecm_marker_contrast(list(cohort.values()), "A", cluster=1)
    assert list(listed.per_sample.index) == ["sample_0", "sample_1", "sample_2"]
    np.testing.assert_array_equal(listed.per_sample.to_numpy(), named.per_sample.to_numpy())


def test_the_inputs_are_not_modified():
    cohort = _cohort()
    before = {sample: adata.uns["ecm_patches"].copy() for sample, adata in cohort.items()}
    keys = {sample: sorted(adata.uns) for sample, adata in cohort.items()}
    mt.tl.cell_ecm_marker_contrast(cohort, "A", cluster=1, ecm_names=MARKERS)
    for sample, adata in cohort.items():
        pd.testing.assert_frame_equal(adata.uns["ecm_patches"], before[sample])
        assert sorted(adata.uns) == keys[sample]


def test_missing_inputs_raise():
    cohort = _cohort()
    with pytest.raises(ValueError, match="No sample"):
        mt.tl.cell_ecm_marker_contrast({}, "A")

    no_patches = cohort["roi_0"].copy()
    del no_patches.uns["ecm_patches"]
    with pytest.raises(ValueError, match="attach_ecm_patches"):
        mt.tl.cell_ecm_marker_contrast({"roi_0": no_patches}, "A")

    with pytest.raises(ValueError, match="cell_kind"):
        mt.tl.cell_ecm_marker_contrast(cohort, "A", cell_attr="cell_kind")

    with pytest.raises(ValueError, match="domain"):
        mt.tl.cell_ecm_marker_contrast(cohort, "A", cluster=1, ecm_attr="domain")

    no_features = cohort["roi_1"].copy()
    no_features.uns["ecm_patches"] = no_features.uns["ecm_patches"].drop(columns=FEATURES)
    with pytest.raises(ValueError, match="feat_"):
        mt.tl.cell_ecm_marker_contrast({"roi_1": no_features}, "A")

    fewer = cohort["roi_2"].copy()
    fewer.uns["ecm_patches"] = fewer.uns["ecm_patches"].drop(columns=["feat_2"])
    with pytest.raises(ValueError, match="same feat_"):
        mt.tl.cell_ecm_marker_contrast({"roi_0": cohort["roi_0"], "roi_2": fewer}, "A")


def test_a_sample_without_contact_is_left_out_of_the_mean():
    cohort = _cohort()
    cohort["roi_0"].obs["cell_type"] = "B"
    result = mt.tl.cell_ecm_marker_contrast(cohort, "A", cluster=1)
    assert result.per_sample.loc["roi_0"].isna().all()
    assert result.counts.loc["roi_0", "n_contacted"] == 0
    assert not result.contacted["roi_0"].any()
    expected = result.per_sample.loc[["roi_1", "roi_2"]].mean()
    np.testing.assert_allclose(result.mean.to_numpy(), expected.to_numpy())
