"""Tests for :func:`mantpy.io.to_spatialdata`.

The public SpatialData contract supports the shape the cohort loaders produce: an
AnnData carrying ``uns['img']`` and an ``uns['ecm_patches']`` DataFrame, with
zero observations on the object itself.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantpy as mt

pytest.importorskip("spatialdata", reason="to_spatialdata needs the [spatial] extra")


def _roi(sid: str, *, n_patches: int = 12, with_patches: bool = True, with_img: bool = True) -> ad.AnnData:
    """An ROI shaped like the ones `fetch.load_balbc_pbs_lung` returns:
    0 observations, patches parked in `uns['ecm_patches']`."""
    adata = ad.AnnData(X=np.empty((0, 3), dtype=np.float32))
    if with_img:
        adata.uns["img"] = np.zeros((16, 20), dtype=np.float32)
    if with_patches:
        rng = np.random.default_rng(abs(hash(sid)) % 2**32)
        adata.uns["ecm_patches"] = pd.DataFrame(
            {
                "x": rng.integers(0, 20, n_patches).astype(float),
                "y": rng.integers(0, 16, n_patches).astype(float),
                "ecm_cluster": rng.integers(-1, 3, n_patches),
                "feat_0": rng.random(n_patches).astype(np.float32),
                "feat_1": rng.random(n_patches).astype(np.float32),
            }
        )
    return adata


def _cohort(n: int = 3, **kw) -> dict[str, ad.AnnData]:
    return {f"roi{i}": _roi(f"roi{i}", **kw) for i in range(1, n + 1)}


def test_cohort_round_trips_to_spatialdata():
    sdata = mt.io.to_spatialdata(_cohort())
    assert set(sdata.images) == {"roi1", "roi2", "roi3"}
    assert sdata.tables["ecm_patches"].n_obs == 36  # 3 ROIs x 12 patches


def test_table_carries_a_valid_instance_key():
    """Regression: `instance_key=None` raised `ValueError` for every real
    cohort, and nothing covered it."""
    t = mt.io.to_spatialdata(_cohort()).tables["ecm_patches"]
    meta = t.uns["spatialdata_attrs"]
    assert meta["instance_key"] == "instance_id"
    assert meta["region_key"] == "region"
    # ids must be unique *within* a region, not globally
    per_region = t.obs.groupby("region", observed=True)["instance_id"]
    assert per_region.apply(lambda s: s.is_unique).all()


def test_patch_centroids_land_in_obsm_spatial():
    """Coordinates must be where scanpy/squidpy look for them."""
    t = mt.io.to_spatialdata(_cohort()).tables["ecm_patches"]
    assert t.obsm["spatial"].shape == (36, 2)
    np.testing.assert_array_equal(t.obsm["spatial"][:, 0], t.obs["x"].to_numpy())
    np.testing.assert_array_equal(t.obsm["spatial"][:, 1], t.obs["y"].to_numpy())


def test_region_lists_only_rois_present_in_the_table():
    """An ROI with an image but no patches is still an image, but must not be
    advertised as a region the table annotates."""
    cohort = _cohort()
    del cohort["roi2"].uns["ecm_patches"]
    sdata = mt.io.to_spatialdata(cohort)
    assert set(sdata.images) == {"roi1", "roi2", "roi3"}
    assert sdata.tables["ecm_patches"].uns["spatialdata_attrs"]["region"] == ["roi1_ecm_patches", "roi3_ecm_patches"]


def test_no_patches_anywhere_yields_images_only():
    sdata = mt.io.to_spatialdata(_cohort(with_patches=False))
    assert len(sdata.images) == 3
    assert "ecm_patches" not in sdata.tables


def test_missing_img_is_a_clear_error():
    cohort = _cohort()
    del cohort["roi2"].uns["img"]
    with pytest.raises(ValueError, match=r"roi2.*uns\['img'\]"):
        mt.io.to_spatialdata(cohort)


def test_mantpy_dataset_delegates_to_the_same_export():
    sdata = mt.MantpyDataset(_cohort()).to_spatialdata()
    assert sdata.tables["ecm_patches"].n_obs == 36


def test_multichannel_uns_img_from_a_real_reader_survives_export():
    """Regression: the two readers write `uns['img']` with different shapes.

    `read_imc` stores a 2-D projection; `read_ecm_image` stores the full
    (C, H, W) stack. `to_spatialdata` assumed 2-D and promoted with np.newaxis,
    turning (C, H, W) into 4-D and failing Image2DModel.parse. Every fixture in
    this file hand-builds a 2-D `uns['img']`, so none of them could catch it —
    this one goes through the real reader.
    """
    rng = np.random.default_rng(0)
    img = rng.random((3, 16, 20), dtype=np.float32)

    adata = mt.io.read_ecm_image(img, marker_names=["Col-I", "Col-IV", "HABP"], sample_id="roi1")
    assert adata.uns["img"].ndim == 3, "read_ecm_image should keep all channels"

    sdata = mt.io.to_spatialdata({"roi1": adata})
    # all three channels must survive; projecting to 2-D would be lossy
    assert sdata.images["roi1"].shape == (3, 16, 20)


def _indexed_cohort():
    cohort = _cohort(n=2, n_patches=3)
    for offset, obj in enumerate(cohort.values()):
        obj.uns["ecm_patches"] = pd.DataFrame(
            {
                "x": [10.0, 1.0, 15.0],
                "y": [12.0, 2.0, 15.0],
                "ecm_cluster": [offset + 1, offset + 3, offset + 5],
                "feat_0": [11.0 + offset, 22.0 + offset, 33.0 + offset],
            },
            index=["patch:49", "patch:17", "patch:88"],
        ).iloc[[0, 1]]
    return cohort


def test_point_instances_and_annotation_consumers_match_reordered_source_ids():
    import spatialdata as sd
    from spatialdata.transformations import get_transformation

    cohort = _indexed_cohort()
    sdata = mt.io.to_spatialdata(cohort)
    for sid, obj in cohort.items():
        element = f"{sid}_ecm_patches"
        source = obj.uns["ecm_patches"]
        assert set(sd.get_element_instances(sdata[element])) == set(source.index)
        for column in ("ecm_cluster", "feat_0"):
            values = sd.get_values(column, sdata=sdata, element_name=element, table_name="ecm_patches")
            np.testing.assert_allclose(values.loc[source.index, column].values, source[column].values)
        assert set(get_transformation(sdata[element], get_all=True)) == {sid}
        assert set(get_transformation(sdata.images[sid], get_all=True)) == {sid}
    assert sdata.tables["ecm_patches"].obs["instance_id"].tolist() == ["patch:49", "patch:17", "patch:49", "patch:17"]


def test_spatial_query_and_zarr_round_trip_preserve_roi_patch_linkage(tmp_path):
    import spatialdata as sd

    source = mt.io.to_spatialdata(_indexed_cohort())
    path = tmp_path / "cohort.zarr"
    source.write(path)
    restored = sd.read_zarr(path)
    for sid in ("roi1", "roi2"):
        values = sd.get_values(
            "ecm_cluster", sdata=restored, element_name=f"{sid}_ecm_patches", table_name="ecm_patches"
        )
        assert set(values.index) == {"patch:17", "patch:49"}
    selected = sd.bounding_box_query(
        restored,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[5, 5],
        target_coordinate_system="roi1",
    )
    assert set(selected.points) == {"roi1_ecm_patches"}
    assert selected.tables["ecm_patches"].obs["instance_id"].tolist() == ["patch:17"]
    assert selected.tables["ecm_patches"].obs["sample_id"].tolist() == ["roi1"]


def test_duplicate_patch_ids_fail_before_creating_an_ambiguous_annotation():
    cohort = _indexed_cohort()
    cohort["roi1"].uns["ecm_patches"].index = ["same", "same"]
    with pytest.raises(ValueError, match="patch identifiers must be unique"):
        mt.io.to_spatialdata(cohort)
