"""Tests for mantpy.io."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantpy as mt
from mantpy._constants import (
    CELLTYPE_COL,
    CONDITION_COL,
    MANTPY_UNS_KEY,
    RAW_LAYER,
    SAMPLE_ID_COL,
    SPATIAL_KEY,
    X_COL,
    Y_COL,
)


class TestReadImc:
    def test_returns_anndata(self, synthetic_img, synthetic_panel, synthetic_cells):
        adata = mt.read_imc(synthetic_img, panel=synthetic_panel, cells=synthetic_cells)
        assert isinstance(adata, ad.AnnData)

    def test_shape(self, synthetic_img, synthetic_panel, synthetic_cells):
        adata = mt.read_imc(synthetic_img, panel=synthetic_panel, cells=synthetic_cells)
        assert adata.n_obs == len(synthetic_cells)
        assert adata.n_vars == len(synthetic_panel)

    def test_obs_columns(self, adata_basic):
        for col in (CELLTYPE_COL, X_COL, Y_COL, SAMPLE_ID_COL, CONDITION_COL):
            assert col in adata_basic.obs.columns, f"Missing obs column: {col}"

    def test_var_columns(self, adata_basic):
        assert "marker_name" in adata_basic.var.columns
        assert "is_ecm" in adata_basic.var.columns

    def test_is_ecm_correct(self, adata_basic):
        assert adata_basic.var.loc["ColIV", "is_ecm"] is True or adata_basic.var["is_ecm"].iloc[0]
        assert not adata_basic.var["is_ecm"].iloc[2]  # CD20

    def test_spatial_key(self, adata_basic):
        assert SPATIAL_KEY in adata_basic.obsm
        assert adata_basic.obsm[SPATIAL_KEY].shape == (adata_basic.n_obs, 2)

    def test_raw_layer(self, adata_basic):
        assert RAW_LAYER in adata_basic.layers
        assert adata_basic.layers[RAW_LAYER].shape == adata_basic.X.shape

    def test_x_is_normalised_min_max(self, adata_basic):
        X = adata_basic.X
        assert X.min() >= 0.0
        assert X.max() <= 1.0 + 1e-6

    def test_uns_mantpy_key(self, adata_basic):
        assert MANTPY_UNS_KEY in adata_basic.uns
        assert "io" in adata_basic.uns[MANTPY_UNS_KEY]

    def test_sample_id_stored(self, adata_basic):
        assert (adata_basic.obs[SAMPLE_ID_COL] == "test_sample").all()

    def test_condition_stored(self, adata_basic):
        assert (adata_basic.obs[CONDITION_COL] == "ctrl").all()

    def test_img_stored_flat(self, adata_basic):
        # backward-compat flat key
        assert "img" in adata_basic.uns
        assert adata_basic.uns["img"].ndim == 2

    def test_img_stored_squidpy_slot(self, adata_basic):
        # Squidpy-compatible spatial slot: uns["spatial"][sample_id]["images"]["hires"]
        assert "spatial" in adata_basic.uns
        sid = adata_basic.obs["sample_id"].iloc[0]
        assert sid in adata_basic.uns["spatial"]
        assert "images" in adata_basic.uns["spatial"][sid]
        hires = adata_basic.uns["spatial"][sid]["images"]["hires"]
        assert hires.ndim == 2

    def test_scalefactors_stored(self, adata_basic):
        sid = adata_basic.obs["sample_id"].iloc[0]
        sf = adata_basic.uns["spatial"][sid]["scalefactors"]
        assert "spot_diameter_fullres" in sf

    def test_accepts_image_container(self, synthetic_img, synthetic_panel, synthetic_cells):
        from mantpy.im import ImageContainer

        ic = ImageContainer(synthetic_img.astype(np.float32))
        adata = mt.read_imc(ic, panel=synthetic_panel, cells=synthetic_cells)
        assert isinstance(adata, ad.AnnData)
        assert adata.n_obs == len(synthetic_cells)

    def test_normalize_znorm(self, synthetic_img, synthetic_panel, synthetic_cells):
        adata = mt.read_imc(
            synthetic_img,
            panel=synthetic_panel,
            cells=synthetic_cells,
            normalize="znorm",
        )
        # z-normed columns should have ~mean 0 (within noise for tiny N)
        assert adata.X.dtype == np.float32

    def test_normalize_none(self, synthetic_img, synthetic_panel, synthetic_cells):
        adata = mt.read_imc(
            synthetic_img,
            panel=synthetic_panel,
            cells=synthetic_cells,
            normalize="none",
        )
        np.testing.assert_array_almost_equal(adata.X, adata.layers[RAW_LAYER])

    def test_panel_mismatch_raises(self, synthetic_img, synthetic_cells):
        bad_panel = pd.DataFrame({"name": ["A", "B"], "ecm": [0, 1]})
        with pytest.raises(ValueError, match="Panel has"):
            mt.read_imc(synthetic_img, panel=bad_panel, cells=synthetic_cells)

    def test_panel_missing_column_raises(self, synthetic_img, synthetic_cells):
        bad_panel = pd.DataFrame({"name": ["A"] * 5})
        with pytest.raises(ValueError, match="ecm"):
            mt.read_imc(synthetic_img, panel=bad_panel, cells=synthetic_cells)

    def test_numpy_img_input(self, synthetic_img, synthetic_panel, synthetic_cells):
        adata = mt.read_imc(synthetic_img, panel=synthetic_panel, cells=synthetic_cells)
        assert adata is not None

    def test_no_cells_ecm_only(self, synthetic_img, synthetic_panel):
        adata = mt.read_imc(synthetic_img, panel=synthetic_panel, cells=None)
        assert adata.n_obs == 0
        assert adata.n_vars == len(synthetic_panel)


class TestReadCodex:
    def test_returns_anndata(self, synthetic_img, synthetic_panel, synthetic_cells):
        adata = mt.read_codex(synthetic_img, panel=synthetic_panel, cells=synthetic_cells)
        assert isinstance(adata, ad.AnnData)

    def test_same_as_read_imc(self, synthetic_img, synthetic_panel, synthetic_cells):
        a1 = mt.read_imc(synthetic_img, panel=synthetic_panel, cells=synthetic_cells)
        a2 = mt.read_codex(synthetic_img, panel=synthetic_panel, cells=synthetic_cells)
        np.testing.assert_array_equal(a1.X, a2.X)


class TestReadEcmImage:
    def test_returns_anndata(self):
        arr = np.random.rand(3, 50, 50).astype(np.float32)
        adata = mt.read_ecm_image(arr)
        assert isinstance(adata, ad.AnnData)

    def test_single_channel_shape(self):
        arr = np.random.rand(1, 50, 50).astype(np.float32)
        adata = mt.read_ecm_image(arr, marker_names=["ColIV"])
        assert adata.n_obs == 0
        assert adata.n_vars == 1

    def test_multi_channel_shape(self):
        arr = np.random.rand(3, 50, 50).astype(np.float32)
        adata = mt.read_ecm_image(arr, marker_names=["ColIV", "FN1", "Laminin"])
        assert adata.n_obs == 0
        assert adata.n_vars == 3

    def test_2d_input_promoted(self):
        arr = np.random.rand(50, 50).astype(np.float32)
        adata = mt.read_ecm_image(arr, marker_names=["ColIV"])
        assert adata.n_vars == 1

    def test_is_ecm_all_true(self):
        arr = np.random.rand(2, 50, 50).astype(np.float32)
        adata = mt.read_ecm_image(arr, marker_names=["ColIV", "FN1"])
        assert adata.var["is_ecm"].all()

    def test_marker_names_stored(self):
        arr = np.random.rand(2, 50, 50).astype(np.float32)
        adata = mt.read_ecm_image(arr, marker_names=["ColIV", "FN1"])
        assert adata.var_names.tolist() == ["ColIV", "FN1"]
        assert adata.var["marker_name"].tolist() == ["ColIV", "FN1"]

    def test_default_marker_names(self):
        arr = np.random.rand(3, 50, 50).astype(np.float32)
        adata = mt.read_ecm_image(arr)
        assert adata.var_names.tolist() == ["channel_0", "channel_1", "channel_2"]

    def test_marker_names_mismatch_raises(self):
        arr = np.random.rand(2, 50, 50).astype(np.float32)
        with pytest.raises(ValueError, match="marker_names has 3"):
            mt.read_ecm_image(arr, marker_names=["A", "B", "C"])

    def test_squidpy_image_slot(self):
        arr = np.random.rand(1, 50, 50).astype(np.float32)
        adata = mt.read_ecm_image(arr, sample_id="wsi_01")
        assert "spatial" in adata.uns
        assert "wsi_01" in adata.uns["spatial"]
        hires = adata.uns["spatial"]["wsi_01"]["images"]["hires"]
        assert hires.shape == (50, 50)

    def test_mantpy_uns_key(self):
        arr = np.random.rand(1, 50, 50).astype(np.float32)
        adata = mt.read_ecm_image(arr)
        assert MANTPY_UNS_KEY in adata.uns
        assert "io" in adata.uns[MANTPY_UNS_KEY]
        assert "read_ecm_image" in adata.uns[MANTPY_UNS_KEY]["io"]

    def test_spatial_key_empty(self):
        arr = np.random.rand(1, 50, 50).astype(np.float32)
        adata = mt.read_ecm_image(arr)
        assert SPATIAL_KEY in adata.obsm
        assert adata.obsm[SPATIAL_KEY].shape == (0, 2)

    def test_accepts_image_container(self):
        from mantpy.im import ImageContainer

        arr = np.random.rand(1, 50, 50).astype(np.float32)
        ic = ImageContainer(arr)
        adata = mt.read_ecm_image(ic, marker_names=["ColIV"])
        assert adata.n_vars == 1

    def test_pipeline_compatible(self):
        arr = np.random.rand(1, 64, 64).astype(np.float32)
        adata = mt.read_ecm_image(arr, marker_names=["ColIV"])
        mt.pp.extract_ecm_patches(
            adata,
            arr,
            patch_size=8,
            features=["mean", "std"],
            ecm_K=None,
        )
        assert "ecm_patches" in adata.uns
        assert len(adata.uns["ecm_patches"]) > 0


class TestCellFeaturesFromMask:
    @pytest.fixture
    def synthetic_mask_and_img(self):
        # 3 cells: 1 disk per ID, distinct fill values per channel.
        from skimage.draw import disk

        H, W = 32, 32
        mask = np.zeros((H, W), dtype=np.int32)
        for cid, (cy, cx) in enumerate([(8, 8), (8, 24), (24, 16)], start=1):
            rr, cc = disk((cy, cx), 3)
            mask[rr, cc] = cid
        # 2 channels, constant fill so per-cell mean equals fill.
        img = np.zeros((2, H, W), dtype=np.float32)
        img[0] = 5.0
        img[1] = 11.0
        return mask, img

    def test_basic_anndata_shape(self, synthetic_mask_and_img):
        mask, img = synthetic_mask_and_img
        adata = mt.io.cell_features_from_mask(
            mask,
            img,
            channel_names=["DAPI", "Col4A1"],
            normalize="none",
        )
        assert adata.n_obs == 3
        assert adata.n_vars == 2
        assert adata.var_names.tolist() == ["DAPI", "Col4A1"]

    def test_means_match_fill(self, synthetic_mask_and_img):
        mask, img = synthetic_mask_and_img
        adata = mt.io.cell_features_from_mask(
            mask,
            img,
            channel_names=["DAPI", "Col4A1"],
            normalize="none",
        )
        np.testing.assert_allclose(adata.X[:, 0], 5.0)
        np.testing.assert_allclose(adata.X[:, 1], 11.0)

    def test_spatial_and_layers(self, synthetic_mask_and_img):
        mask, img = synthetic_mask_and_img
        adata = mt.io.cell_features_from_mask(
            mask,
            img,
            channel_names=["DAPI", "Col4A1"],
            normalize="none",
        )
        assert "spatial" in adata.obsm
        assert adata.obsm["spatial"].shape == (3, 2)
        assert "raw" in adata.layers
        assert "is_ecm" in adata.var.columns

    def test_is_ecm_by_name(self, synthetic_mask_and_img):
        mask, img = synthetic_mask_and_img
        adata = mt.io.cell_features_from_mask(
            mask,
            img,
            channel_names=["DAPI", "Col4A1"],
            is_ecm="Col4A1",
            normalize="none",
        )
        assert adata.var["is_ecm"].tolist() == [False, True]

    def test_is_ecm_by_list(self, synthetic_mask_and_img):
        mask, img = synthetic_mask_and_img
        img3 = np.stack([img[0], img[1], img[1]])  # add a third channel
        adata = mt.io.cell_features_from_mask(
            mask,
            img3,
            channel_names=["DAPI", "Col4A1", "Col1A1"],
            is_ecm=["Col4A1", "Col1A1"],
            normalize="none",
        )
        assert adata.var["is_ecm"].tolist() == [False, True, True]

    def test_arcsinh_clip_normalisation(self, synthetic_mask_and_img):
        mask, img = synthetic_mask_and_img
        adata = mt.io.cell_features_from_mask(
            mask,
            img,
            channel_names=["a", "b"],
            normalize="arcsinh_clip",
        )
        assert float(adata.X.min()) >= 0.0
        assert float(adata.X.max()) <= 1.0 + 1e-6

    def test_empty_mask(self):
        adata = mt.io.cell_features_from_mask(
            np.zeros((16, 16), dtype=np.int32),
            np.zeros((2, 16, 16), dtype=np.float32),
            channel_names=["a", "b"],
        )
        assert adata.n_obs == 0
        assert adata.n_vars == 2

    def test_path_input_matches_array(self, synthetic_mask_and_img, tmp_path):
        import tifffile

        mask, img = synthetic_mask_and_img
        tiff = tmp_path / "stack.tif"
        # Write each channel as a separate page (multi-page TIFF).
        with tifffile.TiffWriter(str(tiff)) as w:
            for c in range(img.shape[0]):
                w.write(img[c])

        ref = mt.io.cell_features_from_mask(
            mask,
            img,
            channel_names=["DAPI", "Col4A1"],
            normalize="none",
        )
        out = mt.io.cell_features_from_mask(
            mask,
            tiff,
            channel_names=["DAPI", "Col4A1"],
            normalize="none",
        )
        np.testing.assert_allclose(out.X, ref.X)
