"""Focused tests for the raw Collagen-IV tutorial input and patch workflow."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

import mantpy as mt


def _write_reference(path):
    np.savez(
        path,
        grid_y=np.array([0, 0, 1], dtype=np.int64),
        grid_x=np.array([0, 1, 0], dtype=np.int64),
        annotation=np.array([0, 1, 2], dtype=np.int64),
        layer_names=np.array(["Mucosa", "Submucosa", "Muscularis"], dtype=str),
    )


def _textured_patch_grid(values: list[list[int]], *, patch_size: int = 4) -> np.ndarray:
    """Make predictable patch means without constant patches/undefined texture."""
    texture = np.indices((patch_size, patch_size)).sum(axis=0) % 2
    image = np.empty((len(values) * patch_size, len(values[0]) * patch_size), dtype=np.uint16)
    for row, row_values in enumerate(values):
        for col, value in enumerate(row_values):
            image[
                row * patch_size : (row + 1) * patch_size,
                col * patch_size : (col + 1) * patch_size,
            ] = value + texture
    return image


@pytest.fixture
def two_channel_ecm_image():
    """Two channels with one shared and two channel-specific bright patches."""
    collagen = _textured_patch_grid([[100, 100], [10, 10]])
    laminin = _textured_patch_grid([[200, 20], [200, 20]])
    return np.stack([collagen, laminin])


class TestLoadColIVIntestine:
    def test_loads_raw_tiff_without_preprocessing(self, tmp_path):
        raw = np.arange(42, dtype=np.uint16).reshape(6, 7)
        image_path = tmp_path / "roi893_colliv.tiff"
        reference_path = tmp_path / "reference.npz"
        tifffile.imwrite(image_path, raw)
        _write_reference(reference_path)

        data = mt.fetch.load_coliv_intestine(image_path, reference_path=reference_path)

        np.testing.assert_array_equal(data.image, raw)
        assert data.image.dtype == raw.dtype
        assert data.raw_image is data.image
        assert data.image_path == image_path.name
        assert data.reference_path == reference_path.name
        assert str(tmp_path) not in repr({"image_path": data.image_path, "reference_path": data.reference_path})
        assert list(data.reference.index) == [
            "patch_y0000_x0000",
            "patch_y0000_x0001",
            "patch_y0001_x0000",
        ]
        assert data.reference["layer"].tolist() == [0, 1, 2]
        assert data.layer_names == ("Mucosa", "Submucosa", "Muscularis")
        assert data.channel_name == "CollIV"
        assert "features" not in data and "graph" not in data

    def test_accepts_raw_array_and_rejects_multichannel_input(self, tmp_path):
        reference_path = tmp_path / "reference.npz"
        _write_reference(reference_path)
        raw = np.arange(30, dtype=np.float32).reshape(5, 6)

        data = mt.fetch.load_coliv_intestine(raw, reference_path=reference_path)

        np.testing.assert_array_equal(data.image, raw)
        assert data.image_path is None
        with pytest.raises(ValueError, match="one raw 2-D ECM channel"):
            mt.fetch.load_coliv_intestine(np.stack([raw, raw]), reference_path=reference_path)

    def test_requires_explicit_reference(self):
        with pytest.raises(FileNotFoundError, match="explicit reference_path"):
            mt.fetch.load_coliv_intestine(np.zeros((4, 4), dtype=np.uint16))


class TestImageECMPatches:
    def test_single_channel_returns_observation_native_anndata(self):
        image = _textured_patch_grid([[10, 20], [30, 40]])
        foreground = np.ones((2, 2), dtype=bool)

        adata = mt.pp.image_ecm_patches(
            image,
            channel_names=["CollIV"],
            patch_size=4,
            foreground_mask=foreground,
            sample_id="roi893",
        )

        assert adata.shape == (4, 35)
        assert list(adata.obs_names) == [
            "patch_y0000_x0000",
            "patch_y0000_x0001",
            "patch_y0001_x0000",
            "patch_y0001_x0001",
        ]
        assert adata.obs["sample_id"].tolist() == ["roi893"] * 4
        np.testing.assert_array_equal(adata.obs[["grid_y", "grid_x"]], [[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_allclose(adata.obsm["spatial"], [[2, 2], [6, 2], [2, 6], [6, 6]])
        assert adata.obsm["image_patches"].shape == (4, 1, 4, 4)
        assert adata.obsm["raw_patch_mean"].shape == (4, 1)
        assert adata.obsp["grid_connectivities"].shape == (4, 4)
        assert np.isfinite(adata.X).all()
        assert "CollIV_summary_mean" in adata.var_names
        assert "CollIV_texture_contrast_dist-1_angle-0.00" in adata.var_names
        assert "CollIV_histogram_bin-0" in adata.var_names
        assert adata.var["marker"].unique().tolist() == ["CollIV"]
        assert set(adata.var["feature_family"]) == {"summary", "texture", "histogram"}
        assert adata.var["feature_source"].unique().tolist() == ["squidpy"]
        assert adata.var["squidpy_feature"].str.contains("_ch-0_").all()

        summary = mt.pp.ecm_patch_summary(adata)
        assert (summary.n_samples, summary.n_patches, summary.n_features) == (1, 4, 35)
        assert summary.feature_backend == "squidpy"
        assert summary.feature_families == ("summary", "texture", "histogram")
        assert summary.channel_names == ("CollIV",)
        assert summary.image_patch_shape == (1, 4, 4)

    def test_multichannel_selection_and_channel_metadata(self, two_channel_ecm_image):
        non_ecm = _textured_patch_grid([[400, 300], [200, 100]])
        stack = np.stack([two_channel_ecm_image[0], non_ecm, two_channel_ecm_image[1]])

        adata = mt.pp.image_ecm_patches(
            stack,
            channel_names=["CollIV", "CD3", "Laminin"],
            is_ecm=np.array([True, False, True]),
            patch_size=4,
            foreground_mask=np.ones((2, 2), dtype=bool),
        )

        assert adata.shape == (4, 70)
        assert adata.obsm["image_patches"].shape == (4, 2, 4, 4)
        assert adata.obsm["raw_patch_mean"].shape == (4, 2)
        assert set(adata.var["marker"]) == {"CollIV", "Laminin"}
        assert not any(name.startswith("CD3_") for name in adata.var_names)
        assert "CollIV_signal_fraction" in adata.obs
        assert "Laminin_signal_fraction" in adata.obs
        metadata = adata.uns["image_ecm_patches"]
        np.testing.assert_array_equal(metadata["channel_names"], ["CollIV", "Laminin"])
        np.testing.assert_array_equal(metadata["selected_channel_indices"], [0, 2])
        assert len(metadata["clip_values"]) == 2
        assert metadata["feature_backend"] == "squidpy"
        assert metadata["features_per_channel"] == 35
        np.testing.assert_array_equal(metadata["feature_families"], ["summary", "texture", "histogram"])
        assert isinstance(metadata["squidpy_version"], str)
        assert np.nanmin(adata.obsm["image_patches"]) >= 0
        assert np.nanmax(adata.obsm["image_patches"]) <= 1

    @pytest.mark.parametrize(
        ("mode", "expected_coordinates", "source"),
        [
            ("any", [[0, 0], [0, 1], [1, 0]], "channel_union"),
            ("all", [[0, 0]], "channel_intersection"),
            ("mean", [[0, 0], [0, 1], [1, 0]], "mean_patch_intensity"),
        ],
    )
    def test_multichannel_foreground_modes(self, two_channel_ecm_image, mode, expected_coordinates, source):
        adata = mt.pp.image_ecm_patches(
            two_channel_ecm_image,
            channel_names=["CollIV", "Laminin"],
            patch_size=4,
            foreground_mode=mode,
        )

        assert adata.obs[["grid_y", "grid_x"]].to_numpy().tolist() == expected_coordinates
        assert adata.uns["image_ecm_patches"]["foreground_source"] == source

    @pytest.mark.parametrize("mask_kind", ["grid", "image"])
    def test_explicit_foreground_mask_overrides_channel_modes(self, two_channel_ecm_image, mask_kind):
        grid_mask = np.array([[False, True], [False, False]])
        mask = np.repeat(np.repeat(grid_mask, 4, axis=0), 4, axis=1) if mask_kind == "image" else grid_mask

        adata = mt.pp.image_ecm_patches(
            two_channel_ecm_image,
            channel_names=["CollIV", "Laminin"],
            patch_size=4,
            foreground_mode="all",
            foreground_mask=mask,
        )

        assert list(adata.obs_names) == ["patch_y0000_x0001"]
        assert adata.uns["image_ecm_patches"]["foreground_source"] == "explicit_mask"

    def test_threshold_provenance_distinguishes_supplied_and_fitted_values(self, two_channel_ecm_image):
        both = mt.pp.image_ecm_patches(
            two_channel_ecm_image,
            channel_names=["CollIV", "Laminin"],
            patch_size=4,
            pixel_thresholds=[0.1, 0.1],
            patch_thresholds=[0.1, 0.1],
        )
        mixed = mt.pp.image_ecm_patches(
            two_channel_ecm_image,
            channel_names=["CollIV", "Laminin"],
            patch_size=4,
            pixel_thresholds=[0.1, 0.1],
        )

        assert both.uns["image_ecm_patches"]["threshold_source"] == "supplied"
        assert both.uns["image_ecm_patches"]["pixel_threshold_source"] == "supplied"
        assert both.uns["image_ecm_patches"]["patch_threshold_source"] == "supplied"
        assert mixed.uns["image_ecm_patches"]["threshold_source"] == "mixed"
        assert mixed.uns["image_ecm_patches"]["pixel_threshold_source"] == "supplied"
        assert mixed.uns["image_ecm_patches"]["patch_threshold_source"] == "two_means"

    def test_uniform_retained_patch_has_finite_features(self):
        image = np.full((4, 4), 7, dtype=np.uint16)

        adata = mt.pp.image_ecm_patches(
            image,
            patch_size=4,
            foreground_mask=np.ones((1, 1), dtype=bool),
        )

        assert adata.shape == (1, 35)
        assert np.isfinite(adata.X).all()

    def test_custom_squidpy_families_kwargs_and_marker_major_order(self, two_channel_ecm_image):
        adata = mt.pp.image_ecm_patches(
            two_channel_ecm_image,
            channel_names=["CollIV", "Laminin"],
            patch_size=4,
            foreground_mask=np.ones((2, 2), dtype=bool),
            feature_families=("summary", "histogram"),
            feature_kwargs={"summary": {"quantiles": (0.25,)}, "histogram": {"bins": 4}},
        )

        assert adata.shape == (4, 14)
        assert adata.var[["marker", "feature_family"]].value_counts(sort=False).to_dict() == {
            ("CollIV", "summary"): 3,
            ("CollIV", "histogram"): 4,
            ("Laminin", "summary"): 3,
            ("Laminin", "histogram"): 4,
        }
        assert adata.var[["marker", "feature_family"]].drop_duplicates().to_records(index=False).tolist() == [
            ("CollIV", "summary"),
            ("CollIV", "histogram"),
            ("Laminin", "summary"),
            ("Laminin", "histogram"),
        ]
        metadata = adata.uns["image_ecm_patches"]
        assert metadata["feature_kwargs_json"] == ('{"histogram": {"bins": 4}, "summary": {"quantiles": [0.25]}}')

    def test_parallel_squidpy_features_match_serial(self):
        image = _textured_patch_grid([[10, 20], [30, 40]])
        kwargs = {
            "channel_names": ["CollIV"],
            "patch_size": 4,
            "foreground_mask": np.ones((2, 2), dtype=bool),
        }

        serial = mt.pp.image_ecm_patches(image, n_jobs=1, **kwargs)
        parallel = mt.pp.image_ecm_patches(image, n_jobs=2, **kwargs)

        np.testing.assert_array_equal(parallel.X, serial.X)
        assert parallel.var.equals(serial.var)
        metadata = parallel.uns["image_ecm_patches"]
        assert metadata["feature_n_jobs"] == 2
        assert metadata["feature_parallel_backend"] == "joblib threads"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"feature_families": ("summary", "unknown")}, "supports only"),
            ({"feature_families": ()}, "at least one"),
            ({"feature_kwargs": {"texture": {}}}, "unselected or unknown"),
            ({"feature_kwargs": {"summary": {"channels": [0]}}}, "Mantpy-managed"),
        ],
    )
    def test_invalid_squidpy_feature_configuration(self, kwargs, message):
        image = _textured_patch_grid([[10]])
        base = {
            "patch_size": 4,
            "foreground_mask": np.ones((1, 1), dtype=bool),
            "feature_families": ("summary",),
        }
        base.update(kwargs)

        with pytest.raises(ValueError, match=message):
            mt.pp.image_ecm_patches(image, **base)


def test_show_image_scales_large_integer_input_in_float32_without_mutation():
    image = np.arange(256, dtype=np.uint16).reshape(16, 16)
    original = image.copy()

    axis = mt.pl.show_image(image, clip_percentile=(0, 99), show=False)
    displayed = np.asarray(axis.images[0].get_array())

    assert displayed.dtype == np.float32
    assert np.isfinite(displayed).all()
    assert displayed.min() >= 0 and displayed.max() <= 1
    np.testing.assert_array_equal(image, original)
