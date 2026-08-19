"""Tests for mantpy.pp."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import mantpy as mt
from mantpy._constants import ECM_IMAGE_KEY, ECM_PATCHES_KEY, MANTPY_UNS_KEY, RAW_LAYER


def test_extract_ecm_patches_cohort_coordinates_shared_preprocessing(monkeypatch) -> None:
    """The cohort wrapper must preserve keys and record shared normalisation."""
    adatas = {
        "a": AnnData(X=np.zeros((1, 1), dtype=np.float32)),
        "b": AnnData(X=np.zeros((1, 1), dtype=np.float32)),
    }
    images = {
        "a": np.ones((1, 2, 2), dtype=np.float32),
        "b": np.full((1, 2, 2), 2.0, dtype=np.float32),
    }

    def fake_extract(adata, image, **kwargs):
        assert image.shape == (1, 2, 2)
        assert kwargs["features"] == ["mean"]
        adata.uns[ECM_PATCHES_KEY] = pd.DataFrame({"feat_0": [float(np.mean(image))]})

    monkeypatch.setattr(mt.pp, "extract_ecm_patches", fake_extract)
    result = mt.pp.extract_ecm_patches_cohort(adatas, images, normalize="none", features=["mean"])

    assert result is not None
    assert set(result) == {"a", "b"}
    assert all(item.uns["ecm_pixel_normalization"] == {"method": "none"} for item in result.values())
    assert all("ecm_pixel_normalization" not in original.uns for original in adatas.values())


def test_extract_ecm_patches_cohort_rejects_mismatched_keys() -> None:
    adata = AnnData(X=np.zeros((1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="Cohort keys differ"):
        mt.pp.extract_ecm_patches_cohort({"a": adata}, {"b": np.zeros((1, 2, 2))}, features=["mean"])


class TestNormalize:
    def test_minmax_inplace(self, adata_basic):
        mt.pp.normalize(adata_basic, method="min-max")
        assert adata_basic.X.min() >= 0.0
        assert adata_basic.X.max() <= 1.0 + 1e-6

    def test_minmax_copy(self, adata_basic):
        original_X = adata_basic.X.copy()
        result = mt.pp.normalize(adata_basic, method="min-max", inplace=False)
        assert result is not None
        assert isinstance(result, AnnData)
        np.testing.assert_array_equal(adata_basic.X, original_X)

    def test_znorm_inplace(self, adata_basic):
        mt.pp.normalize(adata_basic, method="znorm")
        assert adata_basic.X.dtype == np.float32

    def test_znorm_uses_standard_scaler_from_selected_raw_layer(self):
        from sklearn.preprocessing import StandardScaler

        raw = np.array([[1.0, 7.0], [3.0, 7.0], [8.0, 7.0]], dtype=np.float64)
        adata = AnnData(X=np.full_like(raw, 99.0))
        adata.layers["squidpy_features"] = raw.copy()

        mt.pp.normalize(adata, method="znorm", raw_layer="squidpy_features")

        expected = StandardScaler().fit_transform(raw).astype(np.float32)
        np.testing.assert_allclose(adata.X, expected)
        np.testing.assert_array_equal(adata.layers["squidpy_features"], raw)

    def test_repeated_calls_always_use_stable_raw_layer(self):
        raw = np.array([[1.0, 10.0], [2.0, 14.0], [9.0, 18.0]], dtype=np.float32)
        adata = AnnData(X=raw.copy())

        mt.pp.normalize(adata, method="min-max", raw_layer="squidpy_features")
        first_minmax = adata.X.copy()
        mt.pp.normalize(adata, method="znorm", raw_layer="squidpy_features")
        mt.pp.normalize(adata, method="min-max", raw_layer="squidpy_features")

        np.testing.assert_array_equal(adata.X, first_minmax)
        np.testing.assert_array_equal(adata.layers["squidpy_features"], raw)

    def test_znorm_records_serializable_standard_scaler_provenance(self, tmp_path):
        import json

        import anndata
        from packaging.version import Version

        raw = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32)
        adata = AnnData(X=raw.copy())

        mt.pp.normalize(adata, method="znorm", raw_layer="squidpy_features")

        provenance = adata.uns["feature_normalization"]
        assert provenance["method"] == "sklearn.preprocessing.StandardScaler"
        assert provenance["requested_method"] == "znorm"
        assert provenance["with_mean"] is True
        assert provenance["with_std"] is True
        assert provenance["fit_scope"] == "all observations in this AnnData"
        assert provenance["n_obs"] == 3
        assert provenance["n_vars"] == 2
        assert provenance["raw_layer"] == "squidpy_features"
        assert provenance["output"] == "X"
        assert isinstance(provenance["mean"], list)
        assert isinstance(provenance["scale"], list)
        assert isinstance(provenance["sklearn_version"], str)
        json.dumps(provenance)
        # anndata>=0.12 supports the Arrow string indices produced by pandas 3.
        if Version(anndata.__version__) >= Version("0.12"):
            adata.write_h5ad(tmp_path / "normalized.h5ad")

    def test_minmax_records_fitted_range_provenance(self):
        raw = np.array([[2.0, 5.0], [4.0, 5.0]], dtype=np.float32)
        adata = AnnData(X=raw.copy())

        mt.pp.normalize(adata, method="min-max")

        provenance = adata.uns["feature_normalization"]
        assert provenance["method"] == "min-max"
        assert provenance["feature_min"] == [2.0, 5.0]
        assert provenance["feature_max"] == [4.0, 5.0]
        assert provenance["feature_range"] == [0.0, 1.0]
        assert provenance["data_range"] == [2.0, 0.0]

    def test_raw_layer_preserved(self, adata_basic):
        raw_before = adata_basic.layers[RAW_LAYER].copy()
        mt.pp.normalize(adata_basic, method="min-max")
        np.testing.assert_array_equal(adata_basic.layers[RAW_LAYER], raw_before)

    def test_unknown_method_raises(self, adata_basic):
        with pytest.raises(ValueError, match="Unknown method"):
            mt.pp.normalize(adata_basic, method="bogus")

    def test_params_logged(self, adata_basic):
        mt.pp.normalize(adata_basic, method="znorm")
        assert MANTPY_UNS_KEY in adata_basic.uns
        assert "pp" in adata_basic.uns[MANTPY_UNS_KEY]


class TestExtractEcmPatches:
    def test_writes_ecm_patches(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=3, features=["mean"])
        assert ECM_PATCHES_KEY in adata_basic.uns
        assert isinstance(adata_basic.uns[ECM_PATCHES_KEY], pd.DataFrame)

    def test_patch_df_columns(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=3, features=["mean"])
        df = adata_basic.uns[ECM_PATCHES_KEY]
        for col in ("x", "y", "ecm_cluster"):
            assert col in df.columns

    def test_writes_ecm_image(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=3, features=["mean"])
        assert ECM_IMAGE_KEY in adata_basic.uns
        assert adata_basic.uns[ECM_IMAGE_KEY].ndim == 2

    def test_ecm_image_shape_matches_img(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=3, features=["mean"])
        H, W = synthetic_img.shape[1], synthetic_img.shape[2]
        assert adata_basic.uns[ECM_IMAGE_KEY].shape == (H, W)

    def test_cluster_count(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=3, features=["mean"])
        df = adata_basic.uns[ECM_PATCHES_KEY]
        assert df["ecm_cluster"].nunique() <= 3

    def test_inplace_false_returns_copy(self, adata_basic, synthetic_img):
        result = mt.pp.extract_ecm_patches(
            adata_basic, synthetic_img, patch_size=8, ecm_K=3, features=["mean"], inplace=False
        )
        assert result is not None
        assert ECM_PATCHES_KEY not in adata_basic.uns
        assert ECM_PATCHES_KEY in result.uns

    def test_auto_k(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K="auto", features=["mean"])
        assert ECM_PATCHES_KEY in adata_basic.uns

    def test_missing_is_ecm_raises(self, adata_basic, synthetic_img):
        del adata_basic.var["is_ecm"]
        with pytest.raises(ValueError, match="is_ecm"):
            mt.pp.extract_ecm_patches(adata_basic, synthetic_img, features=["mean"])

    def test_params_logged(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=3, features=["mean"])
        assert "extract_ecm_patches" in adata_basic.uns[MANTPY_UNS_KEY]["pp"]

    def test_features_logged(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=3, features=["mean"])
        log = adata_basic.uns[MANTPY_UNS_KEY]["pp"]["extract_ecm_patches"]
        assert "mean" in log["features"]

    def test_std_feature(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=2, features=["std"])
        assert ECM_PATCHES_KEY in adata_basic.uns

    def test_entropy_feature(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=2, features=["entropy"])
        assert ECM_PATCHES_KEY in adata_basic.uns

    def test_coherence_feature(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=2, features=["coherence"])
        assert ECM_PATCHES_KEY in adata_basic.uns

    def test_multiple_features(self, adata_basic, synthetic_img):
        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=2, features=["mean", "coherence"])
        df = adata_basic.uns[ECM_PATCHES_KEY]
        # mean → 2 ECM channels; coherence → 2 ECM channels → 4 feat_ columns
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        assert len(feat_cols) == 4

    def test_custom_callable_feature(self, adata_basic, synthetic_img):
        def my_max(patches):
            # Batched signature: (N, C, P, P) -> (N, C)
            return patches.max(axis=(2, 3))

        mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=2, features=[my_max])
        assert ECM_PATCHES_KEY in adata_basic.uns

    def test_unknown_feature_name_raises(self, adata_basic, synthetic_img):
        with pytest.raises(ValueError, match="Unknown feature name"):
            mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, features=["nonexistent_feature"])

    def test_mask_3d_accepted(self, adata_basic, synthetic_img):
        arr = synthetic_img.astype(np.float32)
        full_mask = np.ones(arr.shape, dtype=bool)
        mt.pp.extract_ecm_patches(adata_basic, arr, patch_size=8, ecm_K=2, features=["mean"], mask=full_mask)
        assert ECM_PATCHES_KEY in adata_basic.uns

    def test_mask_2d_accepted(self, adata_basic, synthetic_img):
        arr = synthetic_img.astype(np.float32)
        H, W = arr.shape[1], arr.shape[2]
        spatial_mask = np.ones((H, W), dtype=bool)
        mt.pp.extract_ecm_patches(adata_basic, arr, patch_size=8, ecm_K=2, features=["mean"], mask=spatial_mask)
        assert ECM_PATCHES_KEY in adata_basic.uns

    def test_single_channel_input_with_ecm_channel(self, adata_basic, synthetic_img):
        # Ground truth: extract patches in multi-channel mode with only ColIV flagged.
        ref = adata_basic.copy()
        ref.var["is_ecm"] = False
        ref.var.loc["ColIV", "is_ecm"] = True
        mt.pp.extract_ecm_patches(
            ref,
            synthetic_img.astype(np.float32),
            patch_size=8,
            ecm_K=None,
            features=["mean"],
        )
        ref_df = ref.uns[ECM_PATCHES_KEY]

        # New path: pass just the ColIV channel as (H, W) + ecm_channel.
        col_img = synthetic_img[0].astype(np.float32)  # (64, 64)
        adata_basic.var["is_ecm"] = False  # nothing flagged on entry
        mt.pp.extract_ecm_patches(
            adata_basic,
            col_img,
            ecm_channel="ColIV",
            patch_size=8,
            ecm_K=None,
            features=["mean"],
        )

        # is_ecm flag now set correctly
        assert adata_basic.var["is_ecm"].tolist() == [True, False, False, False, False]
        # patches written
        assert ECM_PATCHES_KEY in adata_basic.uns
        new_df = adata_basic.uns[ECM_PATCHES_KEY]
        # feat_* columns and patch counts match the multi-channel reference
        feat_cols = sorted(c for c in new_df.columns if c.startswith("feat_"))
        ref_feat_cols = sorted(c for c in ref_df.columns if c.startswith("feat_"))
        assert feat_cols == ref_feat_cols
        assert len(new_df) == len(ref_df)
        np.testing.assert_allclose(new_df[feat_cols].values, ref_df[ref_feat_cols].values, rtol=0, atol=0)
        # protein name in feature metadata. Stored as a JSON string for
        # h5ad-serialisability (anndata can't write list[dict] to disk).
        import json

        feat_meta = adata_basic.uns["ecm_feature_names"]
        if isinstance(feat_meta, str):
            feat_meta = json.loads(feat_meta)
        assert all(m["protein"] == "ColIV" for m in feat_meta)

    def test_single_channel_input_missing_ecm_channel_raises(self, adata_basic, synthetic_img):
        col_img = synthetic_img[0].astype(np.float32)
        with pytest.raises(ValueError, match="ecm_channel"):
            mt.pp.extract_ecm_patches(adata_basic, col_img, patch_size=8, features=["mean"])

    def test_single_channel_input_unknown_ecm_channel_raises(self, adata_basic, synthetic_img):
        col_img = synthetic_img[0].astype(np.float32)
        with pytest.raises(ValueError, match="not in adata.var_names"):
            mt.pp.extract_ecm_patches(
                adata_basic,
                col_img,
                ecm_channel="not_a_var",
                patch_size=8,
                features=["mean"],
            )


class TestExtractEcmPatchesImageContainer:
    """ImageContainer input and tiled extraction path."""

    def test_accepts_image_container(self, adata_basic, img_container):
        mt.pp.extract_ecm_patches(adata_basic, img_container, patch_size=8, ecm_K=3, features=["mean"])
        assert ECM_PATCHES_KEY in adata_basic.uns

    def test_container_produces_same_shape_as_array(self, adata_basic, synthetic_img, img_container):
        import copy

        a1 = copy.deepcopy(adata_basic)
        a2 = copy.deepcopy(adata_basic)
        mt.pp.extract_ecm_patches(a1, synthetic_img, patch_size=8, ecm_K=3, features=["mean"])
        mt.pp.extract_ecm_patches(a2, img_container, patch_size=8, ecm_K=3, features=["mean"])
        assert a1.uns[ECM_IMAGE_KEY].shape == a2.uns[ECM_IMAGE_KEY].shape

    def test_tiled_writes_ecm_patches(self, adata_basic, img_container):
        mt.pp.extract_ecm_patches(
            adata_basic,
            img_container,
            patch_size=8,
            ecm_K=3,
            features=["mean"],
            tiled=True,
            tile_size=32,
        )
        assert ECM_PATCHES_KEY in adata_basic.uns
        assert isinstance(adata_basic.uns[ECM_PATCHES_KEY], pd.DataFrame)

    def test_tiled_writes_ecm_image(self, adata_basic, img_container):
        mt.pp.extract_ecm_patches(
            adata_basic,
            img_container,
            patch_size=8,
            ecm_K=3,
            features=["mean"],
            tiled=True,
            tile_size=32,
        )
        assert ECM_IMAGE_KEY in adata_basic.uns
        assert adata_basic.uns[ECM_IMAGE_KEY].ndim == 2

    def test_tiled_image_shape_matches_original(self, adata_basic, img_container):
        mt.pp.extract_ecm_patches(
            adata_basic,
            img_container,
            patch_size=8,
            ecm_K=3,
            features=["mean"],
            tiled=True,
            tile_size=32,
        )
        assert adata_basic.uns[ECM_IMAGE_KEY].shape == (img_container.height, img_container.width)

    def test_tiled_cluster_count(self, adata_basic, img_container):
        mt.pp.extract_ecm_patches(
            adata_basic,
            img_container,
            patch_size=8,
            ecm_K=3,
            features=["mean"],
            tiled=True,
            tile_size=32,
        )
        df = adata_basic.uns[ECM_PATCHES_KEY]
        assert df["ecm_cluster"].nunique() <= 3

    def test_tiled_params_logged(self, adata_basic, img_container):
        mt.pp.extract_ecm_patches(
            adata_basic,
            img_container,
            patch_size=8,
            ecm_K=3,
            features=["mean"],
            tiled=True,
            tile_size=32,
        )
        log = adata_basic.uns[MANTPY_UNS_KEY]["pp"]["extract_ecm_patches"]
        assert log["tiled"] is True

    def test_tiled_with_overlap(self, adata_basic, img_container):
        mt.pp.extract_ecm_patches(
            adata_basic,
            img_container,
            patch_size=8,
            ecm_K=3,
            features=["mean"],
            tiled=True,
            tile_size=32,
            tile_overlap=8,
        )
        assert ECM_PATCHES_KEY in adata_basic.uns

    def test_channel_mismatch_raises(self, adata_basic):
        from mantpy.im import ImageContainer

        bad_ic = ImageContainer(np.zeros((3, 64, 64), dtype=np.float32))
        with pytest.raises(ValueError, match="channels"):
            mt.pp.extract_ecm_patches(adata_basic, bad_ic, patch_size=8, features=["mean"])


class TestPreprocessEcm:
    """Tests for structural ECM preprocessing filters."""

    @pytest.fixture
    def sample_img(self):
        rng = np.random.default_rng(0)
        return (rng.random((3, 64, 64), dtype=np.float64) * 1000).astype(np.float32)

    @pytest.mark.parametrize("method", ["tophat", "frangi", "sato", "none"])
    def test_preserves_shape_and_dtype(self, sample_img, method):
        kwargs = {"sigmas": (1, 2)} if method in {"frangi", "sato"} else {}
        out = mt.pp.preprocess_ecm(sample_img, method=method, **kwargs)
        assert out.shape == sample_img.shape
        assert out.dtype == np.float32

    def test_none_is_identity(self, sample_img):
        out = mt.pp.preprocess_ecm(sample_img, method="none")
        np.testing.assert_array_equal(out, sample_img)

    def test_channel_indices_only_modifies_selected(self, sample_img):
        out = mt.pp.preprocess_ecm(sample_img, method="tophat", channel_indices=[0])
        np.testing.assert_array_equal(out[1], sample_img[1])
        np.testing.assert_array_equal(out[2], sample_img[2])

    def test_2d_input_promoted(self):
        img2d = np.random.default_rng(1).random((64, 64)).astype(np.float32) * 100
        out = mt.pp.preprocess_ecm(img2d, method="none")
        assert out.shape == (1, 64, 64)

    def test_unknown_method_raises(self, sample_img):
        with pytest.raises(ValueError, match="Unknown method"):
            mt.pp.preprocess_ecm(sample_img, method="bogus")


class TestNewFeatureExtractors:
    """Tests for feat_max, feat_signal_fraction, feat_gradient_magnitude.

    Extractors operate on batched patches: ``(N, C, P, P) -> (N, C)``.
    """

    @pytest.fixture
    def patch_1ch(self):
        # (N=2, C=1, P=16, P=16) batched stack.
        return np.random.default_rng(42).random((2, 1, 16, 16)).astype(np.float32)

    @pytest.fixture
    def patch_3ch(self):
        return np.random.default_rng(42).random((2, 3, 16, 16)).astype(np.float32)

    def test_max_shape(self, patch_1ch, patch_3ch):
        from mantpy._core._patching import feat_max

        assert feat_max(patch_1ch).shape == (2, 1)
        assert feat_max(patch_3ch).shape == (2, 3)

    def test_max_value_correct(self, patch_1ch):
        from mantpy._core._patching import feat_max

        expected = patch_1ch[0, 0].max()
        assert abs(feat_max(patch_1ch)[0, 0] - expected) < 1e-6

    def test_signal_fraction_shape(self, patch_1ch, patch_3ch):
        from mantpy._core._patching import feat_signal_fraction

        assert feat_signal_fraction(patch_1ch).shape == (2, 1)
        assert feat_signal_fraction(patch_3ch).shape == (2, 3)

    def test_signal_fraction_range(self, patch_3ch):
        from mantpy._core._patching import feat_signal_fraction

        result = feat_signal_fraction(patch_3ch)
        assert (result >= 0.0).all()
        assert (result <= 1.0).all()

    def test_gradient_magnitude_shape(self, patch_1ch, patch_3ch):
        from mantpy._core._patching import feat_gradient_magnitude

        assert feat_gradient_magnitude(patch_1ch).shape == (2, 1)
        assert feat_gradient_magnitude(patch_3ch).shape == (2, 3)

    def test_gradient_magnitude_nonneg(self, patch_3ch):
        from mantpy._core._patching import feat_gradient_magnitude

        result = feat_gradient_magnitude(patch_3ch)
        assert (result >= 0.0).all()

    def test_gradient_magnitude_zero_for_constant(self):
        from mantpy._core._patching import feat_gradient_magnitude

        constant = np.ones((1, 1, 16, 16), dtype=np.float32) * 5.0
        result = feat_gradient_magnitude(constant)
        assert abs(result[0, 0]) < 1e-6

    def test_all_seven_features_in_registry(self):
        from mantpy._core._patching import FEATURE_REGISTRY

        expected = {"mean", "std", "entropy", "coherence", "max", "signal_fraction", "gradient_magnitude"}
        assert expected.issubset(set(FEATURE_REGISTRY.keys()))

    def test_all_seven_in_pipeline(self):
        arr = np.random.default_rng(0).random((1, 80, 80)).astype(np.float32)
        adata = mt.read_ecm_image(arr, marker_names=["ColIV"])
        mt.pp.extract_ecm_patches(
            adata,
            arr,
            patch_size=10,
            ecm_K=None,
            features=["mean", "std", "max", "signal_fraction", "entropy", "coherence", "gradient_magnitude"],
        )
        df = adata.uns["ecm_patches"]
        assert df.shape[1] == 10  # x, y, ecm_cluster + 7 features

    def test_remove_background_uses_feature_quantile(self):
        from mantpy._core._patching import remove_background

        features = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        keep = remove_background(features, quantile=0.5)

        np.testing.assert_array_equal(keep, [False, False, True, True])
        assert keep.dtype == np.bool_


class TestGlcmAndSkeletonFeatures:
    """Tests for GLCM texture + skeleton-morphology extractors.

    All extractors use the batched signature: ``(N, C, P, P) -> (N, C)``.
    """

    @pytest.fixture
    def patch_1ch(self):
        return (np.random.default_rng(42).random((2, 1, 24, 24)) * 255).astype(np.float32)

    @pytest.fixture
    def patch_2ch(self):
        return (np.random.default_rng(42).random((2, 2, 24, 24)) * 255).astype(np.float32)

    @pytest.mark.parametrize(
        "name",
        [
            "glcm_contrast",
            "glcm_homogeneity",
            "glcm_energy",
            "glcm_correlation",
        ],
    )
    def test_glcm_returns_NC_shape(self, name, patch_1ch, patch_2ch):
        from mantpy._core._patching import FEATURE_REGISTRY

        fn = FEATURE_REGISTRY[name]
        out1 = fn(patch_1ch)
        out2 = fn(patch_2ch)
        assert out1.shape == (2, 1)
        assert out2.shape == (2, 2)
        assert out1.dtype == np.float32

    def test_glcm_handles_constant_patch(self):
        """Constant patch: contrast should be 0; homogeneity/energy maxed at 1."""
        from mantpy._core._patching import (
            feat_glcm_contrast,
            feat_glcm_energy,
            feat_glcm_homogeneity,
        )

        constant = np.zeros((1, 1, 24, 24), dtype=np.float32)
        assert feat_glcm_contrast(constant)[0, 0] == pytest.approx(0.0)
        assert feat_glcm_homogeneity(constant)[0, 0] == pytest.approx(1.0)
        assert feat_glcm_energy(constant)[0, 0] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "name",
        [
            "skeleton_density",
            "branchpoint_density",
            "mean_branch_length",
        ],
    )
    def test_skeleton_returns_NC_shape(self, name, patch_1ch, patch_2ch):
        from mantpy._core._patching import FEATURE_REGISTRY

        fn = FEATURE_REGISTRY[name]
        assert fn(patch_1ch).shape == (2, 1)
        assert fn(patch_2ch).shape == (2, 2)

    def test_skeleton_zero_on_empty_patch(self):
        from mantpy._core._patching import (
            feat_branchpoint_density,
            feat_mean_branch_length,
            feat_skeleton_density,
        )

        empty = np.zeros((1, 1, 24, 24), dtype=np.float32)
        assert feat_skeleton_density(empty)[0, 0] == pytest.approx(0.0)
        assert feat_branchpoint_density(empty)[0, 0] == pytest.approx(0.0)
        assert feat_mean_branch_length(empty)[0, 0] == pytest.approx(0.0)

    def test_new_features_in_registry(self):
        from mantpy._core._patching import FEATURE_REGISTRY

        expected = {
            "glcm_contrast",
            "glcm_homogeneity",
            "glcm_energy",
            "glcm_correlation",
            "skeleton_density",
            "branchpoint_density",
            "mean_branch_length",
        }
        assert expected.issubset(set(FEATURE_REGISTRY.keys()))

    def test_new_features_flow_through_pipeline(self):
        arr = (np.random.default_rng(0).random((1, 80, 80)) * 255).astype(np.float32)
        adata = mt.read_ecm_image(arr, marker_names=["ColIV"])
        features = [
            "mean",
            "glcm_contrast",
            "glcm_homogeneity",
            "skeleton_density",
            "branchpoint_density",
        ]
        mt.pp.extract_ecm_patches(
            adata,
            arr,
            patch_size=10,
            ecm_K=None,
            features=features,
        )
        df = adata.uns["ecm_patches"]
        # x, y, ecm_cluster + 5 features
        assert df.shape[1] == 3 + len(features)
