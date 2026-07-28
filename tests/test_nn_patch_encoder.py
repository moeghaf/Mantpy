"""Tests for mt.nn.PatchEncoder."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_LIGHTNING = (
    importlib.util.find_spec("pytorch_lightning") is not None or importlib.util.find_spec("lightning") is not None
)

pytestmark = pytest.mark.skipif(not (HAS_TORCH and HAS_LIGHTNING), reason="requires mantpy[patch]")


@pytest.fixture
def patches():
    rng = np.random.default_rng(0)
    return rng.random((64, 1, 16, 16)).astype("float32")


def test_fit_transform_shape(patches):
    from mantpy.nn import PatchEncoder

    enc = PatchEncoder(latent_dim=8).fit(patches, max_epochs=2, accelerator="cpu", random_state=0)
    emb = enc.transform(patches)
    assert emb.shape == (64, 8)
    assert np.isfinite(emb).all()


def test_reproducible_with_seed(patches):
    from mantpy.nn import PatchEncoder

    a = PatchEncoder(latent_dim=4).fit(patches, max_epochs=2, accelerator="cpu", random_state=0).transform(patches)
    b = PatchEncoder(latent_dim=4).fit(patches, max_epochs=2, accelerator="cpu", random_state=0).transform(patches)
    np.testing.assert_allclose(a, b)


def test_autoencoder_objective(patches):
    from mantpy.nn import PatchEncoder

    enc = PatchEncoder(latent_dim=4, objective="autoencoder").fit(
        patches,
        max_epochs=2,
        accelerator="cpu",
        random_state=0,
    )
    assert enc.transform(patches).shape == (64, 4)


def test_invalid_latent_dim_raises():
    from mantpy.nn import PatchEncoder

    with pytest.raises(ValueError):
        PatchEncoder(latent_dim=0)


def test_invalid_objective_raises():
    from mantpy.nn import PatchEncoder

    with pytest.raises(ValueError):
        PatchEncoder(objective="pca")


def test_repr_reports_fitted_state(patches):
    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(latent_dim=4)
    assert "status='unfitted'" in repr(encoder)
    encoder.fit(
        patches,
        max_epochs=1,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
    )
    text = repr(encoder)
    assert "status='fitted'" in text
    assert "input_shape=(1, 16, 16)" in text


def test_non_4d_patches_raise(patches):
    from mantpy.nn import PatchEncoder

    with pytest.raises(ValueError):
        PatchEncoder(latent_dim=4).fit(patches[:, 0], max_epochs=1, accelerator="cpu")  # (N, P, P), no channel axis


def test_transform_before_fit_raises(patches):
    from mantpy.nn import PatchEncoder

    with pytest.raises(RuntimeError):
        PatchEncoder().transform(patches)


def test_fits_on_multiple_ecm_channels():
    from mantpy.nn import PatchEncoder

    rng = np.random.default_rng(0)
    multichannel = rng.random((32, 3, 8, 8), dtype=np.float32)
    embedding = (
        PatchEncoder(latent_dim=5)
        .fit(multichannel, max_epochs=1, batch_size=32, accelerator="cpu", random_state=0)
        .transform(multichannel)
    )

    assert embedding.shape == (32, 5)
    assert np.isfinite(embedding).all()


def test_fit_transform_is_standardised_and_supports_single_patch_inference(patches):
    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(latent_dim=4)
    embedding = encoder.fit_transform(
        patches,
        max_epochs=1,
        batch_size=32,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
    )

    assert isinstance(embedding, np.ndarray)
    np.testing.assert_allclose(embedding.mean(axis=0), 0, atol=2e-5)
    assert encoder.transform(patches[:1]).shape == (1, 4)
    assert len(encoder.history["train_loss"]) == 1


def test_encode_patches_pools_anndata_and_records_provenance(tmp_path):
    import anndata
    from anndata import AnnData
    from packaging.version import Version

    from mantpy.nn import PatchEncoder, encode_patches

    rng = np.random.default_rng(1)
    left = AnnData(np.zeros((12, 0), dtype=np.float32))
    right = AnnData(np.zeros((10, 0), dtype=np.float32))
    left.obsm["image_patches"] = rng.random((12, 2, 8, 8), dtype=np.float32)
    right.obsm["image_patches"] = rng.random((10, 2, 8, 8), dtype=np.float32)
    left.obs["held_out_label"] = "not-read"

    encoder = encode_patches(
        [left, right],
        latent_dim=5,
        max_epochs=1,
        batch_size=16,
        accelerator="cpu",
        random_state=3,
        enable_progress_bar=False,
    )

    assert isinstance(encoder, PatchEncoder)
    assert left.obsm["X_cnn"].shape == (12, 5)
    assert right.obsm["X_cnn"].shape == (10, 5)
    params = left.uns["X_cnn_params"]
    assert params["objective"] == "contrastive"
    assert params["architecture"]["input_shape"] == [2, 8, 8]
    assert params["training"]["n_samples"] == 2
    assert params["training"]["n_patches"] == 22
    assert params["training"]["accelerator"] == "cpu"
    assert params["training"]["resolved_device"] == "cpu"
    assert params["standardization"] == "pooled StandardScaler"
    assert set(params["software"]) == {"mantpy", "torch", "pytorch_lightning", "scikit_learn"}
    assert all(isinstance(version, str) for version in params["software"].values())
    assert "held_out_label" not in str(params)
    # The project requires anndata>=0.12 for Arrow-string serialization, but
    # keep this focused test runnable in older local development environments.
    if Version(anndata.__version__) >= Version("0.12"):
        left.write_h5ad(tmp_path / "encoded.h5ad")


def test_save_load_roundtrip_preserves_features(tmp_path, patches):
    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(latent_dim=6).fit(
        patches,
        max_epochs=1,
        batch_size=32,
        accelerator="cpu",
        random_state=2,
        enable_progress_bar=False,
    )
    expected = encoder.transform(patches[:5])
    encoder.save(tmp_path / "patch_encoder")
    restored = PatchEncoder.load(tmp_path / "patch_encoder")

    assert restored.is_trained
    np.testing.assert_allclose(restored.transform(patches[:5]), expected, rtol=1e-6, atol=1e-6)


def test_pooled_fit_rejects_mismatched_patch_shapes(patches):
    from mantpy.nn import PatchEncoder

    with pytest.raises(ValueError, match="expected shared channel and spatial shape"):
        PatchEncoder().fit(
            [patches, np.ones((4, 2, 16, 16), dtype=np.float32)],
            max_epochs=1,
            accelerator="cpu",
        )


def test_fit_handles_single_patch_tail_batch():
    from mantpy.nn import PatchEncoder

    patches = np.random.default_rng(4).random((33, 1, 8, 8), dtype=np.float32)
    encoder = PatchEncoder(latent_dim=4).fit(
        patches,
        max_epochs=1,
        batch_size=32,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
    )

    assert encoder.transform(patches).shape == (33, 4)
