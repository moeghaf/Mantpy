"""H5AD round-trip tests for Mantpy's public image readers."""

from __future__ import annotations

from collections.abc import Mapping

import anndata as ad
import numpy as np
import pytest

import mantpy as mt
from mantpy._constants import IMAGE_CONTAINER_KEY
from mantpy.im import (
    IMAGE_CONTAINER_SCHEMA,
    IMAGE_CONTAINER_SCHEMA_VERSION,
    ImageContainer,
    as_image_container,
)


def _assert_anndata_native(value) -> None:
    """Assert that a nested value contains no live application objects."""
    if isinstance(value, Mapping):
        for nested in value.values():
            _assert_anndata_native(nested)
        return
    assert isinstance(value, np.ndarray | str | int | float | bool)


def test_image_container_serializable_contract_and_legacy_coercion() -> None:
    image = np.arange(2 * 8 * 9, dtype=np.float32).reshape(2, 8, 9)
    container = ImageContainer(image, channel_names=["ColIV", "FN1"], scale=0.25)
    container.add_layer("preprocessed", image + 1.0)

    payload = container.to_serializable()

    assert payload["schema"] == IMAGE_CONTAINER_SCHEMA
    assert payload["schema_version"] == IMAGE_CONTAINER_SCHEMA_VERSION
    assert payload["default_layer"] == "image"
    assert payload["channel_axis"] == 0
    assert payload["scale"] == 0.25
    assert list(payload["layers"]) == ["image", "preprocessed"]
    assert payload["channel_names"].tolist() == ["ColIV", "FN1"]
    _assert_anndata_native(payload)
    assert as_image_container(container) is container


def test_serialized_container_view_keeps_layer_mapping_in_sync() -> None:
    payload = ImageContainer(np.ones((1, 6, 7), dtype=np.float32)).to_serializable()

    restored = as_image_container(payload)
    restored.add_layer("mask", np.zeros((6, 7), dtype=np.float32))

    assert "mask" in payload["layers"]
    assert payload["layers"]["mask"].shape == (1, 6, 7)
    restored.drop_layer("mask")
    assert "mask" not in payload["layers"]


def test_image_container_rejects_hdf5_unsafe_layer_names() -> None:
    image = np.ones((1, 6, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="without '/'"):
        ImageContainer(image, layer="raw/image")
    with pytest.raises(ValueError, match="non-empty"):
        ImageContainer(image).add_layer("", image)


def test_read_imc_h5ad_round_trip(
    tmp_path,
    synthetic_img,
    synthetic_panel,
    synthetic_cells,
) -> None:
    source = mt.read_imc(
        synthetic_img,
        panel=synthetic_panel,
        cells=synthetic_cells,
        sample_id="round_trip_roi",
    )
    payload = source.uns[IMAGE_CONTAINER_KEY]

    assert isinstance(payload, dict)
    assert not isinstance(payload, ImageContainer)
    _assert_anndata_native(payload)

    path = tmp_path / "read_imc.h5ad"
    source.write_h5ad(path)
    restored_adata = ad.read_h5ad(path)
    restored_image = as_image_container(restored_adata.uns[IMAGE_CONTAINER_KEY])

    np.testing.assert_allclose(restored_adata.X, source.X)
    np.testing.assert_allclose(restored_adata.layers["raw"], source.layers["raw"])
    np.testing.assert_allclose(restored_adata.obsm["spatial"], source.obsm["spatial"])
    np.testing.assert_allclose(restored_adata.uns["img"], source.uns["img"])
    np.testing.assert_allclose(
        restored_adata.uns["spatial"]["round_trip_roi"]["images"]["hires"],
        source.uns["spatial"]["round_trip_roi"]["images"]["hires"],
    )
    np.testing.assert_allclose(restored_image.to_array(), synthetic_img.astype(np.float32))
    assert restored_adata.var_names.tolist() == synthetic_panel["name"].tolist()
    assert restored_adata.obs["sample_id"].astype(str).tolist() == ["round_trip_roi"] * len(synthetic_cells)
    assert restored_adata.uns["mantpy"]["io"]["condition"] is None


def test_read_ecm_image_h5ad_round_trip_preserves_all_layers(tmp_path) -> None:
    rng = np.random.default_rng(13)
    image = rng.random((2, 10, 12), dtype=np.float32)
    preprocessed = image * 0.5
    container = ImageContainer(image, channel_names=["ColIV", "FN1"], scale=0.4)
    container.add_layer("preprocessed", preprocessed)

    source = mt.read_ecm_image(
        container,
        marker_names=["ColIV", "FN1"],
        sample_id="ecm_roi",
    )
    assert isinstance(source.uns[IMAGE_CONTAINER_KEY], dict)

    path = tmp_path / "read_ecm_image.h5ad"
    source.write_h5ad(path)
    restored_adata = ad.read_h5ad(path)
    restored_image = as_image_container(restored_adata.uns[IMAGE_CONTAINER_KEY])

    assert restored_adata.shape == (0, 2)
    assert restored_adata.var_names.tolist() == ["ColIV", "FN1"]
    assert restored_image.layers == ["image", "preprocessed"]
    assert restored_image.channel_names == ["ColIV", "FN1"]
    assert restored_image.scale == 0.4
    np.testing.assert_allclose(restored_image.to_array(), image)
    np.testing.assert_allclose(restored_image.get_layer("preprocessed"), preprocessed)
    np.testing.assert_allclose(restored_adata.uns["img"], image)

    # Public consumers can use the restored mapping without an explicit image
    # argument; this is the compatibility boundary the serialized contract is
    # designed to preserve.
    mt.pp.extract_ecm_patches(
        restored_adata,
        patch_size=4,
        features=["mean"],
        ecm_K=None,
    )
    assert len(restored_adata.uns["ecm_patches"]) > 0
