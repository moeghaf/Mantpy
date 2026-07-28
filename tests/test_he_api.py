"""Focused synthetic tests for the H&E/Visium input and preprocessing API."""

from __future__ import annotations

import io
import json
import sys
import tarfile
import types

import dask.array as da
import numpy as np
import pytest
import zarr
from anndata import AnnData

import mantpy as mt


@pytest.fixture
def synthetic_he() -> np.ndarray:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    image[8:88, 8:88] = (215, 135, 165)
    image[16:32, 16:32] = (75, 30, 105)
    image[56:72, 56:72] = (115, 45, 90)
    return image


def test_image_container_accepts_last_channel_and_rgb_rasters(tmp_path, synthetic_he):
    from PIL import Image

    container = mt.im.ImageContainer(synthetic_he, channel_axis=-1)
    assert container.shape == (3, 96, 96)
    np.testing.assert_array_equal(container.to_array(dtype=np.uint8), np.moveaxis(synthetic_he, -1, 0))

    path = tmp_path / "he.png"
    Image.fromarray(synthetic_he).save(path)
    loaded = mt.im.ImageContainer(path)
    assert loaded.shape == (3, 96, 96)
    assert loaded.channel_names is None


def test_image_container_lazy_rgb_tiff(tmp_path, synthetic_he):
    import tifffile

    assert zarr.__version__
    path = tmp_path / "he.tif"
    tifffile.imwrite(path, synthetic_he, photometric="rgb")
    image = mt.im.ImageContainer(path, lazy=True)
    assert image.shape == (3, 96, 96)
    assert isinstance(image.get_layer("image", compute=False), da.Array)
    assert image.crop(x=10, y=12, w=20, h=18).to_array().shape == (3, 18, 20)


def test_preprocess_he_returns_global_tissue_reference(synthetic_he):
    result = mt.pp.preprocess_he(synthetic_he, downsample=2, min_size_px=1)
    assert result.rgb.shape == (48, 48, 3)
    assert result.H.shape == result.E.shape == result.tissue.shape == (48, 48)
    assert result.H.dtype == result.E.dtype == np.float32
    assert 0.0 <= result.H.min() <= result.H.max() <= 1.0
    assert result.hematoxylin_range[1] > result.hematoxylin_range[0]
    assert result.eosin_range[1] > result.eosin_range[0]
    assert result.params["percentiles"] == (1.0, 99.0)
    assert "preview 48×48" in repr(result)


def test_he_overview_uses_preprocessed_preview(synthetic_he):
    import matplotlib.pyplot as plt

    result = mt.pp.preprocess_he(synthetic_he, downsample=2, min_size_px=1)
    fig = mt.pl.he_overview(result, image_mpp=0.5, show=False)

    assert len(fig.axes) == 4
    assert [axis.get_title() for axis in fig.axes] == [
        "H&E preview",
        "Haematoxylin",
        "Eosin",
        "Tissue mask",
    ]
    plt.close(fig)


def _install_fake_cellpose(monkeypatch, *, device=None):
    class FakeCellposeModel:
        def __init__(self, gpu=True):
            self.gpu = gpu
            if device is not None:
                self.device = device

        def eval(self, tiles, **kwargs):
            masks = []
            for tile in tiles:
                labels = np.zeros(tile.shape, dtype=np.int32)
                labels[4:10, 4:10] = 1
                labels[-10:-4, -10:-4] = 2
                masks.append(labels)
            return masks, None, None

    models = types.ModuleType("cellpose.models")
    models.CellposeModel = FakeCellposeModel
    package = types.ModuleType("cellpose")
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)


def test_tiled_hematoxylin_segmentation_streams_to_anndata(monkeypatch, synthetic_he):
    _install_fake_cellpose(monkeypatch)
    image = mt.im.ImageContainer(synthetic_he, channel_axis=-1, scale=0.5)
    reference = mt.pp.preprocess_he(image, downsample=2, min_size_px=1)

    cells = mt.pp.segment_cells_tiled(
        image,
        stain="hematoxylin",
        stain_reference=reference,
        downsample=2,
        output="anndata",
        image_mpp=0.5,
        tile_size=32,
        overlap=8,
        batch_size=2,
        gpu=False,
    )

    assert cells.n_vars == 0
    assert cells.n_obs > 0
    assert cells.obsm["spatial"].shape == (cells.n_obs, 2)
    np.testing.assert_allclose(cells.obsm["spatial_um"], cells.obsm["spatial"] * 0.5)
    assert cells.obsm["spatial"].max() <= 96
    assert cells.uns["segmentation"]["fullres_shape"] == [96, 96]
    summary = mt.pp.cell_segmentation_summary(cells)
    assert summary.n_cells == cells.n_obs
    assert summary.spatial_keys == ("spatial", "spatial_um")
    assert "CPU (cpu)" in repr(summary)
    assert cells.uns["segmentation"]["gpu_requested"] is False
    assert cells.uns["segmentation"]["device"] == "cpu"
    assert cells.uns["segmentation"]["gpu_used"] is False


def test_tiled_segmentation_records_resolved_cellpose_device(monkeypatch):
    _install_fake_cellpose(monkeypatch, device="cuda:1")
    image = np.ones((32, 32), dtype=np.float32)

    cells = mt.pp.segment_cells_tiled(
        image,
        output="anndata",
        gpu=True,
        tile_size=32,
        overlap=4,
        batch_size=1,
    )

    assert cells.uns["segmentation"]["gpu_requested"] is True
    assert cells.uns["segmentation"]["device"] == "cuda:1"
    assert cells.uns["segmentation"]["gpu_used"] is True


def test_he_ecm_patches_are_observation_native(monkeypatch, synthetic_he):
    _install_fake_cellpose(monkeypatch)
    image = mt.im.ImageContainer(synthetic_he, channel_axis=-1, scale=1.0)
    reference = mt.pp.preprocess_he(image, downsample=1, min_size_px=1)
    cells = AnnData(X=np.empty((2, 0), dtype=np.float32))
    cells.obsm["spatial"] = np.array([[24.0, 24.0], [64.0, 64.0]], dtype=np.float32)

    patches = mt.pp.he_ecm_patches(
        image,
        cells,
        stain_reference=reference,
        patch_size_um=16.0,
        tissue_fraction=0.30,
        strip_patches=2,
        min_size_px=1,
    )

    assert patches.n_vars == 0
    assert {
        "grid_row",
        "grid_col",
        "x0",
        "y0",
        "x1",
        "y1",
        "eosin",
        "hematoxylin",
        "n_nuclei",
        "tissue_fraction",
    }.issubset(patches.obs.columns)
    np.testing.assert_allclose(patches.obsm["spatial_um"], patches.obsm["spatial"])
    summary = patches.uns["he_ecm_patches"]
    assert summary["n_patches_total"] == 36
    assert summary["n_background"] + summary["n_cellular"] + summary["n_ecm"] == 36
    assert summary["eosin_threshold_rule"] == "otsu"
    assert summary["nuclear_density_threshold_rule"] == "median"
    assert np.isfinite(summary["eosin_threshold"])
    assert np.isfinite(summary["nuclear_density_threshold_mm2"])

    compact = mt.pp.he_ecm_patch_summary(patches)
    assert compact.n_patches_total == 36
    assert compact.n_tissue == 36 - summary["n_background"]
    assert compact.n_ecm == patches.n_obs
    assert compact.patch_size_um == 16.0
    assert compact.eosin_threshold_rule == "otsu"
    assert compact.nuclear_density_threshold_rule == "median"
    assert compact.coordinate_keys == ("spatial", "spatial_um")
    assert "AnnData observations" in repr(compact)


def test_he_ecm_patches_accept_fixed_thresholds(synthetic_he):
    image = mt.im.ImageContainer(synthetic_he, channel_axis=-1, scale=1.0)
    reference = mt.pp.preprocess_he(image, downsample=1, min_size_px=1)
    cells = AnnData(X=np.empty((2, 0), dtype=np.float32))
    cells.obsm["spatial"] = np.array([[24.0, 24.0], [64.0, 64.0]], dtype=np.float32)

    patches = mt.pp.he_ecm_patches(
        image,
        cells,
        stain_reference=reference,
        patch_size_um=16.0,
        tissue_fraction=0.0,
        eosin_threshold=-0.1,
        nuclear_density_threshold=1e9,
        strip_patches=2,
        min_size_px=1,
    )

    metadata = patches.uns["he_ecm_patches"]
    assert patches.n_obs == metadata["n_patches_total"] == 36
    assert metadata["n_background"] == metadata["n_cellular"] == 0
    assert metadata["eosin_threshold_rule"] == "fixed"
    assert metadata["eosin_threshold"] == pytest.approx(-0.1)
    assert metadata["nuclear_density_threshold_rule"] == "fixed"
    assert metadata["nuclear_density_threshold"] == pytest.approx(1e9)
    assert metadata["nuclear_density_threshold_mm2"] == pytest.approx(1e9)


@pytest.mark.parametrize(
    "threshold_kwargs",
    [
        {"eosin_threshold": "yen"},
        {"eosin_threshold": np.nan},
        {"eosin_threshold": np.inf},
        {"nuclear_density_threshold": "mean"},
        {"nuclear_density_threshold": -np.inf},
        {"nuclear_density_threshold": True},
    ],
)
def test_he_ecm_patches_reject_invalid_thresholds(synthetic_he, threshold_kwargs):
    image = mt.im.ImageContainer(synthetic_he, channel_axis=-1, scale=1.0)
    reference = mt.pp.preprocess_he(image, downsample=2, min_size_px=1)
    cells = np.empty((0, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="finite numeric value|must be finite"):
        mt.pp.he_ecm_patches(
            image,
            cells,
            stain_reference=reference,
            patch_size_um=16.0,
            min_size_px=1,
            **threshold_kwargs,
        )


def test_load_prostate_he_visium_force_refresh_preserves_direct_10x_loader(
    tmp_path, monkeypatch, synthetic_he
):
    import squidpy as sq
    import tifffile

    scalefactors = {"spot_diameter_fullres": 130.0}

    def fake_download(url, destination, *, min_bytes=1):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name.endswith("_spatial.tar.gz"):
            with tarfile.open(destination, "w:gz") as archive:
                payloads = {
                    "spatial/tissue_positions_list.csv": b"barcode,1,0,0,0,0\n",
                    "spatial/scalefactors_json.json": json.dumps(scalefactors).encode(),
                }
                for name, payload in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
        elif destination.suffix == ".tif":
            tifffile.imwrite(destination, synthetic_he, photometric="rgb")
        else:
            destination.write_bytes(b"synthetic-counts")
        return destination

    def fake_read_visium(*args, library_id, **kwargs):
        visium = AnnData(X=np.ones((3, 2), dtype=np.float32))
        visium.var_names = ["TBCE", "TBCE"]
        visium.var["gene_ids"] = ["ENSG000001", "ENSG000002"]
        visium.obsm["spatial"] = np.array([[0.0, 0.0], [200.0, 0.0], [400.0, 0.0]])
        visium.uns["spatial"] = {library_id: {"scalefactors": scalefactors}}
        return visium

    monkeypatch.setattr(mt.fetch, "_download_with_cache", fake_download)
    monkeypatch.setattr(sq.read, "visium", fake_read_visium)
    data = mt.fetch.load_prostate_he_visium(
        "adeno", cache_dir=tmp_path, force_refresh=True, lazy=False
    )

    assert data.image.shape == (3, 96, 96)
    assert data.image_mpp == pytest.approx(0.5)
    assert data.spot_radius_um == pytest.approx(32.5)
    np.testing.assert_allclose(data.visium.obsm["spatial_um"], data.visium.obsm["spatial"] * 0.5)
    assert data.visium.var_names.tolist() == ["TBCE", "TBCE-1"]
    assert data.visium.var["gene_ids"].tolist() == ["ENSG000001", "ENSG000002"]
    assert data.provenance["provider"] == "10x Genomics"
    assert data.provenance["var_names_make_unique"] == {
        "applied": True,
        "duplicate_count": 1,
        "join": "-",
        "gene_ids_preserved": "var['gene_ids']",
    }
    assert data.provenance["files"] == {name: path.name for name, path in data.paths.items()}
    assert str(tmp_path) not in repr(data.visium.uns["mantpy"]["prostate_he_visium"]["provenance"])
    assert data.paths["spatial"].is_dir()


@pytest.mark.parametrize("sample", ["normal", "acinar"])
def test_load_prostate_he_visium_non_adeno_keeps_direct_10x_route(
    tmp_path, monkeypatch, sample
):
    class DirectRouteSelected(Exception):
        pass

    monkeypatch.setattr(
        mt.datasets,
        "prostate_he_visium",
        lambda **kwargs: pytest.fail(f"canonical loader called with {kwargs}"),
    )

    def direct_download(*args, **kwargs):
        del args, kwargs
        raise DirectRouteSelected

    monkeypatch.setattr(mt.fetch, "_download_with_cache", direct_download)
    with pytest.raises(DirectRouteSelected):
        mt.fetch.load_prostate_he_visium(sample, cache_dir=tmp_path)


def test_load_prostate_cell_segmentation_supports_explicit_path(tmp_path):
    cells = AnnData(X=np.empty((2, 0), dtype=np.float32))
    cells.obsm["spatial"] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    cells.obsm["spatial_um"] = cells.obsm["spatial"] * 0.5
    cells.uns["segmentation"] = {
        "backend": "cellpose-sam",
        "coordinate_only_deposit": True,
    }
    asset = tmp_path / "zenodo_data" / "prostate_he_visium" / "adeno_cellpose_cells.h5ad"
    asset.parent.mkdir(parents=True)
    cells.write_h5ad(asset)

    explicit = mt.fetch.load_prostate_cell_segmentation(asset)
    assert explicit.shape == (2, 0)
    assert explicit.uns["segmentation"]["loaded_from"] == asset.name
    assert str(tmp_path) not in repr(explicit.uns["segmentation"])

@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("variables", "zero variables"),
        ("spatial", "obsm\\['spatial'\\]"),
        ("spatial_um", "obsm\\['spatial'\\]"),
        ("provenance", "uns\\['segmentation'\\]"),
    ],
)
def test_load_prostate_cell_segmentation_rejects_non_coordinate_assets(
    tmp_path, mutation, message
):
    n_vars = 1 if mutation == "variables" else 0
    cells = AnnData(X=np.empty((2, n_vars), dtype=np.float32))
    if mutation != "spatial":
        cells.obsm["spatial"] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    if mutation != "spatial_um":
        cells.obsm["spatial_um"] = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    if mutation != "provenance":
        cells.uns["segmentation"] = {"backend": "cellpose-sam"}
    asset = tmp_path / f"invalid-{mutation}.h5ad"
    cells.write_h5ad(asset)

    with pytest.raises(ValueError, match=message):
        mt.fetch.load_prostate_cell_segmentation(asset)


def test_load_prostate_cell_segmentation_propagates_public_dataset_failure(monkeypatch):
    def unavailable():
        raise RuntimeError("immutable public dataset has not been published yet")

    monkeypatch.setattr(mt.datasets, "prostate_he_visium", unavailable)
    with pytest.raises(RuntimeError, match="has not been published yet"):
        mt.fetch.load_prostate_cell_segmentation()
