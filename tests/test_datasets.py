"""Tests for verified public dataset downloads and one-line loaders."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import tarfile
import tomllib
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import requests
import responses
import tifffile

import mantpy as mt
from mantpy.datasets import _registry
from mantpy.datasets._download import (
    _DownloadProgress,
    _file_spec,
    _NoAmbientAuth,
    _validate_root_manifest,
    prepare_bundle,
    resolve_cache_dir,
    safe_extract_tar,
)
from mantpy.datasets._loaders import _extract_spatial_archive


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, dataset: str, roles: dict[str, tuple[str, str | None, bool]]) -> Path:
    files = []
    for relative, (role, sample_id, quickstart) in roles.items():
        path = root / relative
        entry = {
            "path": relative.replace("\\", "/"),
            "role": role,
            "size": path.stat().st_size,
            "sha256": _digest(path),
        }
        if sample_id is not None:
            entry["sample_id"] = sample_id
        if quickstart:
            entry["quickstart"] = True
        files.append(entry)
    value = {
        "schema_version": 1,
        "dataset": dataset,
        "license": "CC-BY-4.0",
        "provenance": {"provider": "synthetic public fixture"},
        "metadata": {},
        "files": files,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _write_cells(path: Path, sample: str) -> None:
    cells = ad.AnnData(
        X=np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["AEC", "Fibroblast"]}, index=[f"{sample}_0", f"{sample}_1"]),
    )
    cells.obsm["spatial"] = np.array([[1, 2], [3, 4]], dtype=np.float32)
    cells.write_h5ad(path)


def test_coliv_one_line_loader_exposes_verified_quickstarts(tmp_path):
    image = np.arange(30, dtype=np.uint16).reshape(5, 6)
    tifffile.imwrite(tmp_path / "image.tiff", image)
    np.savez(
        tmp_path / "reference.npz",
        grid_y=np.array([0]),
        grid_x=np.array([1]),
        annotation=np.array([2]),
        layer_names=np.array(["Mucosa", "Submucosa", "Muscularis"]),
    )
    quickstart = tmp_path / "quickstart"
    quickstart.mkdir()
    (quickstart / "analysis.h5ad").write_bytes(b"verified-analysis")
    (quickstart / "external_labels.csv").write_text("patch_id,method,label\np0,m,1\n", encoding="utf-8")
    (quickstart / "external_metadata.json").write_text('{"schema_version":1}', encoding="utf-8")
    _manifest(
        tmp_path,
        "coliv_intestine",
        {
            "image.tiff": ("raw_image", None, False),
            "reference.npz": ("reference", None, False),
            "quickstart/analysis.h5ad": ("analysis", None, True),
            "quickstart/external_labels.csv": ("external_labels", None, True),
            "quickstart/external_metadata.json": ("external_metadata", None, True),
        },
    )

    data = mt.datasets.coliv_intestine(source_dir=tmp_path, progressbar=False)

    np.testing.assert_array_equal(data.image, image)
    assert set(data.quickstart) == {"analysis", "external_labels", "external_metadata"}
    assert data.paths["raw_image"] == tmp_path / "image.tiff"
    assert data.provenance["dataset"] == "coliv_intestine"
    assert data.provenance["license"] == "CC-BY-4.0"


def test_lung_loader_preserves_tutorial_schema_without_pickle(tmp_path):
    samples = ["roi_001", "roi_002"]
    panel = pd.DataFrame({"name": ["HABP", "CD3"], "keep": [1, 1], "ecm": [1, 0], "channel": ["001X", "002Y"]})
    panel.to_csv(tmp_path / "panel.csv", index=False)
    pd.DataFrame({"sample_id": samples, "mouse_id": ["mouse_01", "mouse_02"]}).to_csv(
        tmp_path / "metadata.csv", index=False
    )
    roles = {
        "panel.csv": ("panel", None, False),
        "metadata.csv": ("metadata", None, False),
    }
    for index, sample in enumerate(samples):
        _write_cells(tmp_path / f"{sample}.h5ad", sample)
        tifffile.imwrite(tmp_path / f"{sample}.ome.tiff", np.full((2, 4, 5), index + 1, dtype=np.uint16))
        roles[f"{sample}.h5ad"] = ("cells", sample, False)
        roles[f"{sample}.ome.tiff"] = ("raw_image", sample, False)
    pd.DataFrame(
        {
            "sample_id": ["roi_001", "roi_001", "roi_002", "roi_002"],
            "is_artifact": [False, True, False, True],
            "ecm_cluster_artifact": [0, 1, 1, 2],
            "ecm_cluster_pristine": [0, 0, 1, 1],
        }
    ).to_csv(tmp_path / "artifact_overlay.csv", index=False)
    (tmp_path / "artifact_boxes.json").write_text(
        json.dumps({sample: [[0, 0, 1, 1]] for sample in samples}), encoding="utf-8"
    )
    roles["artifact_overlay.csv"] = ("artifact_overlay", None, False)
    roles["artifact_boxes.json"] = ("artifact_boxes", None, False)
    manifest_path = _manifest(tmp_path, "balbc_pbs_lung", roles)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"] = {"showcase_roi": "roi_001", "K_ecm": 3}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    data = mt.datasets.balbc_pbs_lung(source_dir=tmp_path, progressbar=False)

    assert data.roi_names == samples
    assert list(data.cohort_cells) == samples
    assert list(data.images) == samples
    assert data.ecm_names == ["HABP"]
    assert data.target_channel_idx == 0
    assert data.artifact_overlays["roi_001"]["is_artifact"].tolist() == [False, True]
    assert data.artifact_overlays["roi_002"]["artifact_boxes"] == [[0, 0, 1, 1]]
    assert data.quickstart == {}


def test_lung_loader_normalizes_channel_last_tiff(tmp_path):
    panel = pd.DataFrame({"name": ["HABP", "CD3", "DNA"], "keep": [1, 1, 1], "ecm": [1, 0, 0]})
    panel.to_csv(tmp_path / "panel.csv", index=False)
    pd.DataFrame({"sample_id": ["roi_001"]}).to_csv(tmp_path / "metadata.csv", index=False)
    _write_cells(tmp_path / "cells.h5ad", "roi_001")
    image = np.arange(60, dtype=np.uint16).reshape(4, 5, 3)
    tifffile.imwrite(tmp_path / "image.tiff", image, metadata={"axes": "YXC"})
    pd.DataFrame({"sample_id": ["roi_001"], "is_artifact": [False], "ecm_cluster_artifact": [0]}).to_csv(
        tmp_path / "artifact_overlay.csv", index=False
    )
    _manifest(
        tmp_path,
        "balbc_pbs_lung",
        {
            "panel.csv": ("panel", None, False),
            "metadata.csv": ("metadata", None, False),
            "cells.h5ad": ("cells", "roi_001", False),
            "image.tiff": ("raw_image", "roi_001", False),
            "artifact_overlay.csv": ("artifact_overlay", None, False),
        },
    )

    data = mt.datasets.balbc_pbs_lung(source_dir=tmp_path, progressbar=False)

    assert data.images["roi_001"].shape == (3, 4, 5)
    np.testing.assert_array_equal(data.images["roi_001"][0], image[..., 0])


def test_schistosoma_loader_reads_manifested_channel_folders(tmp_path):
    samples = ["roi_001", "roi_002"]
    panel = pd.DataFrame(
        {
            "name": ["Collagen", "Laminin"],
            "channel": ["001X", "002Y"],
            "keep": [1, 1],
            "ecm": [1, 1],
        }
    )
    panel.to_csv(tmp_path / "panel.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": samples,
            "mouse_id": ["mouse_01", "mouse_02"],
            "condition": ["Naive", "Infected"],
            "genotype": ["WT", "KO"],
            "group": ["Naive_WT", "Infected_KO"],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)
    roles = {
        "panel.csv": ("panel", None, False),
        "metadata.csv": ("metadata", None, False),
    }
    for sample in samples:
        folder = tmp_path / "raw" / sample
        folder.mkdir(parents=True)
        for channel, marker in (("001X", "Collagen"), ("002Y", "Laminin")):
            relative = f"raw/{sample}/{sample}_{channel}_{marker}.ome.tiff"
            tifffile.imwrite(tmp_path / relative, np.ones((4, 5), dtype=np.uint16))
            roles[relative] = ("raw_channel", sample, False)
    manifest_path = _manifest(tmp_path, "schistosoma_ecm", roles)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"] = {"best_k": 7, "group_order": ["Naive_WT", "Infected_KO"]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    data = mt.datasets.schistosoma_ecm(source_dir=tmp_path, progressbar=False, verbose=False)

    assert list(data.raw_adatas) == samples
    assert data.images["roi_001"].shape == (2, 4, 5)
    assert data.ecm_names == ["Collagen", "Laminin"]
    assert data.group_order == ["Naive_WT", "Infected_KO"]
    assert data.best_k == 7


def test_prostate_loader_supplies_cells_pathology_and_matrisome(tmp_path, monkeypatch):
    import squidpy as sq

    tifffile.imwrite(tmp_path / "image.tiff", np.ones((8, 9, 3), dtype=np.uint8))
    (tmp_path / "counts.h5").write_bytes(b"synthetic counts")
    with tarfile.open(tmp_path / "spatial.tar.gz", "w:gz") as archive:
        for relative, payload in {
            "spatial/tissue_positions_list.csv": b"a,1,0,0,11,12\nb,1,0,1,21,22\n",
            "spatial/scalefactors_json.json": b'{"spot_diameter_fullres":130}',
        }.items():
            member = tarfile.TarInfo(relative)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    cells = ad.AnnData(X=np.empty((2, 0), dtype=np.float32))
    cells.obsm["spatial"] = np.array([[1, 2], [3, 4]], dtype=np.float32)
    cells.obsm["spatial_um"] = cells.obsm["spatial"].copy()
    cells.uns["segmentation"] = {"method": "Cellpose"}
    cells.write_h5ad(tmp_path / "cells.h5ad")
    (tmp_path / "pathology.csv").write_text("Barcode,Pathology\na,Stroma\n", encoding="utf-8")
    roles = {
        "image.tiff": ("raw_image", None, False),
        "counts.h5": ("counts", None, False),
        "spatial.tar.gz": ("spatial_archive", None, False),
        "cells.h5ad": ("cells", None, False),
        "pathology.csv": ("pathology", None, False),
    }
    manifest_path = _manifest(tmp_path, "prostate_he_visium", roles)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"] = {"library_id": "synthetic", "image_mpp": 0.5, "spot_radius_um": 32.5}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    visium = ad.AnnData(X=np.ones((2, 2), dtype=np.float32), obs=pd.DataFrame(index=["a", "b"]))
    visium.uns["spatial"] = {"synthetic": {"metadata": {"chemistry_description": "synthetic"}}}

    def read_counts_only(*args, **kwargs):
        assert kwargs["load_images"] is False
        assert "source_image_path" not in kwargs
        return visium.copy()

    monkeypatch.setattr(sq.read, "visium", read_counts_only)
    monkeypatch.setattr(
        mt.fetch,
        "fetch_matrisome",
        lambda *args, **kwargs: pd.DataFrame({"gene_symbol": ["COL1A1"]}),
    )

    data = mt.datasets.prostate_he_visium(source_dir=tmp_path, progressbar=False, lazy=False)

    assert data.cells.shape == (2, 0)
    assert data.pathology == tmp_path / "pathology.csv"
    assert data.matrisome["gene_symbol"].tolist() == ["COL1A1"]
    assert data.image_mpp == 0.5
    np.testing.assert_array_equal(data.visium.obsm["spatial"], np.array([[12, 11], [22, 21]]))
    np.testing.assert_array_equal(
        data.visium.obsm["spatial_um"],
        np.array([[6.0, 5.5], [11.0, 10.5]], dtype=np.float32),
    )
    assert data.visium.uns["spatial"]["synthetic"]["images"] == {}
    assert "source_image_path" not in data.visium.uns["spatial"]["synthetic"]["metadata"]

    lazy_data = mt.datasets.prostate_he_visium(
        source_dir=tmp_path,
        progressbar=False,
    )
    assert isinstance(
        lazy_data.image.get_layer("image", compute=False),
        da.Array,
    )

    # The Matrisome table is the one input not covered by the checksummed Zenodo
    # record — it is fetched live from Google Sheets. An upstream change there
    # must not sink the whole loader for callers who only want H&E/Visium/cells.
    def _matrisome_unavailable(*args, **kwargs):
        raise RuntimeError("verified Matrisome checksum mismatch")

    monkeypatch.setattr(mt.fetch, "fetch_matrisome", _matrisome_unavailable)

    with pytest.warns(RuntimeWarning, match="Matrisome"):
        degraded = mt.datasets.prostate_he_visium(
            source_dir=tmp_path, progressbar=False, lazy=False
        )

    assert degraded.matrisome is None
    assert degraded.cells.shape == (2, 0)
    assert degraded.pathology == tmp_path / "pathology.csv"


def _archive_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


def test_public_dataset_registry_pin_is_self_consistent():
    registry = _registry.DATASET_REGISTRY
    datasets = registry["datasets"]
    assert registry["record_id"] == 21538382
    assert isinstance(datasets, dict)
    assert set(datasets) == {
        "balbc_pbs_lung",
        "coliv_intestine",
        "prostate_he_visium",
        "schistosoma_ecm",
    }

    _file_spec(registry["manifest"], label="root manifest")
    for name, spec in datasets.items():
        _file_spec(spec, label=f"archive {name!r}")

    root = {
        "schema_version": 1,
        "record_id": registry["record_id"],
        "license": registry["license"],
        "datasets": datasets,
    }
    payload = (json.dumps(root, indent=2, sort_keys=True) + "\n").encode()
    _, expected_size, expected_sha256 = _file_spec(registry["manifest"], label="root manifest")
    assert len(payload) == expected_size
    assert hashlib.sha256(payload).hexdigest() == expected_sha256


@pytest.mark.parametrize("change", ["record_id", "extra_dataset"])
def test_root_manifest_rejects_registry_identity_drift(tmp_path, change):
    registry = _registry.DATASET_REGISTRY
    manifest = {
        "schema_version": 1,
        "record_id": registry["record_id"],
        "license": registry["license"],
        "datasets": dict(registry["datasets"]),
    }
    if change == "record_id":
        manifest["record_id"] = 1
        error = "pinned record identity"
    else:
        manifest["datasets"]["unexpected"] = {
            "filename": "unexpected.tar.gz",
            "size": 1,
            "sha256": "0" * 64,
        }
        error = "dataset set differs"
    path = tmp_path / "mantpy-datasets-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match=error):
        _validate_root_manifest(path, registry=registry, record_id=21538382)


@responses.activate
def test_pooch_downloads_once_then_reuses_verified_cache(tmp_path, monkeypatch):
    data_payload = b"raw-image"
    bundle_manifest = json.dumps(
        {
            "schema_version": 1,
            "dataset": "coliv_intestine",
            "license": "CC-BY-4.0",
            "provenance": {},
            "files": [
                {
                    "path": "image.tiff",
                    "role": "raw_image",
                    "size": len(data_payload),
                    "sha256": hashlib.sha256(data_payload).hexdigest(),
                }
            ],
        }
    ).encode()
    archive = _archive_bytes({"manifest.json": bundle_manifest, "image.tiff": data_payload})
    archive_spec = {
        "filename": "coliv.tar.gz",
        "size": len(archive),
        "sha256": hashlib.sha256(archive).hexdigest(),
    }
    other_spec = {"filename": "unused.tar.gz", "size": 1, "sha256": "1" * 64}
    datasets = {
        "coliv_intestine": archive_spec,
        "balbc_pbs_lung": other_spec,
        "schistosoma_ecm": other_spec,
        "prostate_he_visium": other_spec,
    }
    root_manifest = json.dumps(
        {"schema_version": 1, "record_id": 123, "license": "CC-BY-4.0", "datasets": datasets}
    ).encode()
    registry = {
        "schema_version": 1,
        "record_id": 123,
        "base_url": "https://zenodo.example",
        "license": "CC-BY-4.0",
        "manifest": {
            "filename": "manifest.json",
            "size": len(root_manifest),
            "sha256": hashlib.sha256(root_manifest).hexdigest(),
        },
        "datasets": datasets,
    }
    monkeypatch.setattr(_registry, "DATASET_REGISTRY", registry)
    responses.get(
        "https://zenodo.example/api/records/123/files/manifest.json/content",
        body=root_manifest,
    )
    responses.get(
        "https://zenodo.example/api/records/123/files/coliv.tar.gz/content",
        body=archive,
    )

    first = prepare_bundle("coliv_intestine", cache_dir=tmp_path, progressbar=False)
    second = prepare_bundle("coliv_intestine", cache_dir=tmp_path, progressbar=False)

    assert first.root == second.root
    assert (first.root / "image.tiff").read_bytes() == data_payload
    assert len(responses.calls) == 2

    (tmp_path / "datasets" / "123" / "coliv.tar.gz").write_bytes(b"interrupted")
    recovered = prepare_bundle("coliv_intestine", cache_dir=tmp_path, progressbar=False)
    assert recovered.root == first.root
    assert len(responses.calls) == 3


@pytest.mark.parametrize(
    ("dataset", "doi"),
    [
        ("coliv_intestine", "HBM893.MCGS.487"),
        ("balbc_pbs_lung", "10.1038/s44320-026-00234-5"),
        ("schistosoma_ecm", "10.1371/journal.ppat.1012928"),
        ("prostate_he_visium", "human-prostate-cancer-adenocarcinoma"),
    ],
)
@responses.activate
def test_first_dataset_download_prints_citation_once(tmp_path, monkeypatch, capsys, dataset, doi):
    data_payload = b"synthetic-public-data"
    bundle_manifest = json.dumps(
        {
            "schema_version": 1,
            "dataset": dataset,
            "license": "CC-BY-4.0",
            "provenance": {},
            "files": [
                {
                    "path": "data.csv",
                    "role": "data",
                    "size": len(data_payload),
                    "sha256": hashlib.sha256(data_payload).hexdigest(),
                }
            ],
        }
    ).encode()
    archive = _archive_bytes({"manifest.json": bundle_manifest, "data.csv": data_payload})
    archive_spec = {
        "filename": f"{dataset}.tar.gz",
        "size": len(archive),
        "sha256": hashlib.sha256(archive).hexdigest(),
    }
    other_spec = {"filename": "unused.tar.gz", "size": 1, "sha256": "1" * 64}
    datasets = {
        "coliv_intestine": other_spec,
        "balbc_pbs_lung": other_spec,
        "schistosoma_ecm": other_spec,
        "prostate_he_visium": other_spec,
        dataset: archive_spec,
    }
    root_manifest = json.dumps(
        {"schema_version": 1, "record_id": 123, "license": "CC-BY-4.0", "datasets": datasets}
    ).encode()
    registry = {
        "schema_version": 1,
        "record_id": 123,
        "base_url": "https://zenodo.example",
        "license": "CC-BY-4.0",
        "manifest": {
            "filename": "manifest.json",
            "size": len(root_manifest),
            "sha256": hashlib.sha256(root_manifest).hexdigest(),
        },
        "datasets": datasets,
    }
    monkeypatch.setattr(_registry, "DATASET_REGISTRY", registry)
    responses.get(
        "https://zenodo.example/api/records/123/files/manifest.json/content",
        body=root_manifest,
    )
    responses.get(
        f"https://zenodo.example/api/records/123/files/{dataset}.tar.gz/content",
        body=archive,
    )

    prepare_bundle(dataset, cache_dir=tmp_path, progressbar=False)
    first_output = capsys.readouterr().out
    prepare_bundle(dataset, cache_dir=tmp_path, progressbar=False)
    cached_output = capsys.readouterr().out

    assert "Please cite" in first_output
    assert doi in first_output
    assert "10.5281/zenodo.21538382" in first_output
    assert cached_output == ""


def test_dataset_downloader_strips_ambient_authorization():
    prepared = requests.Request(
        "GET", "https://zenodo.example/file", headers={"Authorization": "Bearer synthetic-secret"}
    ).prepare()

    result = _NoAmbientAuth()(prepared)

    assert "Authorization" not in result.headers


def test_download_progress_is_lazy_and_descriptive(monkeypatch):
    created = []

    class FakeBar:
        def __init__(self, **kwargs):
            self.total = kwargs["total"]
            self.description = kwargs["desc"]
            self.updates = []
            self.closed = False

        def update(self, amount):
            self.updates.append(amount)

        def reset(self, *, total):
            self.total = total
            self.updates.append("reset")

        def close(self):
            self.closed = True

    def fake_tqdm(**kwargs):
        bar = FakeBar(**kwargs)
        created.append(bar)
        return bar

    monkeypatch.setattr("mantpy.datasets._download.tqdm", fake_tqdm)
    progress = _DownloadProgress("Downloading BALB/c PBS lung dataset")
    progress.total = 128

    assert created == []
    progress.update(64)
    progress.reset()
    progress.update(128)
    progress.close()

    assert len(created) == 1
    assert created[0].description == "Downloading BALB/c PBS lung dataset"
    assert created[0].updates == [64, "reset", 128]
    assert created[0].closed


def test_download_and_lazy_image_dependencies_are_direct_runtime_requirements():
    from packaging.requirements import Requirement

    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    by_name: dict[str, list[Requirement]] = {}
    for entry in pyproject["project"]["dependencies"]:
        requirement = Requirement(entry)
        by_name.setdefault(requirement.name, []).append(requirement)

    # The Zenodo download path and the lazy TIFF/zarr image path must stay
    # direct runtime requirements, never hidden behind an optional extra.
    assert {"dask", "pooch", "squidpy", "tifffile", "tqdm", "zarr"} <= by_name.keys()
    assert "im" not in extras

    # dask needs its array subpackage; tifffile needs its zarr store backend.
    assert any("array" in requirement.extras for requirement in by_name["dask"])
    assert all("zarr" in requirement.extras for requirement in by_name["tifffile"])

    # squidpy >=1.8.2 is load-bearing: it requires spatialdata >=0.7.2, which
    # dropped xarray-schema, whose module-scope `pkg_resources` import fails on
    # setuptools >=82. Dropping below that floor would reintroduce the need for
    # a `setuptools<82` cap on every downstream consumer.
    assert not any(
        requirement.specifier.contains("1.8.1") for requirement in by_name["squidpy"]
    )
    assert "setuptools" not in by_name


def test_fetch_compatibility_wrappers_delegate_to_canonical_loaders(tmp_path, monkeypatch):
    prostate_cells = object()
    sentinels = {
        name: object()
        for name in ("coliv_intestine", "balbc_pbs_lung", "schistosoma_ecm")
    }
    sentinels["prostate_he_visium"] = SimpleNamespace(cells=prostate_cells)
    calls: list[tuple[str, dict[str, object]]] = []

    for name, sentinel in sentinels.items():

        def fake_loader(*, _name=name, _sentinel=sentinel, **kwargs):
            calls.append((_name, kwargs))
            return _sentinel

        monkeypatch.setattr(mt.datasets, name, fake_loader)

    assert mt.fetch.load_coliv_intestine(cache_dir=tmp_path) is sentinels["coliv_intestine"]
    assert mt.fetch.load_balbc_pbs_lung(cache_dir=tmp_path) is sentinels["balbc_pbs_lung"]
    assert (
        mt.fetch.load_schistosoma_ecm_cohort(cache_dir=tmp_path, include_raw=True, verbose=False)
        is sentinels["schistosoma_ecm"]
    )
    assert (
        mt.fetch.load_prostate_he_visium(cache_dir=tmp_path, lazy=False)
        is sentinels["prostate_he_visium"]
    )
    assert mt.fetch.load_prostate_cell_segmentation() is prostate_cells
    assert calls == [
        ("coliv_intestine", {"cache_dir": tmp_path}),
        ("balbc_pbs_lung", {"source_dir": None, "cache_dir": tmp_path}),
        ("schistosoma_ecm", {"cache_dir": tmp_path, "source_dir": None, "verbose": False}),
        ("prostate_he_visium", {"cache_dir": tmp_path, "lazy": False}),
        ("prostate_he_visium", {}),
    ]


def test_prostate_fetch_compatibility_signatures_are_preserved():
    sample_loader = inspect.signature(mt.fetch.load_prostate_he_visium).parameters
    assert list(sample_loader) == ["sample", "cache_dir", "force_refresh", "lazy"]
    assert sample_loader["sample"].default == "adeno"
    assert sample_loader["cache_dir"].default is None
    assert sample_loader["force_refresh"].default is False
    assert sample_loader["lazy"].default is True
    assert all(
        sample_loader[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in ("cache_dir", "force_refresh", "lazy")
    )

    cell_loader = inspect.signature(
        mt.fetch.load_prostate_cell_segmentation
    ).parameters
    assert list(cell_loader) == ["path"]
    assert cell_loader["path"].default is None


def test_lung_fetch_wrapper_forwards_zenodo_dir_without_deprecation(tmp_path, monkeypatch, recwarn):
    source_dir = tmp_path / "verified-source"
    cache_dir = tmp_path / "cache"
    sentinel = object()
    calls = []

    def fake_loader(*, source_dir, cache_dir):
        calls.append({"source_dir": source_dir, "cache_dir": cache_dir})
        return sentinel

    monkeypatch.setattr(mt.datasets, "balbc_pbs_lung", fake_loader)

    assert mt.fetch.load_balbc_pbs_lung(source_dir, cache_dir=cache_dir) is sentinel
    assert calls == [{"source_dir": source_dir, "cache_dir": cache_dir}]
    assert not any(issubclass(warning.category, DeprecationWarning) for warning in recwarn)


def test_cache_precedence(tmp_path, monkeypatch):
    environment = tmp_path / "environment"
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("MANTPY_CACHE", str(environment))

    assert resolve_cache_dir() == environment
    assert resolve_cache_dir(explicit) == explicit


@pytest.mark.parametrize(
    ("member", "message"),
    [
        ("../escape.csv", "Unsafe dataset path"),
        ("C:/escape.csv", "Unsafe dataset path"),
        ("payload.pkl", "Unsupported file type"),
        ("payload.py", "Unsupported file type"),
    ],
)
def test_safe_archive_rejects_traversal_pickle_and_executables(tmp_path, member, message):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = b"unsafe"
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match=message):
        safe_extract_tar(archive, tmp_path / "extract")


def test_prostate_spatial_archive_rejects_non_coordinate_assets(tmp_path):
    archive = tmp_path / "spatial.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, payload in {
            "spatial/tissue_positions_list.csv": b"a,1,0,0,1,2\n",
            "spatial/scalefactors_json.json": b"{}",
            "spatial/unexpected.csv": b"not,part,of,the,contract\n",
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="outside the coordinate-only 10x contract"):
        _extract_spatial_archive(archive, tmp_path / "extracted")


def test_inner_manifest_hash_mismatch_fails_closed(tmp_path):
    (tmp_path / "image.tiff").write_bytes(b"changed")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "coliv_intestine",
                "license": "CC-BY-4.0",
                "files": [
                    {
                        "path": "image.tiff",
                        "role": "raw_image",
                        "size": 7,
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="failed size/SHA-256"):
        prepare_bundle("coliv_intestine", source_dir=tmp_path)


def test_manifest_rejects_machine_specific_paths(tmp_path):
    (tmp_path / "image.tiff").write_bytes(b"image")
    manifest_path = _manifest(
        tmp_path,
        "coliv_intestine",
        {"image.tiff": ("raw_image", None, False)},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"] = {"source": r"Z:\private-source\image.tiff"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-portable manifest metadata"):
        prepare_bundle("coliv_intestine", source_dir=tmp_path)
