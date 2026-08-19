"""Contracts for the download-free synthetic ROI used by the documentation."""

from __future__ import annotations

import numpy as np
import pytest

import mantpy as mt


def test_returns_a_bunch_shaped_like_the_real_loaders():
    roi = mt.datasets.toy_ecm_roi()

    assert roi.image.shape == (5, 96, 96)
    assert roi.image.dtype == np.uint16
    assert list(roi.panel.columns) == ["name", "ecm"]
    assert len(roi.panel) == roi.image.shape[0]
    assert roi.panel["ecm"].sum() == 2
    assert set(roi.cells.columns) == {"celltype", "centroid-0", "centroid-1"}
    assert len(roi.cells) == 40
    # Bunch supports both access styles, like every other loader here.
    assert roi["image"] is roi.image


def test_is_deterministic_for_a_given_seed():
    a = mt.datasets.toy_ecm_roi(seed=7)
    b = mt.datasets.toy_ecm_roi(seed=7)
    c = mt.datasets.toy_ecm_roi(seed=8)

    np.testing.assert_array_equal(a.image, b.image)
    assert not np.array_equal(a.image, c.image)


@pytest.mark.parametrize(("size", "n_cells"), [(32, 4), (64, 12), (96, 40)])
def test_honours_size_and_cell_count(size, n_cells):
    roi = mt.datasets.toy_ecm_roi(size=size, n_cells=n_cells)

    assert roi.image.shape == (5, size, size)
    assert len(roi.cells) == n_cells
    coords = roi.cells[["centroid-0", "centroid-1"]].to_numpy()
    assert coords.min() >= 0
    assert coords.max() <= size - 1


@pytest.mark.parametrize(("kwargs", "match"), [({"size": 8}, "size"), ({"n_cells": 1}, "n_cells")])
def test_rejects_degenerate_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mt.datasets.toy_ecm_roi(**kwargs)


def test_feeds_the_documented_pipeline_without_a_download():
    """The exact call sequence the quickstart documents must work end to end."""
    roi = mt.datasets.toy_ecm_roi()
    adata = mt.io.read_imc(
        roi.image, panel=roi.panel, cells=roi.cells, sample_id="toy", condition="ctrl"
    )
    assert adata.n_obs == 40

    mt.pp.extract_ecm_patches(adata, roi.image, patch_size=8, ecm_K=3, features=["mean"])
    mt.gr.build_cell_graph(adata, k=5)
    mt.gr.build_ecm_graph(adata, k=5)
    mt.gr.build_cell_ecm_graph(adata, k=5)

    joint = adata.uns["cell_ecm_graph"]
    assert joint.number_of_nodes() > adata.n_obs
    assert joint.number_of_edges() > 0


def test_ecm_clusters_are_spatially_coherent():
    """The image carries real structure, so clusters must not be salt-and-pepper.

    Uniform noise scores ~0.33 here at K=3 (chance). Anything near that would
    mean the documentation is teaching the API on meaningless labels.
    """
    roi = mt.datasets.toy_ecm_roi()
    adata = mt.io.read_imc(
        roi.image, panel=roi.panel, cells=roi.cells, sample_id="toy", condition="ctrl"
    )
    mt.pp.extract_ecm_patches(adata, roi.image, patch_size=8, ecm_K=3, features=["mean"])

    patches = adata.uns["ecm_patches"]
    xy = patches[["x", "y"]].to_numpy(float)
    labels = patches["ecm_cluster"].to_numpy()

    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    agreement = (labels[distances.argmin(1)] == labels).mean()

    assert agreement > 0.6, f"clusters look arbitrary (agreement={agreement:.2f})"


def test_ships_no_bytes_in_the_package():
    """The generator exists so the docs need no data files in the wheel."""
    from mantpy.datasets import _toy

    package_dir = __import__("pathlib").Path(_toy.__file__).parent
    payloads = [
        p
        for p in package_dir.rglob("*")
        if p.is_file() and p.suffix not in {".py", ".pyc", ".typed"}
    ]
    assert payloads == []
