"""Shared pytest fixtures for Mantpy tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_panel():
    return pd.DataFrame(
        {
            "name": ["ColIV", "FN", "CD20", "EpCAM", "DAPI"],
            "ecm": [1, 1, 0, 0, 0],
        }
    )


@pytest.fixture
def synthetic_img(rng):
    """(C=5, H=64, W=64) uint16 image."""
    return rng.integers(0, 500, size=(5, 64, 64), dtype=np.uint16)


@pytest.fixture
def synthetic_cells():
    """Small cell DataFrame — 10 cells spread across the image."""
    return pd.DataFrame(
        {
            "celltype": ["T", "B", "T", "Mac", "B", "T", "DC", "Mac", "B", "T"],
            "centroid-0": [10, 20, 30, 40, 50, 12, 22, 32, 42, 52],
            "centroid-1": [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
        }
    )


@pytest.fixture
def adata_basic(synthetic_img, synthetic_panel, synthetic_cells):
    """Minimal AnnData produced by read_imc — all downstream fixtures build on this."""
    import mantpy as mt

    return mt.read_imc(
        synthetic_img,
        panel=synthetic_panel,
        cells=synthetic_cells,
        sample_id="test_sample",
        condition="ctrl",
    )


@pytest.fixture
def adata_with_patches(adata_basic, synthetic_img):
    """AnnData with ECM patches extracted."""
    import mantpy as mt

    mt.pp.extract_ecm_patches(adata_basic, synthetic_img, patch_size=8, ecm_K=3, features=["mean"])
    return adata_basic


@pytest.fixture
def adata_with_graphs(adata_with_patches):
    """AnnData with all three graphs built."""
    import mantpy as mt

    mt.gr.build_cell_graph(adata_with_patches, k=3)
    mt.gr.build_ecm_graph(adata_with_patches, k=3)
    mt.gr.build_cell_ecm_graph(adata_with_patches, k=3)
    return adata_with_patches


@pytest.fixture
def img_container(synthetic_img):
    """ImageContainer wrapping the synthetic 5-channel image."""
    from mantpy.im import ImageContainer

    names = ["ColIV", "FN", "CD20", "EpCAM", "DAPI"]
    return ImageContainer(synthetic_img.astype(np.float32), channel_names=names)
