from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantpy as mt


def _typed_cells() -> ad.AnnData:
    cells = ad.AnnData(
        X=np.empty((3, 0), dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["A", "B", "A"]}, index=["c0", "c1", "c2"]),
    )
    cells.obsm["spatial"] = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
    return cells


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["collagen", "DNA", "fibronectin", "discarded"],
            "keep": [1, 1, 1, 0],
            "ecm ": [1, 0, 1, 0],
        }
    )


def test_input_summary_validates_typed_cells_and_channel_first_image() -> None:
    summary = mt.io.input_summary(
        _typed_cells(),
        np.zeros((3, 4, 5), dtype=np.uint16),
        _panel(),
    )
    assert summary.n_cells == 3
    assert summary.n_cell_vars == 0
    assert summary.spatial_shape == (3, 2)
    assert summary.image_shape == (3, 4, 5)
    assert "not used" in repr(summary)


def test_input_summary_rejects_missing_typed_cell_contract() -> None:
    cells = _typed_cells()
    del cells.obsm["spatial"]
    with pytest.raises(ValueError, match="spatial"):
        mt.io.input_summary(cells, np.zeros((3, 4, 5)), _panel())


def test_panel_summary_strips_ecm_column_and_validates_alignment() -> None:
    retained = _panel().loc[lambda frame: frame["keep"].eq(1)].copy()
    carrier = mt.io.read_imc(
        np.zeros((3, 4, 5), dtype=np.float32),
        retained,
        cells=None,
        normalize="none",
    )
    summary = mt.io.panel_summary(
        _panel(),
        image=np.zeros((3, 4, 5), dtype=np.float32),
        adata=carrier,
    )
    assert summary.acquired_channels == 4
    assert summary.retained_channels == 3
    assert summary.ecm_markers == ("collagen", "fibronectin")
    assert summary.adata_checked


def test_sample_group_map_is_ordered_and_complete() -> None:
    metadata = pd.DataFrame({"sample_id": ["r2", "r1"], "Mouse": [34, 31]})
    assert mt.io.sample_group_map(metadata, group_col="Mouse", samples=["r1", "r2"]) == {
        "r1": "31",
        "r2": "34",
    }
