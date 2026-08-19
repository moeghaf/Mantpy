from __future__ import annotations

import importlib.util

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

import mantpy as mt


def _inputs(*, physical_coordinates: bool = True) -> tuple[ad.AnnData, ad.AnnData]:
    cells = ad.AnnData(
        X=np.empty((4, 0), dtype=np.float32),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}, index=["0", "1", "2", "3"]),
    )
    cells.obsm["spatial"] = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
    ecm = ad.AnnData(
        X=np.arange(8, dtype=np.float32).reshape(4, 2),
        obs=pd.DataFrame(
            {
                "grid_y": [0, 0, 1, 1],
                "grid_x": [0, 1, 0, 1],
                "ecm_cluster": [0, 0, 0, 0],
            },
            index=["0", "1", "2", "3"],
        ),
    )
    ecm.obsm["spatial"] = np.array([[2, 2], [6, 2], [2, 6], [6, 6]], dtype=np.float32)
    if physical_coordinates:
        cells.obsm["spatial_um"] = cells.obsm["spatial"] * 0.5
        ecm.obsm["spatial_um"] = ecm.obsm["spatial"] * 0.5
    return cells, ecm


def test_compose_cell_ecm_graph_uses_sparse_serialisable_schema(tmp_path) -> None:
    cells, ecm = _inputs()

    joint = mt.gr.compose_cell_ecm_graph(cells, ecm, cell_k=1, cell_ecm_radius=3.0)

    assert joint.n_obs == 8
    assert joint.n_vars == 0
    assert list(joint.obs["node_type"].cat.categories) == ["cell", "ecm"]
    assert joint.obs_names.is_unique
    assert joint.obs_names[0] == "cell:0"
    assert joint.obs_names[-1] == "ecm:3"
    np.testing.assert_allclose(joint.obsm["spatial"][:4], cells.obsm["spatial"])
    np.testing.assert_allclose(joint.obsm["spatial_um"][4:], ecm.obsm["spatial_um"])

    keys = {
        "cell_connectivities",
        "ecm_connectivities",
        "cell_ecm_connectivities",
        "joint_connectivities",
    }
    assert keys <= set(joint.obsp)
    for key in keys:
        matrix = joint.obsp[key]
        assert sp.isspmatrix_csr(matrix)
        assert matrix.shape == (8, 8)
        assert (matrix != matrix.T).nnz == 0
        assert not np.any(matrix.diagonal())
        assert set(np.unique(matrix.data)) <= {1.0}

    layers = (
        joint.obsp["cell_connectivities"] + joint.obsp["ecm_connectivities"] + joint.obsp["cell_ecm_connectivities"]
    )
    layers.data[:] = 1
    assert (layers != joint.obsp["joint_connectivities"]).nnz == 0
    assert joint.obsp["ecm_connectivities"].nnz // 2 == 6
    assert joint.uns["cell_ecm_graph"]["format"] == "mantpy_sparse_joint_v1"

    path = tmp_path / "joint.h5ad"
    joint.write_h5ad(path)
    restored = ad.read_h5ad(path)
    assert sp.isspmatrix_csr(restored.obsp["joint_connectivities"])
    assert restored.uns["cell_ecm_graph"]["counts"]["n_nodes"] == 8


def test_compose_derives_physical_coordinates_and_summary_counts() -> None:
    cells, ecm = _inputs(physical_coordinates=False)

    joint = mt.gr.compose_cell_ecm_graph(
        cells,
        ecm,
        cell_k=2,
        cell_ecm_radius=3.0,
        pixel_size_um=0.5,
    )
    summary = mt.gr.joint_graph_summary(joint, sample="section")

    np.testing.assert_allclose(joint.obsm["spatial_um"], joint.obsm["spatial"] * 0.5)
    assert summary.sample == "section"
    assert summary.n_nodes == 8
    assert summary.n_cells == 4
    assert summary.n_ecm == 4
    assert summary.cell_cell_edges == sp.triu(joint.obsp["cell_connectivities"], k=1).nnz
    assert summary.ecm_ecm_edges == 6
    assert summary.cell_ecm_edges == joint.obsp["cell_ecm_connectivities"][:4, 4:].nnz
    assert "cell-ECM" in repr(summary)


def test_sparse_cell_ecm_graph_plotter_accepts_joint_anndata() -> None:
    cells, ecm = _inputs()
    joint = mt.gr.compose_cell_ecm_graph(cells, ecm, cell_k=1, cell_ecm_radius=3.0)

    ax = mt.pl.cell_ecm_graph(joint, max_edges=2, show=False)

    assert ax.get_title() == "Cell-ECM Graph"
    assert len(ax.collections) >= 2
    plt.close(ax.figure)


HAS_PYG = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None


def test_sparse_to_hetero_pyg_when_available() -> None:
    if not HAS_PYG:
        return
    cells, ecm = _inputs()
    joint = mt.gr.compose_cell_ecm_graph(cells, ecm, cell_k=1, cell_ecm_radius=3.0)
    joint.obsm["shared_features"] = np.arange(24, dtype=np.float32).reshape(8, 3)

    data = mt.gr.to_hetero_pyg(joint, node_feature_key="shared_features")

    assert tuple(data["cell"].x.shape) == (4, 3)
    assert tuple(data["ecm"].x.shape) == (4, 3)
    assert tuple(data["cell"].pos.shape) == (4, 2)
    assert data["cell", "cell-cell", "cell"].edge_index.shape[1] == joint.obsp["cell_connectivities"].nnz
    assert data["ecm", "ecm-ecm", "ecm"].edge_index.shape[1] == joint.obsp["ecm_connectivities"].nnz
    assert data["cell", "cell-ecm", "ecm"].edge_index.shape[1] == joint.obsp["cell_ecm_connectivities"][:4, 4:].nnz
    assert data["ecm", "ecm-cell", "cell"].edge_index.shape[1] == joint.obsp["cell_ecm_connectivities"][4:, :4].nnz
