from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

import mantpy as mt


def _joint_input() -> ad.AnnData:
    cells = ad.AnnData(
        X=np.empty((4, 0), dtype=np.float32),
        obs=pd.DataFrame(
            {"cell_type": ["A", "A", "B", "B"]},
            index=["c0", "c1", "c2", "c3"],
        ),
    )
    cells.obsm["spatial"] = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=np.float32)
    cells.uns["ecm_patches"] = pd.DataFrame(
        {
            "x": [0.5, 1.5, 0.5, 1.5],
            "y": [0.5, 0.5, 1.5, 1.5],
            "ecm_cluster": [0, 0, 1, 1],
            "feat_0": [0.0, 0.1, 1.0, 1.1],
        }
    )
    return cells


def test_build_cell_ecm_graphs_uses_squidpy_radius_and_returns_recipe() -> None:
    cohort = {"r1": _joint_input()}
    result = mt.gr.build_cell_ecm_graphs(
        cohort,
        cell_graph="squidpy",
        cell_k=2,
        cell_radius=(0, 3),
        ecm_edge_method="grid",
        ecm_grid_connectivity=8,
        cell_ecm_k=1,
        cell_ecm_Dmax=3,
    )
    assert len(result.summaries) == 1
    assert result.summaries[0].n_cells == 4
    assert result.summaries[0].n_ecm == 4
    assert result.summaries[0].cell_ecm_edges > 0
    assert result.rebuild_kwargs["cell_connectivity_key"] == "spatial_connectivities"
    assert "spatial_connectivities" in cohort["r1"].obsp
    assert "cell_ecm_graph" in cohort["r1"].uns


def test_joint_graph_summary_exposes_edge_types() -> None:
    cohort = {"r1": _joint_input()}
    mt.gr.build_cell_ecm_graphs(
        cohort,
        cell_graph="mantpy",
        cell_k=2,
        cell_Dmax=3,
        ecm_edge_method="grid",
        cell_ecm_k=1,
        cell_ecm_Dmax=3,
    )
    summary = mt.gr.joint_graph_summary(cohort["r1"], sample="r1")
    assert summary.sample == "r1"
    assert summary.n_nodes == summary.n_cells + summary.n_ecm
