"""Focused tests for sparse joint-graph and cross-modal analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import mantpy as mt


def _joint_adata() -> AnnData:
    obs = pd.DataFrame(
        {"node_type": pd.Categorical(["cell", "cell", "cell", "ecm", "ecm", "ecm"])},
        index=pd.Index(np.asarray(["c0", "c1", "c2", "e0", "e1", "e2"], dtype=object), dtype=object),
    )
    adata = AnnData(X=np.zeros((6, 1), dtype=np.float32), obs=obs)
    # Cell--ECM contacts: c0--e0/e1 and c2--e2. Store both orientations.
    row = np.array([0, 0, 2, 3, 4, 5])
    col = np.array([3, 4, 5, 0, 0, 2])
    adata.obsp["cell_ecm_connectivities"] = sp.csr_matrix((np.ones(6), (row, col)), shape=(6, 6))
    # Cell chain c0--c1--c2; ECM nodes deliberately have no cell adjacency.
    row = np.array([0, 1, 1, 2])
    col = np.array([1, 0, 2, 1])
    adata.obsp["cell_connectivities"] = sp.csr_matrix((np.ones(4), (row, col)), shape=(6, 6))
    return adata


def test_cell_ecm_contact_counts_cross_edges_and_preserves_roles() -> None:
    adata = _joint_adata()

    result = mt.tl.cell_ecm_contact(adata)

    np.testing.assert_array_equal(adata.obs.loc[["c0", "c1", "c2"], "ecm_degree"], [2, 0, 1])
    np.testing.assert_array_equal(adata.obs.loc[["c0", "c1", "c2"], "ecm_contact"], [1, 0, 1])
    assert adata.obs.loc[["e0", "e1", "e2"], "ecm_contact"].isna().all()
    assert result.n_cells == 3
    assert result.n_contacted == 2
    assert result.fraction_contacted == pytest.approx(2 / 3)
    assert adata.uns["ecm_contact_params"]["graph_key"] == "cell_ecm_connectivities"


def test_cell_ecm_contact_threshold_and_copy() -> None:
    adata = _joint_adata()

    result = mt.tl.cell_ecm_contact(adata, threshold=1, inplace=False)

    assert "ecm_contact" not in adata.obs
    np.testing.assert_array_equal(result.adata.obs.loc[["c0", "c1", "c2"], "ecm_contact"], [1, 0, 0])


def test_cell_ecm_contact_consumes_compose_cell_ecm_graph_output() -> None:
    cells = AnnData(X=np.empty((3, 0)), obs=pd.DataFrame(index=["c0", "c1", "c2"]))
    cells.obsm["spatial"] = cells.obsm["spatial_um"] = np.array([[0, 0], [2, 0], [8, 0]], dtype=float)
    ecm = AnnData(
        X=np.empty((2, 0)),
        obs=pd.DataFrame({"grid_y": [0, 0], "grid_x": [0, 1]}, index=["e0", "e1"]),
    )
    ecm.obsm["spatial"] = ecm.obsm["spatial_um"] = np.array([[0.5, 0], [2.5, 0]], dtype=float)
    joint = mt.gr.compose_cell_ecm_graph(cells, ecm, cell_k=1, cell_ecm_radius=1.0)

    result = mt.tl.cell_ecm_contact(joint)

    expected = np.asarray(joint.obsp["cell_ecm_connectivities"][:3, 3:].getnnz(axis=1))
    np.testing.assert_array_equal(joint.obs.iloc[:3]["ecm_degree"], expected)
    assert result.n_contacted == 2


def test_smooth_graph_signal_obs_is_mask_aware_and_node_restricted() -> None:
    adata = _joint_adata()
    adata.obs["signal"] = [1.0, 0.0, np.nan, 9.0, 9.0, 9.0]

    result = mt.tl.smooth_graph_signal(
        adata,
        "signal",
        graph_key="cell_connectivities",
        key_added="signal_smooth",
        alpha=1.0,
        n_iter=1,
        node_type="cell",
    )

    # c0 sees c1; c1 sees finite c0 but ignores missing c2. Missing and ECM
    # observations remain outside the immutable support.
    np.testing.assert_allclose(adata.obs.loc[["c0", "c1"], "signal_smooth"], [0.0, 1.0])
    assert adata.obs.loc[["c2", "e0", "e1", "e2"], "signal_smooth"].isna().all()
    assert result.storage == "obs"
    assert result.n_finite == (2,)
    assert adata.uns["signal_smooth_params"]["missing_policy"].startswith("fixed support")


def test_smooth_graph_signal_multiple_vars_writes_ordered_obsm() -> None:
    adata = AnnData(
        X=np.array([[1.0, 10.0], [0.0, 20.0], [3.0, 30.0]], dtype=np.float32),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    row = np.array([0, 1, 1, 2])
    col = np.array([1, 0, 2, 1])
    adata.obsp["spatial_connectivities"] = sp.csr_matrix((np.ones(4), (row, col)), shape=(3, 3))

    result = mt.tl.smooth_graph_signal(
        adata,
        ["G2", "G1"],
        graph_key="spatial_connectivities",
        source="var",
        key_added="X_regulator_smooth",
        alpha=1.0,
        n_iter=1,
    )

    np.testing.assert_allclose(
        adata.obsm["X_regulator_smooth"],
        np.array([[20.0, 0.0], [20.0, 2.0], [20.0, 0.0]]),
    )
    assert result.feature_names == ("G2", "G1")
    assert adata.uns["X_regulator_smooth_params"]["feature_names"] == ["G2", "G1"]


def _transfer_objects() -> tuple[AnnData, AnnData]:
    source = AnnData(
        X=np.array([[10.0], [20.0], [30.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {"score": [1.0, 3.0, 5.0], "use": [True, True, False]},
            index=pd.Index(np.asarray(["s0", "s1", "s2"], dtype=object), dtype=object),
        ),
        var=pd.DataFrame(index=["GENE"]),
    )
    source.obsm["spatial_um"] = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    target = AnnData(X=np.empty((3, 0)), obs=pd.DataFrame(index=["t0", "t1", "t2"]))
    target.obsm["spatial_um"] = np.array([[0.5, 0.0], [4.1, 0.0], [10.0, 0.0]])
    return source, target


def test_transfer_radius_resolves_obs_and_var_and_keeps_unsupported_nan() -> None:
    source, target = _transfer_objects()

    result = mt.tl.transfer_spatial_features(
        source,
        target,
        ["score", "GENE"],
        method="radius",
        radius=0.6,
        source_spatial_key="spatial_um",
        target_spatial_key="spatial_um",
        key_added="X_mapped",
    )

    np.testing.assert_allclose(target.obsm["X_mapped"][:2], [[2.0, 15.0], [5.0, 30.0]])
    assert np.isnan(target.obsm["X_mapped"][2]).all()
    np.testing.assert_array_equal(target.obs["X_mapped_covered"], [True, True, False])
    np.testing.assert_array_equal(target.obs["X_mapped_n_sources"], [2, 1, 0])
    assert result.n_covered == 2
    assert target.uns["X_mapped_params"]["input_source"] == "mixed"


@pytest.mark.parametrize(("aggregation", "expected"), [("sum", 4.0), ("max", 3.0)])
def test_transfer_radius_aggregation(aggregation: str, expected: float) -> None:
    source, target = _transfer_objects()
    one = target[:1].copy()
    mt.tl.transfer_spatial_features(
        source,
        one,
        "score",
        method="radius",
        radius=0.6,
        aggregation=aggregation,
        source_spatial_key="spatial_um",
        target_spatial_key="spatial_um",
        key_added="mapped",
    )
    assert one.obs["mapped"].iloc[0] == pytest.approx(expected)


def test_transfer_nearest_source_mask_and_distance_cap() -> None:
    source, target = _transfer_objects()

    result = mt.tl.transfer_spatial_features(
        source,
        target,
        "score",
        method="nearest",
        max_distance=0.75,
        source_mask="use",
        source_spatial_key="spatial_um",
        target_spatial_key="spatial_um",
        key_added="nearest_score",
        inplace=False,
    )

    assert "nearest_score" not in target.obs
    assert result.target.obs["nearest_score"].iloc[0] == pytest.approx(1.0)
    assert result.target.obs["nearest_score"].iloc[1:].isna().all()
    np.testing.assert_array_equal(result.target.obs["nearest_score_n_sources"], [1, 0, 0])
    assert result.target.uns["nearest_score_params"]["source_mask"] == "use"
