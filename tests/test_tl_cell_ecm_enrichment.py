"""Tests for ``mt.tl.cell_ecm_enrichment`` (within-ROI permutation null).

The null permutes cell-type labels WITHIN each ROI on the fixed cell-ECM
graph, so these tests build small hand-made graphs with a planted
AEC <-> ECM-cluster-1 association and check that the effect size, the
empirical p-value, the per-ROI breakdown, and the background-exclusion
behaviour are all correct.
"""

from __future__ import annotations

import anndata as ad
import networkx as nx
import numpy as np
import pytest

import mantpy as mt

K_ECM = 3
AEC_CLUSTER = 1  # the cluster AEC cells are planted to prefer


def _make_roi(seed: int, *, edges_per_cell: int = 4) -> ad.AnnData:
    """Build a minimal AnnData whose ``uns['cell_ecm_graph']`` carries a
    planted association: AEC cells edge to ECM cluster 1, OTHER cells edge
    to ECM clusters 0 and 2. One AEC also touches a background patch
    (``ecm_cluster == -1``) that must be ignored.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    n_aec, n_other = 15, 15
    for i in range(n_aec):
        G.add_node(f"cell_{i}", node_type="cell", cell_type="AEC")
    for i in range(n_aec, n_aec + n_other):
        G.add_node(f"cell_{i}", node_type="cell", cell_type="OTHER")

    eid = 0
    cluster_nodes: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for cl in (0, 1, 2):
        for _ in range(30):
            name = f"ecm_{eid}"
            eid += 1
            G.add_node(name, node_type="ecm", ecm_cluster=cl)
            cluster_nodes[cl].append(name)
    bg = f"ecm_{eid}"
    G.add_node(bg, node_type="ecm", ecm_cluster=-1)

    for i in range(n_aec):
        for t in rng.choice(cluster_nodes[AEC_CLUSTER], size=edges_per_cell, replace=False):
            G.add_edge(f"cell_{i}", t)
    # One background contact from the first AEC cell — must not be counted.
    G.add_edge("cell_0", bg)

    other_pool = cluster_nodes[0] + cluster_nodes[2]
    for i in range(n_aec, n_aec + n_other):
        for t in rng.choice(other_pool, size=edges_per_cell, replace=False):
            G.add_edge(f"cell_{i}", t)

    a = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
    a.uns["cell_ecm_graph"] = G
    return a


def test_detects_planted_enrichment():
    """AEC is enriched for ECM 1 and depleted from ECM 0 and 2."""
    a = _make_roi(0)
    df = mt.tl.cell_ecm_enrichment(a, cell_type="AEC", K_ecm=K_ECM, n_perm=500)
    by_cluster = df.set_index("cluster")
    assert by_cluster.loc[AEC_CLUSTER, "log2_enr"] > 0, "ECM 1 should be enriched"
    assert bool(by_cluster.loc[AEC_CLUSTER, "significant"]), "ECM 1 should be FDR-significant"
    assert by_cluster.loc[0, "log2_enr"] < 0, "ECM 0 should be depleted"
    assert by_cluster.loc[2, "log2_enr"] < 0, "ECM 2 should be depleted"


def test_background_patches_excluded():
    """The background contact (ecm_cluster == -1) is not counted."""
    a = _make_roi(0)
    df = mt.tl.cell_ecm_enrichment(a, cell_type="AEC", K_ecm=K_ECM, n_perm=100)
    # 15 AEC cells x 4 signal edges each = 60 target edges; the single
    # background edge must be excluded.
    assert int(df["n_obs"].sum()) == 60, "background-class edges leaked into the count"


def test_pooled_across_rois_and_per_roi_written():
    """Cohort call pools edges and writes a per-ROI breakdown to uns."""
    rois = {"roiA": _make_roi(0), "roiB": _make_roi(1)}
    df = mt.tl.cell_ecm_enrichment(rois, cell_type="AEC", K_ecm=K_ECM, n_perm=200)
    assert list(df["cluster"]) == [0, 1, 2]
    assert int(df["n_obs"].sum()) == 120, "two ROIs of 60 target edges should pool to 120"
    per_roi = rois["roiA"].uns["cell_ecm_enrichment_per_roi"]
    assert set(per_roi["roi"]) == {"roiA", "roiB"}, "per-ROI frame should cover both ROIs"
    assert len(per_roi) == 2 * K_ECM, "one row per (ROI, cluster)"


def test_pvalue_floor_is_one_over_nperm_plus_one():
    """A strong planted effect hits the permutation floor 1/(n_perm+1)."""
    a = _make_roi(0)
    n_perm = 200
    df = mt.tl.cell_ecm_enrichment(a, cell_type="AEC", K_ecm=K_ECM, n_perm=n_perm)
    floor = 1.0 / (n_perm + 1)
    assert df.loc[df["cluster"] == AEC_CLUSTER, "p_value"].item() == pytest.approx(floor)


def test_reproducible_with_fixed_seed():
    """Same random_state -> identical p-values (deterministic permutation)."""
    a1, a2 = _make_roi(0), _make_roi(0)
    df1 = mt.tl.cell_ecm_enrichment(a1, cell_type="AEC", K_ecm=K_ECM, n_perm=200, random_state=7)
    df2 = mt.tl.cell_ecm_enrichment(a2, cell_type="AEC", K_ecm=K_ECM, n_perm=200, random_state=7)
    np.testing.assert_array_equal(df1["p_value"].to_numpy(), df2["p_value"].to_numpy())


def test_zero_permutations_returns_descriptive_effects_only():
    """n_perm=0 keeps counts/effects and makes the absence of inference explicit."""
    a = _make_roi(0)
    df = mt.tl.cell_ecm_enrichment(a, cell_type="AEC", K_ecm=K_ECM, n_perm=0)
    assert df.loc[df["cluster"] == AEC_CLUSTER, "log2_enr"].item() > 0
    assert df["p_value"].isna().all()
    assert df["p_fdr"].isna().all()
    assert not df["significant"].any()
    params = a.uns["mantpy"]["tl"]["cell_ecm_enrichment"]
    assert params["null"] == "not_run_descriptive"


def test_negative_permutations_raise():
    a = _make_roi(0)
    with pytest.raises(ValueError, match="non-negative"):
        mt.tl.cell_ecm_enrichment(a, cell_type="AEC", K_ecm=K_ECM, n_perm=-1)


def test_missing_cell_type_raises():
    """A cell type with no incident edges is a clear error, not silent NaNs."""
    a = _make_roi(0)
    with pytest.raises(ValueError, match="incident to cell_type"):
        mt.tl.cell_ecm_enrichment(a, cell_type="GHOST", K_ecm=K_ECM, n_perm=50)


def test_missing_graph_raises():
    """An AnnData without the unified graph raises a helpful error."""
    a = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="cell_ecm_graph"):
        mt.tl.cell_ecm_enrichment(a, cell_type="AEC", K_ecm=K_ECM, n_perm=50)
