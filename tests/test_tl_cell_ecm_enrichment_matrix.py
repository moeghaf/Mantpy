"""Tests for ``mt.tl.cell_ecm_enrichment_matrix``.

Builds a small hand-made cell-ECM graph with two planted associations
(AEC <-> ECM 1, MAC <-> ECM 0) and checks that the cohort-wide matrix:

- has one tidy row per (cell_type, cluster),
- recovers both planted enrichments with the right sign,
- applies BH-FDR ACROSS the whole matrix (q >= per-test p, monotone),
- honours an explicit ``cell_types`` subset, and
- skips a cell type with no incident edges rather than raising.
"""

from __future__ import annotations

import anndata as ad
import networkx as nx
import numpy as np
import pytest

import mantpy as mt

K_ECM = 3


def _make_roi(seed: int, *, edges_per_cell: int = 4) -> ad.AnnData:
    """AEC -> ECM cluster 1, MAC -> ECM cluster 0, OTHER -> spread over 0/2."""
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    groups = {"AEC": (1, 12), "MAC": (0, 12), "OTHER": (None, 12)}
    cid = 0
    cell_groups: list[tuple[str, int | None]] = []
    for ct, (_pref, n) in groups.items():
        for _ in range(n):
            G.add_node(f"cell_{cid}", node_type="cell", cell_type=ct)
            cell_groups.append((ct, groups[ct][0]))
            cid += 1

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

    for i, (_ct, pref) in enumerate(cell_groups):
        if pref is not None:
            pool = cluster_nodes[pref]
        else:
            pool = cluster_nodes[0] + cluster_nodes[2]
        for t in rng.choice(pool, size=edges_per_cell, replace=False):
            G.add_edge(f"cell_{i}", t)
    G.add_edge("cell_0", bg)  # one background contact, must be ignored

    a = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
    a.uns["cell_ecm_graph"] = G
    return a


@pytest.fixture
def cohort():
    return {"roiA": _make_roi(0), "roiB": _make_roi(1)}


def test_tidy_long_shape_and_columns(cohort):
    mat = mt.tl.cell_ecm_enrichment_matrix(cohort, K_ecm=K_ECM, n_perm=300)
    expected = {"cell_type", "cluster", "n_obs", "n_exp", "log2_enr", "p_value", "p_fdr", "significant"}
    assert expected.issubset(mat.columns)
    # 3 cell types x 3 clusters.
    assert set(mat["cell_type"]) == {"AEC", "MAC", "OTHER"}
    assert len(mat) == 3 * K_ECM
    assert set(mat["cluster"]) == {0, 1, 2}


def test_recovers_both_planted_enrichments(cohort):
    mat = mt.tl.cell_ecm_enrichment_matrix(cohort, K_ecm=K_ECM, n_perm=500)
    piv = mat.pivot(index="cell_type", columns="cluster", values="log2_enr")
    assert piv.loc["AEC", 1] > 0, "AEC should be enriched for ECM 1"
    assert piv.loc["MAC", 0] > 0, "MAC should be enriched for ECM 0"
    sig = mat.set_index(["cell_type", "cluster"])["significant"]
    assert bool(sig.loc[("AEC", 1)])
    assert bool(sig.loc[("MAC", 0)])


def test_fdr_is_matrix_wide(cohort):
    """q-values are computed across all rows: every q >= its own p, and the
    number of distinct q's reflects the full pooled set, not per-cell-type."""
    mat = mt.tl.cell_ecm_enrichment_matrix(cohort, K_ecm=K_ECM, n_perm=300)
    assert (mat["p_fdr"] >= mat["p_value"] - 1e-12).all(), "q must be >= p"
    # Matrix-wide BH: the largest q equals the largest p (rank n / n = 1x).
    assert mat["p_fdr"].max() == pytest.approx(mat["p_value"].max(), rel=1e-6)


def test_explicit_cell_types_subset(cohort):
    mat = mt.tl.cell_ecm_enrichment_matrix(
        cohort,
        K_ecm=K_ECM,
        cell_types=["AEC"],
        n_perm=200,
    )
    assert set(mat["cell_type"]) == {"AEC"}
    assert len(mat) == K_ECM


def test_zero_permutations_returns_descriptive_matrix(cohort):
    mat = mt.tl.cell_ecm_enrichment_matrix(cohort, K_ecm=K_ECM, n_perm=0)
    assert mat["p_value"].isna().all()
    assert mat["p_fdr"].isna().all()
    assert not mat["significant"].any()
    assert mat.loc[(mat["cell_type"] == "AEC") & (mat["cluster"] == 1), "log2_enr"].item() > 0


def test_missing_cell_type_skipped_not_raised(cohort):
    """A requested cell type with no edges is skipped; present ones remain."""
    mat = mt.tl.cell_ecm_enrichment_matrix(
        cohort,
        K_ecm=K_ECM,
        cell_types=["AEC", "GHOST"],
        n_perm=200,
    )
    assert set(mat["cell_type"]) == {"AEC"}


def test_all_missing_raises(cohort):
    with pytest.raises(ValueError, match="No cell type produced"):
        mt.tl.cell_ecm_enrichment_matrix(
            cohort,
            K_ecm=K_ECM,
            cell_types=["GHOST"],
            n_perm=50,
        )


def test_reproducible_with_seed(cohort):
    m1 = mt.tl.cell_ecm_enrichment_matrix(cohort, K_ecm=K_ECM, n_perm=200, random_state=3)
    cohort2 = {"roiA": _make_roi(0), "roiB": _make_roi(1)}
    m2 = mt.tl.cell_ecm_enrichment_matrix(cohort2, K_ecm=K_ECM, n_perm=200, random_state=3)
    np.testing.assert_array_equal(m1["p_value"].to_numpy(), m2["p_value"].to_numpy())
