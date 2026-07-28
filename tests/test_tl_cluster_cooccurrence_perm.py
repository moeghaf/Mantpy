"""Tests for :func:`mantpy.tl.cluster_cooccurrence_permutation_test` result NamedTuple.

Critical contract: the new ``CooccurrencePermResult`` NamedTuple must unpack
positionally exactly like the legacy 3-tuple, so existing call sites that
do ``delta, p_uncorr, p_fdr = cluster_cooccurrence_permutation_test(...)``
continue to work bit-identically.
"""

from __future__ import annotations

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd

import mantpy as mt
from mantpy.tl import CooccurrencePermResult


def _make_adata_with_ecm_graph(name: str, cluster_labels: list[int]) -> ad.AnnData:
    n = len(cluster_labels)
    G = nx.Graph()
    rng = np.random.default_rng(hash(name) % (2**32))
    coords = rng.normal(0, 10, size=(n, 2))
    for i in range(n):
        G.add_node(i, x=float(coords[i, 0]), y=float(coords[i, 1]), ecm_cluster=int(cluster_labels[i]))
    # Random edges between adjacent indices
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    if n > 3:
        G.add_edge(0, n - 1)
    adata = ad.AnnData(
        X=np.zeros((n, 1), dtype=np.float32),
        obs=pd.DataFrame(index=[str(i) for i in range(n)]),
    )
    adata.uns["ecm_graph"] = G
    adata.uns["sample_id"] = name
    return adata


def _make_cohort():
    adatas = {}
    for i in range(4):
        labels = [(i + j) % 3 for j in range(20)]
        adatas[f"ko_{i}"] = _make_adata_with_ecm_graph(f"ko_{i}", labels)
    for i in range(4):
        labels = [(i + j + 1) % 3 for j in range(20)]
        adatas[f"wt_{i}"] = _make_adata_with_ecm_graph(f"wt_{i}", labels)
    return adatas


def test_result_is_named_tuple():
    adatas = _make_cohort()
    result = mt.tl.cluster_cooccurrence_permutation_test(
        adatas,
        group_a_rois=[f"ko_{i}" for i in range(4)],
        group_b_rois=[f"wt_{i}" for i in range(4)],
        n_clusters=3,
        n_perm=50,
        seed=42,
    )
    assert isinstance(result, CooccurrencePermResult)
    assert isinstance(result, tuple)


def test_legacy_tuple_unpack_compatible():
    """The legacy ``delta, p_uncorr, p_fdr = ...`` unpack must keep working."""
    adatas = _make_cohort()
    delta, p_uncorr, p_fdr = mt.tl.cluster_cooccurrence_permutation_test(
        adatas,
        group_a_rois=[f"ko_{i}" for i in range(4)],
        group_b_rois=[f"wt_{i}" for i in range(4)],
        n_clusters=3,
        n_perm=50,
        seed=42,
    )
    assert delta.shape == (3, 3)
    assert p_uncorr.shape == (3, 3)
    assert p_fdr.shape == (3, 3)


def test_attribute_access_matches_tuple_position():
    """Field-by-field equality between named attribute access and positional unpack."""
    adatas = _make_cohort()
    r = mt.tl.cluster_cooccurrence_permutation_test(
        adatas,
        group_a_rois=[f"ko_{i}" for i in range(4)],
        group_b_rois=[f"wt_{i}" for i in range(4)],
        n_clusters=3,
        n_perm=50,
        seed=42,
    )
    delta, p_uncorr, p_fdr = r
    np.testing.assert_array_equal(delta, r.delta_obs)
    np.testing.assert_array_equal(p_uncorr, r.p_uncorr)
    np.testing.assert_array_equal(p_fdr, r.p_fdr)


def test_p_fdr_symmetric_and_in_unit_interval():
    adatas = _make_cohort()
    r = mt.tl.cluster_cooccurrence_permutation_test(
        adatas,
        group_a_rois=[f"ko_{i}" for i in range(4)],
        group_b_rois=[f"wt_{i}" for i in range(4)],
        n_clusters=3,
        n_perm=50,
        seed=42,
    )
    np.testing.assert_allclose(r.p_fdr, r.p_fdr.T)
    assert np.all((r.p_fdr >= 0.0) & (r.p_fdr <= 1.0))


def test_single_roi_per_group_does_not_crash():
    """Realistic small-cohort edge case: 1 ROI per group exercises the
    ``max(node_count.sum(), 1.0)`` divide-by-zero guard."""
    adatas = _make_cohort()
    r = mt.tl.cluster_cooccurrence_permutation_test(
        adatas,
        group_a_rois=["ko_0"],
        group_b_rois=["wt_0"],
        n_clusters=3,
        n_perm=20,
        seed=42,
    )
    assert r.delta_obs.shape == (3, 3)
    assert r.p_uncorr.shape == (3, 3)
    assert r.p_fdr.shape == (3, 3)
    assert np.all(np.isfinite(r.delta_obs))
    assert np.all((r.p_uncorr >= 0.0) & (r.p_uncorr <= 1.0))
