"""Tests for :func:`mantpy.tl.compute_cluster_composition`."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

import mantpy as mt
from mantpy.tl import ClusterCompositionResult


def _make_adata(cluster_labels: list[int]) -> ad.AnnData:
    n = len(cluster_labels)
    a = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    a.uns["ecm_patches"] = pd.DataFrame({"ecm_cluster": cluster_labels})
    return a


def _toy_cohort():
    adatas = {
        "r1": _make_adata([0, 0, 1, 1, 2, -1, -1]),  # 2 c0 / 2 c1 / 1 c2 / drop 2 bg
        "r2": _make_adata([0, 1, 1, 2, 2]),
        "r3": _make_adata([0, 0, 1, 2]),
    }
    sample_meta = pd.DataFrame(
        {"group": ["A", "B", "A"]},
        index=pd.Index(["r1", "r2", "r3"], name="sample_id"),
    )
    return adatas, sample_meta


def test_returns_named_tuple():
    adatas, meta = _toy_cohort()
    comp = mt.tl.compute_cluster_composition(adatas, sample_meta=meta)
    assert isinstance(comp, ClusterCompositionResult)
    assert isinstance(comp, tuple)
    assert isinstance(comp.per_sample, pd.DataFrame)
    assert isinstance(comp.per_group_mean, pd.DataFrame)


def test_per_sample_rows_sum_to_one_after_dropping_background():
    """`drop_background=True` (default) renormalises each row."""
    adatas, meta = _toy_cohort()
    comp = mt.tl.compute_cluster_composition(adatas, sample_meta=meta)
    np.testing.assert_allclose(comp.per_sample.sum(axis=1).values, 1.0, atol=1e-9)


def test_drop_background_false_keeps_negative_columns():
    adatas, meta = _toy_cohort()
    comp = mt.tl.compute_cluster_composition(adatas, sample_meta=meta, drop_background=False)
    assert -1 in comp.per_sample.columns


def test_per_group_mean_matches_manual_groupby():
    adatas, meta = _toy_cohort()
    comp = mt.tl.compute_cluster_composition(adatas, sample_meta=meta)
    a_mean = comp.per_sample.loc[["r1", "r3"]].mean(axis=0)
    np.testing.assert_allclose(comp.per_group_mean.loc["A"].values, a_mean.values)


def test_group_order_preserved():
    adatas, meta = _toy_cohort()
    comp = mt.tl.compute_cluster_composition(adatas, sample_meta=meta, group_order=["B", "A"])
    assert comp.per_group_mean.index.tolist() == ["B", "A"]


def test_default_group_order_is_first_appearance():
    """`group_order=None` uses `sample_meta[group_col]` first-appearance order."""
    adatas, meta = _toy_cohort()
    comp = mt.tl.compute_cluster_composition(adatas, sample_meta=meta)
    # sample_meta groups are: A (r1), B (r2), A (r3) → first-appearance = ['A', 'B']
    assert comp.per_group_mean.index.tolist() == ["A", "B"]


def test_cluster_cols_property():
    adatas, meta = _toy_cohort()
    comp = mt.tl.compute_cluster_composition(adatas, sample_meta=meta)
    assert comp.cluster_cols == [0, 1, 2]


def test_custom_group_col():
    """A `group_col=` other than 'group' is honoured."""
    adatas, meta = _toy_cohort()
    meta = meta.rename(columns={"group": "tissue"})
    comp = mt.tl.compute_cluster_composition(adatas, sample_meta=meta, group_col="tissue")
    assert set(comp.per_group_mean.index) == {"A", "B"}


def test_custom_cluster_col():
    """A `cluster_col=` other than 'ecm_cluster' is honoured."""
    a = ad.AnnData(X=np.zeros((4, 1), dtype=np.float32))
    a.uns["ecm_patches"] = pd.DataFrame({"my_label": [0, 0, 1, 1]})
    adatas = {"r1": a}
    meta = pd.DataFrame({"group": ["A"]}, index=pd.Index(["r1"], name="sample_id"))
    comp = mt.tl.compute_cluster_composition(adatas, sample_meta=meta, cluster_col="my_label")
    assert comp.cluster_cols == [0, 1]
