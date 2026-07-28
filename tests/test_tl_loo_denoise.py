"""Tests for the generic leave-one-out denoising evaluator."""

from __future__ import annotations

import numpy as np
import pytest

import mantpy as mt


@pytest.fixture
def loo_pair(adata_with_graphs):
    """Return a cell/ECM pair with at least two signal clusters."""
    cell_adata = adata_with_graphs
    if "cell_type" not in cell_adata.obs.columns:
        cell_adata.obs["cell_type"] = cell_adata.obs.get("celltype", "T").astype(str)
    patches = cell_adata.uns["ecm_patches"].copy()
    signal_clusters = (
        patches.loc[patches["ecm_cluster"].astype(int) >= 0, "ecm_cluster"].nunique()
        if "ecm_cluster" in patches.columns
        else 0
    )
    if signal_clusters < 2:
        n_patches = len(patches)
        patches["ecm_cluster"] = np.tile([0, 1], n_patches // 2 + 1)[:n_patches]
    cell_adata.uns["ecm_patches"] = patches

    ecm_adata = mt.tl.ecm_to_anndata(cell_adata)
    ecm_adata.uns["ecm_patches"] = patches
    return cell_adata, ecm_adata


def test_pristine_only_runs_with_nan_aucs(loo_pair):
    """Without artefact_pool the loop still runs; AUCs are NaN."""
    cell_adata, ecm_adata = loo_pair
    cells_by_sample = {"roi0": cell_adata, "roi1": cell_adata.copy()}
    ecm_by_sample = {"roi0": ecm_adata, "roi1": ecm_adata.copy()}

    curves, summary, scored = mt.tl.loo_denoise_evaluation(
        cells_by_sample=cells_by_sample,
        ecm_by_sample=ecm_by_sample,
        artefact_pool=None,
        graph_kwargs={"cell_k": 3, "ecm_k": 3, "cell_ecm_k": 3},
        verbose=False,
    )
    assert set(summary["fold"]) == {"roi0", "roi1"}
    assert summary["roc_auc"].isna().all()
    assert curves.empty
    assert set(scored) == {"roi0", "roi1"}


def test_with_artefact_pool_returns_auc_curves(loo_pair):
    cell_adata, ecm_adata = loo_pair
    art_ecm = ecm_adata.copy()
    patches = art_ecm.uns["ecm_patches"].copy()
    signal_indices = np.where(patches["ecm_cluster"].astype(int).to_numpy() >= 0)[0]
    is_artifact = np.zeros(len(patches), dtype=bool)
    is_artifact[signal_indices[: max(1, len(signal_indices) // 2)]] = True
    patches["is_artifact"] = is_artifact
    art_ecm.uns["ecm_patches"] = patches

    cells_by_sample = {"roi0": cell_adata, "roi1": cell_adata.copy()}
    ecm_by_sample = {"roi0": ecm_adata, "roi1": ecm_adata.copy()}
    artefact_pool = {"roi0": art_ecm, "roi1": art_ecm.copy()}

    curves, summary, scored = mt.tl.loo_denoise_evaluation(
        cells_by_sample=cells_by_sample,
        ecm_by_sample=ecm_by_sample,
        artefact_pool=artefact_pool,
        graph_kwargs={"cell_k": 3, "ecm_k": 3, "cell_ecm_k": 3},
        verbose=False,
    )
    assert {"fold", "roc_auc", "pr_auc"}.issubset(summary.columns)
    assert {"fold", "kind", "x", "y"}.issubset(curves.columns)
    assert set(curves["kind"].unique()) == {"roc", "pr"}
    for ecm in scored.values():
        assert "anomaly_score" in ecm.uns["ecm_patches"].columns


def test_missing_roi_raises(loo_pair):
    cell_adata, ecm_adata = loo_pair
    with pytest.raises(KeyError, match="missing from ecm_by_sample"):
        mt.tl.loo_denoise_evaluation(
            cells_by_sample={"roi0": cell_adata, "roi1": cell_adata},
            ecm_by_sample={"roi0": ecm_adata},
            verbose=False,
        )


def test_empty_cohort_raises():
    with pytest.raises(ValueError, match="empty"):
        mt.tl.loo_denoise_evaluation(
            cells_by_sample={},
            ecm_by_sample={},
            verbose=False,
        )


def test_default_factory_uses_cell_context(loo_pair, monkeypatch):
    """The no-factory path explicitly constructs the cell-context baseline."""
    contexts: list[str] = []

    class SpyBaseline:
        device = "cpu"

        def __init__(self, *, context: str):
            contexts.append(context)

        def fit(self, _pristine_pairs, **_kwargs):
            return self

    monkeypatch.setattr(mt.nn, "NeighbourCompositionBaseline", SpyBaseline)
    cell_adata, ecm_adata = loo_pair
    mt.tl.loo_denoise_evaluation(
        cells_by_sample={"roi0": cell_adata, "roi1": cell_adata.copy()},
        ecm_by_sample={"roi0": ecm_adata, "roi1": ecm_adata.copy()},
        graph_kwargs={"cell_k": 3, "ecm_k": 3, "cell_ecm_k": 3},
        verbose=False,
    )

    assert contexts == ["cell", "cell"]
