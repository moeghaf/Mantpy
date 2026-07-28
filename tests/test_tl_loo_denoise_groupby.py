"""Tests for the leave-one-group-out option of ``mt.tl.loo_denoise_evaluation``.

The default (``groupby=None``, ``group_map=None``) must stay byte-for-byte the
classic leave-one-ROI-out. Supplying a ``group_map`` (roi -> group) turns the
cross-validation into leave-one-group-out: ROIs sharing a group are held out
together and none of them may leak into that fold's training set.

These tests stub out the graph rebuild (``ensure_cell_ecm_graph``) and the
model, because the fold-*partitioning* logic — which ROIs train on which — is
independent of the graphs and the scorer. A recording stub model captures the
exact training ROIs per fold so the no-leakage invariant can be asserted
directly.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantpy as mt
import mantpy.tl as tl_mod

# Six ROIs across two mice (matches the BALB/c lung cohort layout):
#   m1 -> ROIs 1,2,3 ; m2 -> ROIs 4,5,6.
ROI_NAMES = [f"roi_{i}" for i in range(1, 7)]
MOUSE = {f"roi_{i}": ("m1" if i <= 3 else "m2") for i in range(1, 7)}


def _make_cell_adata(name: str) -> ad.AnnData:
    """Minimal cell AnnData carrying obs['Mouse'] and an ecm_patches stub."""
    a = ad.AnnData(np.zeros((4, 1), dtype=np.float32))
    a.obs["cell_type"] = ["AEC", "AEC", "OTHER", "OTHER"]
    a.obs["Mouse"] = MOUSE[name]
    a.uns["ecm_patches"] = pd.DataFrame({"ecm_cluster": [0, 1, 2], "is_artifact": [False, False, False]})
    return a


def _make_ecm_adata() -> ad.AnnData:
    a = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
    a.uns["ecm_patches"] = pd.DataFrame({"ecm_cluster": [0, 1, 2], "is_artifact": [False, False, False]})
    return a


@pytest.fixture
def cohort():
    cells = {r: _make_cell_adata(r) for r in ROI_NAMES}
    ecm = {r: _make_ecm_adata() for r in ROI_NAMES}
    return cells, ecm


class _RecordingModel:
    """Stub denoiser: records which ROIs (by a stamped uns key) it trained on."""

    device = "cpu"
    train_log: list[set[str]] = []

    def __init__(self, **_: Any) -> None:
        pass

    def fit(self, pristine_pairs, **_: Any) -> _RecordingModel:
        seen = {cell.uns["_roi_name"] for cell, _ecm in pristine_pairs}
        _RecordingModel.train_log.append(seen)
        return self


@pytest.fixture(autouse=True)
def _stub_graph_and_reset(monkeypatch):
    """No-op the graph rebuild; the fold logic doesn't touch the graph."""
    monkeypatch.setattr(tl_mod, "ensure_cell_ecm_graph", lambda *a, **k: None, raising=False)
    # ensure_cell_ecm_graph is imported lazily inside the fold helper via
    # `from mantpy.gr import ensure_cell_ecm_graph`; patch the source too.
    monkeypatch.setattr("mantpy.gr.ensure_cell_ecm_graph", lambda *a, **k: None)
    _RecordingModel.train_log = []
    yield


def _stamp_names(cells: dict[str, ad.AnnData]) -> None:
    for name, a in cells.items():
        a.uns["_roi_name"] = name


def test_group_folds_never_train_on_same_group_roi(cohort):
    """Leave-one-MOUSE-out: each fold's training ROIs exclude the held mouse."""
    cells, ecm = cohort
    _stamp_names(cells)
    curves, summary, _scored = mt.tl.loo_denoise_evaluation(
        cells_by_sample=cells,
        ecm_by_sample=ecm,
        artefact_pool=None,
        roi_names=ROI_NAMES,
        model_factory=_RecordingModel,
        group_map=MOUSE,
        verbose=False,
    )
    # Two folds (m1, m2); each held-out mouse trains only on the other mouse.
    assert len(_RecordingModel.train_log) == 2, "expected one fit per mouse"
    for trained in _RecordingModel.train_log:
        mice = {MOUSE[r] for r in trained}
        assert len(mice) == 1, "a single fold trained across both mice (leakage)"
    # Every ROI still scored individually -> 6 summary rows tagged by group.
    assert len(summary) == len(ROI_NAMES)
    assert "group" in summary.columns
    assert set(summary["group"]) == {"m1", "m2"}
    # Each held-out ROI's group must differ from its fold's training mice.
    for _, row in summary.iterrows():
        assert MOUSE[row["fold"]] == row["group"]


def test_groupby_obs_column_matches_explicit_group_map(cohort):
    """groupby='Mouse' reads obs and yields the same folds as group_map."""
    cells, ecm = cohort
    _stamp_names(cells)
    _, summary_obs, _ = mt.tl.loo_denoise_evaluation(
        cells_by_sample=cells,
        ecm_by_sample=ecm,
        artefact_pool=None,
        roi_names=ROI_NAMES,
        model_factory=_RecordingModel,
        groupby="Mouse",
        verbose=False,
    )
    log_obs = [set(s) for s in _RecordingModel.train_log]

    _RecordingModel.train_log = []
    _, summary_map, _ = mt.tl.loo_denoise_evaluation(
        cells_by_sample=cells,
        ecm_by_sample=ecm,
        artefact_pool=None,
        roi_names=ROI_NAMES,
        model_factory=_RecordingModel,
        group_map=MOUSE,
        verbose=False,
    )
    log_map = [set(s) for s in _RecordingModel.train_log]

    assert log_obs == log_map
    pd.testing.assert_frame_equal(summary_obs.reset_index(drop=True), summary_map.reset_index(drop=True))


def test_default_is_leave_one_roi_out_unchanged(cohort):
    """No groupby/group_map -> 6 folds, each trains on the other 5 ROIs, no
    'group' column (the classic path)."""
    cells, ecm = cohort
    _stamp_names(cells)
    _, summary, _ = mt.tl.loo_denoise_evaluation(
        cells_by_sample=cells,
        ecm_by_sample=ecm,
        artefact_pool=None,
        roi_names=ROI_NAMES,
        model_factory=_RecordingModel,
        verbose=False,
    )
    assert len(_RecordingModel.train_log) == len(ROI_NAMES), "one fold per ROI"
    for held, trained in zip(ROI_NAMES, _RecordingModel.train_log, strict=False):
        assert held not in trained, "held-out ROI leaked into its own training"
        assert trained == set(ROI_NAMES) - {held}
    assert "group" not in summary.columns
    assert list(summary["fold"]) == ROI_NAMES


def test_group_map_missing_roi_raises(cohort):
    cells, ecm = cohort
    _stamp_names(cells)
    bad_map = {r: MOUSE[r] for r in ROI_NAMES if r != "roi_6"}
    with pytest.raises(KeyError, match="missing from group_map"):
        mt.tl.loo_denoise_evaluation(
            cells_by_sample=cells,
            ecm_by_sample=ecm,
            artefact_pool=None,
            roi_names=ROI_NAMES,
            model_factory=_RecordingModel,
            group_map=bad_map,
            verbose=False,
        )


def test_groupby_missing_column_raises(cohort):
    cells, ecm = cohort
    _stamp_names(cells)
    with pytest.raises(KeyError, match="not found"):
        mt.tl.loo_denoise_evaluation(
            cells_by_sample=cells,
            ecm_by_sample=ecm,
            artefact_pool=None,
            roi_names=ROI_NAMES,
            model_factory=_RecordingModel,
            groupby="NoSuchColumn",
            verbose=False,
        )


def test_single_group_raises(cohort):
    """One group can't be held out — there'd be no training ROIs."""
    cells, ecm = cohort
    _stamp_names(cells)
    one_group = {r: "only" for r in ROI_NAMES}
    with pytest.raises(ValueError, match="no training ROIs"):
        mt.tl.loo_denoise_evaluation(
            cells_by_sample=cells,
            ecm_by_sample=ecm,
            artefact_pool=None,
            roi_names=ROI_NAMES,
            model_factory=_RecordingModel,
            group_map=one_group,
            verbose=False,
        )
