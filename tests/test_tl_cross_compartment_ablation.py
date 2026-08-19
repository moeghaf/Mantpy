"""Tests for ``mt.tl.cross_compartment_ablation``.

The function sweeps the *compartment* axis (cell / ECM / joint neighbour
context + label-frequency prior) through the shared leave-one-ROI-out harness
and returns a tidy ``[model, fold, roc_auc, pr_auc]`` frame. Two layers of
test:

1. **Assembly** — with ``loo_denoise_evaluation`` stubbed, assert the wrapper
   runs one fold-CV per requested context (+ prior), tags each block with the
   ``model`` name, threads ``group_map`` / ``groupby`` straight through, and
   returns the tidy long shape.
2. **Integration** — on a synthetic cell-ECM grid with a planted, spatially
   coherent artefact, the *real* baselines run through the real harness and the
   cell-context model separates the artefacts (ROC-AUC above chance).
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import mantpy as mt
import mantpy.tl as tl_mod
from mantpy._constants import EDGE_TYPE_CE, EDGE_TYPE_EE, NODE_TYPE_CELL, NODE_TYPE_ECM

ROI_NAMES = [f"roi_{i}" for i in range(1, 5)]
MOUSE = {f"roi_{i}": ("m1" if i <= 2 else "m2") for i in range(1, 5)}


# --------------------------------------------------------------------------- #
# Layer 1: assembly with loo_denoise_evaluation stubbed.
# --------------------------------------------------------------------------- #
@pytest.fixture
def stub_loo(monkeypatch):
    """Replace loo_denoise_evaluation with a recorder returning canned AUCs."""
    calls: list[dict[str, Any]] = []

    def fake_loo(
        *,
        cells_by_sample,
        ecm_by_sample,
        artefact_pool,
        roi_names,
        model_factory,
        model_kwargs,
        fit_kwargs,
        graph_kwargs,
        denoise_kwargs,
        groupby=None,
        group_map=None,
        verbose=False,
    ):
        calls.append(
            {
                "model_kwargs": model_kwargs,
                "factory": model_factory,
                "groupby": groupby,
                "group_map": group_map,
                "graph_kwargs": graph_kwargs,
            }
        )
        rois = list(roi_names) if roi_names is not None else sorted(cells_by_sample)
        summary = pd.DataFrame(
            {
                "fold": rois,
                "roc_auc": np.linspace(0.8, 0.95, len(rois)),
                "pr_auc": np.linspace(0.7, 0.9, len(rois)),
            }
        )
        if group_map is not None or groupby is not None:
            gm = group_map or {r: MOUSE[r] for r in rois}
            summary["group"] = [gm[r] for r in rois]
        return pd.DataFrame(), summary, {}

    monkeypatch.setattr(tl_mod, "loo_denoise_evaluation", fake_loo)
    return calls


@pytest.fixture
def dummy_cohort():
    cells = {}
    for r in ROI_NAMES:
        a = ad.AnnData(np.zeros((2, 1), dtype=np.float32))
        a.obs["cell_type"] = ["AEC", "OTHER"]
        a.obs["Mouse"] = MOUSE[r]
        cells[r] = a
    ecm = {r: ad.AnnData(np.zeros((1, 1), dtype=np.float32)) for r in ROI_NAMES}
    art = {r: ad.AnnData(np.zeros((1, 1), dtype=np.float32)) for r in ROI_NAMES}
    return cells, ecm, art


def test_tidy_long_shape_and_model_names(stub_loo, dummy_cohort):
    cells, ecm, art = dummy_cohort
    out = mt.tl.cross_compartment_ablation(
        cells,
        ecm,
        artefact_pool=art,
        roi_names=ROI_NAMES,
    )
    # 3 contexts + prior, 4 folds each = 16 rows.
    assert list(out.columns)[:2] == ["model", "fold"]
    assert set(out["model"]) == {"cell", "ecm", "joint", "prior"}
    assert len(out) == 4 * len(ROI_NAMES)
    assert {"roc_auc", "pr_auc"}.issubset(out.columns)
    # One CV run per model.
    assert len(stub_loo) == 4
    ctx_seen = [c["model_kwargs"].get("context") for c in stub_loo[:3]]
    assert ctx_seen == ["cell", "ecm", "joint"]
    assert stub_loo[3]["model_kwargs"] == {}  # prior has no context kwarg


def test_contexts_subset_and_no_prior(stub_loo, dummy_cohort):
    cells, ecm, art = dummy_cohort
    out = mt.tl.cross_compartment_ablation(
        cells,
        ecm,
        artefact_pool=art,
        roi_names=ROI_NAMES,
        contexts=("cell",),
        include_prior=False,
    )
    assert set(out["model"]) == {"cell"}
    assert len(stub_loo) == 1


def test_group_map_threaded_through_and_group_column(stub_loo, dummy_cohort):
    cells, ecm, art = dummy_cohort
    out = mt.tl.cross_compartment_ablation(
        cells,
        ecm,
        artefact_pool=art,
        roi_names=ROI_NAMES,
        group_map=MOUSE,
    )
    # group_map reached every inner CV call...
    assert all(c["group_map"] == MOUSE for c in stub_loo)
    # ...and the group column survives into the tidy frame, after fold.
    assert "group" in out.columns
    assert list(out.columns[:3]) == ["model", "fold", "group"]
    assert set(out["group"]) == {"m1", "m2"}


def test_groupby_threaded_through(stub_loo, dummy_cohort):
    cells, ecm, art = dummy_cohort
    mt.tl.cross_compartment_ablation(
        cells,
        ecm,
        artefact_pool=art,
        roi_names=ROI_NAMES,
        groupby="Mouse",
    )
    assert all(c["groupby"] == "Mouse" for c in stub_loo)


def test_graph_kwargs_threaded_through(stub_loo, dummy_cohort):
    cells, ecm, art = dummy_cohort
    gk = {"cell_k": 5, "ecm_edge_method": "grid"}
    mt.tl.cross_compartment_ablation(
        cells,
        ecm,
        artefact_pool=art,
        roi_names=ROI_NAMES,
        graph_kwargs=gk,
    )
    assert all(c["graph_kwargs"] == gk for c in stub_loo)


def test_missing_artefact_pool_raises(dummy_cohort):
    cells, ecm, _art = dummy_cohort
    with pytest.raises(ValueError, match="artefact_pool"):
        mt.tl.cross_compartment_ablation(cells, ecm, artefact_pool=None)


def test_invalid_context_raises(dummy_cohort):
    cells, ecm, art = dummy_cohort
    with pytest.raises(ValueError, match="contexts"):
        mt.tl.cross_compartment_ablation(
            cells,
            ecm,
            artefact_pool=art,
            contexts=("bogus",),
        )


# --------------------------------------------------------------------------- #
# Layer 2: real baselines through the real harness on a synthetic grid.
# --------------------------------------------------------------------------- #
_R = _C = 7
_FLIP_TO = 2
_FLIP = [(0, 0), (1, 1), (2, 0), (0, 5), (1, 6), (5, 1), (6, 5)]


def _true_cluster(r: int, c: int) -> int:
    if c <= 2:
        return 0
    if c >= 4:
        return 1
    return 2


def _build_pair(corrupt: bool) -> tuple[ad.AnnData, ad.AnnData]:
    def pid(r: int, c: int) -> int:
        return r * _C + c

    G = nx.Graph()
    for r in range(_R):
        for c in range(_C):
            G.add_node(f"ecm_{pid(r, c)}", node_type=NODE_TYPE_ECM, ecm_cluster=_true_cluster(r, c))
    for r in range(_R):
        for c in range(_C):
            if c + 1 < _C:
                G.add_edge(f"ecm_{pid(r, c)}", f"ecm_{pid(r, c + 1)}", edge_type=EDGE_TYPE_EE)
            if r + 1 < _R:
                G.add_edge(f"ecm_{pid(r, c)}", f"ecm_{pid(r + 1, c)}", edge_type=EDGE_TYPE_EE)
    cell_types: list[str] = []
    cid = 0
    for r in range(_R):
        for c in range(_C):
            tc = _true_cluster(r, c)
            for _ in range(2):
                G.add_node(f"cell_{cid}", node_type=NODE_TYPE_CELL, cell_type=f"T{tc}")
                G.add_edge(f"cell_{cid}", f"ecm_{pid(r, c)}", edge_type=EDGE_TYPE_CE)
                cell_types.append(f"T{tc}")
                cid += 1
    labels = np.array([_true_cluster(r, c) for r in range(_R) for c in range(_C)])
    is_art = np.zeros(labels.size, dtype=bool)
    if corrupt:
        for r, c in _FLIP:
            labels[pid(r, c)] = _FLIP_TO
            is_art[pid(r, c)] = True
    ecm = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
    ecm.uns["ecm_patches"] = pd.DataFrame({"ecm_cluster": labels, "is_artifact": is_art})
    cells = ad.AnnData(np.zeros((len(cell_types), 1), dtype=np.float32))
    cells.obs["cell_type"] = cell_types
    cells.uns["cell_ecm_graph"] = G
    return cells, ecm


def test_integration_real_baselines(monkeypatch):
    """End-to-end: real baselines through the real harness; cell beats chance.

    The synthetic grids already carry a hand-built cell-ECM graph, so we stub
    the graph rebuild to keep it (avoids needing spatial coords / patch x,y).
    """
    monkeypatch.setattr("mantpy.gr.ensure_cell_ecm_graph", lambda *a, **k: None)

    rois = ["a", "b", "c"]
    cells_by_sample, ecm_by_sample, artefact_pool = {}, {}, {}
    for r in rois:
        cells, ecm = _build_pair(corrupt=False)
        cells_art, ecm_art = _build_pair(corrupt=True)
        cells_by_sample[r] = cells
        ecm_by_sample[r] = ecm
        artefact_pool[r] = ecm_art

    out = mt.tl.cross_compartment_ablation(
        cells_by_sample,
        ecm_by_sample,
        artefact_pool=artefact_pool,
        roi_names=rois,
        contexts=("cell", "ecm"),
        include_prior=True,
    )
    assert set(out["model"]) == {"cell", "ecm", "prior"}
    assert len(out) == 3 * len(rois)
    cell_mean = out.loc[out.model == "cell", "roc_auc"].mean()
    assert cell_mean > 0.7, f"cell-context mean ROC-AUC = {cell_mean:.3f}"
