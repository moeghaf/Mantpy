"""Tests for the ECM-cluster denoising scorers.

Each baseline must satisfy the duck-typed contract consumed by
:func:`mantpy.tl.loo_denoise_evaluation` (``fit`` / ``score_anomaly`` / a
``device`` attribute) and must detect a planted artefact on a small synthetic
cell-ECM grid. The internal prior protects the optional label-frequency
reference, while the public scorer protects neighbour-composition behaviour.
"""

from __future__ import annotations

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import pytest

import mantpy as mt
from mantpy._constants import EDGE_TYPE_CE, EDGE_TYPE_EE, NODE_TYPE_CELL, NODE_TYPE_ECM
from mantpy.nn._baselines import PriorFrequencyBaseline

_R = _C = 7
_FLIP_TO = 2  # the rare cluster artefacts are flipped to
_FLIP_PATCHES = [(0, 0), (1, 1), (2, 0), (0, 5), (1, 6), (5, 1), (6, 5)]

BASELINES = [
    PriorFrequencyBaseline,
    mt.nn.NeighbourCompositionBaseline,
]


def _true_cluster(r: int, c: int) -> int:
    if c <= 2:
        return 0
    if c >= 4:
        return 1
    return 2  # middle column — the rare cluster


def _build_pair(corrupt: bool) -> tuple[ad.AnnData, ad.AnnData]:
    """Build a (cell, ecm) pair sharing one fixed graph; vary only the labels.

    When ``corrupt`` is True, a scattered set of cluster-0/1 patches are flipped
    to the rare cluster 2 and flagged in ``is_artifact``.
    """

    def pid(r: int, c: int) -> int:
        return r * _C + c

    G = nx.Graph()
    for r in range(_R):
        for c in range(_C):
            G.add_node(
                f"ecm_{pid(r, c)}",
                node_type=NODE_TYPE_ECM,
                ecm_cluster=_true_cluster(r, c),
            )
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
        for r, c in _FLIP_PATCHES:
            labels[pid(r, c)] = _FLIP_TO
            is_art[pid(r, c)] = True

    ecm = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
    ecm.uns["ecm_patches"] = pd.DataFrame({"ecm_cluster": labels, "is_artifact": is_art})
    cells = ad.AnnData(np.zeros((len(cell_types), 1), dtype=np.float32))
    cells.obs["cell_type"] = cell_types
    cells.uns["cell_ecm_graph"] = G
    return cells, ecm


@pytest.fixture
def cohort():
    pristine = [_build_pair(corrupt=False)]
    held_cells, held_ecm = _build_pair(corrupt=True)
    return pristine, held_cells, held_ecm


@pytest.mark.parametrize("factory", BASELINES, ids=lambda f: f.__name__)
def test_baseline_detects_planted_artefact(factory, cohort):
    """Each baseline separates the planted artefacts (ROC-AUC well above chance)."""
    pristine, held_cells, held_ecm = cohort
    model = factory()
    model.fit(pristine, cluster_key="ecm_cluster", celltype_key="cell_type")
    out = model.score_anomaly(held_cells, held_ecm, truth_col="is_artifact")
    assert out["roc_auc"] > 0.7, f"{factory.__name__} ROC-AUC = {out['roc_auc']:.3f}"


@pytest.mark.parametrize("factory", BASELINES, ids=lambda f: f.__name__)
def test_score_anomaly_returns_full_metric_dict(factory, cohort):
    """The returned dict satisfies the shared anomaly-scoring contract."""
    pristine, held_cells, held_ecm = cohort
    model = factory()
    assert model.device == "cpu"
    model.fit(pristine, cluster_key="ecm_cluster", celltype_key="cell_type")
    out = model.score_anomaly(held_cells, held_ecm, truth_col="is_artifact")
    for key in ("roc_auc", "pr_auc", "fpr", "tpr", "precision", "recall", "scores"):
        assert key in out, f"{factory.__name__} missing '{key}'"
    assert "anomaly_score" in out["scores"].columns


@pytest.mark.parametrize("factory", BASELINES, ids=lambda f: f.__name__)
def test_fit_tolerates_loo_forwarded_kwargs(factory, cohort):
    """loo_denoise_evaluation forwards epochs/lr/extra_feat_cols — must not error."""
    pristine, _held_cells, _held_ecm = cohort
    model = factory()
    returned = model.fit(
        pristine,
        cluster_key="ecm_cluster",
        celltype_key="cell_type",
        epochs=10,
        lr=5e-4,
        extra_feat_cols=[],
    )
    assert returned is model, "fit should return self for chaining"


@pytest.mark.parametrize("factory", BASELINES, ids=lambda f: f.__name__)
def test_missing_truth_col_raises(factory, cohort):
    pristine, held_cells, held_ecm = cohort
    model = factory().fit(pristine, cluster_key="ecm_cluster", celltype_key="cell_type")
    with pytest.raises(KeyError, match="missing"):
        model.score_anomaly(held_cells, held_ecm, truth_col="not_a_column")


def test_prior_baseline_untrained_raises(cohort):
    _pristine, held_cells, held_ecm = cohort
    with pytest.raises(RuntimeError, match="not trained"):
        PriorFrequencyBaseline().score_anomaly(held_cells, held_ecm)


def test_neighbour_composition_untrained_raises(cohort):
    _pristine, held_cells, held_ecm = cohort
    with pytest.raises(RuntimeError, match="not trained"):
        mt.nn.NeighbourCompositionBaseline().score_anomaly(held_cells, held_ecm)


@pytest.mark.parametrize("context", ["cell", "ecm", "joint"])
def test_neighbour_composition_contexts_detect(context, cohort):
    """All three LR variants (cell / ecm / joint) detect the planted artefact."""
    pristine, held_cells, held_ecm = cohort
    model = mt.nn.NeighbourCompositionBaseline(context=context)
    model.fit(pristine, cluster_key="ecm_cluster", celltype_key="cell_type")
    out = model.score_anomaly(held_cells, held_ecm, truth_col="is_artifact")
    assert out["roc_auc"] > 0.7, f"context={context} ROC-AUC = {out['roc_auc']:.3f}"


def test_invalid_context_raises():
    with pytest.raises(ValueError, match="context"):
        mt.nn.NeighbourCompositionBaseline(context="bogus")


def test_neighbour_composition_defaults_to_cell_context():
    assert mt.nn.NeighbourCompositionBaseline().context == "cell"


def test_cell_context_beats_prior(cohort):
    """Cell context should outperform the label-frequency reference here."""
    pristine, held_cells, held_ecm = cohort
    prior = PriorFrequencyBaseline().fit(pristine, cluster_key="ecm_cluster", celltype_key="cell_type")
    neighbour = mt.nn.NeighbourCompositionBaseline().fit(
        pristine,
        cluster_key="ecm_cluster",
        celltype_key="cell_type",
    )
    prior_auc = prior.score_anomaly(held_cells, held_ecm, truth_col="is_artifact")["roc_auc"]
    neighbour_auc = neighbour.score_anomaly(held_cells, held_ecm, truth_col="is_artifact")["roc_auc"]
    assert neighbour_auc >= prior_auc
