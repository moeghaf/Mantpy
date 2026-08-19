from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

import mantpy as mt
import mantpy.nn._baselines as nn_baselines

GRAPH_KWARGS = {
    "cell_k": 1,
    "cell_Dmax": 5,
    "ecm_edge_method": "grid",
    "ecm_grid_connectivity": 8,
    "cell_ecm_k": 1,
    "cell_ecm_Dmax": 5,
}


def _cohorts() -> tuple[dict[str, ad.AnnData], dict[str, ad.AnnData], dict[str, ad.AnnData]]:
    cells_by_sample: dict[str, ad.AnnData] = {}
    ecm_by_sample: dict[str, ad.AnnData] = {}
    artefact_pool: dict[str, ad.AnnData] = {}
    for sample, shift in (("r1", 0.0), ("r2", 0.2)):
        cells = ad.AnnData(
            X=np.empty((4, 0), dtype=np.float32),
            obs=pd.DataFrame(
                {"cell_type": ["A", "A", "B", "B"]},
                index=[f"{sample}_c{i}" for i in range(4)],
            ),
        )
        cells.obsm["spatial"] = np.array(
            [[0, 0], [2, 0], [0, 2], [2, 2]],
            dtype=np.float32,
        )
        pristine = pd.DataFrame(
            {
                "x": [0.5, 1.5, 0.5, 1.5],
                "y": [0.5, 0.5, 1.5, 1.5],
                "ecm_cluster": [0, 0, 1, 1],
                "feat_0": [0.0 + shift, 0.1 + shift, 1.0 + shift, 1.1 + shift],
            }
        )
        ecm = ad.AnnData(X=np.empty((0, 0), dtype=np.float32))
        ecm.uns["ecm_patches"] = pristine.copy()
        corrupt = ecm.copy()
        corrupted = pristine.copy()
        corrupted["is_artifact"] = [True, False, True, False]
        corrupted.loc[corrupted["is_artifact"], "ecm_cluster"] = [1, 0]
        corrupt.uns["ecm_patches"] = corrupted
        corrupt.uns["artifact_boxes"] = [{"center": (1.0, 1.0), "half": 0.5}]
        cells.uns["ecm_patches"] = pristine.copy()
        cells_by_sample[sample] = cells
        ecm_by_sample[sample] = ecm
        artefact_pool[sample] = corrupt
    mt.gr.build_cell_ecm_graphs(
        cells_by_sample,
        cell_graph="mantpy",
        cell_k=1,
        cell_Dmax=5,
        ecm_edge_method="grid",
        cell_ecm_k=1,
        cell_ecm_Dmax=5,
    )
    return cells_by_sample, ecm_by_sample, artefact_pool


class _FakeModel:
    def __init__(self, context: str = "cell", **_: object) -> None:
        self.context = context
        self.K_ecm = 2

    def fit(self, *_: object, **__: object) -> _FakeModel:
        return self

    def score(self, _: ad.AnnData, ecm: ad.AnnData) -> pd.DataFrame:
        patches = ecm.uns["ecm_patches"]
        actual = patches["ecm_cluster"].to_numpy(dtype=int)
        truth = patches.get("is_artifact", pd.Series(False, index=patches.index)).to_numpy(dtype=bool)
        predicted = actual.copy()
        predicted[truth] = 1 - predicted[truth]
        anomaly = np.where(truth, 0.9, 0.1)
        return pd.DataFrame(
            {
                "anomaly_score": anomaly,
                "denoised_cluster": predicted,
                "p_actual": 1 - anomaly,
                "p_pred": anomaly,
            }
        )

    def score_anomaly(
        self,
        cells: ad.AnnData,
        ecm: ad.AnnData,
        *,
        truth_col: str,
        signal_only: bool,
    ) -> dict[str, object]:
        from sklearn.metrics import (
            average_precision_score,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )

        patches = ecm.uns["ecm_patches"]
        mask = patches["ecm_cluster"].to_numpy(dtype=int) >= 0 if signal_only else np.ones(len(patches), bool)
        y_true = patches[truth_col].to_numpy(dtype=bool)[mask]
        y_score = np.where(y_true, 0.9, 0.1)
        if np.unique(y_true).size < 2:
            fpr = tpr = np.array([0.0, 1.0])
            precision = recall = np.array([1.0, 0.0])
            roc_auc = pr_auc = float("nan")
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            roc_auc = float(roc_auc_score(y_true, y_score))
            pr_auc = float(average_precision_score(y_true, y_score))
        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "y_true": y_true,
            "y_score": y_score,
            "scores": self.score(cells, ecm).loc[mask].reset_index(drop=True),
        }


def test_denoise_held_out_roi_runs_the_staged_workflow() -> None:
    cells, ecm, artefact = _cohorts()
    result = mt.tl.denoise_held_out_roi(
        cells,
        ecm,
        held_out="r1",
        target_pool=artefact,
        model_factory=_FakeModel,
        graph_kwargs=GRAPH_KWARGS,
        denoise_kwargs={"spatial_min_neighbours": 0},
    )
    assert result.n_relabelled == 2
    assert "denoised_cluster" in result.ecm.uns["ecm_patches"]
    assert "cell_ecm_graph" in result.cells.uns
    assert "relabelled    2" in repr(result)


def test_loo_reconstruction_evaluation_scores_completed_labels() -> None:
    cells, ecm, artefact = _cohorts()
    result = mt.tl.loo_reconstruction_evaluation(
        cells,
        ecm,
        artefact_pool=artefact,
        model_factory=_FakeModel,
        graph_kwargs=GRAPH_KWARGS,
        denoise_kwargs={"spatial_min_neighbours": 0},
    )
    assert list(result["roi"]) == ["r1", "r2"]
    assert np.allclose(result["corrupted_fraction"], 0.5)
    assert np.allclose(result["before_accuracy"], 0.5)
    assert np.allclose(result["after_accuracy"], 1.0)
    assert np.allclose(result["artifact_recovered"], 1.0)
    assert np.allclose(result["correction_precision"], 1.0)
    assert np.allclose(result["pristine_relabel_rate"], 0.0)

    summary = mt.tl.summarize_reconstruction_evaluation(result)
    assert summary.n_samples == 2
    assert summary.before_accuracy == 0.5
    assert summary.after_accuracy == 1.0
    assert "50.0% -> 100.0%" in repr(summary)


def test_ecm_neighbour_label_agreement_scores_selected_signal_neighbours() -> None:
    cells, _, artefact = _cohorts()
    summary = mt.tl.ecm_neighbour_label_agreement(cells, artefact)
    assert summary.n_samples == 2
    assert summary.n_patches == 4
    assert summary.n_scored == 4
    assert np.isclose(summary.mean, 1 / 3)
    assert np.allclose(summary.per_sample["mean_agreement"], 1 / 3)
    assert "ECM-neighbour label agreement" in repr(summary)


def test_topology_sensitivity_returns_one_column_per_topology() -> None:
    cells, _, _ = _cohorts()
    result = mt.tl.cell_ecm_topology_sensitivity(
        cells,
        cell_type="A",
        K_ecm=2,
        topologies=("knn", "delaunay"),
        knn_k=1,
        knn_Dmax=5,
        n_perm=10,
        random_state=0,
    )
    assert list(result.columns) == ["knn", "delaunay"]
    assert list(result.index) == [0, 1]


def test_pristine_flag_rate_and_cross_compartment_ablation(monkeypatch) -> None:
    cells, ecm, artefact = _cohorts()
    pristine = mt.tl.loo_pristine_flag_rate(
        cells,
        ecm,
        artefact_pool=artefact,
        model_factory=_FakeModel,
        graph_kwargs=GRAPH_KWARGS,
    )
    assert len(pristine) == 2
    assert np.allclose(pristine["pristine_flagged"], 0.0)
    pristine_summary = mt.tl.summarize_pristine_flag_rate(pristine)
    assert pristine_summary.mean == 0.0

    monkeypatch.setattr(mt.nn, "NeighbourCompositionBaseline", _FakeModel)
    monkeypatch.setattr(nn_baselines, "PriorFrequencyBaseline", _FakeModel)
    ablation = mt.tl.cross_compartment_ablation(
        cells,
        ecm,
        artefact_pool=artefact,
        contexts=("cell", "ecm", "joint"),
        include_prior=True,
        graph_kwargs=GRAPH_KWARGS,
    )
    assert set(ablation["model"]) == {"cell", "ecm", "joint", "prior"}
    assert len(ablation) == 8
    roc = mt.tl.ablation_roc_curves(ablation, model="cell")
    assert set(roc.curves["kind"]) == {"roc", "pr"}
    assert len(roc.summary) == 2

    summary = mt.tl.grouped_metric_summary(ablation, groupby="model", metrics=("roc_auc", "pr_auc"))
    assert set(summary["metric"]) == {"roc_auc", "pr_auc"}
    assert set(summary["n"]) == {2}
