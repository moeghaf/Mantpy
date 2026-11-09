"""Graph layers, distance kernels, label corruption and reconstruction scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

import mantpy as mt

FIXTURE = Path(__file__).parent / "data" / "lung_roi004_corruption.csv"


def _toy(seed: int = 0) -> AnnData:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 60, size=(40, 2))
    adata = AnnData(X=np.zeros((40, 1), dtype=np.float32))
    adata.obs["cell_type"] = rng.choice(["A", "B", "C"], size=40)
    adata.obsm["spatial"] = xy
    gx, gy = np.meshgrid(np.arange(5, 60, 10), np.arange(5, 60, 10))
    patches = pd.DataFrame({"x": gx.ravel().astype(float), "y": gy.ravel().astype(float)})
    patches["ecm_cluster"] = rng.integers(-1, 3, size=len(patches))
    patches["feat_0"] = rng.normal(size=len(patches))
    patches.index = patches.index.astype(str)
    adata.uns["ecm_patches"] = patches
    mt.gr.ensure_cell_ecm_graph(
        adata, cell_edge_method="radius", cell_r=15,
        ecm_edge_method="grid", ecm_grid_connectivity=8,
        cell_ecm_edge_method="radius", cell_ecm_r=15,
    )
    return adata


def test_graph_layers_match_the_stored_graph():
    adata = _toy()
    graph = adata.uns["cell_ecm_graph"]
    layers = mt.gr.graph_layers(adata)

    n_cells, n_patches = adata.n_obs, len(adata.uns["ecm_patches"])
    assert layers.cell.shape == (n_cells, n_cells)
    assert layers.ecm.shape == (n_patches, n_patches)
    assert layers.cell_ecm.shape == (n_cells, n_patches)
    assert list(layers.cell_ids) == list(adata.obs_names)
    assert list(layers.ecm_ids) == list(adata.uns["ecm_patches"].index)
    assert (layers.cell != layers.cell.T).nnz == 0
    assert (layers.ecm != layers.ecm.T).nnz == 0
    kinds = [attrs["edge_type"] for *_, attrs in graph.edges(data=True)]
    assert layers.cell.nnz == 2 * kinds.count("cell-cell")
    assert layers.ecm.nnz == 2 * kinds.count("ecm-ecm")
    assert layers.cell_ecm.nnz == kinds.count("cell-ecm") > 0
    for u, v, attrs in graph.edges(data=True):
        if attrs["edge_type"] != "cell-ecm":
            continue
        cell, patch = (u, v) if str(u).startswith("cell_") else (v, u)
        i, j = int(str(cell).split("_")[1]), int(str(patch).split("_")[1])
        assert layers.cell_ecm[i, j] == 1.0
        assert layers.cell_ecm_distances[i, j] == attrs["distance"]
    np.testing.assert_array_equal(layers.cell_ecm.indices, layers.cell_ecm_distances.indices)


def test_graph_layers_reject_a_stale_graph():
    adata = _toy()
    adata.uns["ecm_patches"] = adata.uns["ecm_patches"].iloc[:-1]
    with pytest.raises(ValueError, match="rebuild"):
        mt.gr.graph_layers(adata)


def test_distance_weights_kernels_and_stored_zeros():
    d = sp.csr_matrix((np.array([0.0, 2.0, 4.0]), (np.array([0, 0, 1]), np.array([0, 1, 1]))), shape=(2, 2))

    uniform = mt.gr.distance_weights(d)
    assert uniform.nnz == 3
    np.testing.assert_array_equal(uniform.data, [1.0, 1.0, 1.0])
    gaussian = mt.gr.distance_weights(d, kernel="gaussian")
    np.testing.assert_array_equal(gaussian.data, np.exp(-(d.data ** 2) / (2.0 * 2.0 ** 2)))
    np.testing.assert_array_equal(
        mt.gr.distance_weights(np.array([1.0, 2.0]), kernel="inverse_square"), [1.0, 0.25],
    )
    with pytest.raises(ValueError, match="min_distance"):
        mt.gr.distance_weights(d, kernel="inverse")
    clipped = mt.gr.distance_weights(d, kernel="inverse", min_distance=1.0)
    np.testing.assert_array_equal(clipped.data, [1.0, 0.5, 0.25])
    assert d.data[0] == 0.0


def test_context_features_equal_the_array_level_engine():
    adata = _toy()
    layers = mt.gr.graph_layers(adata)
    patches = adata.uns["ecm_patches"]
    weights = mt.gr.distance_weights(layers.cell_ecm_distances, kernel="gaussian")

    table = mt.tl.context_features(
        adata, context="joint", hops=2, normalization="counts", edge_weight="gaussian",
        cell_categories=["A", "B", "C"], ecm_categories=range(3),
    )

    assert list(table.index) == list(patches.index)
    assert table.shape == (len(patches), 15)
    expected = mt.tl.neighbourhood_features(
        weights.T, adata.obs["cell_type"].astype(str).to_numpy(), categories=["A", "B", "C"],
        normalization="counts", exclude_self=False,
    )
    np.testing.assert_array_equal(table.iloc[:, :3].to_numpy(), expected.to_numpy())
    relabelled = mt.tl.context_features(
        adata, context="ecm", ecm_categories=range(3), ecm_labels=np.zeros(len(patches), dtype=int),
    )
    assert relabelled["ecm:1"].sum() == 0
    assert relabelled["ecm:0"].max() == 1.0


def test_corrupt_labels_reproduces_the_published_lung_realisation():
    """Fig. 4h of the Mantpy paper: lung ROI 004, mixed corruption, seed 3."""
    table = pd.read_csv(FIXTURE)

    result = mt.tl.corrupt_labels(
        table["pristine"].to_numpy(), table[["x", "y"]].to_numpy(float),
        categories=range(3), random_state=3,
    )

    np.testing.assert_array_equal(result.labels, table["observed"].to_numpy())
    np.testing.assert_array_equal(result.is_corrupted, table["is_artifact"].to_numpy(bool))
    np.testing.assert_array_equal(result.in_block(), table["in_block"].to_numpy(bool))
    assert result.n_requested == int(result.is_corrupted.sum()) == 150
    assert len(result.blocks) == 1
    block = result.blocks.iloc[0]
    assert (block.side, block.new_label, block.n_replaced) == (150.0, 2, 75)
    np.testing.assert_allclose([block.centre_x, block.centre_y], [155.52157598941494, 100.19116978095953])
    assert int((result.block_id == 0).sum()) == 75


@pytest.mark.parametrize("block_fraction", [0.0, 0.5, 1.0])
def test_corrupt_labels_budget_background_and_overlay(block_fraction):
    table = pd.read_csv(FIXTURE)
    labels = table["pristine"].to_numpy()
    xy = table[["x", "y"]].to_numpy(float)

    result = mt.tl.corrupt_labels(
        labels, xy, categories=range(3), block_fraction=block_fraction, random_state=7,
    )

    assert int(result.is_corrupted.sum()) == round(0.1 * int((labels >= 0).sum()))
    np.testing.assert_array_equal(result.is_corrupted, result.labels != labels)
    assert not result.is_corrupted[labels < 0].any()
    assert (len(result.blocks) == 0) == (block_fraction == 0.0)
    overlay = result.to_overlay()
    assert {"is_artifact", "ecm_cluster_artifact", "ecm_cluster_pristine", "artifact_boxes"} <= set(overlay)
    again = mt.tl.corrupt_labels(
        labels, xy, categories=range(3), block_fraction=block_fraction, random_state=7,
    )
    np.testing.assert_array_equal(result.labels, again.labels)


def test_label_anomaly_scores_match_reconstruct_labels_and_adjust_for_the_prior():
    rng = np.random.default_rng(1)
    proba = rng.dirichlet(np.ones(3), size=30)
    observed = rng.integers(-1, 3, size=30)
    prior = np.array([0.6, 0.3, 0.1])

    for score in ("raw_neglogp", "minmax_neglogp", "one_minus_p", "margin"):
        np.testing.assert_array_equal(
            mt.tl.label_anomaly_scores(observed, proba, categories=range(3), score=score),
            mt.tl.reconstruct_labels(
                observed, proba, categories=range(3), threshold=0.0, score=score,
            ).anomaly_scores,
        )
    raw = mt.tl.label_anomaly_scores(observed, proba, categories=range(3))
    adjusted = mt.tl.label_anomaly_scores(observed, proba, categories=range(3), prior=prior)
    signal = observed >= 0
    np.testing.assert_array_equal(adjusted[signal], raw[signal] + np.log(prior[observed[signal]]))
    assert not adjusted[~signal].any()
    keyed = pd.Series(prior[::-1], index=[2, 1, 0])
    np.testing.assert_array_equal(
        adjusted, mt.tl.label_anomaly_scores(observed, proba, categories=range(3), prior=keyed),
    )
    with pytest.raises(ValueError, match="raw_neglogp"):
        mt.tl.label_anomaly_scores(observed, proba, categories=range(3), prior=prior, score="margin")
    with pytest.raises(ValueError, match="summing to one"):
        mt.tl.label_anomaly_scores(observed, proba, categories=range(3), prior=[1, 1, 1])


def test_score_label_reconstruction_matches_the_published_lung_values():
    """Fig. 4h,i: accuracy 0.900 to 0.951 and 72% of replaced labels corrected."""
    table = pd.read_csv(FIXTURE)

    scored = mt.tl.score_label_reconstruction(
        table["pristine"], table["observed"], table["cleaned"],
        scores=table["is_artifact"].astype(float),
    )

    metrics = scored.metrics
    assert (metrics["n_signal"], metrics["n_artifact"], metrics["n_relabelled"]) == (1503, 150, 143)
    assert metrics["before_accuracy"] == pytest.approx(1353 / 1503)
    assert metrics["after_accuracy"] == pytest.approx(1429 / 1503)
    assert metrics["artifact_recovered"] == pytest.approx(0.72)
    assert metrics["roc_auc"] == 1.0
    assert scored.per_class["n_corrupted"].tolist() == [123, 19, 8]
    assert scored.per_class["n_restored"].tolist() == [90, 17, 1]
    empty = mt.tl.score_label_reconstruction([0, 1], [0, 1], [0, 1])
    assert np.isnan(empty.metrics["artifact_recovered"])
