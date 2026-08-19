"""Tests for mantpy.tl."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mantpy as mt
from mantpy._constants import (
    INTERACTION_TEST_KEY,
    MANTPY_UNS_KEY,
    NEIGHBOURHOOD_CLUSTERS_KEY,
)


class TestInteractionTest:
    def test_returns_dataframe(self, adata_with_graphs):
        result = mt.tl.interaction_test(adata_with_graphs, n_iter=10)
        assert isinstance(result, pd.DataFrame)

    def test_stores_in_uns(self, adata_with_graphs):
        mt.tl.interaction_test(adata_with_graphs, n_iter=10)
        assert INTERACTION_TEST_KEY in adata_with_graphs.uns

    def test_sigval_values(self, adata_with_graphs):
        mt.tl.interaction_test(adata_with_graphs, n_iter=20)
        sigval = adata_with_graphs.uns[INTERACTION_TEST_KEY]
        assert set(sigval.values.flat).issubset({-1, 0, 1})

    def test_index_are_cell_types(self, adata_with_graphs):
        mt.tl.interaction_test(adata_with_graphs, n_iter=10)
        sigval = adata_with_graphs.uns[INTERACTION_TEST_KEY]
        expected = {"T", "B", "Mac", "DC"}
        assert set(sigval.index).issubset(expected)

    def test_always_returns_dataframe(self, adata_with_graphs):
        result = mt.tl.interaction_test(adata_with_graphs, n_iter=10)
        assert isinstance(result, pd.DataFrame)

    def test_missing_graph_raises(self, adata_with_patches):
        with pytest.raises(ValueError, match="cell_ecm_graph"):
            mt.tl.interaction_test(adata_with_patches, n_iter=5)

    def test_params_logged(self, adata_with_graphs):
        mt.tl.interaction_test(adata_with_graphs, n_iter=10)
        assert "interaction_test" in adata_with_graphs.uns[MANTPY_UNS_KEY]["tl"]


class TestNeighbourhoodClustering:
    def test_writes_obs_column(self, adata_with_graphs):
        mt.tl.neighbourhood_clustering(adata_with_graphs, n_clusters=2)
        assert NEIGHBOURHOOD_CLUSTERS_KEY in adata_with_graphs.obs.columns

    def test_label_count(self, adata_with_graphs):
        mt.tl.neighbourhood_clustering(adata_with_graphs, n_clusters=2)
        n_unique = adata_with_graphs.obs[NEIGHBOURHOOD_CLUSTERS_KEY].nunique()
        assert n_unique <= 2

    def test_centroids_stored(self, adata_with_graphs):
        mt.tl.neighbourhood_clustering(adata_with_graphs, n_clusters=2)
        assert NEIGHBOURHOOD_CLUSTERS_KEY + "_centroids" in adata_with_graphs.uns

    def test_inplace_false(self, adata_with_graphs):
        result = mt.tl.neighbourhood_clustering(adata_with_graphs, n_clusters=2, inplace=False)
        assert result is not None
        assert NEIGHBOURHOOD_CLUSTERS_KEY not in adata_with_graphs.obs.columns

    def test_without_ecm(self, adata_with_graphs):
        mt.tl.neighbourhood_clustering(adata_with_graphs, n_clusters=2, include_ecm=False)
        assert NEIGHBOURHOOD_CLUSTERS_KEY in adata_with_graphs.obs.columns

    def test_missing_graph_raises(self, adata_with_patches):
        with pytest.raises(ValueError, match="cell_ecm_graph"):
            mt.tl.neighbourhood_clustering(adata_with_patches)

    def test_params_logged(self, adata_with_graphs):
        mt.tl.neighbourhood_clustering(adata_with_graphs, n_clusters=2)
        assert "neighbourhood_clustering" in adata_with_graphs.uns[MANTPY_UNS_KEY]["tl"]

    def test_ecm_only_graph_raises_informative(self, adata_with_patches):
        """neighbourhood_clustering on an ECM-only graph raises with a clear message."""
        mt.gr.build_ecm_graph(adata_with_patches, k=3)
        mt.gr.build_cell_ecm_graph(adata_with_patches)
        with pytest.raises(ValueError, match="No cell nodes"):
            mt.tl.neighbourhood_clustering(adata_with_patches, n_clusters=2)


class TestEcmToAnndata:
    def test_basic_shape(self, adata_with_patches):
        """Returns AnnData with rows = patches, columns = features."""
        ecm_adata = mt.tl.ecm_to_anndata(adata_with_patches)
        n_patches = len(adata_with_patches.uns["ecm_patches"])
        feat_cols = [c for c in adata_with_patches.uns["ecm_patches"].columns if c.startswith("feat_")]
        assert ecm_adata.n_obs == n_patches
        assert ecm_adata.n_vars == len(feat_cols)
        assert ecm_adata.X.dtype == np.float32

    def test_obsm_spatial_and_obs(self, adata_with_patches):
        """obsm['spatial'] holds centroids, obs preserves ecm_cluster as categorical."""
        ecm_adata = mt.tl.ecm_to_anndata(adata_with_patches)
        assert "spatial" in ecm_adata.obsm
        assert ecm_adata.obsm["spatial"].shape == (ecm_adata.n_obs, 2)
        assert "ecm_cluster" in ecm_adata.obs.columns
        assert ecm_adata.obs["ecm_cluster"].dtype.name == "category"
        # Coordinates should match the source DataFrame.
        src = adata_with_patches.uns["ecm_patches"][["x", "y"]].to_numpy()
        np.testing.assert_allclose(ecm_adata.obsm["spatial"], src)

    def test_provenance_and_var_names(self, adata_with_patches):
        """Provenance is stored; var_names are descriptive when feature meta exists."""
        ecm_adata = mt.tl.ecm_to_anndata(adata_with_patches)
        prov = ecm_adata.uns["mantpy_provenance"]
        assert prov["patches_key"] == "ecm_patches"
        assert prov["n_patches"] == ecm_adata.n_obs
        assert prov["source_sample"] == adata_with_patches.uns.get("sample_id")
        # ecm_feature_names from extract_ecm_patches gives extractor + protein,
        # so var_names should be like "mean__ColIV", not "0".
        assert all("__" in vn or vn.isdigit() is False for vn in ecm_adata.var_names)

    def test_missing_patches_key_raises(self, adata_basic):
        """Calling without ecm_patches in uns raises with a clear message."""
        with pytest.raises(ValueError, match="ecm_patches"):
            mt.tl.ecm_to_anndata(adata_basic)


class TestTopEnrichedCluster:
    def _toy_enr(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cluster": [0, 1, 2, 3],
                "log2_enr": [-3.5, 2.3, -1.4, 0.7],
                "significant": [True, True, True, False],
            }
        )

    def test_positive_significant(self):
        assert mt.tl.top_enriched_cluster(self._toy_enr()) == 1

    def test_negative_significant(self):
        assert mt.tl.top_enriched_cluster(self._toy_enr(), direction="negative") == 0

    def test_require_significant_false_picks_non_sig_row(self):
        df = pd.DataFrame(
            {
                "cluster": [0, 1],
                "log2_enr": [0.5, 4.0],
                "significant": [True, False],
            }
        )
        assert mt.tl.top_enriched_cluster(df, require_significant=False) == 1
        # With the default filter the non-sig row is excluded.
        assert mt.tl.top_enriched_cluster(df, require_significant=True) == 0

    def test_no_positive_rows_raises(self):
        df = pd.DataFrame({"cluster": [0, 1], "log2_enr": [-1.0, -2.0], "significant": [True, True]})
        with pytest.raises(ValueError, match="positive log2"):
            mt.tl.top_enriched_cluster(df)

    def test_no_negative_rows_raises(self):
        df = pd.DataFrame({"cluster": [0, 1], "log2_enr": [1.0, 2.0], "significant": [True, True]})
        with pytest.raises(ValueError, match="negative log2"):
            mt.tl.top_enriched_cluster(df, direction="negative")

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            mt.tl.top_enriched_cluster(self._toy_enr(), direction="sideways")

    def test_missing_log2_col_raises(self):
        df = pd.DataFrame({"cluster": [0], "significant": [True]})
        with pytest.raises(KeyError, match="log2_enr"):
            mt.tl.top_enriched_cluster(df)

    def test_missing_sig_col_raises_when_required(self):
        df = pd.DataFrame({"cluster": [0, 1], "log2_enr": [1.0, -1.0]})
        with pytest.raises(KeyError, match="significant"):
            mt.tl.top_enriched_cluster(df)

    def test_custom_column_names(self):
        df = pd.DataFrame(
            {
                "k": [0, 1, 2],
                "lr": [-1.0, 3.0, 0.5],
                "sig": [True, True, False],
            }
        )
        assert mt.tl.top_enriched_cluster(df, cluster_col="k", log2_col="lr", sig_col="sig") == 1


class _StubScorer:
    """Minimal duck-typed scorer used to test ``mt.tl.denoise_ecm_clusters``."""

    K_ecm = 3

    def __init__(self, anomaly, denoised, p_pred):
        self.anomaly = np.asarray(anomaly, dtype=np.float32)
        self.denoised = np.asarray(denoised, dtype=int)
        self.p_pred = np.asarray(p_pred, dtype=np.float32)

    def score(self, cell_adata, ecm_adata):
        return pd.DataFrame(
            {
                "anomaly_score": self.anomaly,
                "denoised_cluster": self.denoised,
                "p_actual": 1.0 - self.anomaly,
                "p_pred": self.p_pred,
            }
        )


def _toy_ecm_pair(n_patches: int = 6, K_signal: int = 3, seed: int = 0):
    """Build a tiny (cell_adata, ecm_adata) pair for tl-layer tests."""
    rng = np.random.default_rng(seed)
    from anndata import AnnData

    cell_adata = AnnData(
        X=rng.normal(size=(4, 1)).astype(np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
    )
    cell_adata.obs["cell_type"] = ["A", "B", "A", "B"]

    xy = rng.uniform(0, 100, size=(n_patches, 2)).astype(np.float32)
    cluster = np.array([0, 1, 2, 0, 1, 2])[:n_patches]
    patches = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "ecm_cluster": cluster,
            "feat_marker0": rng.normal(size=n_patches).astype(np.float32),
        }
    )
    ecm_adata = AnnData(
        X=patches[["feat_marker0"]].to_numpy(dtype=np.float32),
        obs=pd.DataFrame(index=[str(i) for i in range(n_patches)]),
    )
    ecm_adata.uns["ecm_patches"] = patches
    return cell_adata, ecm_adata


class TestDenoiseEcmClusters:
    def test_writes_anomaly_and_denoised(self):
        cell_adata, ecm_adata = _toy_ecm_pair()
        # Six patches at (rng-drawn) positions ~50-100 px apart. Disable the
        # spatial-coherence filter to exercise the anomaly-threshold logic
        # in isolation.
        denoiser = _StubScorer(
            anomaly=[0.9, 0.1, 0.8, 0.2, 0.5, 0.4],
            denoised=[1, 1, 0, 0, 1, 2],
            p_pred=[0.95, 0.20, 0.10, 0.99, 0.30, 0.99],
        )
        mt.tl.denoise_ecm_clusters(
            cell_adata,
            ecm_adata,
            model=denoiser,
            anomaly_threshold=0.30,
            spatial_min_neighbours=0,
        )
        patches = ecm_adata.uns["ecm_patches"]
        assert "anomaly_score" in patches.columns
        assert "denoised_cluster" in patches.columns
        denoised = np.asarray(patches["denoised_cluster"]).astype(int)
        actual = np.asarray(patches["ecm_cluster"]).astype(int)
        # Patch 0: anomaly 0.9 >= 0.30, original 0 -> predicted 1 (signal). Flip.
        assert denoised[0] == 1
        # Patch 1: anomaly 0.1 < 0.30 -> retain original (1).
        assert denoised[1] == actual[1]
        # Patch 3: anomaly 0.2 < 0.30 -> retain original (0).
        assert denoised[3] == actual[3]

    def test_spatial_coherence_filter_suppresses_isolated_candidates(self):
        cell_adata, ecm_adata = _toy_ecm_pair()
        # All six patches would otherwise be candidates, but the toy points
        # are far apart relative to spatial_radius=2.0, so each candidate is
        # isolated -> no relabels accepted.
        denoiser = _StubScorer(
            anomaly=[0.9] * 6,
            denoised=[1, 0, 1, 2, 0, 1],
            p_pred=[0.9] * 6,
        )
        mt.tl.denoise_ecm_clusters(
            cell_adata,
            ecm_adata,
            model=denoiser,
            anomaly_threshold=0.30,
            spatial_radius=2.0,
            spatial_min_neighbours=1,
        )
        patches = ecm_adata.uns["ecm_patches"]
        np.testing.assert_array_equal(
            np.asarray(patches["denoised_cluster"]).astype(int),
            np.asarray(patches["ecm_cluster"]).astype(int),
        )

    def test_bad_threshold_raises(self):
        cell_adata, ecm_adata = _toy_ecm_pair()
        with pytest.raises(ValueError, match="anomaly_threshold"):
            mt.tl.denoise_ecm_clusters(
                cell_adata,
                ecm_adata,
                model=_StubScorer([0.5] * 6, [0] * 6, [0.5] * 6),
                anomaly_threshold=1.5,
            )

    def test_non_model_object_raises(self):
        cell_adata, ecm_adata = _toy_ecm_pair()
        with pytest.raises(TypeError, match="must be a fitted scorer"):
            mt.tl.denoise_ecm_clusters(cell_adata, ecm_adata, model="not a model")

    def test_copy_returns_new_anndata(self):
        cell_adata, ecm_adata = _toy_ecm_pair()
        denoiser = _StubScorer([0.5] * 6, [0] * 6, [0.9] * 6)
        out = mt.tl.denoise_ecm_clusters(
            cell_adata,
            ecm_adata,
            model=denoiser,
            copy=True,
            spatial_min_neighbours=0,
        )
        assert out is not None
        assert out is not ecm_adata
        assert "anomaly_score" in out.uns["ecm_patches"].columns
        assert "anomaly_score" not in ecm_adata.uns["ecm_patches"].columns
