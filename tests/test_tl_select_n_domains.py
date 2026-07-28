"""Tests for the label-free domain-count selector: graph_modularity, cluster_coherence, select_n_domains."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import mantpy as mt


def _two_blobs(n=60, sep=20.0, seed=0):
    """Two spatially separated blobs with labels 0/1."""
    rng = np.random.default_rng(seed)
    a = rng.normal([0, 0], 1.0, (n, 2))
    b = rng.normal([sep, 0], 1.0, (n, 2))
    coords = np.vstack([a, b])
    labels = np.r_[np.zeros(n, int), np.ones(n, int)]
    return coords, labels


def _three_bands(per=40, seed=0):
    """Three stacked spatial bands with near-onehot separable features (true K = 3)."""
    rng = np.random.default_rng(seed)
    coords, feats, truth = [], [], []
    for band in range(3):
        xs, ys = np.meshgrid(np.arange(8), np.arange(per // 8))
        coords.append(np.column_stack([xs.ravel(), ys.ravel() + band * 10]))
        oneh = np.eye(3)[band] + rng.normal(0, 0.02, (xs.size, 3))
        feats.append(oneh)
        truth.append(np.full(xs.size, band))
    return np.vstack(coords), np.vstack(feats).astype(np.float32), np.concatenate(truth)


class TestGraphModularity:
    def test_separated_clusters_high(self):
        coords, labels = _two_blobs()
        assert mt.tl.graph_modularity(labels, coords=coords, k=6) > 0.3

    def test_in_unit_interval(self):
        coords, labels = _two_blobs()
        q = mt.tl.graph_modularity(labels, coords=coords, k=6)
        assert 0.0 <= q <= 1.0

    def test_single_cluster_is_zero(self):
        coords, _ = _two_blobs()
        labels = np.zeros(coords.shape[0], int)
        assert mt.tl.graph_modularity(labels, coords=coords, k=6) == pytest.approx(0.0, abs=1e-9)

    def test_random_labels_below_true(self):
        coords, labels = _two_blobs()
        rng = np.random.default_rng(1)
        shuffled = rng.permutation(labels)
        assert mt.tl.graph_modularity(shuffled, coords=coords, k=6) < mt.tl.graph_modularity(labels, coords=coords, k=6)

    def test_precomputed_adjacency_matches_coords(self):
        coords, labels = _two_blobs()
        adj = mt.tl._knn_adjacency(coords, 6)
        q_coords = mt.tl.graph_modularity(labels, coords=coords, k=6)
        q_adj = mt.tl.graph_modularity(labels, adjacency=adj)
        assert q_coords == pytest.approx(q_adj, abs=1e-12)

    def test_empty_graph_is_zero(self):
        labels = np.array([0, 1, 0, 1])
        empty = sp.csr_matrix((4, 4))
        assert mt.tl.graph_modularity(labels, adjacency=empty) == 0.0

    def test_requires_coords_or_adjacency(self):
        with pytest.raises(ValueError, match="coords"):
            mt.tl.graph_modularity(np.array([0, 1]))


class TestClusterCoherence:
    def _two_chains(self):
        # nodes 0-2 chain A, nodes 3-5 chain B, no edge between chains
        rows = [0, 1, 3, 4]
        cols = [1, 2, 4, 5]
        a = sp.coo_matrix((np.ones(8), (rows + cols, cols + rows)), shape=(6, 6)).tocsr()
        return a

    def test_contiguous_clusters_min_one(self):
        adj = self._two_chains()
        labels = np.array([0, 0, 0, 1, 1, 1])  # each cluster == one whole chain
        assert mt.tl.cluster_coherence(labels, adj)["min"] == pytest.approx(1.0)

    def test_fragmented_cluster_lowers_min(self):
        adj = self._two_chains()
        labels = np.zeros(6, int)  # one cluster spanning both disconnected chains
        coh = mt.tl.cluster_coherence(labels, adj)
        assert coh["min"] == pytest.approx(0.5)

    def test_min_le_mean(self):
        adj = self._two_chains()
        labels = np.array([0, 0, 1, 1, 1, 1])
        coh = mt.tl.cluster_coherence(labels, adj)
        assert coh["min"] <= coh["mean"] + 1e-12


class TestSelectNDomains:
    def test_recovers_true_k(self):
        coords, feats, _ = _three_bands()
        sel = mt.tl.select_n_domains(feats, coords, k_range=range(2, 7), random_state=0)
        assert sel["n_domains"] == 3

    def test_table_schema(self):
        coords, feats, _ = _three_bands()
        sel = mt.tl.select_n_domains(feats, coords, k_range=range(2, 6), random_state=0)
        assert isinstance(sel["table"], pd.DataFrame)
        assert set(sel["table"].columns) == {"n_domains", "modularity", "coherence_min", "score"}
        assert list(sel["table"]["n_domains"]) == [2, 3, 4, 5]

    def test_labels_length_and_selected_in_range(self):
        coords, feats, _ = _three_bands()
        sel = mt.tl.select_n_domains(feats, coords, k_range=range(2, 7), random_state=0)
        assert sel["labels"].shape[0] == coords.shape[0]
        assert sel["n_domains"] in range(2, 7)

    def test_custom_adjacency_for_coherence(self):
        coords, feats, _ = _three_bands()
        adj = mt.tl._knn_adjacency(coords, 8)
        sel = mt.tl.select_n_domains(feats, coords, adjacency=adj, k_range=range(2, 6), random_state=0)
        assert sel["n_domains"] in range(2, 6)

    def test_empty_k_range_raises(self):
        coords, feats, _ = _three_bands()
        with pytest.raises(ValueError, match="k_range"):
            mt.tl.select_n_domains(feats, coords, k_range=[])
