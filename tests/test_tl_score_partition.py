"""Tests for mt.tl.score_partition."""

from __future__ import annotations

import numpy as np
import pytest

import mantpy as mt


def test_perfect_recovery_is_permutation_invariant():
    truth = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([7, 7, 3, 3, 9, 9])  # same partition, arbitrary cluster ids
    r = mt.tl.score_partition(pred, truth)
    assert r["macro_f1"] == 1.0
    assert r["ari"] == 1.0
    assert r["classes"] == [0, 1, 2]


def test_over_clustering_not_penalised():
    truth = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1, 2, 2])  # reference class 0 split into two clusters
    assert mt.tl.score_partition(pred, truth)["macro_f1"] == 1.0


def test_per_class_f1_length_matches_n_classes():
    truth = np.array([0, 1, 2, 3, 4, 0])
    pred = np.array([0, 1, 2, 3, 4, 0])
    r = mt.tl.score_partition(pred, truth)
    assert len(r["per_class_f1"]) == 5


def test_string_reference_labels():
    truth = np.array(["a", "a", "b", "b"])
    pred = np.array([1, 1, 0, 0])
    r = mt.tl.score_partition(pred, truth)
    assert r["macro_f1"] == 1.0
    assert r["classes"] == ["a", "b"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        mt.tl.score_partition(np.array([0, 1]), np.array([0, 1, 2]))


def test_collapsing_to_one_cluster_drops_macro_f1():
    truth = np.array([0, 0, 1, 1, 2, 2])
    pred = np.zeros(6, dtype=int)  # everything in one cluster — misses the minority classes
    assert mt.tl.score_partition(pred, truth)["macro_f1"] < 0.5
