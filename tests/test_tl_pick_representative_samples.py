"""Tests for the cohort-selection helpers in mantpy.tl."""

from __future__ import annotations

import pandas as pd
import pytest

import mantpy as mt


def _toy_meta():
    return pd.DataFrame(
        {"group": ["A", "A", "A", "B", "B"]},
        index=pd.Index(["s1", "s2", "s3", "s4", "s5"], name="sample_id"),
    )


def test_median_selector_picks_middle_by_score():
    meta = _toy_meta()
    # Group A scored 10, 50, 30 → sorted [s1=10, s3=30, s2=50] → median = s3
    scores = {"s1": 10, "s2": 50, "s3": 30, "s4": 100, "s5": 5}
    picks = mt.tl.pick_representative_samples(meta, scores)
    assert picks == {"A": "s3", "B": "s4"}


def test_max_and_min_selectors():
    meta = _toy_meta()
    scores = {"s1": 10, "s2": 50, "s3": 30, "s4": 100, "s5": 5}
    assert mt.tl.pick_representative_samples(meta, scores, selector="max")["A"] == "s2"
    assert mt.tl.pick_representative_samples(meta, scores, selector="min")["A"] == "s1"


def test_first_selector_ignores_scores():
    meta = _toy_meta()
    picks = mt.tl.pick_representative_samples(meta, {}, selector="first")
    assert picks == {"A": "s1", "B": "s4"}


def test_selector_per_group_overrides_default():
    meta = _toy_meta()
    scores = {"s1": 10, "s2": 50, "s3": 30, "s4": 100, "s5": 5}
    picks = mt.tl.pick_representative_samples(meta, scores, selector="median", selector_per_group={"A": "first"})
    assert picks["A"] == "s1"
    assert picks["B"] == "s4"  # median of 2-element B = index 1 = s4 (higher score)


def test_min_score_filter_drops_low_scorers():
    meta = _toy_meta()
    scores = {"s1": 10, "s2": 50, "s3": 30, "s4": 100, "s5": 5}
    picks = mt.tl.pick_representative_samples(meta, scores, min_score=20)
    assert picks["A"] == "s2"


def test_min_score_falls_back_when_filter_empties_group():
    meta = _toy_meta()
    scores = {"s1": 10, "s2": 50, "s3": 30, "s4": 100, "s5": 5}
    picks = mt.tl.pick_representative_samples(meta, scores, min_score=1000)
    assert picks["A"] == "s3"


def test_overrides_applied_last():
    meta = _toy_meta()
    scores = {"s1": 10, "s2": 50, "s3": 30, "s4": 100, "s5": 5}
    picks = mt.tl.pick_representative_samples(meta, scores, overrides={"A": "s1"})
    assert picks["A"] == "s1"
    assert picks["B"] == "s4"  # B unaffected by override


def test_unknown_selector_raises():
    meta = _toy_meta()
    with pytest.raises(ValueError, match="Unknown selector"):
        mt.tl.pick_representative_samples(meta, {}, selector="not_a_selector")


def test_lesion_size_by_sample_only_kept():
    lesions = {
        "r1": [{"n_nodes": 10, "kept": True}, {"n_nodes": 5, "kept": False}],
        "r2": [{"n_nodes": 8, "kept": True}],
    }
    sizes = mt.tl.lesion_size_by_sample(lesions)
    assert sizes == {"r1": 10, "r2": 8}


def test_lesion_size_by_sample_include_dropped():
    lesions = {"r1": [{"n_nodes": 10, "kept": True}, {"n_nodes": 5, "kept": False}]}
    sizes = mt.tl.lesion_size_by_sample(lesions, only_kept=False)
    assert sizes == {"r1": 15}
