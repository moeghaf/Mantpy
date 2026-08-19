"""Merging sample metadata must not duplicate columns the table already has.

`sample_meta` is indexed by sample_id and, in every shipped dataset, repeats
sample_id as a column as well. Merging it on `right_index=True` therefore
collided with the table's own `sample_id`, and pandas emitted `sample_id_x`
and `sample_id_y` beside it. Downstream code then failed on the duplicate.
"""

from __future__ import annotations

import pandas as pd
import pytest

import mantpy as mt


@pytest.fixture
def records():
    import networkx as nx

    def rec(n):
        g = nx.path_graph(n)
        return {"subgraph": g, "coords": {i: (float(i), 0.0) for i in g}, "kept": True}

    return {"roi_001": [rec(6)], "roi_002": [rec(5)]}


def _meta(with_sample_id_column: bool):
    data = {"mouse_id": ["m1", "m1"], "condition": ["Infected"] * 2}
    if with_sample_id_column:
        data = {"sample_id": ["roi_001", "roi_002"], **data}
    return pd.DataFrame(data, index=pd.Index(["roi_001", "roi_002"], name="sample_id"))


def test_no_suffixed_duplicate_columns(records):
    df = mt.tl.lesion_topology_stats_df(records, sample_meta=_meta(True))

    assert "sample_id_x" not in df.columns
    assert "sample_id_y" not in df.columns
    assert list(df.columns).count("sample_id") == 1


def test_metadata_columns_still_arrive(records):
    df = mt.tl.lesion_topology_stats_df(records, sample_meta=_meta(True))

    assert {"mouse_id", "condition"} <= set(df.columns)
    assert set(df.loc[df.sample_id == "roi_001", "mouse_id"]) == {"m1"}


def test_works_without_a_redundant_sample_id_column(records):
    df = mt.tl.lesion_topology_stats_df(records, sample_meta=_meta(False))

    assert list(df.columns).count("sample_id") == 1
    assert {"mouse_id", "condition"} <= set(df.columns)


def test_downstream_insert_of_an_existing_column_is_now_the_caller_s_problem(records):
    """Guard the exact downstream failure: a duplicate blocks DataFrame.insert."""
    df = mt.tl.lesion_topology_stats_df(records, sample_meta=_meta(True))

    with pytest.raises(ValueError, match="already exists"):
        df.insert(1, "mouse_id", df["sample_id"])
