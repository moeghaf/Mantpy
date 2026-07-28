from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from mantpy import gr


def _ecm_carrier(offset: float = 0.0) -> ad.AnnData:
    carrier = ad.AnnData(X=np.empty((0, 1), dtype=np.float32))
    carrier.uns["ecm_patches"] = pd.DataFrame(
        {
            "x": np.array([0, 10, 0, 10], dtype=float) + offset,
            "y": [0.0, 0.0, 10.0, 10.0],
            "ecm_cluster": [0, 0, 1, 1],
        }
    )
    return carrier


def test_build_ecm_graphs_returns_counts_and_rebuild_recipe():
    cohort = {"roi_a": _ecm_carrier(), "roi_b": _ecm_carrier(2.0)}

    result = gr.build_ecm_graphs(cohort, edge_method="delaunay")

    assert result.samples == ("roi_a", "roi_b")
    assert result.n_nodes == 8
    assert result.n_edges == sum(a.uns["ecm_graph"].number_of_edges() for a in cohort.values())
    assert result.rebuild_kwargs["edge_method"] == "delaunay"
    assert "ECM graph cohort" in repr(result)
    assert all("ecm_graph" in a.uns for a in cohort.values())


def test_build_ecm_graphs_rejects_an_empty_cohort():
    try:
        gr.build_ecm_graphs({}, edge_method="delaunay")
    except ValueError as error:
        assert "empty" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected an empty-cohort error.")
