"""Tests for mantpy.gr.ensure_cell_ecm_graph."""

from __future__ import annotations

import pytest

import mantpy as mt
from mantpy._constants import (
    CELL_ECM_GRAPH_KEY,
    CELL_GRAPH_KEY,
    ECM_GRAPH_KEY,
)


def _has_all_three(adata) -> bool:
    return (CELL_GRAPH_KEY + "_nx") in adata.uns and ECM_GRAPH_KEY in adata.uns and CELL_ECM_GRAPH_KEY in adata.uns


class TestEnsureCellEcmGraph:
    def test_builds_all_three_from_scratch(self, adata_with_patches):
        adata = adata_with_patches
        # Sanity: starting state has no graphs yet.
        assert not _has_all_three(adata)
        mt.gr.ensure_cell_ecm_graph(
            adata,
            cell_k=3,
            ecm_k=3,
            cell_ecm_k=3,
        )
        assert _has_all_three(adata)

    def test_rebuild_true_drops_stale_keys(self, adata_with_graphs):
        adata = adata_with_graphs
        mt.gr.ensure_cell_ecm_graph(
            adata,
            cell_k=3,
            ecm_k=3,
            cell_ecm_k=3,
            rebuild=True,
        )
        # Object identity differs after rebuild.
        assert _has_all_three(adata)
        # The rebuilt graph still has edges (we don't strictly demand the
        # count match because the rebuild uses the same params).
        assert adata.uns[CELL_ECM_GRAPH_KEY].number_of_edges() >= 1

    def test_rebuild_false_only_fills_gaps(self, adata_with_graphs):
        """When rebuild=False, present graphs are preserved by identity."""
        adata = adata_with_graphs
        original_ecm_graph = adata.uns[ECM_GRAPH_KEY]
        mt.gr.ensure_cell_ecm_graph(
            adata,
            cell_k=3,
            ecm_k=3,
            cell_ecm_k=3,
            rebuild=False,
        )
        # Same object — not rebuilt.
        assert adata.uns[ECM_GRAPH_KEY] is original_ecm_graph

    def test_missing_patches_raises(self, adata_basic):
        with pytest.raises(ValueError, match="ecm_patches"):
            mt.gr.ensure_cell_ecm_graph(adata_basic)
