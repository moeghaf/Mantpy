"""Tests for the unified mantpy.gr.build_graph dispatcher."""

from __future__ import annotations

import pytest
from networkx.utils import graphs_equal

import mantpy as mt
from mantpy._constants import (
    CELL_ECM_GRAPH_KEY,
    CELL_GRAPH_KEY,
    ECM_GRAPH_KEY,
)


class TestBuildGraphDispatch:
    def test_cell_mode_writes_obsp(self, adata_with_patches):
        mt.gr.build_graph(adata_with_patches, mode="cell", k=3)
        assert CELL_GRAPH_KEY in adata_with_patches.obsp

    def test_ecm_mode_writes_uns(self, adata_with_patches):
        mt.gr.build_graph(adata_with_patches, mode="ecm", k=3)
        assert ECM_GRAPH_KEY in adata_with_patches.uns

    def test_cell_ecm_mode_merges_layers(self, adata_with_patches):
        # cell_ecm is a thin wrapper over build_cell_ecm_graph, which merges the
        # already-built cell + ECM layers; build the prerequisites first.
        mt.gr.build_graph(adata_with_patches, mode="cell", k=3)
        mt.gr.build_graph(adata_with_patches, mode="ecm", k=3)
        mt.gr.build_graph(adata_with_patches, mode="cell_ecm", k=3)
        assert CELL_ECM_GRAPH_KEY in adata_with_patches.uns

    def test_invalid_mode_raises(self, adata_with_patches):
        with pytest.raises(ValueError, match="mode must be one of"):
            mt.gr.build_graph(adata_with_patches, mode="bipartite")

    def test_dispatch_equivalent_to_typed_builder(self, adata_with_patches):
        # build_graph(mode='cell') must be byte-identical to calling the typed
        # builder directly with the same kwargs.
        a1 = adata_with_patches.copy()
        a2 = adata_with_patches.copy()
        mt.gr.build_graph(a1, mode="cell", k=3)
        mt.gr.build_cell_graph(a2, k=3)
        diff = a1.obsp[CELL_GRAPH_KEY] != a2.obsp[CELL_GRAPH_KEY]
        assert diff.nnz == 0


class TestBuildGraphEdgeMethod:
    def test_string_edge_method_passthrough(self, adata_with_patches):
        mt.gr.build_graph(adata_with_patches, mode="cell", edge_method="delaunay")
        assert CELL_GRAPH_KEY in adata_with_patches.obsp

    def test_mapping_selects_mode_entry(self, adata_with_patches):
        # A per-interaction mapping applies only its entry for ``mode``.
        a1 = adata_with_patches.copy()
        a2 = adata_with_patches.copy()
        mt.gr.build_graph(a1, mode="ecm", edge_method={"cell": "knn", "ecm": "delaunay"})
        mt.gr.build_ecm_graph(a2, edge_method="delaunay")
        assert graphs_equal(a1.uns[ECM_GRAPH_KEY], a2.uns[ECM_GRAPH_KEY])

    def test_mapping_without_mode_key_uses_builder_default(self, adata_with_patches):
        # No entry for ``mode`` → fall through to the builder's own default rule.
        a1 = adata_with_patches.copy()
        a2 = adata_with_patches.copy()
        mt.gr.build_graph(a1, mode="ecm", edge_method={"cell": "delaunay"}, k=3)
        mt.gr.build_ecm_graph(a2, k=3)
        assert graphs_equal(a1.uns[ECM_GRAPH_KEY], a2.uns[ECM_GRAPH_KEY])


class TestBuildGraphInplace:
    def test_inplace_false_returns_copy(self, adata_with_patches):
        out = mt.gr.build_graph(adata_with_patches, mode="cell", k=3, inplace=False)
        assert out is not None
        assert CELL_GRAPH_KEY in out.obsp
        assert CELL_GRAPH_KEY not in adata_with_patches.obsp
