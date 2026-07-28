"""Tests for mantpy.pl — all tests use a non-interactive backend."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before other imports

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection

import mantpy as mt


def test_ecm_centroid_table_returns_plot_source_data(adata_with_patches) -> None:
    table = mt.pl.ecm_centroid_table(adata_with_patches)
    assert {"cluster", "marker", "feature", "n_patches", "mean_intensity", "z_score"} <= set(table)
    assert len(table) > 0


def test_node_value_overlay_retains_spatial_and_value_encoding() -> None:
    import matplotlib.pyplot as plt

    positions = np.asarray([[1.0, 2.0], [3.0, 5.0], [6.0, 9.0]])
    values = np.asarray([0.0, 0.5, 1.0])
    edge_segments = np.asarray([[[1.0, 2.0], [3.0, 5.0]]])
    figure, axis = plt.subplots()

    output = mt.pl.node_value_overlay(
        axis,
        positions,
        values,
        edges=edge_segments,
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        bbox=(0.0, 7.0, 1.0, 10.0),
        header="Attribution",
    )

    assert output is axis
    assert isinstance(axis.collections[0], LineCollection)
    assert isinstance(axis.collections[-1], PathCollection)
    np.testing.assert_allclose(axis.collections[0].get_segments(), edge_segments)
    np.testing.assert_allclose(axis.collections[-1].get_offsets(), positions)
    np.testing.assert_allclose(axis.collections[-1].get_array(), values)
    np.testing.assert_allclose(axis.collections[-1].get_sizes(), [6.0, 20.0, 34.0])
    assert axis.get_xlim() == (0.0, 7.0)
    assert axis.get_ylim() == (10.0, 1.0)
    assert axis.get_title() == "Attribution"
    assert not any(spine.get_visible() for spine in axis.spines.values())
    assert figure is axis.figure


def test_ecm_graph_overlay_returns_overlay_axis(adata_with_graphs) -> None:
    axes = mt.pl.ecm_graph_overlay(adata_with_graphs, img=np.zeros((64, 64)), show=False)
    assert len(axes) == 1
    assert isinstance(axes[0], Axes)


def test_niche_bubble_table_returns_edge_weighted_source_data() -> None:
    adata = AnnData(X=np.zeros((1, 1), dtype=np.float32))
    adata.uns["ecm_patches"] = pd.DataFrame({"feat_0": [2.0], "feat_1": [1.0], "ecm_cluster": [0]})
    graph = nx.Graph()
    graph.add_node("cell_0", node_type="cell", cell_type="T")
    graph.add_node("ecm_0", node_type="ecm", ecm_cluster=0, ecm_index=0)
    graph.add_edge("cell_0", "ecm_0")
    adata.uns["cell_ecm_graph"] = graph

    table = mt.pl.niche_bubble_table(adata, cell_type="T", focus_cluster=0, marker_names=["A", "B"])

    assert table["marker"].tolist() == ["A", "B"]
    assert table["n_edges"].tolist() == [1, 1]
    np.testing.assert_allclose(table["proportion"].sum(), 1.0)


@pytest.fixture(autouse=True)
def _close_plots():
    """Close all figures after every test to avoid resource leaks."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


class TestCellGraph:
    def test_returns_axes(self, adata_with_graphs):
        ax = mt.pl.cell_graph(adata_with_graphs, show=False)
        assert isinstance(ax, Axes)

    def test_missing_graph_raises(self, adata_basic):
        with pytest.raises(ValueError, match="cell_graph_nx"):
            mt.pl.cell_graph(adata_basic, show=False)


class TestEcmGraph:
    def test_returns_axes(self, adata_with_graphs):
        ax = mt.pl.ecm_graph(adata_with_graphs, show=False)
        assert isinstance(ax, Axes)

    def test_missing_graph_raises(self, adata_basic):
        with pytest.raises(ValueError, match="ecm_graph"):
            mt.pl.ecm_graph(adata_basic, show=False)


class TestCellEcmGraph:
    def test_returns_axes(self, adata_with_graphs):
        ax = mt.pl.cell_ecm_graph(adata_with_graphs, show=False)
        assert isinstance(ax, Axes)

    def test_missing_graph_raises(self, adata_basic):
        with pytest.raises(ValueError, match="cell_ecm_graph"):
            mt.pl.cell_ecm_graph(adata_basic, show=False)


class TestInteractionHeatmap:
    def test_returns_axes(self, adata_with_graphs):
        mt.tl.interaction_test(adata_with_graphs, n_iter=5)
        ax = mt.pl.interaction_heatmap(adata_with_graphs, show=False)
        assert isinstance(ax, Axes)

    def test_missing_key_raises(self, adata_with_graphs):
        with pytest.raises(ValueError, match="interaction_test"):
            mt.pl.interaction_heatmap(adata_with_graphs, show=False)


class TestEcmImage:
    def test_returns_axes(self, adata_with_patches):
        ax = mt.pl.ecm_image(adata_with_patches, show=False)
        assert isinstance(ax, Axes)

    def test_missing_image_raises(self, adata_basic):
        with pytest.raises(ValueError, match="ecm_image"):
            mt.pl.ecm_image(adata_basic, show=False)


class TestNeighbourhoodClusters:
    def test_returns_axes(self, adata_with_graphs):
        mt.tl.neighbourhood_clustering(adata_with_graphs, n_clusters=2)
        ax = mt.pl.neighbourhood_clusters(adata_with_graphs, show=False)
        assert isinstance(ax, Axes)

    def test_missing_column_raises(self, adata_basic):
        with pytest.raises(ValueError, match="neighbourhood_clusters"):
            mt.pl.neighbourhood_clusters(adata_basic, show=False)


class TestSmokeTest:
    """Full end-to-end smoke test matching the plan verification example."""

    def test_full_pipeline(self, synthetic_img, synthetic_panel, synthetic_cells):
        import mantpy as mt

        adata = mt.read_imc(synthetic_img, panel=synthetic_panel, cells=synthetic_cells)
        mt.pp.extract_ecm_patches(adata, synthetic_img, patch_size=8, ecm_K=3, features=["mean"])
        mt.gr.build_cell_graph(adata, k=3)
        mt.gr.build_ecm_graph(adata, k=3)
        mt.gr.build_cell_ecm_graph(adata, k=3)
        mt.tl.interaction_test(adata, n_iter=10)
        mt.tl.neighbourhood_clustering(adata, n_clusters=2)

        ax = mt.pl.cell_ecm_graph(adata, show=False)
        assert isinstance(ax, Axes)

        ax2 = mt.pl.interaction_heatmap(adata, show=False)
        assert isinstance(ax2, Axes)


class TestImagePanel:
    def test_path_input(self, tmp_path):
        import matplotlib.pyplot as plt
        from PIL import Image

        png = tmp_path / "p.png"
        Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8)).save(png)

        _, ax = plt.subplots()
        out_ax = mt.pl.image_panel(ax, png, title="hello", title_color="red")
        assert out_ax is ax
        assert len(ax.images) == 1
        title = ax.title
        assert title.get_text() == "hello"
        assert title.get_color() == "red"

    def test_ndarray_input(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        arr = np.full((8, 8, 3), 50, dtype=np.uint8)
        mt.pl.image_panel(ax, arr, title="x", title_color="#0066AA")
        assert len(ax.images) == 1
        assert ax.title.get_text() == "x"

    def test_no_title(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        mt.pl.image_panel(ax, arr)
        assert ax.title.get_text() == ""

    def test_rejects_unknown_input_type(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        with pytest.raises(TypeError, match="path, ndarray, or PIL.Image"):
            mt.pl.image_panel(ax, 42)


class TestGraphTriptych:
    def test_returns_three_axes(self, adata_with_graphs):
        axes = mt.pl.graph_triptych(adata_with_graphs, show=False)
        assert len(axes) == 3
        for ax in axes:
            assert isinstance(ax, Axes)

    def test_default_titles(self, adata_with_graphs):
        axes = mt.pl.graph_triptych(adata_with_graphs, show=False)
        assert [ax.get_title() for ax in axes] == ["Cell graph", "ECM graph", "Cell-ECM graph"]

    def test_subset_panels(self, adata_with_graphs):
        axes = mt.pl.graph_triptych(adata_with_graphs, panels=("cell", "cell_ecm"), show=False)
        assert len(axes) == 2
        assert axes[0].get_title() == "Cell graph"
        assert axes[1].get_title() == "Cell-ECM graph"

    def test_single_panel(self, adata_with_graphs):
        axes = mt.pl.graph_triptych(adata_with_graphs, panels=("ecm",), show=False)
        assert len(axes) == 1
        assert axes[0].get_title() == "ECM graph"

    def test_custom_titles(self, adata_with_graphs):
        axes = mt.pl.graph_triptych(adata_with_graphs, titles=("A", "B", "C"), show=False)
        assert [ax.get_title() for ax in axes] == ["A", "B", "C"]

    def test_provided_axes(self, adata_with_graphs):
        import matplotlib.pyplot as plt

        fig, ax_arr = plt.subplots(1, 3, figsize=(15, 5))
        axes = mt.pl.graph_triptych(adata_with_graphs, axes=list(ax_arr), show=False)
        assert axes[0] is ax_arr[0]
        assert axes[1] is ax_arr[1]
        assert axes[2] is ax_arr[2]

    def test_panel_kwargs_forwarded(self, adata_with_graphs):
        # edge_alpha=0.0 should still produce a valid Axes
        axes = mt.pl.graph_triptych(
            adata_with_graphs,
            panel_kwargs={"cell_ecm": {"edge_alpha": 0.0}},
            show=False,
        )
        assert len(axes) == 3

    def test_empty_panels_raises(self, adata_with_graphs):
        with pytest.raises(ValueError, match="panels"):
            mt.pl.graph_triptych(adata_with_graphs, panels=(), show=False)

    def test_unknown_panel_raises(self, adata_with_graphs):
        with pytest.raises(ValueError, match="unknown panel"):
            mt.pl.graph_triptych(adata_with_graphs, panels=("nope",), show=False)

    def test_title_length_mismatch_raises(self, adata_with_graphs):
        with pytest.raises(ValueError, match="titles has length"):
            mt.pl.graph_triptych(adata_with_graphs, titles=("A", "B"), show=False)

    def test_axes_length_mismatch_raises(self, adata_with_graphs):
        import matplotlib.pyplot as plt

        fig, ax_arr = plt.subplots(1, 2)
        with pytest.raises(ValueError, match="axes has length"):
            mt.pl.graph_triptych(adata_with_graphs, axes=list(ax_arr), show=False)

    def test_missing_graph_raises(self, adata_basic):
        with pytest.raises(ValueError, match="cell_graph_nx"):
            mt.pl.graph_triptych(adata_basic, show=False)


# ── multi-fold classifier ROC ───────────────────────────


class TestClassifierRocCurves:
    def _toy_curves(self):
        rows = []
        for fold in ("A", "B", "C"):
            for x, y in zip(np.linspace(0, 1, 11), np.linspace(0, 1, 11) ** 0.5, strict=False):
                rows.append({"fold": fold, "kind": "roc", "x": x, "y": y})
        return pd.DataFrame(rows)

    def _toy_summary(self):
        return pd.DataFrame({"fold": ["A", "B", "C"], "roc_auc": [0.9, 0.85, 0.88]})

    def test_returns_axes_with_curves_mode(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        out = mt.pl.classifier_roc(ax, curves=self._toy_curves(), summary=self._toy_summary())
        assert isinstance(out, Axes)
        # Mean overlay + chance line + per-fold (3) = at least 5 lines.
        assert len(ax.get_lines()) >= 5

    def test_mutual_exclusion_raises(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Pass exactly one"):
            mt.pl.classifier_roc(ax)

    def test_missing_columns_raise(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        bad = pd.DataFrame({"fold": ["A"], "x": [0.1], "y": [0.2]})
        with pytest.raises(ValueError, match="missing required columns"):
            mt.pl.classifier_roc(ax, curves=bad)


class TestPrivateSaveAndShow:
    def test_writes_only_requested_file_when_suffix_given(self, tmp_path):
        import matplotlib.pyplot as plt

        from mantpy.pl import _save_and_show

        figure, axis = plt.subplots()
        axis.plot([0, 1], [1, 0])
        output = tmp_path / "demo.png"
        _save_and_show(figure, output, False)
        assert output.is_file()
        assert output.stat().st_size > 0
        assert not (tmp_path / "demo.pdf").exists()

    def test_writes_pdf_and_png_when_no_suffix(self, tmp_path):
        import matplotlib.pyplot as plt

        from mantpy.pl import _save_and_show

        figure, axis = plt.subplots()
        axis.plot([0, 1], [1, 0])
        output_stem = tmp_path / "demo_stem"
        _save_and_show(figure, output_stem, False)
        assert output_stem.with_suffix(".pdf").is_file()
        assert output_stem.with_suffix(".png").is_file()
