"""Plotting functions for Mantpy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

if TYPE_CHECKING:
    from collections.abc import Sequence

from mantpy._constants import (
    CELL_ECM_GRAPH_KEY,
    CELL_GRAPH_KEY,
    ECM_GRAPH_KEY,
    ECM_IMAGE_KEY,
    ECM_PATCHES_KEY,
    EDGE_TYPE_CC,
    EDGE_TYPE_CE,
    EDGE_TYPE_EE,
    IMAGE_CONTAINER_KEY,
    INTERACTION_TEST_KEY,
    NEIGHBOURHOOD_CLUSTERS_KEY,
    NODE_TYPE_CELL,
    NODE_TYPE_ECM,
)
from mantpy.im import as_image_container

__all__ = [
    "categorical_palette",
    "cell_ecm_enrichment_bars",
    "cell_ecm_enrichment_heatmap",
    "cell_ecm_enrichment_per_roi",
    "cell_ecm_graph",
    "cell_graph",
    "channel_overlay_on_neighbours",
    "classifier_roc",
    "cross_compartment_ablation_bars",
    "ecm_centroid_table",
    "ecm_centroid_heatmap",
    "ecm_cluster_comparison",
    "ecm_resolution_selection",
    "ecm_graph",
    "ecm_graph_overlay",
    "ecm_image",
    "graph_triptych",
    "he_overview",
    "image_panel",
    "interaction_heatmap",
    "neighbourhood_clusters",
    "niche_bubble",
    "niche_bubble_table",
    "node_value_overlay",
    "patch_domain_map",
    "plot_cluster_map",
    "plot_delta_masked",
    "plot_lesion_metric_view",
    "plot_mean_composition",
    "plot_marker_otsu_composite",
    "show_image",
]


def he_overview(
    result: Any,
    *,
    image_mpp: float = 1.0,
    figsize: tuple[float, float] = (12.0, 3.0),
    save: str | Path | None = None,
    show: bool = True,
):
    """Plot the RGB, H, E and tissue previews from :func:`mt.pp.preprocess_he`.

    The preview is already downsampled; ``image_mpp`` and the recorded
    downsample factor are used only to express its extent in micrometres.
    Analytical image tiles are not read again.
    """
    required = ("rgb", "hematoxylin", "eosin", "tissue_mask", "params")
    missing = [name for name in required if not hasattr(result, name)]
    if missing:
        raise TypeError(
            "result must be returned by mt.pp.preprocess_he; missing "
            f"attributes: {missing}."
        )
    if not np.isfinite(image_mpp) or image_mpp <= 0:
        raise ValueError("image_mpp must be a positive finite value.")

    downsample = int(result.params.get("downsample", 1))
    height, width = np.asarray(result.rgb).shape[:2]
    extent = (0.0, width * downsample * image_mpp, height * downsample * image_mpp, 0.0)
    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    panels = (
        (result.rgb, "H&E preview", None),
        (result.hematoxylin, "Haematoxylin", "Blues"),
        (result.eosin, "Eosin", "magma"),
        (result.tissue_mask, "Tissue mask", "gray"),
    )
    for ax, (values, title, cmap) in zip(axes, panels, strict=True):
        ax.imshow(values, cmap=cmap, extent=extent, interpolation="nearest")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    _save_and_show(fig, save, show)
    return fig

# Default qualitative, colourblind-safe palette.
_CELL_PALETTE = [
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
    "#A65628",
    "#F781BF",
    "#999999",
    "#66C2A5",
    "#FC8D62",
]
_ECM_PALETTE = [
    "#1B9E77",
    "#D95F02",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
    "#666666",
]


def categorical_palette(
    categories: Sequence[Any],
    *,
    colors: Sequence[str] | None = None,
) -> dict[Any, str]:
    """Return a stable category-to-colour mapping in first-seen order."""
    ordered = list(dict.fromkeys(categories))
    source = list(colors or (*_CELL_PALETTE, *_ECM_PALETTE))
    if not source:
        raise ValueError("colors is empty; supply at least one colour.")
    return {category: source[index % len(source)] for index, category in enumerate(ordered)}


def cell_graph(
    adata: AnnData,
    *,
    color: str = "cell_type",
    graph_key: str = CELL_GRAPH_KEY + "_nx",
    node_size: float = 75,
    node_marker: str = "o",
    node_edgecolor: str | None = "black",
    node_linewidth: float = 0.5,
    palette: dict[str, str] | list[str] | None = None,
    cmap: str | None = None,
    edge_alpha: float = 0.3,
    edge_width: float = 2.0,
    edge_color: str = "#aaaaaa",
    edge_attr: str | None = None,
    edge_cmap: str | None = None,
    ax_off: bool = False,
    ax: Axes | None = None,
    save: str | Path | None = None,
    show: bool = True,
    **kwargs,
) -> Axes:
    """Plot the cell-cell spatial graph.

    Parameters
    ----------
    adata
        AnnData with ``adata.uns[graph_key]``.
    color
        Node attribute to colour by (default ``"cell_type"``).
        When ``cmap`` is set, the attribute must hold numeric values.
    graph_key
        Key of the NetworkX graph in ``adata.uns``.
    node_size
        Marker area in points² (matplotlib ``s``).
    node_marker
        Matplotlib marker string for nodes — e.g. ``"o"`` (circle, default),
        ``"^"`` (triangle-up), ``"s"`` (square), ``"D"`` (diamond).
    node_edgecolor
        Colour for the thin border drawn around each node marker.  Defaults to
        ``"black"``.  Set to ``None`` to remove the border entirely.
    node_linewidth
        Width of the node border in points (default ``0.5``).
    palette
        Override node colours for **categorical** colouring (``cmap=None``).
        Accepts a ``{label: hex}`` dict or a list of colours.
    cmap
        Matplotlib colormap name for **continuous** colouring (e.g.
        ``"plasma"``, ``"viridis"``).  When set, a colorbar is added.
    edge_alpha
        Transparency of edges (0 = invisible, 1 = opaque).
    edge_width
        Line width of edges in points (default ``2``).
    edge_color
        Hex or named colour for edges.
    ax
        Existing Axes to draw on; created if ``None``.
    save
        File path to save the figure (e.g. ``"cell_graph.png"``).
        Supports any format recognised by matplotlib (PNG, PDF, SVG …).
    show
        Call ``plt.show()`` after drawing.

    Returns
    -------
    matplotlib Axes
    """
    if graph_key not in adata.uns:
        raise ValueError(f"Key '{graph_key}' not found in adata.uns. Run `mt.gr.build_cell_graph(adata)` first.")

    G = adata.uns[graph_key]
    _ax_provided = ax is not None
    fig, ax = _get_ax(ax)

    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    lc = _draw_edges(G, pos, edge_alpha, edge_width, edge_color, ax, edge_attr=edge_attr, edge_cmap=edge_cmap)
    if lc is not None:
        fig.colorbar(lc, ax=ax, label=edge_attr, fraction=0.046)

    nodes = list(G.nodes())
    coords = np.array([pos[n] for n in nodes])
    ec = node_edgecolor if node_edgecolor is not None else "none"

    if cmap is not None:
        vals = np.array([float(G.nodes[n].get(color, 0)) for n in nodes])
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=vals,
            cmap=cmap,
            s=node_size,
            marker=node_marker,
            zorder=3,
            edgecolors=ec,
            linewidths=node_linewidth,
            **kwargs,
        )
        fig.colorbar(sc, ax=ax, label=color, fraction=0.046)
    else:
        node_labels = [str(G.nodes[n].get(color, "unknown")) for n in nodes]
        pal = _make_palette(node_labels, _CELL_PALETTE, override=palette)
        colors = [pal[lbl] for lbl in node_labels]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=node_size,
            marker=node_marker,
            zorder=3,
            edgecolors=ec,
            linewidths=node_linewidth,
            **kwargs,
        )
        _add_legend(ax, pal, title=color)

    ax.autoscale()
    if not _ax_provided:
        ax.set_title("Cell Graph")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    ax.invert_yaxis()  # pixel coords have origin top-left
    if ax_off:
        ax.axis("off")
    _save_and_show(fig, save, show)
    return ax


def ecm_graph(
    adata: AnnData,
    *,
    color: str = "ecm_cluster",
    graph_key: str = ECM_GRAPH_KEY,
    node_size: float = 75,
    node_marker: str = "^",
    node_edgecolor: str | None = "black",
    node_linewidth: float = 0.5,
    palette: dict[str, str] | list[str] | None = None,
    cmap: str | None = None,
    edge_alpha: float = 0.3,
    edge_width: float = 2.0,
    edge_color: str = "#aaaaaa",
    edge_attr: str | None = None,
    edge_cmap: str | None = None,
    exclude_clusters: tuple[int, ...] = (-1,),
    ax_off: bool = False,
    ax: Axes | None = None,
    save: str | Path | None = None,
    show: bool = True,
    **kwargs,
) -> Axes:
    """Plot the ECM spatial graph.

    Parameters
    ----------
    adata
        AnnData with ``adata.uns[graph_key]``.
    color
        Node attribute to colour by (default ``"ecm_cluster"``).
        When ``cmap`` is set, the attribute must hold numeric (float) values.
    graph_key
        Key of the NetworkX ECM graph.
    node_size
        Marker area in points² (matplotlib ``s``).
    node_marker
        Matplotlib marker for ECM nodes — default ``"^"`` (triangle-up).
        Other options: ``"o"`` (circle), ``"s"`` (square), ``"D"`` (diamond),
        ``"v"`` (triangle-down), ``"P"`` (plus-filled).
    node_edgecolor
        Colour for the thin border drawn around each node marker.  Defaults to
        ``"black"``.  Set to ``None`` to remove the border entirely.
    node_linewidth
        Width of the node border in points (default ``0.5``).
    palette
        Override node colours for **categorical** colouring (``cmap=None``).
        Accepts a ``{label: hex}`` dict or a list of colours.  Ignored when
        ``cmap`` is set.
    cmap
        Matplotlib colormap name for **continuous** colouring (e.g.
        ``"plasma"``, ``"viridis"``, ``"magma"``).  When set, ``color`` is
        read as a float per node and a colorbar is added.  Set to ``None``
        (default) for categorical/palette colouring.
    edge_alpha
        Transparency of edges (0 = invisible, 1 = opaque).
    edge_width
        Line width of edges in points (default ``2``).
    edge_color
        Flat colour for edges when ``edge_cmap`` is ``None``.
    edge_attr
        Edge attribute to map onto a colormap — e.g. ``"weight"``,
        ``"feat_fwd"`` (first component), or any scalar stored on edges.
        Requires ``edge_cmap`` to also be set.
    edge_cmap
        Colormap for edge colouring (e.g. ``"viridis"``, ``"coolwarm"``).
        When set together with ``edge_attr``, edges are coloured by the
        attribute value and a colorbar is added.  ``edge_color`` is ignored.
    ax
        Existing Axes to draw on; created if ``None``.
    save
        File path to save the figure (e.g. ``"ecm_graph.pdf"``).
        Supports any format recognised by matplotlib (PNG, PDF, SVG …).
    show
        Call ``plt.show()`` after drawing.

    Returns
    -------
    matplotlib Axes
    """
    if graph_key not in adata.uns:
        raise ValueError(f"Key '{graph_key}' not found in adata.uns. Run `mt.gr.build_ecm_graph(adata)` first.")

    G = adata.uns[graph_key]
    _ax_provided = ax is not None
    fig, ax = _get_ax(ax)

    # Filter background / noise clusters (default: exclude cluster -1).
    _excluded = set(exclude_clusters) | {str(c) for c in exclude_clusters}
    kept = {n for n, d in G.nodes(data=True) if str(d.get(color, "?")) not in _excluded}
    G_draw = G.subgraph(kept)

    pos = {n: (d["x"], d["y"]) for n, d in G_draw.nodes(data=True)}
    lc = _draw_edges(G_draw, pos, edge_alpha, edge_width, edge_color, ax, edge_attr=edge_attr, edge_cmap=edge_cmap)
    if lc is not None:
        fig.colorbar(lc, ax=ax, label=edge_attr, fraction=0.046)

    nodes = list(G_draw.nodes())
    coords = np.array([pos[n] for n in nodes]) if nodes else np.empty((0, 2))
    ec = node_edgecolor if node_edgecolor is not None else "none"

    if nodes:
        if cmap is not None:
            vals = np.array([float(G_draw.nodes[n].get(color, 0)) for n in nodes])
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=vals,
                cmap=cmap,
                s=node_size,
                marker=node_marker,
                zorder=3,
                edgecolors=ec,
                linewidths=node_linewidth,
                **kwargs,
            )
            fig.colorbar(sc, ax=ax, label=color, fraction=0.046)
        else:
            node_labels = [str(G_draw.nodes[n].get(color, "?")) for n in nodes]
            pal = _make_palette(node_labels, _ECM_PALETTE, override=palette)
            colors = [pal[lbl] for lbl in node_labels]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=colors,
                s=node_size,
                marker=node_marker,
                zorder=3,
                edgecolors=ec,
                linewidths=node_linewidth,
                **kwargs,
            )
            _add_legend(ax, pal, title=color)

    ax.autoscale()
    if not _ax_provided:
        # standalone figure: set labels and title
        ax.set_title("ECM Graph")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    ax.invert_yaxis()  # pixel coords have origin top-left
    if ax_off:
        ax.axis("off")
    _save_and_show(fig, save, show)
    return ax


def ecm_graph_overlay(
    adata: AnnData,
    img: np.ndarray | None = None,
    *,
    layer: str | None = None,
    color: str = "ecm_cluster",
    graph_key: str = ECM_GRAPH_KEY,
    node_size: float = 75,
    node_marker: str = "^",
    node_edgecolor: str | None = "black",
    node_linewidth: float = 0.5,
    palette: dict[str, str] | list[str] | None = None,
    cmap: str | None = None,
    img_cmap: str = "magma",
    img_alpha: float = 0.7,
    edge_alpha: float = 0.3,
    edge_width: float = 2.0,
    edge_color: str = "white",
    edge_attr: str | None = None,
    edge_cmap: str | None = None,
    side_by_side: bool = False,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
    show: bool = True,
    **kwargs,
) -> list[Axes]:
    """Plot the ECM graph overlaid on the raw image, with an optional side-by-side panel.

    Parameters
    ----------
    adata
        AnnData with ``adata.uns[graph_key]``.
    img
        Background image as a 2-D ``(H, W)`` array (single channel).  Typically
        the raw TIFF channel or the Frangi-enhanced image ``enhanced[0]``.

        If ``None`` (default), reads from the attached ``ImageContainer``:
        uses the layer specified by ``layer`` (default: ``"preprocessed"``
        if available, else ``"image"``), taking channel 0.
    layer
        Which ImageContainer layer to use when ``img=None``.  Default: auto-
        selects ``"preprocessed"`` > ``"image"``.
    color
        Node attribute to colour by.  When ``cmap`` is set, must be numeric.
    graph_key
        Key of the NetworkX ECM graph in ``adata.uns``.
    node_size
        Node marker area in points² (default ``75``).
    node_marker
        Matplotlib marker for nodes (default ``"^"`` triangle-up).
    node_edgecolor
        Border colour for node markers.  ``None`` removes the border.
    node_linewidth
        Width of the node border in points.
    palette
        Categorical colour override — ``{label: hex}`` dict or list.  Ignored
        when ``cmap`` is set.
    cmap
        Colormap for continuous node colouring (e.g. ``"plasma"``).  When set,
        a colorbar is added.
    img_cmap
        Matplotlib colormap for the background image (default ``"magma"``).
    img_alpha
        Opacity of the background image (0 = transparent, 1 = opaque).
    edge_alpha
        Edge transparency.
    edge_width
        Edge line width in points.
    edge_color
        Edge colour (default ``"white"`` for visibility against dark backgrounds).
    side_by_side
        If ``True``, create a two-panel figure: left = graph-only, right = overlay.
    figsize
        Figure size override.  Defaults to ``(7, 7)`` for single panel or
        ``(14, 7)`` for side-by-side.
    save
        File path to save the figure.
    show
        Call ``plt.show()`` after drawing.

    Returns
    -------
    list of matplotlib Axes — one element (overlay only) or two (side-by-side).
    """
    if graph_key not in adata.uns:
        raise ValueError(f"Key '{graph_key}' not found in adata.uns. Run `mt.gr.build_ecm_graph(adata)` first.")

    G = adata.uns[graph_key]

    # Resolve image
    if img is None:
        if IMAGE_CONTAINER_KEY not in adata.uns:
            raise ValueError(
                "No `img` provided and no ImageContainer attached to adata. "
                "Either pass `img` explicitly or use `mt.io.read_imc()` / "
                "`mt.io.read_ecm_image()` to attach an ImageContainer."
            )
        ic = as_image_container(adata.uns[IMAGE_CONTAINER_KEY])
        if layer is not None:
            layer_arr = ic.get_layer(layer)
        elif ic.has_layer("preprocessed"):
            layer_arr = ic.get_layer("preprocessed")
        else:
            layer_arr = ic.to_array()
        arr = layer_arr[0]  # first channel → (H, W)
    else:
        arr = np.squeeze(np.asarray(img))
    if arr.ndim != 2:
        raise ValueError(
            f"img must be a 2-D (H, W) array, got shape {arr.shape}. Pass a single channel, e.g. enhanced[0] or raw."
        )

    n_panels = 2 if side_by_side else 1
    fs = figsize or (7 * n_panels, 7)
    fig, axes_raw = plt.subplots(1, n_panels, figsize=fs)
    axes: list[Axes] = list(np.atleast_1d(axes_raw))

    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    nodes = list(G.nodes())
    coords = np.array([pos[n] for n in nodes])
    ec = node_edgecolor if node_edgecolor is not None else "none"

    def _scatter_nodes(ax_: Axes, extra_kw: dict) -> None:
        if cmap is not None:
            vals = np.array([float(G.nodes[n].get(color, 0)) for n in nodes])
            sc = ax_.scatter(
                coords[:, 0],
                coords[:, 1],
                c=vals,
                cmap=cmap,
                s=node_size,
                marker=node_marker,
                zorder=3,
                edgecolors=ec,
                linewidths=node_linewidth,
                **extra_kw,
            )
            fig.colorbar(sc, ax=ax_, label=color, fraction=0.046)
        else:
            node_labels = [str(G.nodes[n].get(color, "?")) for n in nodes]
            pal = _make_palette(node_labels, _ECM_PALETTE, override=palette)
            colors_ = [pal[lbl] for lbl in node_labels]
            ax_.scatter(
                coords[:, 0],
                coords[:, 1],
                c=colors_,
                s=node_size,
                marker=node_marker,
                zorder=3,
                edgecolors=ec,
                linewidths=node_linewidth,
                **extra_kw,
            )
            _add_legend(ax_, pal, title=color)

    if side_by_side:
        # Left panel — graph only (no background image)
        lc0 = _draw_edges(G, pos, edge_alpha, edge_width, "#aaaaaa", axes[0], edge_attr=edge_attr, edge_cmap=edge_cmap)
        if lc0 is not None:
            fig.colorbar(lc0, ax=axes[0], label=edge_attr, fraction=0.046)
        _scatter_nodes(axes[0], {})
        axes[0].autoscale()
        axes[0].set_title("ECM Graph")
        axes[0].set_xlabel("x (px)")
        axes[0].set_ylabel("y (px)")
        axes[0].invert_yaxis()

    # Overlay panel (always rightmost / only)
    ax_ov = axes[-1]
    ax_ov.imshow(arr, cmap=img_cmap, alpha=img_alpha, aspect="auto")
    lc_ov = _draw_edges(G, pos, edge_alpha, edge_width, edge_color, ax_ov, edge_attr=edge_attr, edge_cmap=edge_cmap)
    if lc_ov is not None:
        fig.colorbar(lc_ov, ax=ax_ov, label=edge_attr, fraction=0.046)
    _scatter_nodes(ax_ov, kwargs)
    ax_ov.set_title("ECM Graph — overlay")
    ax_ov.invert_yaxis()
    ax_ov.axis("off")

    plt.tight_layout()
    _save_and_show(fig, save, show)
    return axes


def cell_ecm_graph(
    adata: AnnData,
    *,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    cell_color: str = "cell_type",
    ecm_color: str = "ecm_cluster",
    node_size_cell: float = 75,
    node_size_ecm: float = 75,
    node_size: float | None = None,
    cell_marker: str = "o",
    ecm_marker: str = "^",
    node_edgecolor: str | None = "black",
    node_linewidth: float = 0.5,
    cell_palette: dict[str, str] | list[str] | None = None,
    ecm_palette: dict[str, str] | list[str] | None = None,
    edge_alpha: float = 0.25,
    edge_width: float = 2.0,
    edge_color_cc: str | None = None,
    edge_color_ee: str | None = None,
    edge_color_ce: str | None = None,
    exclude_ecm_clusters: tuple[int, ...] = (-1,),
    max_edges: int | None = 100_000,
    random_state: int = 0,
    ax_off: bool = False,
    ax: Axes | None = None,
    save: str | Path | None = None,
    show: bool = True,
    **kwargs,
) -> Axes:
    """Plot the unified cell-ECM heterogeneous graph.

    Cell nodes are coloured by ``cell_color``, ECM nodes by ``ecm_color``.
    Edges are coloured by type: cell-cell (grey), ecm-ecm (green), cell-ecm (orange).

    Parameters
    ----------
    adata
        AnnData with ``adata.uns[graph_key]``.
    graph_key
        Key of the unified NetworkX graph.
    cell_color
        Node attribute for cell node colours.
    ecm_color
        Node attribute for ECM node colours.
    node_size_cell
        Marker area for cell nodes (points²).
    node_size_ecm
        Marker area for ECM nodes (points²).
    cell_marker
        Matplotlib marker for cell nodes — default ``"o"`` (circle).
    ecm_marker
        Matplotlib marker for ECM nodes — default ``"^"`` (triangle-up).
        Other useful choices: ``"s"`` (square), ``"D"`` (diamond), ``"v"``.
    node_edgecolor
        Colour for the border drawn around each node marker.  Defaults to
        ``"black"``.  Set to ``None`` to remove the border.
    node_linewidth
        Width of the node border in points (default ``0.5``).
    cell_palette
        Override cell node colours — ``dict`` ``{label: hex}`` or ``list``
        of colours cycled in sorted-label order.  ``None`` = default palette.
    ecm_palette
        Override ECM node colours — same format as ``cell_palette``.
    edge_alpha
        Transparency of edges (0 = invisible, 1 = opaque).
    edge_width
        Line width of edges in points.
    edge_color_cc
        Override colour for cell-cell edges.  ``None`` (default) keeps
        the historical grey ``"#666666"``.
    edge_color_ee
        Override colour for ecm-ecm edges.  ``None`` (default) keeps
        the historical green ``"#2ca02c"``.
    edge_color_ce
        Override colour for cell-ecm edges.  ``None`` (default) keeps
        the historical orange ``"#ff7f0e"``.
    max_edges
        Maximum number of edges drawn per edge layer for a sparse
        observation-native whole-section graph. Edges are sampled only for
        display; the analytical graph is unchanged. ``None`` draws all edges.
    random_state
        Seed for deterministic display-edge sampling.
    ax
        Existing Axes to draw on; created if ``None``.
    save
        File path to save the figure.  Supports PNG, PDF, SVG, etc.
    show
        Call ``plt.show()`` after drawing.

    Returns
    -------
    matplotlib Axes
    """
    if graph_key not in adata.uns:
        raise ValueError(f"Key '{graph_key}' not found in adata.uns. Run `mt.gr.build_cell_ecm_graph(adata)` first.")

    G = adata.uns[graph_key]
    if isinstance(G, dict) and G.get("format") == "mantpy_sparse_joint_v1":
        return _sparse_cell_ecm_graph(
            adata,
            graph=G,
            cell_color=cell_color,
            ecm_color=ecm_color,
            node_size_cell=node_size if node_size is not None else node_size_cell,
            node_size_ecm=node_size if node_size is not None else node_size_ecm,
            cell_marker=cell_marker,
            ecm_marker=ecm_marker,
            node_edgecolor=node_edgecolor,
            node_linewidth=node_linewidth,
            cell_palette=cell_palette,
            ecm_palette=ecm_palette,
            edge_alpha=edge_alpha,
            edge_width=edge_width,
            edge_color_cc=edge_color_cc,
            edge_color_ee=edge_color_ee,
            edge_color_ce=edge_color_ce,
            exclude_ecm_clusters=exclude_ecm_clusters,
            max_edges=max_edges,
            random_state=random_state,
            ax_off=ax_off,
            ax=ax,
            save=save,
            show=show,
        )
    _ax_provided = ax is not None
    fig, ax = _get_ax(ax)

    # node_size convenience param: overrides both node_size_cell and node_size_ecm.
    if node_size is not None:
        node_size_cell = node_size
        node_size_ecm = node_size

    _ecm_excluded = set(exclude_ecm_clusters) | {str(c) for c in exclude_ecm_clusters}

    cell_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == NODE_TYPE_CELL]
    ecm_nodes = [
        n
        for n, d in G.nodes(data=True)
        if d.get("node_type") == NODE_TYPE_ECM and str(d.get(ecm_color, "?")) not in _ecm_excluded
    ]

    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}

    cell_labels = [str(G.nodes[n].get(cell_color, "?")) for n in cell_nodes]
    ecm_labels = [str(G.nodes[n].get(ecm_color, "?")) for n in ecm_nodes]

    cell_pal = _make_palette(cell_labels, _CELL_PALETTE, override=cell_palette)
    ecm_pal = _make_palette(ecm_labels, _ECM_PALETTE, override=ecm_palette)

    _excluded_set = set(G.nodes()) - set(cell_nodes) - set(ecm_nodes)
    edge_type_color = {
        EDGE_TYPE_CC: edge_color_cc if edge_color_cc is not None else "#666666",
        EDGE_TYPE_EE: edge_color_ee if edge_color_ee is not None else "#2ca02c",
        EDGE_TYPE_CE: edge_color_ce if edge_color_ce is not None else "#ff7f0e",
    }
    for n1, n2, ed in G.edges(data=True):
        if n1 in _excluded_set or n2 in _excluded_set:
            continue
        etype = ed.get("edge_type", EDGE_TYPE_CC)
        color = edge_type_color.get(etype, "#aaaaaa")
        x1, y1 = pos[n1]
        x2, y2 = pos[n2]
        ax.plot([x1, x2], [y1, y2], color=color, alpha=edge_alpha, linewidth=edge_width, zorder=1)

    ec = node_edgecolor if node_edgecolor is not None else "none"

    if cell_nodes:
        cell_pos = np.array([pos[n] for n in cell_nodes])
        cell_colors_rgba = [cell_pal[lbl] for lbl in cell_labels]
        ax.scatter(
            cell_pos[:, 0],
            cell_pos[:, 1],
            c=cell_colors_rgba,
            s=node_size_cell,
            marker=cell_marker,
            zorder=3,
            label="Cell",
            edgecolors=ec,
            linewidths=node_linewidth,
        )

    if ecm_nodes:
        ecm_pos = np.array([pos[n] for n in ecm_nodes])
        ecm_colors_rgba = [ecm_pal[lbl] for lbl in ecm_labels]
        ax.scatter(
            ecm_pos[:, 0],
            ecm_pos[:, 1],
            c=ecm_colors_rgba,
            s=node_size_ecm,
            marker=ecm_marker,
            zorder=2,
            label="ECM",
            edgecolors=ec,
            linewidths=node_linewidth,
            alpha=0.7,
        )

    cell_handles = [matplotlib.patches.Patch(color=c, label=lbl) for lbl, c in cell_pal.items()]
    ecm_handles = [matplotlib.patches.Patch(color=c, label=f"ECM {lbl}") for lbl, c in ecm_pal.items()]
    ax.legend(
        handles=cell_handles + ecm_handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    if not _ax_provided:
        ax.set_title("Cell-ECM Graph")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    ax.invert_yaxis()  # pixel coords have origin top-left
    if ax_off:
        ax.axis("off")
    _save_and_show(fig, save, show)
    return ax


def _sparse_cell_ecm_graph(
    adata: AnnData,
    *,
    graph: dict[str, Any],
    cell_color: str,
    ecm_color: str,
    node_size_cell: float,
    node_size_ecm: float,
    cell_marker: str,
    ecm_marker: str,
    node_edgecolor: str | None,
    node_linewidth: float,
    cell_palette: dict[str, str] | list[str] | None,
    ecm_palette: dict[str, str] | list[str] | None,
    edge_alpha: float,
    edge_width: float,
    edge_color_cc: str | None,
    edge_color_ee: str | None,
    edge_color_ce: str | None,
    exclude_ecm_clusters: tuple[int, ...],
    max_edges: int | None,
    random_state: int,
    ax_off: bool,
    ax: Axes | None,
    save: str | Path | None,
    show: bool,
) -> Axes:
    """Draw a sparse joint AnnData graph with vectorised line collections."""
    import scipy.sparse as sp

    _ax_provided = ax is not None
    fig, ax = _get_ax(ax)
    spatial_key = str(graph.get("spatial_key", "spatial"))
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    node_types = adata.obs["node_type"].astype(str).to_numpy()
    cell_mask = node_types == NODE_TYPE_CELL
    ecm_mask = node_types == NODE_TYPE_ECM
    cell_labels = (
        adata.obs[cell_color].astype(str).to_numpy()[cell_mask]
        if cell_color in adata.obs
        else np.repeat("cell", int(cell_mask.sum()))
    )
    ecm_labels_all = (
        adata.obs[ecm_color].astype(str).to_numpy() if ecm_color in adata.obs else np.repeat("0", adata.n_obs)
    )
    excluded = {str(value) for value in exclude_ecm_clusters}
    ecm_keep = ecm_mask & ~np.isin(ecm_labels_all, list(excluded))
    keep = cell_mask | ecm_keep
    ecm_labels = ecm_labels_all[ecm_keep]
    cell_pal = _make_palette(list(cell_labels), _CELL_PALETTE, override=cell_palette)
    ecm_pal = _make_palette(list(ecm_labels), _ECM_PALETTE, override=ecm_palette)

    connectivity_keys = graph["connectivities"]
    layers = (
        (str(connectivity_keys["cell"]), edge_color_cc or "#666666"),
        (str(connectivity_keys["ecm"]), edge_color_ee or "#2ca02c"),
        (str(connectivity_keys["cell_ecm"]), edge_color_ce or "#ff7f0e"),
    )
    rng = np.random.default_rng(random_state)
    for key, color in layers:
        matrix = sp.triu(adata.obsp[key], k=1, format="coo")
        layer_keep = keep[matrix.row] & keep[matrix.col]
        rows, cols = matrix.row[layer_keep], matrix.col[layer_keep]
        if max_edges is not None and rows.size > max_edges:
            selection = np.sort(rng.choice(rows.size, size=max_edges, replace=False))
            rows, cols = rows[selection], cols[selection]
        if rows.size:
            segments = np.stack([coords[rows], coords[cols]], axis=1)
            ax.add_collection(LineCollection(segments, colors=color, alpha=edge_alpha, linewidths=edge_width, zorder=1))

    edge = node_edgecolor if node_edgecolor is not None else "none"
    if cell_mask.any():
        ax.scatter(
            coords[cell_mask, 0],
            coords[cell_mask, 1],
            c=[cell_pal[label] for label in cell_labels],
            s=node_size_cell,
            marker=cell_marker,
            edgecolors=edge,
            linewidths=node_linewidth,
            zorder=3,
        )
    if ecm_keep.any():
        ax.scatter(
            coords[ecm_keep, 0],
            coords[ecm_keep, 1],
            c=[ecm_pal[label] for label in ecm_labels],
            s=node_size_ecm,
            marker=ecm_marker,
            edgecolors=edge,
            linewidths=node_linewidth,
            alpha=0.7,
            zorder=2,
        )
    handles = [matplotlib.patches.Patch(color=color, label=label) for label, color in cell_pal.items()]
    handles.extend(matplotlib.patches.Patch(color=color, label=f"ECM {label}") for label, color in ecm_pal.items())
    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, fontsize=8)
    if not _ax_provided:
        ax.set_title("Cell-ECM Graph")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
    ax.autoscale_view()
    ax.invert_yaxis()
    if ax_off:
        ax.axis("off")
    _save_and_show(fig, save, show)
    return ax


_GRAPH_TRIPTYCH_PANELS = {
    "cell": (cell_graph, "Cell graph"),
    "ecm": (ecm_graph, "ECM graph"),
    "cell_ecm": (cell_ecm_graph, "Cell-ECM graph"),
}


def graph_triptych(
    adata: AnnData,
    *,
    panels: tuple[str, ...] = ("cell", "ecm", "cell_ecm"),
    titles: tuple[str, ...] | None = None,
    figsize: tuple[float, float] | None = None,
    axes: list[Axes] | None = None,
    panel_kwargs: dict[str, dict[str, Any]] | None = None,
    save: str | Path | None = None,
    show: bool = True,
    **kwargs,
) -> list[Axes]:
    """Render cell, ECM, and cell-ECM graph panels side-by-side.

    Convenience wrapper around :func:`cell_graph`, :func:`ecm_graph`, and
    :func:`cell_ecm_graph` — the three plotters always appear together when
    summarising a unified-graph ROI, so this single call replaces the
    `plt.subplots(1, 3)` + three individual plotter calls + title-setting
    loop that every spatial-graph tutorial cell rewrites by hand.

    Parameters
    ----------
    adata
        AnnData with the cell graph, ECM graph, and unified cell-ECM graph
        already built (typically via :func:`mt.gr.build_cell_graph`,
        :func:`mt.gr.build_ecm_graph`, :func:`mt.gr.build_cell_ecm_graph`).
    panels
        Sub-sequence of ``("cell", "ecm", "cell_ecm")`` controlling which
        panels appear, in left-to-right order.
    titles
        Optional per-panel titles.  ``None`` uses ``"Cell graph"``,
        ``"ECM graph"``, ``"Cell-ECM graph"``.  Pass a tuple matching
        ``len(panels)`` to override.
    figsize
        Figure size in inches when ``axes`` is ``None``.  Defaults to
        ``(5 * len(panels), 5)``.
    axes
        Optional list of pre-existing matplotlib Axes (one per panel).
        When ``None`` a fresh figure with ``constrained_layout=True`` is
        created.  Must have length ``len(panels)`` when provided.
    panel_kwargs
        Per-panel keyword overrides, keyed by panel name.  E.g.
        ``{"cell_ecm": {"edge_alpha": 0.1}}`` to pass ``edge_alpha=0.1``
        only to the cell-ECM graph plotter.
    save
        File path to save the figure.
    show
        Call ``plt.show()`` after drawing.
    **kwargs
        Extra keyword arguments forwarded to every panel's plotter.

    Returns
    -------
    list[matplotlib.axes.Axes]
        One Axes per requested panel, in the order given by ``panels``.

    Examples
    --------
    >>> import mantpy as mt  # doctest: +SKIP
    >>> mt.gr.build_cell_graph(adata)
    >>> mt.gr.build_ecm_graph(adata)
    >>> mt.gr.build_cell_ecm_graph(adata)
    >>> mt.pl.graph_triptych(adata)
    """
    if not panels:
        raise ValueError("panels is empty: expected at least one panel name. Pass e.g. panels=('cell',).")
    unknown = [p for p in panels if p not in _GRAPH_TRIPTYCH_PANELS]
    if unknown:
        raise ValueError(f"unknown panel name(s) {unknown!r}: expected a subset of {tuple(_GRAPH_TRIPTYCH_PANELS)}.")
    if titles is not None and len(titles) != len(panels):
        raise ValueError(
            f"titles has length {len(titles)} but panels has length {len(panels)}; "
            f"pass one title per panel or omit `titles`."
        )

    panel_kwargs = panel_kwargs or {}

    if axes is None:
        if figsize is None:
            figsize = (5 * len(panels), 5)
        fig, ax_arr = plt.subplots(1, len(panels), figsize=figsize, constrained_layout=True)
        if len(panels) == 1:
            ax_list = [ax_arr]
        else:
            ax_list = list(ax_arr)
    else:
        if len(axes) != len(panels):
            raise ValueError(
                f"axes has length {len(axes)} but panels has length {len(panels)}; pass one Axes per panel."
            )
        fig = axes[0].get_figure()
        ax_list = list(axes)

    for i, name in enumerate(panels):
        plotter, default_title = _GRAPH_TRIPTYCH_PANELS[name]
        extra = {**kwargs, **panel_kwargs.get(name, {})}
        plotter(adata, ax=ax_list[i], show=False, **extra)
        title = default_title if titles is None else titles[i]
        ax_list[i].set_title(title)

    _save_and_show(fig, save, show)
    return ax_list


def interaction_heatmap(
    adata: AnnData,
    *,
    key: str = INTERACTION_TEST_KEY,
    condition: str | None = None,
    cmap: str = "RdBu_r",
    ax: Axes | None = None,
    save: str | Path | None = None,
    show: bool = True,
    **kwargs,
) -> Axes:
    """Plot interaction significance as a heatmap.

    Parameters
    ----------
    adata
        AnnData with ``adata.uns[key]``.
    key
        Key of the interaction result DataFrame.
    condition
        Optional title annotation.
    cmap
        Matplotlib colormap.
    ax, save, show
        See :func:`cell_graph`.

    Returns
    -------
    matplotlib Axes
    """
    if key not in adata.uns:
        raise ValueError(f"Key '{key}' not found in adata.uns. Run `mt.tl.interaction_test(adata)` first.")

    sigval = adata.uns[key]
    fig, ax = _get_ax(ax, figsize=(max(6, len(sigval.columns) * 0.8), max(4, len(sigval) * 0.7)))

    vabs = max(1, abs(sigval.values).max())
    sns.heatmap(
        sigval,
        cmap=cmap,
        center=0,
        vmin=-vabs,
        vmax=vabs,
        linewidths=0.5,
        annot=True,
        fmt="d",
        ax=ax,
        **kwargs,
    )
    title = "Interaction Test"
    if condition:
        title += f" — {condition}"
    ax.set_title(title)
    ax.set_xlabel("ECM Cluster")
    ax.set_ylabel("Cell Type")
    _save_and_show(fig, save, show)
    return ax


def ecm_image(
    adata: AnnData,
    *,
    ax: Axes | None = None,
    save: str | Path | None = None,
    show: bool = True,
    **kwargs,
) -> Axes:
    """Visualise the ECM cluster label image as a spatial map.

    Parameters
    ----------
    adata
        AnnData with ``adata.uns['ecm_image']``.
    ax, save, show
        See :func:`cell_graph`.

    Returns
    -------
    matplotlib Axes
    """
    if ECM_IMAGE_KEY not in adata.uns:
        raise ValueError(
            f"Key '{ECM_IMAGE_KEY}' not found in adata.uns. Run `mt.pp.extract_ecm_patches(adata, img)` first."
        )

    img = adata.uns[ECM_IMAGE_KEY]
    n_clusters = int(img.max()) + 1
    cmap = matplotlib.colors.ListedColormap(["#cccccc"] + _ECM_PALETTE[:n_clusters])

    fig, ax = _get_ax(ax)
    ax.imshow(img, cmap=cmap, interpolation="nearest", vmin=-1, vmax=n_clusters - 1)
    ax.set_title("ECM Cluster Map")
    ax.axis("off")
    _save_and_show(fig, save, show)
    return ax


def neighbourhood_clusters(
    adata: AnnData,
    *,
    key: str = NEIGHBOURHOOD_CLUSTERS_KEY,
    ax: Axes | None = None,
    save: str | Path | None = None,
    show: bool = True,
    **kwargs,
) -> Axes:
    """Scatter plot of cells coloured by neighbourhood cluster (IDIN).

    Parameters
    ----------
    adata
        AnnData with ``adata.obs[key]``.
    key
        Column in ``adata.obs`` holding cluster labels.
    ax, save, show
        See :func:`cell_graph`.

    Returns
    -------
    matplotlib Axes
    """
    if key not in adata.obs.columns:
        raise ValueError(f"Column '{key}' not found in adata.obs. Run `mt.tl.neighbourhood_clustering(adata)` first.")

    from mantpy._constants import SPATIAL_KEY

    coords = adata.obsm[SPATIAL_KEY]
    labels = adata.obs[key].astype(str).values
    palette = _make_palette(labels, _CELL_PALETTE)

    fig, ax = _get_ax(ax)
    for lbl, color in palette.items():
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1], c=[color], label=f"IDIN {lbl}", s=60, edgecolors="white", linewidths=0.3
        )

    ax.set_title("Neighbourhood Clusters (IDIN)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, fontsize=8)
    _save_and_show(fig, save, show)
    return ax


def ecm_resolution_selection(
    selection: Any,
    *,
    ax: Axes | None = None,
    save: str | Path | None = None,
    show: bool = True,
) -> Axes:
    """Show how a Leiden resolution — and therefore the ECM state count — was chosen.

    :func:`mantpy.pp.select_ecm_leiden_resolution` optimises the *resolution*,
    scoring each candidate by Calinski-Harabasz. The number of ECM states is
    whatever Leiden emits at the winning resolution; it is recorded but never
    itself optimised. This plot makes that explicit, so a reader can see the
    margin the selection won by rather than taking the resulting count on
    trust.

    Parameters
    ----------
    selection
        :class:`~mantpy.pp.ECMLeidenResolutionSelection` from
        :func:`mantpy.pp.select_ecm_leiden_resolution`.
    ax, save, show
        See :func:`cell_graph`.

    Returns
    -------
    matplotlib Axes
        The Calinski-Harabasz axis. The cluster-count axis is its twin, so
        ``ax.get_shared_x_axes()`` reaches both.
    """
    table = selection.table.sort_values("resolution")
    resolutions = table["resolution"].to_numpy(dtype=float)
    scores = table["calinski_harabasz"].to_numpy(dtype=float)
    counts = table["n_clusters"].to_numpy(dtype=int)

    fig, ax = _get_ax(ax, figsize=(5.2, 3.4))
    score_colour = "#1f4e79"
    count_colour = "#c0504d"

    ax.plot(resolutions, scores, "-o", color=score_colour, markersize=4, linewidth=1.2, zorder=3)
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("Calinski-Harabasz score", color=score_colour)
    ax.tick_params(axis="y", labelcolor=score_colour)

    chosen = float(selection.selected_resolution)
    ax.axvline(chosen, color="0.4", linestyle="--", linewidth=0.9, zorder=1)
    best = float(scores[np.argmin(np.abs(resolutions - chosen))])
    ax.plot([chosen], [best], "o", color=score_colour, markersize=9, markerfacecolor="none", markeredgewidth=1.6, zorder=4)

    twin = ax.twinx()
    twin.step(resolutions, counts, where="mid", color=count_colour, linewidth=1.1, alpha=0.85, zorder=2)
    twin.set_ylabel("ECM states resolved", color=count_colour)
    twin.tick_params(axis="y", labelcolor=count_colour)
    twin.set_ylim(0, max(counts.max() + 1, 2))

    ax.set_title(
        f"resolution {chosen:g} → {int(selection.selected_n_clusters)} ECM states"
        f"  (k={selection.effective_n_neighbors}, {selection.subset} patches)",
        fontsize=9,
    )
    ax.set_zorder(twin.get_zorder() + 1)
    ax.patch.set_visible(False)

    _save_and_show(fig, save, show)
    return ax


# ---------------------------------------------------------------------------
# Convenience comparison / overview functions
# ---------------------------------------------------------------------------


def show_image(
    img: np.ndarray | Any,
    *,
    layer: str | None = None,
    channel: int = 0,
    cmap: str = "inferno",
    clip_percentile: tuple[float, float] | None = (1.0, 99.5),
    mask_zero: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5, 5),
    title: str | None = None,
    interpolation: str = "lanczos",
    save: str | Path | None = None,
    show: bool = True,
) -> Axes:
    """Display a single-channel image with percentile contrast rescaling.

    The "clip foreground to [p_lo, p_hi] then linearly rescale to [0, 1]"
    pattern is ubiquitous in spatial-omics figures.  This helper saves
    users from reinventing it every time.

    Parameters
    ----------
    img
        One of:

        - ``np.ndarray`` of shape ``(H, W)`` or ``(C, H, W)``
        - :class:`~mantpy.im.ImageContainer`
        - :class:`~anndata.AnnData` with ``uns['image_container']``
    layer
        Layer name to display when ``img`` is an ImageContainer or AnnData.
        ``None`` → uses ``"preprocessed"`` if present, else ``"image"``.
    channel
        Channel index (for ``(C, H, W)`` inputs).
    cmap
        Matplotlib colormap.
    clip_percentile
        ``(p_lo, p_hi)`` percentiles used to clip intensities before
        rescaling to ``[0, 1]``.  ``None`` disables clipping (use raw data
        range).
    mask_zero
        If ``True``, zero-valued pixels are excluded from percentile
        calculation (avoids the background dominating the statistics) and
        set back to zero after rescaling.
    ax
        Existing axes to draw into.  When ``None`` a new figure is created.
    figsize, title, save, show, interpolation
        Standard figure parameters.

    Returns
    -------
    The matplotlib Axes containing the image.

    Examples
    --------
    >>> mt.pl.show_image(adata)  # raw ColIV, auto-clip
    >>> mt.pl.show_image(adata, layer="preprocessed")  # Frangi response
    >>> mt.pl.show_image(arr, clip_percentile=(2, 98))  # tighter clip
    """
    from mantpy.im import ImageContainer as _IC

    # ---- resolve input to a 2-D array ---------------------------------------
    if isinstance(img, AnnData):
        if IMAGE_CONTAINER_KEY not in img.uns:
            raise ValueError(
                f"AnnData has no ImageContainer at uns['{IMAGE_CONTAINER_KEY}']. "
                "Load it with mt.read_ecm_image / mt.read_imc first."
            )
        ic = as_image_container(img.uns[IMAGE_CONTAINER_KEY])
        arr = _pick_layer(ic, layer)
    elif isinstance(img, _IC):
        arr = _pick_layer(img, layer)
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        raise TypeError(f"img must be ndarray, ImageContainer, or AnnData; got {type(img).__name__}.")

    if arr.ndim == 3:
        if channel >= arr.shape[0]:
            raise IndexError(f"channel={channel} but image has {arr.shape[0]} channels.")
        arr = arr[channel]
    elif arr.ndim != 2:
        raise ValueError(f"Expected (H, W) or (C, H, W), got shape {arr.shape}.")

    # Keep one float32 display buffer. In-place scaling avoids a second,
    # float64-promoted full-image temporary for large microscopy planes.
    disp = arr.astype(np.float32, copy=True)

    # ---- contrast rescale ---------------------------------------------------
    if clip_percentile is not None:
        p_lo, p_hi = clip_percentile
        source = disp[disp > 0] if mask_zero else disp
        if source.size > 0:
            lo, hi = (np.float32(value) for value in np.percentile(source, [p_lo, p_hi]))
            del source
            if hi > lo:
                np.clip(disp, lo, hi, out=disp)
                np.subtract(disp, lo, out=disp)
                np.divide(disp, np.float32(hi - lo + 1e-8), out=disp)
            else:
                np.divide(disp, np.float32(disp.max() + 1e-8), out=disp)

    # ---- draw ---------------------------------------------------------------
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ax.imshow(disp, cmap=cmap, interpolation=interpolation, origin="upper", vmin=0, vmax=1)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if created_fig:
        _save_and_show(fig, save, show)
    return ax


def _pick_layer(ic: Any, layer: str | None) -> np.ndarray:
    """Resolve which layer of an ImageContainer to display."""
    if layer is not None:
        return ic.get_layer(layer)
    if ic.has_layer("preprocessed"):
        return ic.get_layer("preprocessed")
    return ic.to_array()


def _get_ax(
    ax: Axes | None,
    figsize: tuple[float, float] = (8, 7),
) -> tuple[plt.Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _make_palette(
    labels: list[str],
    base: list[str],
    override: dict[str, str] | list[str] | None = None,
) -> dict[str, str]:
    unique = sorted(set(labels))
    if override is None:
        return {lbl: base[i % len(base)] for i, lbl in enumerate(unique)}
    if isinstance(override, dict):
        # Normalise keys to str so that int-keyed dicts (e.g.
        # ECM_PALETTE_LUNG_PUBLISHED = {0: "#...", 1: "#..."}) merge
        # correctly with the string-keyed default built from node labels.
        override_str = {str(k): v for k, v in override.items()}
        default = {lbl: base[i % len(base)] for i, lbl in enumerate(unique)}
        return {**default, **override_str}
    # list — cycle through in label order
    return {lbl: override[i % len(override)] for i, lbl in enumerate(unique)}


def _draw_edges(
    G: nx.Graph,
    pos: dict,
    edge_alpha: float,
    edge_width: float,
    edge_color: str,
    ax: Axes,
    *,
    edge_attr: str | None = None,
    edge_cmap: str | None = None,
) -> LineCollection | None:
    """Draw graph edges.  Returns a LineCollection when edge_attr+edge_cmap are set."""
    edges = list(G.edges(data=True))
    if not edges:
        return None

    segments = [[(pos[u][0], pos[u][1]), (pos[v][0], pos[v][1])] for u, v, _ in edges]

    if edge_attr is not None and edge_cmap is not None:
        vals = []
        for _u, _v, d in edges:
            raw = d.get(edge_attr)
            # feat_fwd is a 1-D array — pick first element when attr is a vector
            if raw is None:
                raw = d.get("feat_fwd")
                if raw is not None:
                    # find position of edge_attr name in EDGE_FEATURE_REGISTRY order
                    raw = float(np.asarray(raw).ravel()[0])
                else:
                    raw = 0.0
            vals.append(float(np.asarray(raw).ravel()[0]) if hasattr(raw, "__len__") else float(raw))

        vals_arr = np.array(vals, dtype=np.float32)
        lc = LineCollection(segments, array=vals_arr, cmap=edge_cmap, linewidth=edge_width, alpha=edge_alpha, zorder=1)
        ax.add_collection(lc)
        return lc
    else:
        lc = LineCollection(segments, color=edge_color, linewidth=edge_width, alpha=edge_alpha, zorder=1)
        ax.add_collection(lc)
        return None


def _add_legend(ax: Axes, palette: dict[str, str], title: str) -> None:
    handles = [matplotlib.patches.Patch(color=c, label=lbl) for lbl, c in palette.items()]
    ax.legend(
        handles=handles,
        title=title,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )


def _save_and_show(fig: plt.Figure, save: str | Path | None, show: bool) -> None:
    """Apply Mantpy's private save/show convention for package plotters."""
    if save is not None:
        path = Path(save)
        if path.suffix:
            fig.savefig(str(path), bbox_inches="tight", dpi=300)
        else:
            fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight")
            fig.savefig(
                str(path.with_suffix(".png")),
                bbox_inches="tight",
                dpi=300,
            )
    if show:
        plt.show()


def image_panel(
    ax: Axes,
    image: str | Path | np.ndarray | Any,
    *,
    title: str | None = None,
    title_color: str = "black",
    title_fontsize: float = 9,
    title_weight: str = "bold",
    title_pad: float = 4,
    border_color: str | None = "#CCCCCC",
    border_width: float = 0.4,
    facecolor: str | None = "black",
    interpolation: str = "nearest",
    aspect: str = "auto",
) -> Axes:
    """Render a saved image (path / PIL / ndarray) onto an Axes.

    Bundles the recurring "open image → imshow → axis off → border → coloured
    title" pattern into one call for composites that mix pre-rendered tiles
    with live plots.

    The ``border_color`` / ``border_width`` hooks make the spines visible
    again after ``ax.axis("off")`` so the panel keeps a consistent frame.

    Parameters
    ----------
    ax
        Target Axes (typically created from a gridspec).
    image
        Path (``str`` / :class:`~pathlib.Path`), :class:`PIL.Image.Image`, or
        a ``(H, W)`` / ``(H, W, 3)`` / ``(H, W, 4)`` :class:`numpy.ndarray`.
        Paths and PIL images are converted to RGB.
    title
        Optional title text — drawn in ``title_color`` (per-panel colour
        override is useful when adjacent panels have different palettes.
    title_color, title_fontsize, title_weight, title_pad
        Title styling.
    border_color, border_width
        Frame around the panel.  Pass ``border_color=None`` or
        ``border_width=0`` to omit.
    facecolor
        Axis background color (showed where the image doesn't cover the
        whole panel — useful for back-projection figures with black
        backgrounds).  Pass ``None`` to leave the default.
    interpolation, aspect
        Forwarded to ``ax.imshow``.

    Returns
    -------
    matplotlib.axes.Axes
        The same Axes that was passed in.

    Examples
    --------
    >>> for ax, (gene, ecm_cluster, png_path) in zip(  # doctest: +SKIP
    ...     spatial_axes,
    ...     spatial_pairs,
    ... ):
    ...     mt.pl.image_panel(
    ...         ax,
    ...         png_path,
    ...         title=gene,
    ...         title_color=CLUSTER_COLORS[ecm_cluster],
    ...     )
    """
    # Resolve the image to an ndarray.
    if isinstance(image, np.ndarray):
        img = image
    else:
        from PIL import Image as _PILImage

        if isinstance(image, _PILImage.Image):
            img = np.asarray(image.convert("RGB"))
        elif isinstance(image, str | Path):
            img = np.asarray(_PILImage.open(Path(image)).convert("RGB"))
        else:
            raise TypeError(f"image_panel: image must be a path, ndarray, or PIL.Image; got {type(image).__name__}.")

    ax.imshow(img, aspect=aspect, interpolation=interpolation)
    if facecolor is not None:
        ax.set_facecolor(facecolor)
    ax.axis("off")
    if border_color is not None and border_width > 0:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(border_width)
            sp.set_edgecolor(border_color)

    if title is not None:
        ax.set_title(
            title,
            fontsize=title_fontsize,
            fontweight=title_weight,
            color=title_color,
            pad=title_pad,
        )

    return ax


# -----------------------------------------------------------------------------
# Classifier panels
# -----------------------------------------------------------------------------


def classifier_roc(
    ax: Axes | None = None,
    df: pd.DataFrame | None = None,
    *,
    curves: pd.DataFrame | None = None,
    summary: pd.DataFrame | None = None,
    fold_col: str | None = "fold",
    prob_col: str = "prob",
    label_col: str = "label",
    pos_label: int | str = 1,
    kind: str = "roc",
    fold_cmap: str | None = "Greens",
    fold_alpha: float = 0.85,
    fold_color: str | None = None,
    mean_color: str | None = None,
    mean_lw: float = 2.6,
    show_pooled: bool = True,
    pooled_color: str = "black",
    pooled_lw: float = 2.4,
    chance_line: bool = True,
    title: str | None = "ROC",
    xlabel: str = "False positive rate",
    ylabel: str = "True positive rate",
    legend: bool = True,
    legend_fontsize: float = 8,
    figsize: tuple[float, float] = (3.4, 3.4),
    save: str | Path | None = None,
    show: bool = False,
) -> Axes:
    """Per-fold ROC curves with optional pooled curve, from a long-format DataFrame.

    Two input modes:

    * **Per-sample mode** (default): pass ``df`` with per-sample
      ``prob_col`` + ``label_col`` (and optional ``fold_col``) — the
      function calls scikit-learn's ``roc_curve`` / ``roc_auc_score`` to
      reconstruct the curves and pooled AUC.
    * **Pre-computed curves mode**: pass ``curves`` with columns
      ``[fold, kind, x, y]`` (the schema written by leave-one-out CV
      pipelines such as the one bundled with
      :func:`mantpy.fetch.load_balbc_pbs_lung`). Per-fold curves are
      plotted directly and an interpolated mean curve is overlaid. When
      ``summary`` is also provided (columns ``[fold, roc_auc, pr_auc]``)
      it is used for the per-fold + mean AUCs shown in the legend.

    Parameters
    ----------
    df
        Per-sample predictions DataFrame. Mutually exclusive with ``curves``.
    curves
        Pre-computed long-form curve DataFrame
        (one row per ``(fold, x, y)`` point). Mutually exclusive with ``df``.
    summary
        Optional per-fold AUC summary used in pre-computed-curves mode.
    kind
        Curve kind to plot in pre-computed-curves mode
        (matches the ``kind`` column of ``curves``; usually ``"roc"``).
    fold_color, mean_color
        Per-fold and mean overlay colours used in pre-computed-curves mode.
        Default to ``mantpy.palette.ROC_FOLD`` / ``mantpy.palette.ROC_MEAN``.
    mean_lw
        Line width of the mean overlay curve.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that was drawn into.
    """
    from mantpy import _palette as _pal

    if (df is None) == (curves is None):
        raise ValueError("Pass exactly one of `df` (per-sample mode) or `curves` (pre-computed-curves mode).")
    fig, ax = _get_ax(ax, figsize=figsize)

    if curves is not None:
        result = _classifier_roc_from_curves(
            ax,
            curves,
            summary=summary,
            kind=kind,
            fold_color=fold_color or _pal.ROC_FOLD,
            mean_color=mean_color or _pal.ROC_MEAN,
            mean_lw=mean_lw,
            chance_line=chance_line,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
            legend_fontsize=legend_fontsize,
        )
        _save_and_show(fig, save, show)
        return result

    from sklearn.metrics import roc_auc_score, roc_curve

    fold_aucs: list[float] = []
    if fold_col is not None and fold_col in df.columns:
        fold_ids = sorted(df[fold_col].dropna().unique())
        cmap = plt.get_cmap(fold_cmap) if fold_cmap else None
        for k_idx, k in enumerate(fold_ids):
            sub = df[df[fold_col] == k]
            if sub[label_col].nunique() < 2:
                continue
            y_true = (sub[label_col] == pos_label).astype(int).values
            fpr, tpr, _ = roc_curve(y_true, sub[prob_col].values)
            auc_k = roc_auc_score(y_true, sub[prob_col].values)
            fold_aucs.append(auc_k)
            color = cmap(0.30 + k_idx * 0.13) if cmap is not None else f"C{k_idx}"
            ax.plot(fpr, tpr, color=color, linewidth=1.5, alpha=fold_alpha, label=f"Fold {k}  AUC = {auc_k:.3f}")

    if show_pooled and df[label_col].nunique() >= 2:
        y_all = (df[label_col] == pos_label).astype(int).values
        pooled_auc = roc_auc_score(y_all, df[prob_col].values)
        fpr_p, tpr_p, _ = roc_curve(y_all, df[prob_col].values)
        ax.plot(
            fpr_p, tpr_p, color=pooled_color, linewidth=pooled_lw, alpha=0.95, label=f"Pooled  AUC = {pooled_auc:.3f}"
        )

    if chance_line:
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.45)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        if fold_aucs:
            ax.set_title(f"{title}   mean AUC = {np.mean(fold_aucs):.2f} ± {np.std(fold_aucs):.2f}", pad=8, fontsize=10)
        else:
            ax.set_title(title, pad=8, fontsize=10)
    if legend:
        ax.legend(loc="lower right", fontsize=legend_fontsize)
    _save_and_show(fig, save, show)
    return ax


def _classifier_roc_from_curves(
    ax: Axes,
    curves: pd.DataFrame,
    *,
    summary: pd.DataFrame | None,
    kind: str,
    fold_color: str,
    mean_color: str,
    mean_lw: float,
    chance_line: bool,
    title: str | None,
    xlabel: str,
    ylabel: str,
    legend: bool,
    legend_fontsize: float,
) -> Axes:
    required = {"fold", "kind", "x", "y"}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"`curves` is missing required columns: {sorted(missing)}. Schema: {sorted(required)}.")

    sub = curves[curves["kind"] == kind]
    if sub.empty:
        raise ValueError(
            f"No rows with kind={kind!r} in `curves`. Available kinds: {sorted(curves['kind'].unique().tolist())}"
        )

    fold_ids = list(pd.unique(sub["fold"]))
    x_grid = np.linspace(0.0, 1.0, 201)
    stack: list[np.ndarray] = []
    per_fold_auc: dict[Any, float] = {}
    if summary is not None and "fold" in summary.columns:
        auc_col = "roc_auc" if kind == "roc" else "pr_auc"
        if auc_col in summary.columns:
            per_fold_auc = dict(zip(summary["fold"], summary[auc_col], strict=False))

    for fold in fold_ids:
        f = sub[sub["fold"] == fold]
        xs = f["x"].to_numpy(dtype=float)
        ys = f["y"].to_numpy(dtype=float)
        order = np.argsort(xs, kind="stable")
        xs = xs[order]
        ys = ys[order]
        # Dedup ties on x (keep max y at each unique x) before interpolating.
        xu: list[float] = []
        yu: list[float] = []
        last_x: float | None = None
        for xv, yv in zip(xs, ys, strict=False):
            if last_x is None or xv != last_x:
                xu.append(float(xv))
                yu.append(float(yv))
                last_x = float(xv)
            elif yv > yu[-1]:
                yu[-1] = float(yv)
        ax.plot(xs, ys, color=fold_color, lw=1.0, alpha=0.75, zorder=2)
        stack.append(np.interp(x_grid, np.asarray(xu), np.asarray(yu)))

    mean_curve = np.mean(stack, axis=0) if stack else None
    if mean_curve is not None:
        if per_fold_auc:
            mean_auc = float(np.mean(list(per_fold_auc.values())))
            mean_label = f"mean AUC = {mean_auc:.3f}  (n = {len(fold_ids)} folds)"
        else:
            # Trapezoidal AUC of the interpolated mean curve as a fallback.
            mean_auc = float(np.trapz(mean_curve, x_grid))
            mean_label = f"mean AUC = {mean_auc:.3f}"
        ax.plot(x_grid, mean_curve, color=mean_color, lw=mean_lw, zorder=4, label=mean_label)

    if chance_line:
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.7, linestyle="--", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    if legend:
        ax.legend(
            loc="lower right",
            fontsize=legend_fontsize,
            frameon=True,
            framealpha=0.95,
            edgecolor="#ccc",
        )
    return ax


# -----------------------------------------------------------------------------
# Spatial overlays
# -----------------------------------------------------------------------------


def node_value_overlay(
    ax: Axes,
    pos: np.ndarray,
    values: np.ndarray,
    *,
    edges: np.ndarray | None = None,
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    size_range: tuple[float, float] = (6, 34),
    size_by_value: bool = True,
    edge_color: str = "#aaaaaa",
    edge_alpha: float = 0.45,
    edge_lw: float = 0.35,
    node_edgecolor: str = "black",
    node_edge_lw: float = 0.2,
    node_alpha: float = 0.9,
    header: str | None = None,
    header_color: str = "white",
    header_bg: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    pad: float = 30.0,
    invert_y: bool = True,
    show_spines: bool = False,
) -> Axes:
    """Scatter ``pos`` coloured (and optionally sized) by ``values``, plus optional edges.

    Designed for per-node attribution overlays (IG, SHAP, GNN node-mask, …).
    The returned ``ax`` carries the underlying ``PathCollection`` as its last
    ``ax.collections[-1]`` entry, so a shared colorbar can be added downstream
    via ``fig.colorbar(ax.collections[-1], ax=…)``.
    """
    pos = np.asarray(pos)
    values = np.asarray(values, dtype=float)
    if edges is not None and len(edges):
        lc = LineCollection(
            np.asarray(edges).tolist(), colors=edge_color, linewidths=edge_lw, alpha=edge_alpha, zorder=1
        )
        ax.add_collection(lc)

    if size_by_value and np.nanmax(values) > np.nanmin(values):
        rng = np.nanmax(values) - np.nanmin(values)
        norm = (values - np.nanmin(values)) / rng
        sizes = norm * (size_range[1] - size_range[0]) + size_range[0]
    else:
        sizes = np.full_like(values, size_range[0])
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=values,
        cmap=cmap,
        s=sizes,
        vmin=vmin,
        vmax=vmax,
        alpha=node_alpha,
        edgecolors=node_edgecolor,
        linewidths=node_edge_lw,
        zorder=2,
    )

    if bbox is not None:
        x0, x1, y0, y1 = bbox
    else:
        x0 = pos[:, 0].min() - pad
        x1 = pos[:, 0].max() + pad
        y0 = pos[:, 1].min() - pad
        y1 = pos[:, 1].max() + pad
    ax.set_xlim(x0, x1)
    ax.set_ylim((y1, y0) if invert_y else (y0, y1))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if not show_spines:
        for sp in ax.spines.values():
            sp.set_visible(False)
    if header is not None:
        ax.set_title(
            header, color=header_color, fontsize=11, fontweight="bold", pad=4, backgroundcolor=header_bg or "#666666"
        )
    return ax


# ---------------------------------------------------------------------------
# ECM cluster centroid heatmap.
# Per-cluster mean of `feat_*` mean-feature columns, column-z-scored,
# rendered as a markers x clusters heatmap with signed-decimal annotations.
# Works on a single AnnData or pooled across a cohort.
# ---------------------------------------------------------------------------
def ecm_centroid_table(
    adata_or_adatas: AnnData | dict[str, AnnData] | list[AnnData],
    *,
    ecm_names: list[str] | None = None,
    K_ecm: int | None = None,
) -> pd.DataFrame:
    """Return the tidy values used by :func:`ecm_centroid_heatmap`.

    The returned table contains one row per ECM cluster and marker, including
    the number of contributing patches, the raw cluster mean, and the marker-
    wise z-score plotted by :func:`ecm_centroid_heatmap`.
    """
    if isinstance(adata_or_adatas, AnnData):
        adatas = [adata_or_adatas]
    elif isinstance(adata_or_adatas, dict):
        adatas = list(adata_or_adatas.values())
    else:
        adatas = list(adata_or_adatas)

    frames = []
    for adata in adatas:
        if ECM_PATCHES_KEY not in adata.uns:
            raise ValueError(
                f"Key '{ECM_PATCHES_KEY}' not found in adata.uns. "
                "Run `mt.pp.extract_ecm_patches(adata)` first."
            )
        frames.append(adata.uns[ECM_PATCHES_KEY])
    patches = pd.concat(frames, ignore_index=True)

    feat_cols = [column for column in patches.columns if column.startswith("feat_")]
    if not feat_cols:
        raise ValueError("No `feat_*` columns found in ecm_patches.")
    n_markers = len(ecm_names) if ecm_names is not None else len(feat_cols)
    if n_markers > len(feat_cols):
        raise ValueError(
            f"ecm_names has {n_markers} entries but only {len(feat_cols)} feat_* columns exist."
        )
    mean_cols = feat_cols[:n_markers]
    markers = list(ecm_names) if ecm_names is not None else mean_cols

    if K_ecm is None:
        K_ecm = int(patches["ecm_cluster"].max()) + 1
    cluster_levels = list(range(K_ecm))
    centroids = np.zeros((K_ecm, n_markers), dtype=np.float64)
    counts = np.zeros(K_ecm, dtype=np.int64)
    for cluster in cluster_levels:
        mask = patches["ecm_cluster"] == cluster
        counts[cluster] = int(mask.sum())
        if mask.any():
            centroids[cluster] = patches.loc[mask, mean_cols].mean(axis=0).values

    z_scores = (centroids - centroids.mean(axis=0)) / (centroids.std(axis=0) + 1e-9)
    records = []
    for cluster in cluster_levels:
        for marker_index, (feature, marker) in enumerate(zip(mean_cols, markers, strict=True)):
            records.append(
                {
                    "cluster": cluster,
                    "marker": marker,
                    "feature": feature,
                    "n_patches": int(counts[cluster]),
                    "mean_intensity": float(centroids[cluster, marker_index]),
                    "z_score": float(z_scores[cluster, marker_index]),
                }
            )
    return pd.DataFrame.from_records(records)


def ecm_centroid_heatmap(
    adata_or_adatas: AnnData | dict[str, AnnData] | list[AnnData],
    *,
    ecm_names: list[str] | None = None,
    K_ecm: int | None = None,
    cmap: str = "RdBu_r",
    signed_text: bool = True,
    text_fontsize: int = 8,
    label_fontsize: int = 10,
    cluster_colors: dict[int, str] | list[str] | None = None,
    cluster_label_template: str = "ECM {k}",
    rotation: int = 45,
    show_colorbar: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    save: str | Path | None = None,
    show: bool = True,
) -> Axes:
    """ECM-cluster centroid heatmap (markers x clusters, z-scored per marker).

    Pools ``adata.uns['ecm_patches']`` across one or many AnnDatas, takes the
    per-cluster mean of the ``feat_*`` mean-intensity columns (first
    ``n_markers`` ``feat_*`` columns, matching the order in ``ecm_names``),
    z-scores down columns (i.e. per marker, across clusters), and renders
    the resulting matrix with optional signed-decimal text annotations.

    Cells are square (``aspect="equal"``).  The heatmap is anchored to the
    centre of its axes so the natural wide-short proportion does not
    "stretch" against the available panel.

    Parameters
    ----------
    adata_or_adatas
        Single AnnData, list, or ``{label: AnnData}`` mapping.  Each must
        carry ``adata.uns['ecm_patches']`` with an ``ecm_cluster`` column
        and ``feat_*`` columns.
    ecm_names
        Marker names corresponding to the first ``len(ecm_names)``
        ``feat_*`` columns.  When ``None``, the column names themselves
        are used.
    K_ecm
        Number of signal clusters (0..K_ecm-1).  Inferred from the data
        when ``None``.
    cmap, signed_text, text_fontsize, label_fontsize
        Visual options.
    cluster_colors
        ``{k: hex}`` or list of hex strings used to colour each y-tick
        label.  When provided, cluster labels render in bold + the cluster
        colour to anchor them to the rest of the figure's ECM palette.
    cluster_label_template
        Format string for y-tick labels, e.g. ``"ECM {k}"``.
    rotation
        Rotation of x-tick (marker) labels.
    show_colorbar
        Append a thin colorbar to the right of the heatmap.
    ax, save, show
        Standard mantpy plotting controls.

    Returns
    -------
    matplotlib Axes
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    table = ecm_centroid_table(
        adata_or_adatas,
        ecm_names=ecm_names,
        K_ecm=K_ecm,
    )
    markers = list(dict.fromkeys(table["marker"].tolist()))
    cluster_levels = list(dict.fromkeys(table["cluster"].astype(int).tolist()))
    n_markers = len(markers)
    z = (
        table.pivot(index="cluster", columns="marker", values="z_score")
        .reindex(index=cluster_levels, columns=markers)
        .to_numpy(dtype=float)
    )
    z_lim = float(np.max(np.abs(z))) if z.size else 1.0

    # Heatmap is wide-and-short by nature; the module default (8, 7) leaves
    # it swimming in whitespace, which is why callers reached for plt.subplots.
    fig, ax = _get_ax(ax, figsize=figsize or (4.2, 3.0))
    im = ax.imshow(z, aspect="equal", cmap=cmap, vmin=-z_lim, vmax=z_lim)
    ax.set_anchor("C")

    ax.set_xticks(range(n_markers))
    ax.set_xticklabels(markers, rotation=rotation, ha="right", fontsize=label_fontsize)
    ax.set_yticks(cluster_levels)
    ax.set_yticklabels(
        [cluster_label_template.format(k=k) for k in cluster_levels],
        fontsize=label_fontsize,
    )

    if cluster_colors is not None:
        if isinstance(cluster_colors, dict):
            ck = lambda k: cluster_colors.get(k, "#000000")
        else:
            ck = lambda k: cluster_colors[k % len(cluster_colors)]
        for k in cluster_levels:
            ax.get_yticklabels()[k].set_color(ck(k))
            ax.get_yticklabels()[k].set_fontweight("bold")

    if signed_text:
        _cmap_fn = plt.get_cmap(cmap)
        for i in cluster_levels:
            for j in range(n_markers):
                v = float(z[i, j])
                # Sample the actual rendered colour to pick legible text colour.
                _rgba = _cmap_fn((v + z_lim) / (2 * z_lim + 1e-12))
                _lum = 0.2126 * _rgba[0] + 0.7152 * _rgba[1] + 0.0722 * _rgba[2]
                ax.text(
                    j,
                    i,
                    f"{v:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=text_fontsize,
                    color="white" if _lum < 0.45 else "black",
                )

    ax.set_xticks(np.arange(n_markers + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(K_ecm + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    if show_colorbar:
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.05)
        _cbar = plt.colorbar(im, cax=cax, label="z")
        _cbar.ax.tick_params(labelsize=label_fontsize)
        _cbar.set_label("z", fontsize=label_fontsize)
    if title is not None:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    _save_and_show(fig, save, show)
    return ax


def cross_compartment_ablation_bars(
    results: pd.DataFrame,
    *,
    order: Sequence[str] | None = None,
    model_col: str = "model",
    metric_col: str = "roc_auc",
    colors: dict[str, str] | Sequence[str] | None = None,
    chance: float | None = 0.5,
    jitter: bool = True,
    random_state: int | None = 0,
    ylabel: str = "Artefact-detection ROC-AUC",
    ylim: tuple[float, float] | None = (0.3, 1.0),
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    save: str | Path | None = None,
    show: bool = True,
) -> Axes:
    """Plot fold-level results from ``tl.cross_compartment_ablation``.

    Bars show the mean, error bars the between-fold standard deviation, and
    optional jittered points retain every fold. Column names are configurable,
    so the helper also works for other tidy model-by-fold metric tables.
    """
    for column in (model_col, metric_col):
        if column not in results:
            raise KeyError(f"Column {column!r} not found in results.")
    models = list(dict.fromkeys(results[model_col].astype(str))) if order is None else list(order)
    missing = [model for model in models if model not in set(results[model_col].astype(str))]
    if missing:
        raise ValueError(f"Models {missing} do not occur in results[{model_col!r}].")

    if colors is None:
        palette = {model: _ECM_PALETTE[i % len(_ECM_PALETTE)] for i, model in enumerate(models)}
    elif isinstance(colors, dict):
        palette = {model: colors.get(model, "#888888") for model in models}
    else:
        color_list = list(colors)
        if not color_list:
            raise ValueError("colors is empty; supply at least one colour.")
        palette = {model: color_list[i % len(color_list)] for i, model in enumerate(models)}

    default_figsize = (max(3.2, 0.75 * len(models) + 1.2), 3.2)
    fig, ax = _get_ax(ax, figsize=figsize or default_figsize)
    rng = np.random.default_rng(random_state)
    model_values = results[model_col].astype(str)
    for i, model in enumerate(models):
        values = results.loc[model_values == model, metric_col].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        color = palette[model]
        ax.bar(i, mean, width=0.62, color=color, alpha=0.6, edgecolor=color)
        ax.errorbar(i, mean, yerr=std, color=color, capsize=3, lw=1.2)
        if jitter:
            x = np.full(values.size, i, dtype=float) + rng.uniform(-0.12, 0.12, values.size)
            ax.scatter(x, values, s=14, color=color, edgecolor="white", linewidth=0.4, zorder=3)
    if chance is not None:
        ax.axhline(chance, ls=":", lw=0.9, color="0.5")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title is not None:
        ax.set_title(title, loc="left", fontsize=9)
    _save_and_show(fig, save, show)
    return ax


def cell_ecm_enrichment_per_roi(
    data: AnnData | pd.DataFrame,
    *,
    key: str = "cell_ecm_enrichment_per_roi",
    cluster_colors: dict[int, str] | Sequence[str] | None = None,
    cluster_labels: dict[int, str] | Sequence[str] | None = None,
    sample_labels: dict[str, str] | Sequence[str] | str | None = None,
    xtick_rotation: float = 0,
    xlabel: str = "Sample",
    ylabel: str = "log2(observed / expected)",
    title: str | None = None,
    figsize: tuple[float, float] = (4.4, 3.0),
    ax: Axes | None = None,
    save: str | Path | None = None,
    show: bool = True,
) -> Axes:
    """Plot the per-sample table stored by ``tl.cell_ecm_enrichment``."""
    if isinstance(data, AnnData):
        if key not in data.uns:
            raise KeyError(f"adata.uns[{key!r}] is missing. Run mt.tl.cell_ecm_enrichment first.")
        table = data.uns[key]
    else:
        table = data
    required = {"roi", "cluster", "log2_enr"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Per-ROI enrichment table is missing columns {sorted(missing)}.")
    pivot = table.pivot(index="roi", columns="cluster", values="log2_enr")
    clusters = [int(cluster) for cluster in pivot.columns]

    fig, ax = _get_ax(ax, figsize=figsize)
    for cluster in clusters:
        if cluster_colors is None:
            color = _ECM_PALETTE[cluster % len(_ECM_PALETTE)]
        elif isinstance(cluster_colors, dict):
            color = cluster_colors.get(cluster, "#888888")
        else:
            color = cluster_colors[cluster % len(cluster_colors)]
        if cluster_labels is None:
            label = f"ECM {cluster}"
        elif isinstance(cluster_labels, dict):
            label = str(cluster_labels.get(cluster, f"ECM {cluster}"))
        else:
            label = str(cluster_labels[cluster]) if cluster < len(cluster_labels) else f"ECM {cluster}"
        ax.plot(range(len(pivot)), pivot[cluster].to_numpy(), "o-", color=color, label=label)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(pivot)))
    if sample_labels is None:
        tick_labels = pivot.index.astype(str).tolist()
    elif isinstance(sample_labels, str):
        if sample_labels != "suffix":
            raise ValueError("String sample_labels must be 'suffix'.")
        tick_labels = [str(sample).rsplit("_", 1)[-1] for sample in pivot.index]
    elif isinstance(sample_labels, dict):
        tick_labels = [sample_labels.get(str(sample), str(sample)) for sample in pivot.index]
    else:
        tick_labels = list(sample_labels)
        if len(tick_labels) != len(pivot):
            raise ValueError("sample_labels must have one entry per plotted sample.")
    ax.set_xticklabels(
        tick_labels,
        rotation=xtick_rotation,
        ha="right" if xtick_rotation else "center",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, frameon=False)
    if title is not None:
        ax.set_title(title, loc="left", fontsize=9)
    _save_and_show(fig, save, show)
    return ax


def ecm_cluster_comparison(
    cell_adata: AnnData,
    ecm_adata: AnnData,
    *,
    graph_kwargs: dict[str, Any] | None = None,
    cluster_key: str = "ecm_cluster",
    denoised_key: str = "denoised_cluster",
    palette: dict[int, str] | Sequence[str] | None = None,
    titles: tuple[str, str] = ("Corrupted input", "Denoised"),
    marker_size: float = 1.4,
    artifact_boxes: list[dict[str, Any]] | None = None,
    figsize: tuple[float, float] = (7.4, 3.6),
    axes: Sequence[Axes] | None = None,
    save: str | Path | None = None,
    show: bool = True,
) -> list[Axes]:
    """Compare an ECM cluster graph before and after denoising.

    ``ecm_adata`` must contain both the observed and denoised patch labels.
    The cleaned graph is rebuilt on a copy with ``graph_kwargs``; neither input
    object is modified. Artefact boxes are read from ``ecm_adata.uns`` unless
    explicitly supplied.
    """
    patches = ecm_adata.uns.get(ECM_PATCHES_KEY)
    if patches is None:
        raise KeyError(f"ecm_adata.uns[{ECM_PATCHES_KEY!r}] is missing.")
    for column in (cluster_key, denoised_key):
        if column not in patches:
            raise KeyError(f"Patch column {column!r} is missing.")

    corrupted = cell_adata.copy()
    corrupted.uns[ECM_PATCHES_KEY] = patches.copy()
    corrupted.uns[ECM_PATCHES_KEY][cluster_key] = patches[cluster_key].astype(int).to_numpy()
    cleaned = cell_adata.copy()
    cleaned.uns[ECM_PATCHES_KEY] = patches.copy()
    cleaned.uns[ECM_PATCHES_KEY][cluster_key] = patches[denoised_key].astype(int).to_numpy()

    from mantpy.gr import ensure_cell_ecm_graph

    recipe = dict(graph_kwargs or {})
    recipe["rebuild"] = True
    ensure_cell_ecm_graph(corrupted, **recipe)
    ensure_cell_ecm_graph(cleaned, **recipe)

    if axes is None:
        fig, axes_array = plt.subplots(1, 2, figsize=figsize)
        axes_list = list(axes_array)
    else:
        if len(axes) != 2:
            raise ValueError(f"axes must contain two axes (got {len(axes)}).")
        axes_list = list(axes)
        fig = axes_list[0].figure
    panel_kwargs = {"ecm": {"ax_off": True, "node_marker": "s", "palette": palette}}
    for adata, ax, title in zip((corrupted, cleaned), axes_list, titles, strict=True):
        graph_triptych(
            adata,
            panels=("ecm",),
            titles=(title,),
            axes=[ax],
            node_size=marker_size,
            edge_alpha=0.85,
            edge_width=0.4,
            node_linewidth=0.25,
            panel_kwargs=panel_kwargs,
            show=False,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    boxes = ecm_adata.uns.get("artifact_boxes") if artifact_boxes is None else artifact_boxes
    if boxes:
        from matplotlib.patches import Rectangle

        for ax in axes_list:
            for box in boxes:
                cx, cy = box["center"]
                half = box.get("half", 50)
                ax.add_patch(
                    Rectangle(
                        (cx - half, cy - half),
                        2 * half,
                        2 * half,
                        fill=False,
                        edgecolor="black",
                        lw=1.0,
                        zorder=6,
                    )
                )
    _save_and_show(fig, save, show)
    return axes_list


def cell_ecm_enrichment_heatmap(
    matrix_df: pd.DataFrame,
    *,
    ax: Axes | None = None,
    cell_type_col: str = "cell_type",
    cluster_col: str = "cluster",
    value_col: str = "log2_enr",
    qvalue_col: str = "p_fdr",
    cluster_labels: dict[int, str] | list[str] | None = None,
    order_by_cluster: int | None = None,
    highlight_rows: str | list[str] | None = None,
    highlight_color: str = "#D55E00",
    cmap: str = "RdBu_r",
    vmax: float = 4.0,
    annotate: bool = True,
    star_thresholds: tuple[float, float, float] = (1e-3, 1e-2, 0.05),
    cbar: bool = True,
    xtick_rotation: float = 0,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
    show: bool = True,
) -> Axes:
    """Cell-type x ECM-cluster log2-enrichment heatmap with BH-FDR stars.

    Consumes the tidy long frame returned by
    :func:`mantpy.tl.cell_ecm_enrichment_matrix` (columns ``cell_type``,
    ``cluster``, ``log2_enr``, ``p_fdr``) and renders the full cell<->ECM
    association landscape as a diverging heatmap: each cell shows the
    ``log2(observed/expected)`` enrichment, annotated with its effect size
    and significance stars (``***`` ``**`` ``*`` from ``star_thresholds``).

    This is the matrix counterpart of :func:`cell_ecm_enrichment_bars`, which
    plots a single focal cell type's bar chart from ``adata.uns``.

    Parameters
    ----------
    matrix_df
        Tidy frame from :func:`mantpy.tl.cell_ecm_enrichment_matrix`.
    ax
        Existing axes to draw on; a new figure is made when ``None``.
    cell_type_col, cluster_col, value_col, qvalue_col
        Column names in ``matrix_df``.
    cluster_labels
        Optional pretty x-tick labels per cluster (``dict`` keyed by cluster
        int, or a ``list`` indexed by cluster). Defaults to ``ECM {k}``.
    order_by_cluster
        If given, sort the cell-type rows by descending enrichment in that
        ECM cluster (e.g. ``1`` to lead with the basement-membrane niche).
        ``None`` keeps the frame's row order.
    highlight_rows
        Cell-type name(s) whose y-tick label is drawn bold in
        ``highlight_color`` (e.g. ``'AEC'``).
    highlight_color
        Colour for the highlighted row label(s).
    cmap, vmax
        Diverging colormap and symmetric colour limit (``+/- vmax``); values
        are clipped to ``[-vmax, vmax]`` for display.
    annotate
        Write ``log2`` value + stars in each cell.
    star_thresholds
        Three ascending q cut-offs for ``***`` / ``**`` / ``*``.
    cbar
        Draw the colorbar.
    title, save, show
        Standard mantpy plotting controls.

    Returns
    -------
    matplotlib Axes.

    Examples
    --------
    >>> import mantpy as mt  # doctest: +SKIP
    >>> mat = mt.tl.cell_ecm_enrichment_matrix(cohort, K_ecm=3)  # doctest: +SKIP
    >>> mt.pl.cell_ecm_enrichment_heatmap(mat, order_by_cluster=1, highlight_rows="AEC")  # doctest: +SKIP
    """
    for col in (cell_type_col, cluster_col, value_col):
        if col not in matrix_df.columns:
            raise KeyError(
                f"column {col!r} not found in matrix_df. Available: "
                f"{list(matrix_df.columns)}. Pass the output of "
                "mt.tl.cell_ecm_enrichment_matrix."
            )

    clusters = sorted(int(c) for c in matrix_df[cluster_col].unique())
    L = matrix_df.pivot(index=cell_type_col, columns=cluster_col, values=value_col).reindex(columns=clusters)
    if qvalue_col in matrix_df.columns:
        Q = matrix_df.pivot(index=cell_type_col, columns=cluster_col, values=qvalue_col).reindex(columns=clusters)
    else:
        Q = None

    if order_by_cluster is not None and order_by_cluster in L.columns:
        order = L[order_by_cluster].fillna(-np.inf).sort_values(ascending=False).index
        L = L.reindex(order)
        if Q is not None:
            Q = Q.reindex(order)

    cell_rows = list(L.index)
    Lv = L.to_numpy(dtype=float)
    Qv = Q.to_numpy(dtype=float) if Q is not None else None

    if cluster_labels is None:
        xlabels = [f"ECM {k}" for k in clusters]
    elif isinstance(cluster_labels, dict):
        xlabels = [cluster_labels.get(k, f"ECM {k}") for k in clusters]
    else:
        xlabels = [cluster_labels[k] if k < len(cluster_labels) else f"ECM {k}" for k in clusters]

    h = 0.34 * len(cell_rows) + 1.2
    default_figsize = (0.9 * len(clusters) + 2.0, h)
    fig, ax = _get_ax(ax, figsize=figsize or default_figsize)

    im = ax.imshow(np.clip(Lv, -vmax, vmax), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(xlabels, fontsize=8, rotation=xtick_rotation, ha="right" if xtick_rotation else "center")
    ax.set_yticks(range(len(cell_rows)))
    ax.set_yticklabels(cell_rows, fontsize=8)

    if highlight_rows is not None:
        targets = {highlight_rows} if isinstance(highlight_rows, str) else set(highlight_rows)
        for lbl in ax.get_yticklabels():
            if lbl.get_text() in targets:
                lbl.set_color(highlight_color)
                lbl.set_fontweight("bold")

    if annotate:
        t1, t2, t3 = star_thresholds
        for i in range(len(cell_rows)):
            for j in range(len(clusters)):
                v = Lv[i, j]
                if not np.isfinite(v):
                    continue
                star = ""
                if Qv is not None and np.isfinite(Qv[i, j]):
                    q = Qv[i, j]
                    star = "***" if q < t1 else ("**" if q < t2 else ("*" if q < t3 else ""))
                shade = abs(np.clip(v, -vmax, vmax)) / vmax
                ax.text(
                    j,
                    i,
                    f"{v:.1f}{star}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if shade > 0.55 else "black",
                )

    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label(r"$\log_2$(observed / expected)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    if title is not None:
        ax.set_title(title, fontsize=9, loc="left")

    _save_and_show(fig, save, show)
    return ax


# ----------------------------------------------------------------------
# Re-export plotting helpers so users discover them on `mt.pl.<name>`.
# Kept at end-of-file (E402) to preserve the package import order.
# ----------------------------------------------------------------------
from mantpy._plot_helpers import (  # noqa: E402  imported after core plotters
    cell_ecm_enrichment_bars,
    channel_overlay_on_neighbours,
    niche_bubble,
    niche_bubble_table,
    patch_domain_map,
    plot_cluster_map,
    plot_delta_masked,
    plot_lesion_metric_view,
    plot_marker_otsu_composite,
    plot_mean_composition,
)
