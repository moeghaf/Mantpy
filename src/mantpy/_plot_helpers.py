"""Reusable composite plotting helpers.

These functions are exposed under ``mantpy.pl`` (see the ``__init__`` of
``pl.py`` for re-export).  They live in a separate module to keep ``pl.py``
focused on small, single-purpose plotters.

- :func:`cell_ecm_enrichment_bars` -- bar chart of log2(obs/exp) per ECM
  cluster with BH-FDR significance stars, paired with
  :func:`mantpy.tl.cell_ecm_enrichment`.
- :func:`niche_bubble` -- radial bubble plot of marker proportions in
  ECM patches that a chosen cell type touches via the cell-ECM graph.
- :func:`channel_overlay_on_neighbours` -- IMC raw-intensity overlay
  restricted to ECM patches connected to a target cell type via the
  cell-ECM graph, with target-cell scatter on top.

Import-order note
-----------------
This module imports ``mantpy.gr`` and ``mantpy.tl`` at module load (for the
``plot_lesion_metric_view`` panel).  ``pl.py`` then transitively imports
this module at the bottom of its own file.  ``mantpy/__init__.py`` must
therefore load ``gr`` and ``tl`` **before** ``pl``; reordering would
trigger a circular-import failure at package import time.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.image import AxesImage
from matplotlib.patches import Circle
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mantpy._constants import (
    CELL_ECM_GRAPH_KEY,
    ECM_PATCHES_KEY,
    NODE_TYPE_CELL,
    NODE_TYPE_ECM,
)
from mantpy._palette import ECM_CLUSTERS
from mantpy.gr import largest_component_radius_reconnect as _largest_component_radius_reconnect
from mantpy.im import as_image_container as _as_image_container
from mantpy.tl import lesion_central_void as _lesion_central_void

# Neutral chrome colours for composite panels. Kept here (not in
# ``mantpy._palette``) because they are not semantic palette anchors —
# they are layout-only greys for backgrounds, grids, spines, and tick
# annotations.
_NEUTRAL_BG_DARK = "#0d0d0d"  # plot_cluster_map facecolor
_NEUTRAL_BG_LIGHT = "#ffffff"  # plot_lesion_metric_view facecolor + delta-mask "bad" colour
_NEUTRAL_EDGE = "#ffffff"  # cluster-map white edge layer
_NEUTRAL_OUTLIER = "#444444"  # -1 / unclassified scatter, central-void cores
_NEUTRAL_GRID = "#bbbbbb"  # lesion subgraph edges
_NEUTRAL_FAINT = "#cccccc"  # below-max-k-core dots, background scatter
_NEUTRAL_SPINE = "#666666"  # panel spines on dark / light backgrounds
_NEUTRAL_VOID_FILL = "#BBBBBB"  # filled-triangle alpha-shape interior

_DEFAULT_ECM_PALETTE = [
    "#1B9E77",
    "#D95F02",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
    "#666666",
]

# High-contrast distinct hues for an N-marker composite (so markers stand out and
# co-localised abundant markers do not all blend to yellow).
_OTSU_DEFAULT_COLORS = tuple(ECM_CLUSTERS[1:])


def _get_ax(ax, figsize=(8, 7)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _save_and_show(fig, save, show):
    if save is not None:
        p = Path(save)
        if p.suffix:
            fig.savefig(str(p), bbox_inches="tight", dpi=300)
        else:
            fig.savefig(str(p.with_suffix(".pdf")), bbox_inches="tight")
            fig.savefig(str(p.with_suffix(".png")), bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    # show=False leaves the figure open — see the same pattern in pl.py.


def _ecm_node_index(node_id, node_attrs: dict | None = None) -> int | None:
    """Extract the integer ECM patch index from a node identifier.

    Mantpy graph builders use ``"ecm_<idx>"`` as the node key; ``<idx>``
    is the row index into ``adata.uns['ecm_patches']``.  Falls back to an
    explicit ``idx`` attribute in ``node_attrs`` when present.
    """
    try:
        return int(str(node_id).split("_")[-1])
    except (ValueError, IndexError, AttributeError):
        pass
    if node_attrs is not None:
        idx = node_attrs.get("idx")
        if idx is not None:
            try:
                return int(idx)
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# log2 enrichment bars with FDR significance stars
# ---------------------------------------------------------------------------
def cell_ecm_enrichment_bars(
    adata: AnnData,
    *,
    key: str = "cell_ecm_enrichment",
    alpha: float = 0.05,
    cluster_colors=None,
    cluster_labels: dict[int, str] | list[str] | None = None,
    banner_label=None,
    banner_color: str = "#d33232",
    banner_text_color: str = "white",
    title: str | None = None,
    annotate: str = "stars",
    qfloor: float = 2e-4,
    star_fontsize: float = 14,
    effect_fontsize: float = 8,
    tick_fontsize: float = 11,
    tick_rotation: float = 0,
    ylabel_fontsize: float = 11,
    banner_fontsize: float = 13,
    title_fontsize: float = 12,
    figsize: tuple[float, float] = (8, 7),
    ax: Axes | None = None,
    ax_banner: Axes | None = None,
    save=None,
    show: bool = True,
) -> Axes:
    """Bar chart of log2(observed / expected) ECM-cluster enrichment for a focal cell type.

    Reads ``adata.uns[key]`` written by ``mt.tl.cell_ecm_enrichment``.
    The ``annotate`` switch chooses the bar annotation style:

    - ``'stars'`` (default) — BH-FDR stars: ``***`` p<0.001, ``**`` p<0.01,
      ``*`` p<alpha, ``ns`` otherwise. Backwards-compatible.
    - ``'effect'`` prints only the signed log2 effect size, with no
      significance threshold, p/q value or star annotation.
    - ``'effect_q'`` — the signed log2 effect size and floored permutation q
      (e.g. ``+2.1`` / ``q<2e-4``), avoiding floor-saturated stars when ``n_perm`` makes every
      significant bar hit the same ``***``.

    Parameters
    ----------
    adata
        AnnData carrying the enrichment table at ``adata.uns[key]``.
    key
        ``adata.uns`` key for the enrichment DataFrame.
    alpha
        BH-FDR threshold for significance stars / the ``effect_q`` colour
        cut.
    annotate
        ``'stars'`` (default), ``'effect'`` (signed effect only) or
        ``'effect_q'`` (signed effect + floored q).
    qfloor
        The permutation q-value floor printed as ``q<{qfloor}`` in the
        ``'effect_q'`` style (default ``2e-4``, i.e. ``n_perm=5000``).
    cluster_colors
        Per-cluster bar colour map (``dict`` or ``list``).
    cluster_labels
        Optional pretty x-tick labels per cluster (``dict`` keyed by cluster
        int, or a ``list`` indexed by cluster). Defaults to ``ECM {k}``.
        Mirrors :func:`mantpy.pl.cell_ecm_enrichment_heatmap`.
    banner_label
        Optional bold label rendered in a coloured banner above the bars
        (e.g. the focal cell type name).  Falls back to no banner.
    banner_color, banner_text_color
        Background / foreground colours for the optional banner.
    ax_banner
        Existing banner axes; auto-created above ``ax`` when ``None``.
    star_fontsize, effect_fontsize
        Annotation font sizes for the ``'stars'`` and effect-label styles,
        respectively.
    ax, save, show
        Standard mantpy plotting controls.

    Returns
    -------
    matplotlib Axes (the bar-chart axes).
    """
    if annotate not in ("stars", "effect", "effect_q"):
        raise ValueError(f"annotate={annotate!r}: choose 'stars', 'effect' or 'effect_q'.")
    if key not in adata.uns:
        raise ValueError(
            "Key '" + key + "' not found in adata.uns. Run mt.tl.cell_ecm_enrichment(adata, cell_type=...) first."
        )
    df = adata.uns[key]
    K = len(df)
    obs_enr = df["log2_enr"].to_numpy()
    if annotate in ("stars", "effect_q") and "p_fdr" not in df:
        raise ValueError(f"annotate={annotate!r} requires a 'p_fdr' column.")
    p_fdr = df["p_fdr"].to_numpy() if "p_fdr" in df else np.full(K, np.nan)

    if cluster_colors is None:
        colors = [_DEFAULT_ECM_PALETTE[k % len(_DEFAULT_ECM_PALETTE)] for k in range(K)]
    elif isinstance(cluster_colors, dict):
        colors = [cluster_colors.get(k, _DEFAULT_ECM_PALETTE[k % len(_DEFAULT_ECM_PALETTE)]) for k in range(K)]
    else:
        colors = [cluster_colors[k % len(cluster_colors)] for k in range(K)]

    fig, ax = _get_ax(ax, figsize=figsize)
    ax.bar(np.arange(K), obs_enr, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", lw=0.7)
    y_max = float(max(abs(obs_enr.min()), abs(obs_enr.max()), 1.0))
    for k in range(K):
        up = obs_enr[k] >= 0
        if annotate == "effect":
            label = f"{obs_enr[k]:+.1f}"
            y_text = obs_enr[k] + (0.12 * y_max if up else -0.12 * y_max)
            ax.text(
                k,
                y_text,
                label,
                ha="center",
                va="bottom" if up else "top",
                fontsize=effect_fontsize,
                color="black",
            )
            continue
        if annotate == "effect_q":
            # Signed effect plus floored permutation q-value.
            # A q at or below the floor (permutation-saturated) prints
            # "q<{floor}" in compact scientific form, matching draw_aec_bars.
            qk = p_fdr[k]
            floor_txt = f"{qfloor:.0e}".replace("e-0", "e-").replace("e+0", "e+")
            if qk <= qfloor:
                qtxt = f"q<{floor_txt}"
            elif qk < alpha:
                qtxt = f"q={qk:.1e}"
            else:
                qtxt = "ns"
            label = f"{obs_enr[k]:+.1f}\n{qtxt}"
            y_text = obs_enr[k] + (0.12 * y_max if up else -0.12 * y_max)
            ax.text(
                k,
                y_text,
                label,
                ha="center",
                va="bottom" if up else "top",
                fontsize=effect_fontsize,
                color="black" if qk < alpha else "#999999",
            )
            continue

        if p_fdr[k] < 0.001:
            stars = "***"
        elif p_fdr[k] < 0.01:
            stars = "**"
        elif p_fdr[k] < alpha:
            stars = "*"
        else:
            stars = "ns"
        y_text = obs_enr[k] + (0.15 * y_max if up else -0.30 * y_max)
        ax.text(
            k,
            y_text,
            stars,
            ha="center",
            va="center",
            fontsize=star_fontsize,
            fontweight="bold",
            color="black" if stars != "ns" else "#777777",
        )
    if cluster_labels is None:
        xtick_labels = ["ECM " + str(k) for k in range(K)]
    elif isinstance(cluster_labels, dict):
        xtick_labels = [str(cluster_labels.get(k, "ECM " + str(k))) for k in range(K)]
    else:
        xtick_labels = [str(cluster_labels[k]) if k < len(cluster_labels) else "ECM " + str(k) for k in range(K)]
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(xtick_labels, fontsize=tick_fontsize, fontweight="bold")
    if tick_rotation:
        plt.setp(ax.get_xticklabels(), rotation=tick_rotation, ha="right", rotation_mode="anchor")
    for k in range(K):
        ax.get_xticklabels()[k].set_color(colors[k])
    ax.set_ylabel("log$_2$ (observed / expected)", fontsize=ylabel_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_ylim(-y_max * 1.30, y_max * 1.30)
    ax.grid(axis="y", alpha=0.25)
    for s in ax.spines.values():
        s.set_edgecolor("#666666")

    if banner_label is not None and ax_banner is None:
        ax_banner = inset_axes(
            ax,
            width="100%",
            height="9%",
            loc="lower center",
            bbox_to_anchor=(0, 1.02, 1, 0.10),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
    if ax_banner is not None and banner_label is not None:
        ax_banner.set_facecolor(banner_color)
        ax_banner.set_xticks([])
        ax_banner.set_yticks([])
        for s in ax_banner.spines.values():
            s.set_visible(True)
            s.set_edgecolor("black")
            s.set_linewidth(0.8)
        ax_banner.text(
            0.5,
            0.5,
            str(banner_label),
            transform=ax_banner.transAxes,
            ha="center",
            va="center",
            fontsize=banner_fontsize,
            fontweight="bold",
            color=banner_text_color,
        )
    if title is not None:
        ax.set_title(title, loc="left", fontsize=title_fontsize, fontweight="bold")
    _save_and_show(fig, save, show)
    return ax


# ---------------------------------------------------------------------------
# Radial niche bubble and its exact source-data table
# ---------------------------------------------------------------------------
def niche_bubble_table(
    adata_or_adatas,
    *,
    cell_type: str,
    focus_cluster: int,
    marker_names=None,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    cell_attr: str = "cell_type",
) -> pd.DataFrame:
    """Return marker intensities and proportions used by :func:`niche_bubble`.

    Values are pooled over cell--ECM edges connecting ``cell_type`` cells to
    ECM patches assigned to ``focus_cluster``. The edge count is repeated on
    each marker row so source-data exports retain the contributing unit count.
    """
    if isinstance(adata_or_adatas, AnnData):
        adatas = [adata_or_adatas]
    elif isinstance(adata_or_adatas, dict):
        adatas = list(adata_or_adatas.values())
    else:
        adatas = list(adata_or_adatas)

    intens_sum = None
    weight = 0
    mean_cols_ref = None
    for adata in adatas:
        if graph_key not in adata.uns or ECM_PATCHES_KEY not in adata.uns:
            continue
        graph = adata.uns[graph_key]
        patches = adata.uns[ECM_PATCHES_KEY]
        feat_cols = [column for column in patches.columns if column.startswith("feat_")]
        n_markers = len(marker_names) if marker_names is not None else len(feat_cols)
        if n_markers > len(feat_cols):
            raise ValueError(f"marker_names has {n_markers} entries but only {len(feat_cols)} feat_* columns exist.")
        mean_cols = feat_cols[:n_markers]
        if mean_cols_ref is None:
            mean_cols_ref = mean_cols
        elif mean_cols != mean_cols_ref:
            raise ValueError("All inputs must expose the same ordered ECM feature columns.")
        if intens_sum is None:
            intens_sum = np.zeros(n_markers, dtype=np.float64)
        attrs = dict(graph.nodes(data=True))
        for u, v in graph.edges():
            attrs_u, attrs_v = attrs.get(u, {}), attrs.get(v, {})
            type_u = attrs_u.get("node_type")
            type_v = attrs_v.get("node_type")
            if type_u == NODE_TYPE_CELL and type_v == NODE_TYPE_ECM:
                cell_node, ecm_node, ecm_id = attrs_u, attrs_v, v
            elif type_v == NODE_TYPE_CELL and type_u == NODE_TYPE_ECM:
                cell_node, ecm_node, ecm_id = attrs_v, attrs_u, u
            else:
                continue
            if str(cell_node.get(cell_attr, "")) != str(cell_type):
                continue
            cluster = ecm_node.get("ecm_cluster", -1)
            try:
                cluster = int(cluster)
            except (TypeError, ValueError):
                continue
            if cluster != int(focus_cluster):
                continue
            index = _ecm_node_index(ecm_id, ecm_node)
            if index is None or not (0 <= index < len(patches)):
                continue
            values = patches.iloc[index][mean_cols].values.astype(float)
            intens_sum += np.clip(values, 0, None)
            weight += 1

    if intens_sum is None or weight == 0 or mean_cols_ref is None:
        raise ValueError("No cell-ECM edges from " + str(cell_type) + " into cluster " + str(focus_cluster) + ".")

    mean_intensities = intens_sum / weight
    proportions = mean_intensities / max(mean_intensities.sum(), 1e-9)
    markers = list(marker_names) if marker_names is not None else mean_cols_ref
    return pd.DataFrame(
        {
            "cell_type": str(cell_type),
            "focus_cluster": int(focus_cluster),
            "marker": markers,
            "feature": mean_cols_ref,
            "n_edges": int(weight),
            "mean_intensity": mean_intensities,
            "proportion": proportions,
        }
    )


def niche_bubble(
    adata_or_adatas,
    *,
    cell_type: str,
    focus_cluster: int,
    marker_names=None,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    cell_attr: str = "cell_type",
    cmap: str = "Greens",
    centre_color: str = "#d33232",
    centre_label=None,
    bubble_r_min: float = 0.05,
    bubble_r_max: float = 0.25,
    ring_radius: float = 1.0,
    cmap_lo: float = 0.30,
    cmap_hi: float = 0.95,
    show_colorbar: bool = True,
    title: str | None = None,
    centre_radius: float = 0.30,
    centre_fontsize: float = 14,
    label_fontsize: float = 9,
    cbar_label_fontsize: float = 10,
    cbar_tick_fontsize: float = 8,
    title_fontsize: float = 12,
    xlim: float = 1.9,
    figsize: tuple[float, float] = (8, 7),
    ax: Axes | None = None,
    save=None,
    show: bool = True,
) -> Axes:
    """Radial bubble plot of marker proportions around a focal cell type within a focus ECM cluster.

    For each ECM patch of cluster ``focus_cluster`` that ``cell_type`` cells
    touch via the cell-ECM graph, mean per-marker intensities are pooled,
    converted to proportions, and rendered as bubbles arranged in a circle
    around a central node labelled with the cell type.  Bubble radius
    encodes proportion; fill colour samples a sequential cmap.

    Parameters
    ----------
    adata_or_adatas
        AnnData, list, or dict carrying ``cell_ecm_graph`` + ``ecm_patches``.
    cell_type
        Focal cell type (matched on ``cell_attr``).
    focus_cluster
        ECM cluster index to focus on.
    marker_names
        Marker labels for the first ``len(marker_names)`` ``feat_*`` mean
        columns.  ``None`` falls back to column names.
    cmap, centre_color, centre_label
        Visual controls.  ``centre_label`` defaults to ``cell_type``.
    bubble_r_min, bubble_r_max, ring_radius
        Geometry of the radial arrangement (axes coords).
    cmap_lo, cmap_hi
        Lower / upper fraction of the colormap to sample bubble colours
        from (avoids the very-light and very-dark ends).
    show_colorbar
        Append a thin colorbar to the right of the axes.
    ax, save, show
        Standard mantpy plotting controls.

    Returns
    -------
    matplotlib Axes
    """
    table = niche_bubble_table(
        adata_or_adatas,
        cell_type=cell_type,
        focus_cluster=focus_cluster,
        marker_names=marker_names,
        graph_key=graph_key,
        cell_attr=cell_attr,
    )
    eps = 1e-9
    proportions = table["proportion"].to_numpy(dtype=float)
    n_m = len(proportions)
    marker_names = table["marker"].astype(str).tolist()
    centre_label = centre_label if centre_label is not None else str(cell_type)

    fig, ax = _get_ax(ax, figsize=figsize)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-xlim, xlim)
    ax.set_aspect("equal")
    ax.axis("off")
    order = np.argsort(proportions)[::-1]
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n_m, endpoint=False)
    max_p = max(proportions.max(), eps)
    bubble_r = bubble_r_min + (bubble_r_max - bubble_r_min) * (proportions / max_p) ** 0.7
    if isinstance(cmap, str) and mcolors.is_color_like(cmap):
        cmap_fn = mcolors.LinearSegmentedColormap.from_list("mantpy_niche", ["#ffffff", cmap])
    else:
        cmap_fn = plt.get_cmap(cmap)
    for ang in angles:
        x = ring_radius * np.cos(ang)
        y = ring_radius * np.sin(ang)
        ax.plot([0, x], [0, y], color="#aaaaaa", lw=1.0, alpha=0.5, zorder=1)
    centre_r = centre_radius
    ax.add_patch(Circle((0, 0), centre_r, facecolor=centre_color, edgecolor="black", linewidth=1.6, zorder=3))
    ax.text(0, 0, centre_label, ha="center", va="center", fontsize=centre_fontsize, fontweight="bold", color="white")
    for ang, idx, br in zip(angles, order, bubble_r[order], strict=False):
        marker = marker_names[idx]
        x = ring_radius * np.cos(ang)
        y = ring_radius * np.sin(ang)
        rel = float(proportions[idx] / max_p)
        col = cmap_fn(cmap_lo + (cmap_hi - cmap_lo) * rel)
        ax.add_patch(Circle((x, y), br, facecolor=col, edgecolor="black", linewidth=0.9, zorder=4))
        lx = (ring_radius + br + 0.10) * np.cos(ang)
        ly = (ring_radius + br + 0.10) * np.sin(ang)
        ha = "left" if np.cos(ang) > 0.15 else ("right" if np.cos(ang) < -0.15 else "center")
        va = "bottom" if np.sin(ang) > 0.15 else ("top" if np.sin(ang) < -0.15 else "center")
        ax.text(
            lx,
            ly,
            marker + "\n" + ("%0.1f" % (100 * proportions[idx])) + "%",
            ha=ha,
            va=va,
            fontsize=label_fontsize,
            fontweight="bold",
            linespacing=1.05,
        )
    if show_colorbar:
        sm = mpl.cm.ScalarMappable(cmap=cmap_fn, norm=mcolors.Normalize(vmin=0, vmax=100 * max_p))
        sm.set_array([])
        cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.05)
        cb = plt.colorbar(sm, cax=cax)
        cb.set_label("% of ECM signal", fontsize=cbar_label_fontsize)
        cb.ax.tick_params(labelsize=cbar_tick_fontsize)
    if title is not None:
        ax.set_title(title, loc="left", fontsize=title_fontsize, fontweight="bold")
    _save_and_show(fig, save, show)
    return ax


# ---------------------------------------------------------------------------
# Channel overlay restricted to cell-ECM-graph neighbours of a target cell type
# ---------------------------------------------------------------------------
def channel_overlay_on_neighbours(
    adata: AnnData,
    *,
    img: np.ndarray,
    channels,
    cell_type: str,
    channel_colors=None,
    channel_names=None,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    cell_attr: str = "cell_type",
    patch_half: int = 5,
    clip=(1.0, 99.5),
    scatter_color: str = "#d33232",
    scatter_size: float = 8,
    scatter_alpha: float = 0.5,
    scatter_edgecolor: str = "none",
    scatter_linewidth: float = 0.0,
    show_legend: bool = True,
    legend_fontsize: float = 9,
    pixel_size: float | None = None,
    scalebar_um: float | None = None,
    scalebar_color: str = "white",
    scalebar_loc: str = "lower right",
    scalebar_fontsize: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 7),
    ax: Axes | None = None,
    save=None,
    show: bool = True,
) -> Axes:
    """Composite raw-channel IMC overlay restricted to ECM patches a chosen cell type touches.

    Patches are those the cell type touches via the cell-ECM graph.
    Builds a binary patch mask from the cell-ECM-graph neighbours of the
    focal cell type, percentile-stretches each requested channel within
    the masked pixels, additively blends the channels with ``channel_colors``,
    and draws the target cells as a scatter on top.

    Parameters
    ----------
    adata
        AnnData with the cell-ECM graph + ``ecm_patches`` table containing
        ``x`` / ``y`` columns.
    img
        Raw IMC stack ``(C, H, W)`` numpy array.
    channels
        Channel indices into ``img`` (ints) or marker names resolved against
        ``adata.var_names``.
    cell_type
        Focal cell type for the patch mask.
    channel_colors
        Hex colours per channel.  Defaults: red, green, blue, magenta.
    channel_names
        Labels for the in-image legend.
    patch_half
        Half-extent in pixels of each ECM patch (grid stride is usually
        ``2 * patch_half``).
    clip
        ``(lo_pct, hi_pct)`` percentiles for the per-channel stretch,
        computed on masked pixels only.
    scatter_color, scatter_size, scatter_alpha, scatter_edgecolor, scatter_linewidth, show_legend
        Cell-overlay styling and in-image legend toggle.
    pixel_size
        Physical size of one pixel in micrometres.  Required to draw a
        scale bar; ``None`` disables it.
    scalebar_um
        Length of the scale bar in micrometres.  When set (together with
        ``pixel_size``), an anchored size bar is drawn on the image.
    scalebar_color, scalebar_loc, scalebar_fontsize
        Scale-bar colour, matplotlib anchor location (e.g. ``"lower right"``)
        and label font size (defaults to ``legend_fontsize``).
    ax, save, show
        Standard mantpy plotting controls.

    Returns
    -------
    matplotlib Axes
    """
    if graph_key not in adata.uns:
        raise ValueError("Key '" + graph_key + "' not found in adata.uns.")
    if ECM_PATCHES_KEY not in adata.uns:
        raise ValueError("Key '" + ECM_PATCHES_KEY + "' not found in adata.uns.")
    if img.ndim != 3:
        raise ValueError("`img` must be (C, H, W).")
    H, W = img.shape[1], img.shape[2]

    chan_idx = []
    var_names = list(adata.var_names)
    for c in channels:
        if isinstance(c, str):
            if c not in var_names:
                raise ValueError("Channel '" + c + "' not in adata.var_names")
            chan_idx.append(var_names.index(c))
        else:
            chan_idx.append(int(c))
    if channel_names is None:
        channel_names = [str(c) for c in channels]
    if channel_colors is None:
        channel_colors = ["#ff3333", "#33dd33", "#3366ff", "#dd33dd"][: len(chan_idx)]

    patches = adata.uns[ECM_PATCHES_KEY]
    if "x" not in patches.columns or "y" not in patches.columns:
        raise ValueError("ecm_patches must contain `x` and `y` columns.")

    G = adata.uns[graph_key]
    attrs = dict(G.nodes(data=True))

    cell_xy_list = []
    for _n, d in attrs.items():
        if d.get("node_type") == NODE_TYPE_CELL and str(d.get(cell_attr, "")) == str(cell_type):
            cell_xy_list.append((d.get("x"), d.get("y")))

    touched_idx = set()
    for u, v in G.edges():
        au, av = attrs.get(u, {}), attrs.get(v, {})
        t_u = au.get("node_type")
        t_v = av.get("node_type")
        if t_u == NODE_TYPE_CELL and t_v == NODE_TYPE_ECM:
            cell_n, ecm_n, ecm_id = au, av, v
        elif t_v == NODE_TYPE_CELL and t_u == NODE_TYPE_ECM:
            cell_n, ecm_n, ecm_id = av, au, u
        else:
            continue
        if str(cell_n.get(cell_attr, "")) != str(cell_type):
            continue
        idx = _ecm_node_index(ecm_id, ecm_n)
        if idx is not None:
            touched_idx.add(idx)

    mask = np.zeros((H, W), dtype=bool)
    for i in touched_idx:
        if 0 <= i < len(patches):
            x_c = float(patches.iloc[i]["x"])
            y_c = float(patches.iloc[i]["y"])
            x0 = max(int(round(x_c - patch_half)), 0)
            x1 = min(int(round(x_c + patch_half)), W)
            y0 = max(int(round(y_c - patch_half)), 0)
            y1 = min(int(round(y_c + patch_half)), H)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True

    rgb = np.zeros((H, W, 3), dtype=np.float64)
    for ci, col_hex in zip(chan_idx, channel_colors, strict=False):
        chan = img[ci].astype(np.float64)
        vals = chan[mask]
        if vals.size == 0:
            continue
        lo = float(np.percentile(vals, clip[0]))
        hi = float(np.percentile(vals, clip[1]))
        s = np.clip((chan - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        s[~mask] = 0.0
        rgb_col = np.array(
            [
                int(col_hex.lstrip("#")[0:2], 16) / 255.0,
                int(col_hex.lstrip("#")[2:4], 16) / 255.0,
                int(col_hex.lstrip("#")[4:6], 16) / 255.0,
            ]
        )
        rgb += s[..., None] * rgb_col[None, None, :]
    rgb = np.clip(rgb, 0, 1)

    fig, ax = _get_ax(ax, figsize=figsize)
    ax.set_facecolor("black")
    ax.imshow(rgb, origin="upper", extent=(0, W, H, 0), interpolation="nearest")
    if cell_xy_list:
        cell_xy = np.asarray(cell_xy_list, dtype=float)
        ax.scatter(
            cell_xy[:, 0],
            cell_xy[:, 1],
            s=scatter_size,
            facecolor=scatter_color,
            edgecolor=scatter_edgecolor,
            linewidth=scatter_linewidth,
            alpha=scatter_alpha,
            zorder=3,
        )
    if show_legend:
        for i, (name, col) in enumerate(zip(channel_names, channel_colors, strict=False)):
            ax.text(
                0.025,
                0.97 - i * 0.05,
                name,
                transform=ax.transAxes,
                color=col,
                fontsize=legend_fontsize,
                fontweight="bold",
                ha="left",
                va="top",
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 2.0},
            )
        y_t = 0.97 - len(channel_names) * 0.05
        ax.text(
            0.025,
            y_t,
            str(cell_type),
            transform=ax.transAxes,
            color=scatter_color,
            fontsize=legend_fontsize,
            fontweight="bold",
            ha="left",
            va="top",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 2.0},
        )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#666666")
    if scalebar_um is not None:
        if pixel_size is None:
            raise ValueError("`pixel_size` (micrometres per pixel) is required when `scalebar_um` is set.")
        from matplotlib.font_manager import FontProperties
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

        bar_px = float(scalebar_um) / float(pixel_size)
        sb_fs = scalebar_fontsize if scalebar_fontsize is not None else legend_fontsize
        bar = AnchoredSizeBar(
            ax.transData,
            bar_px,
            f"{scalebar_um:g} µm",
            scalebar_loc,
            pad=0.3,
            sep=2,
            borderpad=0.4,
            color=scalebar_color,
            frameon=False,
            size_vertical=max(bar_px * 0.03, 1.0),
            fontproperties=FontProperties(size=sb_fs),
        )
        ax.add_artist(bar)
    if title is not None:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="black")
    _save_and_show(fig, save, show)
    return ax


# ---------------------------------------------------------------------------
# Cohort composition and topology panels
# ---------------------------------------------------------------------------
def plot_mean_composition(
    ax: Axes,
    group_mean: pd.DataFrame,
    ecm_cols,
    *,
    palette,
    group_order,
    target_cluster,
) -> Axes:
    """Stacked-bar of mean ECM-cluster composition per experimental group.

    One stacked bar per entry of ``group_order`` (e.g. ``['Naive_KO',
    'Naive_WT', 'Infected_KO', 'Infected_WT']``).  The bar for each ECM
    cluster is taken from ``group_mean[cl]``; the ``target_cluster`` bar
    is drawn with a heavier edge to identify the highlighted cluster.

    Parameters
    ----------
    ax
        Axis to draw on.
    group_mean
        DataFrame indexed by ``group_order`` with one column per cluster
        in ``ecm_cols``; cells are mean cluster fractions per group.
    ecm_cols
        Iterable of cluster identifiers (ints or strings) — columns of
        ``group_mean`` rendered as stacked-bar segments in order.
    palette
        ``dict[int|str, str]`` mapping cluster → hex colour.
    group_order
        Ordered list of group labels (x-axis category order).
    target_cluster
        Cluster identifier whose bar is drawn with a heavier 1.4-pt edge
        (the rest are 0.6-pt) so the highlighted cluster stands out.

    Returns
    -------
    matplotlib.axes.Axes
        The same ``ax``.

    Examples
    --------
    >>> # plot_mean_composition(ax, group_mean, ecm_cols,
    >>> #                        palette=PALETTE,
    >>> #                        group_order=GROUP_ORDER,
    >>> #                        target_cluster=TARGET_CLUSTER)  # doctest: +SKIP
    """
    x = np.arange(len(group_order))
    bottoms = np.zeros(len(group_order))
    for cl in ecm_cols:
        vals = group_mean[cl].values.astype(float)
        is_target = cl == target_cluster
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            color=palette[cl],
            edgecolor="black",
            linewidth=1.4 if is_target else 0.6,
            label=f"ECM {cl}",
        )
        bottoms += vals
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, len(group_order) - 0.5)
    ax.set_title("Mean ECM composition per group", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8, title="ECM cluster")
    return ax


def plot_delta_masked(
    ax: Axes,
    delta,
    p_fdr=None,
    *,
    palette,
    K: int,
    cmap=None,
    alpha_fdr: float = 0.05,
    mask_nonsignificant: bool = True,
    vlim: float | None = None,
    fontsize: float = 9,
    tick_fontsize: float = 10,
) -> AxesImage:
    """FDR-masked Δ heatmap of cluster co-occurrence (KO − WT).

    Renders a ``K × K`` matrix of ``delta`` values, masking cells with
    ``p_fdr >= alpha_fdr`` to white.  Each significant cell is annotated
    with the Δ value plus a star count (``*``: ``p<alpha_fdr``, ``**``:
    ``p<0.01``, ``***``: ``p<0.001``).  Cluster tick labels are coloured
    per ``palette`` to match the rest of the figure's cluster encoding.

    Set ``mask_nonsignificant=False`` for a purely descriptive heatmap. In
    that mode all contrasts are shown, no p-value or star annotation is drawn,
    and ``p_fdr`` may be ``None``.

    Parameters
    ----------
    ax
        Axis to draw on.
    delta
        ``K × K`` ndarray of pairwise Δ values (e.g. Δlog₂ KO − WT).
    p_fdr
        ``K × K`` ndarray of BH-FDR-adjusted p-values, aligned with
        ``delta``.
    palette
        Cluster colour map (``dict[int, str]``).
    K
        Cluster count (row/column extent).
    cmap
        Diverging colormap.  ``None`` → ``plt.cm.RdBu_r``.  A *copy* is
        made before ``set_bad('white')`` so the caller's cmap is not
        mutated.
    alpha_fdr
        FDR threshold for masking + base star.
    mask_nonsignificant
        Apply FDR masking and annotations. ``False`` disables inferential
        display and does not require ``p_fdr``.
    vlim
        Symmetric colour-axis half-range.  ``None`` → ``max(|delta|, 1.0)``.
    fontsize, tick_fontsize
        Standard matplotlib text controls.

    Returns
    -------
    matplotlib.image.AxesImage
        The image artist (so the caller can attach a colorbar).
    """
    if cmap is None:
        cmap = plt.cm.RdBu_r
    cmap = cmap.copy()
    cmap.set_bad("white")
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (K, K):
        raise ValueError(f"delta must have shape {(K, K)}; got {delta.shape}.")
    if vlim is None:
        vlim = float(max(np.abs(delta).max(), 1.0))
    if mask_nonsignificant:
        if p_fdr is None:
            raise ValueError("p_fdr is required when mask_nonsignificant=True.")
        p_fdr = np.asarray(p_fdr, dtype=float)
        if p_fdr.shape != delta.shape:
            raise ValueError(f"p_fdr must have shape {delta.shape}; got {p_fdr.shape}.")
        displayed = np.where(p_fdr < alpha_fdr, delta, np.nan)
    else:
        displayed = delta
    im = ax.imshow(displayed, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"ECM {k}" for k in range(K)], rotation=30, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels([f"ECM {k}" for k in range(K)], fontsize=tick_fontsize)
    for k in range(K):
        for tl in (ax.get_xticklabels()[k], ax.get_yticklabels()[k]):
            tl.set_color(palette[k])
            tl.set_fontweight("bold")
    if mask_nonsignificant:
        for i in range(K):
            for j in range(K):
                if p_fdr[i, j] >= alpha_fdr:
                    continue
                stars = "***" if p_fdr[i, j] < 0.001 else ("**" if p_fdr[i, j] < 0.01 else "*")
                ax.text(
                    j,
                    i,
                    f"{delta[i, j]:+.1f}\n{stars}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight="bold",
                    linespacing=0.85,
                    color="white" if abs(delta[i, j]) > vlim * 0.55 else "black",
                )
    ax.set_xticks(np.arange(K + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(K + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    return im


def plot_cluster_map(
    ax: Axes,
    adata: AnnData,
    *,
    palette,
    best_k: int,
    target_cluster,
    show_background: bool = True,
    edge_alpha: float = 0.30,
    graph_key: str = "ecm_graph",
    image_key: str = "image_container",
    cluster_attr: str = "ecm_cluster",
) -> Axes:
    """Per-ROI ECM-graph scatter map with cluster-coloured nodes.

    Reads ``adata.uns[graph_key]`` (a NetworkX graph with ``x``/``y`` and
    ``ecm_cluster`` node attributes) and plots:

    * a thin white edge layer (skipped where both endpoints are unclassified),
    * optional grey dots for the ``-1`` (background) cluster,
    * coloured dots for clusters ``0..best_k-1`` excluding ``target_cluster``,
    * larger, black-outlined dots for the ``target_cluster`` (foreground).

    If ``adata.uns[image_key]`` is an ``mt.im.ImageContainer``, its
    ``.to_array()`` shape fixes the axis extent so the scatter sits in
    image coordinates; otherwise ``invert_yaxis()`` is called so the plot
    still reads "image-like" (origin at top-left).

    Parameters
    ----------
    ax
        Axis to draw on.
    adata
        AnnData with a graph at ``adata.uns[graph_key]``.
    palette
        Cluster colour map (``dict[int, str]``).
    best_k
        Number of valid ECM clusters; clusters with index outside
        ``[0, best_k)`` are not coloured (except ``-1``, which is grey).
    target_cluster
        Cluster identifier rendered with the heavier black-outlined
        marker.  Notebook default is ``5`` (ECM-5 lesion cluster).
    show_background
        Whether to render nodes assigned to the ``-1`` background cluster.
        When hidden, edges touching a background node are hidden as well.
    edge_alpha
        Alpha for the white edge layer.
    graph_key, image_key, cluster_attr
        Mantpy ``adata.uns`` / node-attribute keys.

    Returns
    -------
    matplotlib.axes.Axes
        The same ``ax``.
    """
    ax.set_facecolor(_NEUTRAL_BG_DARK)
    G = adata.uns[graph_key]
    xy = {v: (d["x"], d["y"]) for v, d in G.nodes(data=True)}
    cl = {v: int(d.get(cluster_attr, -1)) for v, d in G.nodes(data=True)}
    for u, v in G.edges():
        if (cl[u] < 0 and cl[v] < 0) or (not show_background and (cl[u] < 0 or cl[v] < 0)):
            continue
        x0, y0 = xy[u]
        x1, y1 = xy[v]
        ax.plot([x0, x1], [y0, y1], color=_NEUTRAL_EDGE, lw=0.35, alpha=edge_alpha, zorder=2)
    by_cl = {}
    for v, lbl in cl.items():
        by_cl.setdefault(lbl, []).append(xy[v])
    if show_background and -1 in by_cl:
        xs, ys = zip(*by_cl[-1], strict=False)
        ax.scatter(xs, ys, c=_NEUTRAL_OUTLIER, s=2, edgecolor="none", zorder=3)
    for lbl in sorted(c for c in by_cl if 0 <= c < best_k and c != target_cluster):
        xs, ys = zip(*by_cl[lbl], strict=False)
        ax.scatter(xs, ys, c=palette[lbl], s=5, edgecolor="none", zorder=4, alpha=0.9)
    if target_cluster in by_cl:
        xs, ys = zip(*by_cl[target_cluster], strict=False)
        ax.scatter(xs, ys, c=palette[target_cluster], s=10, edgecolor="black", linewidth=0.35, zorder=6)
    try:
        img = _as_image_container(adata.uns[image_key]).to_array()
        ax.set_xlim(0, img.shape[2])
        ax.set_ylim(img.shape[1], 0)
    except (KeyError, AttributeError):
        ax.invert_yaxis()
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(_NEUTRAL_SPINE)
    return ax


def plot_lesion_metric_view(
    ax: Axes,
    adata: AnnData,
    metric: str,
    *,
    palette,
    radius_px: float,
    target_cluster,
    best_k: int,
    cmap=None,
    metric_vmin: float | None = None,
    metric_vmax: float | None = None,
    image_key: str = "image_container",
    graph_key: str = "ecm_graph",
    cluster_attr: str = "ecm_cluster",
) -> Axes:
    """Per-ROI lesion metric overlay (size / degree / k-core / void area).

    Draws the radius-reconnected biggest lesion of ``target_cluster`` on
    a light background of the ROI's other-cluster ECM patches, with
    nodes coloured by one of four pre-computed topology metrics:

    * ``'n_nodes'`` — constant fill = node count of the lesion;
    * ``'mean_degree'`` — per-node degree in the radius-reconnected graph;
    * ``'central_hole_area'`` — fills the largest bounded un-filled
      component of the alpha shape (the lesion's "doughnut hole");
    * ``'max_k_core'`` — highlights nodes in the maximum k-core only.

    Pulls the lesion via :func:`mantpy.gr.largest_component_radius_reconnect`
    and the central-void geometry via :func:`mantpy.tl.lesion_central_void`,
    so the metric value at the per-axis level matches the per-row
    summary table bit-for-bit.

    Parameters
    ----------
    ax
        Axis to draw on.
    adata
        AnnData with an ECM graph at ``adata.uns[graph_key]`` and
        optionally an image container at ``adata.uns[image_key]`` to fix
        the axis extent.
    metric
        One of ``'n_nodes'``, ``'mean_degree'``, ``'central_hole_area'``,
        ``'max_k_core'``.
    palette
        Cluster colour map — only consulted by the background scatter
        layer.
    radius_px
        Spatial reconnect radius (px) for the lesion graph.  Forwarded to
        :func:`~mantpy.gr.largest_component_radius_reconnect` and to
        :func:`~mantpy.tl.lesion_central_void` (the latter derives
        ``alpha_sq = (radius / 2)²``).
    target_cluster
        Cluster identifier whose largest reconnected component is the
        lesion under study.
    best_k
        Cluster count — anything outside ``[0, best_k)`` or equal to
        ``target_cluster`` is excluded from the background layer.
    cmap
        Sequential / diverging colormap for the metric ramp.  ``None`` →
        ``plt.cm.RdBu_r``.
    metric_vmin, metric_vmax
        Optional colour-axis bounds.  ``None`` chooses per-metric
        sensible defaults for the selected metric.
    image_key, graph_key, cluster_attr
        Mantpy ``uns`` / node-attribute keys.

    Returns
    -------
    matplotlib.axes.Axes
        The same ``ax``.
    """
    if cmap is None:
        cmap = plt.cm.RdBu_r
    ax.set_facecolor(_NEUTRAL_BG_LIGHT)
    G = adata.uns[graph_key]
    xy = {v: (d["x"], d["y"]) for v, d in G.nodes(data=True)}
    cl = {v: int(d.get(cluster_attr, -1)) for v, d in G.nodes(data=True)}
    bg_xy = [xy[v] for v, c in cl.items() if 0 <= c < best_k and c != target_cluster]
    if bg_xy:
        xs, ys = zip(*bg_xy, strict=False)
        ax.scatter(xs, ys, c=_NEUTRAL_FAINT, s=1.5, alpha=0.45, edgecolor="none", zorder=2)

    rec = _largest_component_radius_reconnect(
        adata,
        target_cluster=target_cluster,
        radius=radius_px,
    )
    if rec is not None:
        G_big = rec["subgraph"]
        big_coords = rec["coords"]
        biggest = list(G_big.nodes())
        big_n = rec["n_nodes"]
        if metric != "max_k_core":
            local_pos = {orig: k for k, orig in enumerate(biggest)}
            for u, v in G_big.edges():
                ui, vi = local_pos[u], local_pos[v]
                ax.plot(
                    [big_coords[ui, 0], big_coords[vi, 0]],
                    [big_coords[ui, 1], big_coords[vi, 1]],
                    color=_NEUTRAL_GRID,
                    lw=0.35,
                    alpha=0.55,
                    zorder=3,
                )
        if metric == "n_nodes":
            vmin = 0 if metric_vmin is None else metric_vmin
            vmax = max(big_n, 1) if metric_vmax is None else metric_vmax
            ax.scatter(
                big_coords[:, 0],
                big_coords[:, 1],
                c=np.full(big_n, big_n, dtype=float),
                cmap=cmap,
                s=4,
                edgecolor="none",
                vmin=vmin,
                vmax=vmax,
                zorder=5,
            )
        elif metric == "mean_degree":
            degrees = np.array([G_big.degree(o) for o in biggest], dtype=float)
            vmin = 0 if metric_vmin is None else metric_vmin
            vmax = max(int(degrees.max()), 1) if metric_vmax is None else metric_vmax
            ax.scatter(
                big_coords[:, 0],
                big_coords[:, 1],
                c=degrees,
                cmap=cmap,
                s=5,
                edgecolor="none",
                vmin=vmin,
                vmax=vmax,
                zorder=5,
            )
        elif metric == "central_hole_area":
            void = _lesion_central_void(big_coords, radius=radius_px)
            if void.kept_tris:
                polys = [big_coords[list(t)] for t in void.kept_tris]
                pc = PatchCollection(
                    [MplPolygon(p, closed=True) for p in polys],
                    facecolor=_NEUTRAL_VOID_FILL,
                    alpha=0.55,
                    edgecolor="none",
                    zorder=2,
                )
                ax.add_collection(pc)
            ax.scatter(big_coords[:, 0], big_coords[:, 1], c=_NEUTRAL_OUTLIER, s=2, edgecolor="none", zorder=5)
            if void.component and void.d_tri is not None:
                vmin = 0 if metric_vmin is None else metric_vmin
                vmax = max(void.area * 1.1, 1.0) if metric_vmax is None else metric_vmax
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                color = cmap(norm(void.area))
                polys = [big_coords[void.d_tri.simplices[si]] for si in void.component]
                pc = PatchCollection(
                    [MplPolygon(p, closed=True) for p in polys],
                    facecolor=color,
                    alpha=0.95,
                    edgecolor=color,
                    linewidth=0.6,
                    zorder=6,
                )
                ax.add_collection(pc)
        elif metric == "max_k_core":
            Gc = G_big.copy()
            Gc.remove_edges_from(nx.selfloop_edges(Gc))
            cores = nx.core_number(Gc) if Gc.number_of_nodes() else {}
            max_k = max(cores.values()) if cores else 0
            core_vals = np.array([cores.get(o, 0) for o in biggest], dtype=float)
            is_max = core_vals == max_k
            if (~is_max).any():
                ax.scatter(
                    big_coords[~is_max, 0],
                    big_coords[~is_max, 1],
                    c=_NEUTRAL_FAINT,
                    s=1.5,
                    alpha=0.45,
                    edgecolor="none",
                    zorder=4,
                )
            if is_max.any():
                vmin = 0 if metric_vmin is None else metric_vmin
                vmax = max(float(max_k), 1.0) if metric_vmax is None else metric_vmax
                color_vals = np.full(int(is_max.sum()), float(max_k))
                ax.scatter(
                    big_coords[is_max, 0],
                    big_coords[is_max, 1],
                    c=color_vals,
                    cmap=cmap,
                    s=8,
                    edgecolor="none",
                    vmin=vmin,
                    vmax=vmax,
                    zorder=6,
                )
    try:
        img = _as_image_container(adata.uns[image_key]).to_array()
        ax.set_xlim(0, img.shape[2])
        ax.set_ylim(img.shape[1], 0)
    except (KeyError, AttributeError):
        ax.invert_yaxis()
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(_NEUTRAL_SPINE)
    return ax


def patch_domain_map(
    ax: Axes,
    ys: np.ndarray,
    xs: np.ndarray,
    colors: np.ndarray,
    shape: tuple[int, int],
    *,
    crop: bool = True,
    crop_pad: int = 2,
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    title: str | None = None,
    title_color: str = "#1a1a1a",
    edge_color: str | None = None,
) -> Axes:
    """Raster map of per-patch values on the image-patch lattice.

    Paints ``colors`` (one RGB triplet per patch) at patch pixel coordinates ``(ys, xs)`` on a
    ``shape = (H, W)`` canvas and shows it with ``imshow``. Use it for spatial-domain maps (one
    colour per cluster) or per-cluster highlights (one cluster coloured, the rest greyed), e.g.
    when visualising :func:`mantpy.tl.select_n_domains` output for image-patch data.

    Parameters
    ----------
    ax
        Axis to draw on.
    ys, xs
        Integer patch pixel coordinates, each shape ``(n_patches,)``.
    colors
        Per-patch RGB colours, shape ``(n_patches, 3)`` in ``[0, 1]``.
    shape
        ``(H, W)`` of the full patch lattice.
    crop, crop_pad
        Crop to the tissue bounding box (with a ``crop_pad`` margin).
    background
        RGB fill for empty (non-patch) pixels.
    title, title_color, edge_color
        Optional bold title and a coloured panel border.

    Returns
    -------
    matplotlib.axes.Axes
        The same ``ax``.
    """
    h, w = shape
    canvas = np.ones((h, w, 3), dtype=float) * np.asarray(background, dtype=float)
    canvas[ys, xs] = colors
    if crop:
        y0 = max(int(ys.min()) - crop_pad, 0)
        y1 = min(int(ys.max()) + crop_pad + 1, h)
        x0 = max(int(xs.min()) - crop_pad, 0)
        x1 = min(int(xs.max()) + crop_pad + 1, w)
        canvas = canvas[y0:y1, x0:x1]
    ax.imshow(canvas, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if edge_color is not None:
        for s in ax.spines.values():
            s.set_edgecolor(edge_color)
            s.set_linewidth(1.1)
    if title is not None:
        ax.set_title(title, fontsize=7.0, fontweight="bold", color=title_color, pad=2.5)
    return ax


def plot_marker_otsu_composite(
    ax: Axes,
    adata: AnnData,
    markers: Sequence[str],
    *,
    colors: Sequence[str] | None = None,
    image_key: str = "image_container",
    clip_pct: float = 99.5,
    alpha: float = 0.9,
    blend: str = "max",
    normalize: bool = True,
    show_legend: bool = False,
) -> Axes:
    """Multiplexed multi-marker composite via per-channel Otsu thresholding.

    For each marker channel resolved in ``adata.var_names``: ``arcsinh`` transform,
    clip at the ``clip_pct`` percentile of positive pixels, Otsu-threshold to a
    binary foreground mask, optionally contrast-stretch the in-mask signal to the
    full range (``normalize``), then paint a distinct colour onto a black canvas.
    ``blend='max'`` (lighten -- default) keeps each marker's colour crisp and avoids
    the wash-out of additive blending in dense regions; ``blend='add'`` sums.
    Gives a clean "gestalt" of the matrix the ECM clustering ingests -- the
    per-cluster identity is made precise elsewhere with
    :func:`mantpy.pl.ecm_centroid_heatmap`.

    Parameters
    ----------
    ax
        Axis to draw on.
    adata
        AnnData with an image container at ``adata.uns[image_key]`` whose
        ``.to_array()`` is ``(C, H, W)``, and ``var_names`` containing ``markers``.
    markers
        Channel names to composite (e.g. the ECM markers used for clustering).
    colors
        One colour per marker.  Defaults to a CVD-safe Okabe-Ito set.
    image_key, clip_pct, alpha
        Image-container key, per-channel clip percentile, and additive blend weight.
    show_legend
        Append a marker-to-colour legend at the right of the axis.

    Returns
    -------
    matplotlib.axes.Axes
        The same ``ax``.
    """
    try:
        from skimage.filters import threshold_otsu
    except ImportError as e:  # pragma: no cover
        raise ImportError("plot_marker_otsu_composite requires scikit-image.") from e

    cols = list(colors) if colors is not None else list(_OTSU_DEFAULT_COLORS)
    img = _as_image_container(adata.uns[image_key]).to_array()  # (C, H, W)
    var_names = list(adata.var_names)
    out = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)
    for m, c in zip(markers, cols, strict=False):
        if m not in var_names:
            continue
        ch = np.arcsinh(img[var_names.index(m)].astype(np.float32))
        pos = ch[ch > 0]
        if pos.size == 0:
            continue
        norm = np.clip(ch / max(float(np.percentile(pos, clip_pct)), 1e-6), 0.0, 1.0)
        nz = norm[norm > 0]
        thr = float(threshold_otsu(nz)) if nz.size > 50 else 0.5
        mask = norm >= thr
        if normalize:  # contrast-stretch the in-mask signal to [0, 1] for a crisp composite
            w = np.clip((norm - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0) * mask
        else:
            w = norm * mask
        contrib = np.asarray(mcolors.to_rgb(c), dtype=np.float32)[None, None, :] * (w * alpha)[..., None]
        out = np.maximum(out, contrib) if blend == "max" else out + contrib
    ax.imshow(np.clip(out, 0.0, 1.0))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if show_legend:
        import matplotlib.patches as mpatches

        handles = [mpatches.Patch(color=c, label=m)
                   for m, c in zip(markers, cols, strict=False) if m in var_names]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=7, title="ECM markers", title_fontsize=7)
    return ax
