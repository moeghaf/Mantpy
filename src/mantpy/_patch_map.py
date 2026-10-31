"""Observation-aligned adapter for Mantpy's patch-lattice renderer."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb


def patch_map(
    data: AnnData | pd.DataFrame,
    color="ecm_cluster",
    *,
    patch_size=None,
    palette=None,
    ax=None,
    cmap="viridis",
    crop=False,
    background="white",
    title=None,
    vmin=None,
    vmax=None,
    origin="upper",
    extent=None,
    shape=None,
    categorical=None,
    background_label=-1,
    colorbar=False,
):
    """Draw categorical labels or continuous values on the original patch grid.

    ``color`` is an observation column, aligned vector or ID-indexed Series.
    Explicit ``grid_y/grid_x`` or ``grid_row/grid_col`` columns take precedence
    over spatial centroids. AnnData's recorded full ``patch_grid_shape`` keeps
    empty boundary rows and columns. ``extent`` supports physical-coordinate
    overlays; the default is the usual imshow lattice in grid-index units.
    Missing values and the categorical background label leave empty pixels.
    """
    from mantpy._plot_helpers import patch_domain_map
    from mantpy.pl import categorical_palette

    table = data.obs if isinstance(data, AnnData) else data
    metadata = data.uns.get("image_ecm_patches", {}) if isinstance(data, AnnData) else {}
    if not table.index.is_unique:
        raise ValueError("Patch IDs must be unique.")
    if isinstance(color, str):
        if color not in table:
            raise KeyError(f"Patch table has no column {color!r}.")
        values = table[color]
        if categorical is None:
            categorical = (
                palette is not None
                or color.endswith(("cluster", "label"))
                or isinstance(values.dtype, pd.CategoricalDtype)
            )
    elif isinstance(color, pd.Series):
        if not color.index.is_unique or not color.index.isin(table.index).all():
            raise ValueError("color Series must have unique patch IDs belonging to data.")
        values = color.reindex(table.index)
    else:
        values = pd.Series(np.asarray(color), index=table.index)
    if len(values) != len(table):
        raise ValueError("color must contain one value per patch.")
    if categorical is None:
        categorical = palette is not None or not pd.api.types.is_numeric_dtype(values)
    if {"grid_y", "grid_x"} <= set(table):
        ys, xs = table["grid_y"].to_numpy(), table["grid_x"].to_numpy()
    elif {"grid_row", "grid_col"} <= set(table):
        ys, xs = table["grid_row"].to_numpy(), table["grid_col"].to_numpy()
    else:
        size = patch_size if patch_size is not None else metadata.get("patch_size")
        if size is None or not np.isfinite(size) or size <= 0:
            raise ValueError("Supply patch_size when explicit patch-grid columns are absent.")
        if isinstance(data, AnnData) and "spatial" in data.obsm:
            xy = np.asarray(data.obsm["spatial"])
        elif {"x", "y"} <= set(table):
            xy = table[["x", "y"]].to_numpy()
        else:
            raise KeyError("Patch-grid columns or spatial centroids are required.")
        xs, ys = np.floor(xy[:, 0] / size), np.floor(xy[:, 1] / size)
    ys, xs = np.asarray(ys), np.asarray(xs)
    if not np.isfinite(ys).all() or not np.isfinite(xs).all():
        raise ValueError("Patch-grid coordinates must be finite.")
    if (ys < 0).any() or (xs < 0).any() or (ys != np.floor(ys)).any() or (xs != np.floor(xs)).any():
        raise ValueError("Patch-grid coordinates must be non-negative integers.")
    ys, xs = ys.astype(int), xs.astype(int)
    if len(set(zip(ys, xs, strict=True))) != len(table):
        raise ValueError("Multiple patches occupy the same grid position.")
    shape = shape if shape is not None else metadata.get("patch_grid_shape")
    if shape is None:
        if not len(table):
            raise ValueError("Supply shape for an empty patch table.")
        shape = (int(ys.max()) + 1, int(xs.max()) + 1)
    shape = tuple(int(value) for value in shape)
    if len(shape) != 2 or min(shape) < 1 or (ys >= shape[0]).any() or (xs >= shape[1]).any():
        raise ValueError("shape must contain every patch-grid position.")
    valid = values.notna().to_numpy(copy=True)
    colors = np.tile(to_rgb(background), (len(table), 1))
    scalar = None
    if categorical:
        if background_label is not None:
            valid &= values.ne(background_label).fillna(False).to_numpy(dtype=bool)
        categories = list(pd.unique(values[valid]))
        mapping = palette if isinstance(palette, Mapping) else categorical_palette(categories, colors=palette)
        for i in np.flatnonzero(valid):
            if values.iloc[i] not in mapping:
                raise KeyError(f"palette has no colour for {values.iloc[i]!r}.")
            colors[i] = to_rgb(mapping[values.iloc[i]])
    else:
        numbers = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
        valid &= np.isfinite(numbers)
        lower = vmin if vmin is not None else (float(numbers[valid].min()) if valid.any() else 0.0)
        upper = vmax if vmax is not None else (float(numbers[valid].max()) if valid.any() else 1.0)
        scalar = ScalarMappable(norm=Normalize(lower, upper), cmap=cmap)
        colors[valid] = scalar.to_rgba(numbers[valid])[:, :3]
    if ax is None:
        _, ax = plt.subplots()
    # Keep the shared renderer responsible for the canvas and interpolation.
    patch_domain_map(ax, ys, xs, colors, shape, crop=crop and len(table) > 0, background=to_rgb(background), title=title)
    image = ax.images[-1]
    image.origin = origin
    if origin not in {"upper", "lower"}:
        raise ValueError("origin must be 'upper' or 'lower'.")
    if crop and len(table):
        y0, y1 = max(int(ys.min()) - 2, 0), min(int(ys.max()) + 3, shape[0])
        x0, x1 = max(int(xs.min()) - 2, 0), min(int(xs.max()) + 3, shape[1])
    else:
        y0, y1, x0, x1 = 0, shape[0], 0, shape[1]
    if extent is None:
        resolved_extent = (x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5)
        if origin == "lower":
            resolved_extent = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)
    else:
        left, right, bottom, top = extent
        if origin == "upper":
            vertical = (top + (bottom - top) * y1 / shape[0], top + (bottom - top) * y0 / shape[0])
        else:
            vertical = (bottom + (top - bottom) * y0 / shape[0], bottom + (top - bottom) * y1 / shape[0])
        resolved_extent = (
            left + (right - left) * x0 / shape[1],
            left + (right - left) * x1 / shape[1],
            *vertical,
        )
    image.set_extent(resolved_extent)
    if colorbar and scalar is not None:
        ax.figure.colorbar(scalar, ax=ax)
    return ax
