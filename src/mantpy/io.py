"""I/O functions for reading spatial omics data into AnnData."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantpy._constants import (
    CELLTYPE_COL,
    CONDITION_COL,
    IMAGE_CONTAINER_KEY,
    MANTPY_UNS_KEY,
    RAW_LAYER,
    SAMPLE_ID_COL,
    SPATIAL_KEY,
    X_COL,
    Y_COL,
)

if TYPE_CHECKING:
    import spatialdata

    from mantpy.im import ImageContainer

__all__ = [
    "InputSummary",
    "PanelSummary",
    "cell_features_from_mask",
    "input_summary",
    "panel_summary",
    "read_codex",
    "read_ecm_image",
    "read_imc",
    "read_imc_folder",
    "sample_group_map",
    "to_spatialdata",
]


@dataclass(frozen=True)
class InputSummary:
    """Validated overview of the three inputs to a cell--ECM analysis."""

    n_cells: int
    n_cell_vars: int
    celltype_key: str
    spatial_shape: tuple[int, int]
    image_shape: tuple[int, int, int]
    image_dtype: str
    image_range: tuple[float, float]
    panel_shape: tuple[int, int]
    panel_columns: tuple[str, ...]

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Cell--ECM inputs",
                f"  typed cells   {self.n_cells:,} observations; labels in obs[{self.celltype_key!r}]",
                f"  coordinates   obsm['spatial'] shape {self.spatial_shape}",
                f"  cell .X       {self.n_cell_vars} variables (not used by Mantpy's joint graph)",
                f"  image         {self.image_shape} {self.image_dtype}; range {self.image_range[0]:g}--{self.image_range[1]:g}",
                f"  panel         {self.panel_shape}; columns {list(self.panel_columns)}",
            ]
        )


def input_summary(
    cells: AnnData,
    image: str | Path | np.ndarray | ImageContainer,
    panel: str | Path | pd.DataFrame,
    *,
    celltype_key: str = "cell_type",
    spatial_key: str = "spatial",
) -> InputSummary:
    """Validate and summarize the typed cells, image stack, and panel.

    This checks Mantpy's general joint-analysis input contract without reading
    cell marker intensities: cell labels in ``obs``, cell centroids in
    ``obsm['spatial']``, a channel-first image, and a panel table. Channel-level
    panel alignment is checked separately by :func:`panel_summary`.
    """
    if celltype_key not in cells.obs:
        raise ValueError(f"Typed-cell AnnData is missing obs[{celltype_key!r}].")
    if spatial_key not in cells.obsm:
        raise ValueError(f"Typed-cell AnnData is missing obsm[{spatial_key!r}].")
    spatial = np.asarray(cells.obsm[spatial_key])
    if spatial.shape != (cells.n_obs, 2):
        raise ValueError(f"obsm[{spatial_key!r}] must have shape (n_cells, 2); got {spatial.shape}.")

    from mantpy.im import ImageContainer

    if isinstance(image, ImageContainer):
        image_array = image.to_array()
    elif isinstance(image, np.ndarray):
        image_array = image
    else:
        image_array = _load_img_array(image)
    if image_array.ndim != 3:
        raise ValueError(f"Image must be channel-first (C, H, W); got shape {image_array.shape}.")

    panel_df = _load_dataframe(panel)
    panel_df.columns = [str(column).strip() for column in panel_df.columns]
    return InputSummary(
        n_cells=cells.n_obs,
        n_cell_vars=cells.n_vars,
        celltype_key=celltype_key,
        spatial_shape=(int(spatial.shape[0]), int(spatial.shape[1])),
        image_shape=tuple(int(value) for value in image_array.shape),
        image_dtype=str(image_array.dtype),
        image_range=(float(np.nanmin(image_array)), float(np.nanmax(image_array))),
        panel_shape=(int(panel_df.shape[0]), int(panel_df.shape[1])),
        panel_columns=tuple(map(str, panel_df.columns)),
    )


@dataclass(frozen=True)
class PanelSummary:
    """Validated alignment summary for a panel, image, and optional AnnData."""

    acquired_channels: int
    retained_channels: int
    image_channels: int | None
    ecm_markers: tuple[str, ...]
    non_ecm_markers: tuple[str, ...]
    adata_checked: bool

    @property
    def n_ecm(self) -> int:
        """Number of retained ECM markers."""
        return len(self.ecm_markers)

    @property
    def n_non_ecm(self) -> int:
        """Number of retained non-ECM markers."""
        return len(self.non_ecm_markers)

    def __repr__(self) -> str:
        image = "not checked" if self.image_channels is None else f"{self.image_channels}  [matches panel]"
        adata = "not checked" if not self.adata_checked else "var_names and var['is_ecm'] match"
        markers = ", ".join(self.ecm_markers) or "none"
        return "\n".join(
            [
                "Panel summary",
                f"  acquired channels       {self.acquired_channels}",
                f"  retained panel rows     {self.retained_channels}",
                f"  retained image planes   {image}",
                f"  ECM markers             {self.n_ecm}",
                f"  retained non-ECM        {self.n_non_ecm}",
                f"  AnnData                  {adata}",
                "",
                f"ECM markers: {markers}",
            ]
        )


def panel_summary(
    panel: str | Path | pd.DataFrame,
    *,
    image: str | Path | np.ndarray | ImageContainer | None = None,
    adata: AnnData | None = None,
    keep_col: str = "keep",
    name_col: str = "name",
    ecm_col: str = "ecm",
) -> PanelSummary:
    """Validate and summarize panel-to-image-to-AnnData channel alignment.

    Column names are stripped before validation, so upstream names such as
    ``"ecm "`` are handled identically to :func:`read_imc`. When ``keep_col``
    is present, only rows equal to ``1`` are expected in the retained image
    stack and AnnData variables.

    Parameters
    ----------
    panel
        Panel CSV path or DataFrame. Must contain marker names and an ECM flag.
    image
        Optional channel-first image array, TIFF path, or ImageContainer. When
        supplied, its channel count must match the retained panel rows.
    adata
        Optional AnnData carrying the retained channels in ``var_names`` and
        their boolean mask in ``var['is_ecm']``.
    keep_col, name_col, ecm_col
        Panel column names after surrounding whitespace is stripped.

    Returns
    -------
    PanelSummary
        A compact notebook-friendly representation plus structured counts and
        marker-name tuples.
    """
    panel_df = _load_dataframe(panel)
    panel_df.columns = [str(c).strip() for c in panel_df.columns]
    missing = [c for c in (name_col, ecm_col) if c not in panel_df.columns]
    if missing:
        raise ValueError(f"Panel is missing required columns {missing}; available: {list(panel_df.columns)}.")

    retained_mask = panel_df[keep_col].eq(1) if keep_col in panel_df.columns else pd.Series(True, index=panel_df.index)
    retained = panel_df.loc[retained_mask].reset_index(drop=True)
    retained_names = retained[name_col].astype(str).tolist()
    retained_ecm = retained[ecm_col].eq(1).to_numpy(dtype=bool)

    image_channels: int | None = None
    if image is not None:
        from mantpy.im import ImageContainer

        if isinstance(image, ImageContainer):
            image_channels = int(image.n_channels)
        elif isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(f"Image must be channel-first (C, H, W); got shape {image.shape}.")
            image_channels = int(image.shape[0])
        else:
            import tifffile

            with tifffile.TiffFile(str(image)) as tif:
                shape = tuple(tif.series[0].shape)
            if len(shape) != 3:
                raise ValueError(f"Image must be channel-first (C, H, W); TIFF reports shape {shape}.")
            image_channels = int(shape[0])
        if image_channels != len(retained):
            raise ValueError(
                f"Retained panel has {len(retained)} rows but image has {image_channels} channels. "
                "Filter and order the panel to match the image stack."
            )

    if adata is not None:
        if adata.n_vars != len(retained):
            raise ValueError(f"Retained panel has {len(retained)} rows but AnnData has {adata.n_vars} variables.")
        if list(map(str, adata.var_names)) != retained_names:
            raise ValueError("AnnData var_names do not match the retained panel marker names in order.")
        if "is_ecm" not in adata.var:
            raise ValueError("AnnData is missing var['is_ecm']; create it with mt.io.read_imc().")
        observed_ecm = adata.var["is_ecm"].to_numpy(dtype=bool)
        if not np.array_equal(observed_ecm, retained_ecm):
            raise ValueError("AnnData var['is_ecm'] does not match the retained panel ECM flags.")

    return PanelSummary(
        acquired_channels=len(panel_df),
        retained_channels=len(retained),
        image_channels=image_channels,
        ecm_markers=tuple(np.asarray(retained_names, dtype=object)[retained_ecm].tolist()),
        non_ecm_markers=tuple(np.asarray(retained_names, dtype=object)[~retained_ecm].tolist()),
        adata_checked=adata is not None,
    )


def sample_group_map(
    metadata: pd.DataFrame,
    *,
    group_col: str,
    samples: Sequence[str] | None = None,
    sample_col: str = "sample_id",
) -> dict[str, str]:
    """Build a validated ``sample -> group`` map from a metadata table."""
    missing = [c for c in (sample_col, group_col) if c not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns {missing}; available: {list(metadata.columns)}.")
    frame = metadata[[sample_col, group_col]].dropna().copy()
    frame[sample_col] = frame[sample_col].astype(str)
    frame[group_col] = frame[group_col].astype(str)
    if frame[sample_col].duplicated().any():
        dupes = sorted(frame.loc[frame[sample_col].duplicated(keep=False), sample_col].unique())
        raise ValueError(f"Metadata contains duplicate sample identifiers: {dupes}.")
    mapping = dict(zip(frame[sample_col], frame[group_col], strict=True))
    if samples is not None:
        requested = [str(s) for s in samples]
        missing_samples = [s for s in requested if s not in mapping]
        if missing_samples:
            raise ValueError(f"Metadata has no {group_col!r} value for samples {missing_samples}.")
        mapping = {s: mapping[s] for s in requested}
    return mapping


def read_imc(
    img: str | Path | np.ndarray | ImageContainer,
    panel: str | Path | pd.DataFrame,
    cells: str | Path | pd.DataFrame | None = None,
    *,
    normalize: Literal["min-max", "znorm", "none"] = "min-max",
    cell_type_col: str = "celltype",
    x_col: str = "centroid-1",
    y_col: str = "centroid-0",
    sample_id: str | None = None,
    condition: str | None = None,
) -> AnnData:
    """Read IMC (Imaging Mass Cytometry) data into an AnnData object.

    Parameters
    ----------
    img
        Path to a ``(C, H, W)`` TIFF file, a numpy array of that shape, or
        an :class:`~mantpy.im.ImageContainer`.
    panel
        Path to a CSV or a DataFrame with columns ``name`` (marker name) and
        ``ecm`` (1 = ECM marker, 0 = non-ECM marker).
    cells
        Optional path to a CSV or DataFrame with per-cell measurements.  Must
        contain ``cell_type_col``, ``x_col``, and ``y_col`` columns.  Pass
        ``None`` for ECM-only analysis (no cell observations).
    normalize
        Per-channel normalisation applied to ``adata.X``.  ``"none"`` skips it.
    cell_type_col
        Column in ``cells`` that holds cell-type labels.
    x_col
        Column in ``cells`` for x-centroid (default ``"centroid-1"``).
    y_col
        Column in ``cells`` for y-centroid (default ``"centroid-0"``).
    sample_id
        String identifier for this ROI / sample.
    condition
        Experimental condition label.

    Returns
    -------
    AnnData
        Cells × markers AnnData with the full Mantpy storage contract populated.

    Notes
    -----
    Storage contract written by this function:

    - ``adata.X``               — normalised per-cell mean expression
    - ``adata.obs``             — cell_type, x, y, sample_id, condition
    - ``adata.var``             — marker_name, is_ecm
    - ``adata.obsm['spatial']`` — (N, 2) pixel coordinates
    - ``adata.layers['raw']``   — raw (unnormalised) expression
    - ``adata.uns['spatial'][sample_id]['images']['hires']``
                                — max-projected visualisation image (Squidpy-compatible slot)
    - ``adata.uns['image_container']``
                                — H5AD-safe image mapping; recover the object interface with
                                  :func:`mantpy.im.as_image_container`
    - ``adata.uns['mantpy']``   — params log
    """
    from mantpy.im import ImageContainer as _IC

    # ---- normalise img → ImageContainer ---------------------------------
    if isinstance(img, _IC):
        ic = img
    else:
        ic = _IC(_load_img_array(img))

    img_array = ic.to_array()  # (C, H, W)  – used for cell detection / expression

    panel_df = _load_dataframe(panel)
    panel_df.columns = [str(c).strip() for c in panel_df.columns]
    _validate_panel(panel_df)

    n_channels = img_array.shape[0]
    if len(panel_df) != n_channels:
        raise ValueError(f"Panel has {len(panel_df)} rows but image has {n_channels} channels. They must match 1-to-1.")

    sid = sample_id if sample_id is not None else "sample_0"

    var = pd.DataFrame(index=pd.RangeIndex(n_channels))
    var["marker_name"] = panel_df["name"].values
    var["is_ecm"] = (panel_df["ecm"] == 1).values
    var.index = var["marker_name"].astype(str)

    if cells is None:
        # ECM-only mode — zero observations, no cell data
        obs = pd.DataFrame(index=pd.Index([], dtype=str))
        X_norm = np.empty((0, n_channels), dtype=np.float32)
        X_raw = np.empty((0, n_channels), dtype=np.float32)
        spatial = np.empty((0, 2), dtype=np.float32)
    else:
        cells_df = _load_cells(cells, x_col, y_col)
        X_raw = _extract_cell_expression(img_array, cells_df, x_col, y_col)
        X_norm = _normalize_expression(X_raw, method=normalize)

        obs = pd.DataFrame(index=pd.RangeIndex(len(cells_df)))
        obs[CELLTYPE_COL] = cells_df[cell_type_col].values if cell_type_col in cells_df.columns else "unknown"
        obs[X_COL] = cells_df[x_col].values.astype(float)
        obs[Y_COL] = cells_df[y_col].values.astype(float)
        obs[SAMPLE_ID_COL] = sid
        obs[CONDITION_COL] = condition if condition is not None else "unknown"
        obs.index = obs.index.astype(str)
        spatial = np.column_stack([obs[X_COL].values, obs[Y_COL].values])

    adata = ad.AnnData(X=X_norm.astype(np.float32), obs=obs, var=var)
    adata.layers[RAW_LAYER] = X_raw.astype(np.float32)
    adata.obsm[SPATIAL_KEY] = spatial.astype(np.float32)

    # ---- Squidpy-compatible image slot ----------------------------------
    # adata.uns["spatial"][sample_id]["images"]["hires"] = (H, W) float32
    # adata.uns["spatial"][sample_id]["scalefactors"]["spot_diameter_fullres"] = 1.0
    # This mirrors the Visium/Squidpy AnnData convention so squidpy.pl.spatial_scatter
    # and similar tools work without modification.
    vis_img = ic.max_projection()  # (H, W) float32, normalised 0..1
    if "spatial" not in adata.uns:
        adata.uns["spatial"] = {}
    adata.uns["spatial"][sid] = {
        "images": {"hires": vis_img},
        "scalefactors": {"spot_diameter_fullres": 1.0, "tissue_hires_scalef": 1.0},
    }
    # Also keep the flat key for backward compat with existing pl.py code
    adata.uns["img"] = vis_img

    # Store only AnnData-native values. Consumers recover the ImageContainer
    # interface with mantpy.im.as_image_container().
    adata.uns[IMAGE_CONTAINER_KEY] = ic.to_serializable()

    adata.uns[MANTPY_UNS_KEY] = {
        "io": {
            "normalize": normalize,
            "cell_type_col": cell_type_col,
            "x_col": x_col,
            "y_col": y_col,
            "sample_id": sid,
            "condition": condition,
            "n_cells": adata.n_obs,
            "n_channels": n_channels,
        }
    }

    return adata


def read_imc_folder(
    folder: str | Path,
    panel: str | Path | pd.DataFrame,
    cells: str | Path | pd.DataFrame | None = None,
    *,
    channel_col: str = "channel",
    pattern: str = "*.tiff",
    sample_id: str | None = None,
    condition: str | None = None,
    **read_imc_kwargs,
) -> AnnData:
    """Read an IMC ROI stored as one single-channel TIFF per marker.

    Many IMC exports (PancDB, MCD-extract, custom acquisition scripts) write a
    folder per ROI containing one ``<prefix>_<metal>_<marker>.tiff`` file per
    panel channel.  This reader matches each panel row's ``channel`` value
    (e.g. ``160Gd``) against the filenames in ``folder``, stacks the matched
    images into a ``(C, H, W)`` array in panel order, and delegates to
    :func:`read_imc`.

    Parameters
    ----------
    folder
        Directory containing one TIFF per channel.
    panel
        CSV path or DataFrame with columns ``channel`` (metal tag),
        ``name`` (marker), and ``ecm`` (1 = ECM marker).
    cells
        Optional per-cell measurements forwarded to :func:`read_imc`.
    channel_col
        Column in ``panel`` that holds the metal tag used to match filenames.
        Default ``"channel"``.
    pattern
        Glob pattern for TIFF discovery within ``folder``.  Default
        ``"*.tiff"``.  Use ``"*.ome.tiff"`` if needed.
    sample_id
        Forwarded to :func:`read_imc`.  Defaults to the folder name.
    condition
        Forwarded to :func:`read_imc`.
    **read_imc_kwargs
        Any other :func:`read_imc` keyword arguments (``normalize``,
        ``cell_type_col``, ``x_col``, ``y_col``, ...).

    Returns
    -------
    AnnData
        Cells x markers AnnData with the Mantpy storage contract; identical
        to what :func:`read_imc` returns.

    Raises
    ------
    FileNotFoundError
        If a panel row's channel tag has no matching file in ``folder``.
    ValueError
        If two files match the same channel, or images disagree on ``(H, W)``.

    Examples
    --------
    >>> import mantpy as mt  # doctest: +SKIP
    >>> adata = mt.io.read_imc_folder(  # doctest: +SKIP
    ...     "path/to/imc/sample_01",
    ...     panel="path/to/imc/panel.csv",
    ...     pattern="*.ome.tiff",
    ...     condition="Infected",
    ... )
    """
    import tifffile

    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder_path}")

    panel_df = _load_dataframe(panel)
    panel_df.columns = [str(c).strip() for c in panel_df.columns]
    _validate_panel(panel_df)
    if channel_col not in panel_df.columns:
        raise ValueError(
            f"panel is missing the channel column '{channel_col}'. Available columns: {list(panel_df.columns)}."
        )

    files = sorted(folder_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder_path}.")

    # Match each panel row to exactly one file by metal tag substring.
    matched: list[Path] = []
    for metal in panel_df[channel_col].astype(str):
        token = f"_{metal}_"
        hits = [f for f in files if token in f.name]
        if not hits:
            raise FileNotFoundError(
                f"No file in {folder_path} matches channel '{metal}' (looking for token '{token}' in filenames)."
            )
        if len(hits) > 1:
            raise ValueError(
                f"Multiple files in {folder_path} match channel '{metal}': "
                f"{[h.name for h in hits]}. Tighten the panel or rename files."
            )
        matched.append(hits[0])

    first = tifffile.imread(str(matched[0]))
    if first.ndim != 2:
        raise ValueError(f"Expected 2-D single-channel images; got {first.shape} for {matched[0].name}.")
    H, W = first.shape
    stack = np.empty((len(matched), H, W), dtype=np.float32)
    stack[0] = first.astype(np.float32, copy=False)
    for i, p in enumerate(matched[1:], start=1):
        arr = tifffile.imread(str(p))
        if arr.shape != (H, W):
            raise ValueError(
                f"Image shape mismatch in {folder_path}: {matched[0].name} is {(H, W)} but {p.name} is {arr.shape}."
            )
        stack[i] = arr.astype(np.float32, copy=False)

    sid = sample_id if sample_id is not None else folder_path.name

    return read_imc(
        stack,
        panel_df,
        cells,
        sample_id=sid,
        condition=condition,
        **read_imc_kwargs,
    )


def to_spatialdata(
    adatas: dict[str, AnnData],
) -> spatialdata.SpatialData:
    """Bundle multiple ROI AnnData objects into a :class:`spatialdata.SpatialData`.

    Each AnnData must have been produced by :func:`read_imc` (or
    :func:`read_codex`) so that ``adata.uns['img']`` exists.

    Parameters
    ----------
    adatas
        Mapping of ``{sample_id: AnnData}``.

    Returns
    -------
    spatialdata.SpatialData
        A container with each ROI stored as ``sdata.images[sample_id]``. When
        at least one ROI has ``adata.uns['ecm_patches']``, the concatenated
        patch table is available as ``sdata.tables["ecm_patches"]``.

    Notes
    -----
    Requires the ``[spatial]`` optional dependency::

        pip install mantpy[spatial]
    """
    try:
        import spatialdata as sd
        from spatialdata.models import Image2DModel
    except ImportError as exc:
        raise ImportError("to_spatialdata requires spatialdata. Install with: pip install mantpy[spatial]") from exc

    images: dict = {}
    ecm_patch_frames: list[pd.DataFrame] = []

    for sid, adata in adatas.items():
        if "img" not in adata.uns:
            raise ValueError(f"ROI '{sid}' is missing adata.uns['img']. Ensure it was created with mt.read_imc().")
        # uns['img'] is written by two readers with different shapes:
        # read_imc stores a 2-D (H, W) visualisation projection, while
        # read_ecm_image stores the full (C, H, W) stack. Promote the 2-D case
        # to single-channel rather than projecting the 3-D case down, which
        # would silently drop every channel but one.
        img = adata.uns["img"]
        if img.ndim not in (2, 3):
            raise ValueError(f"ROI '{sid}': adata.uns['img'] must be (H, W) or (C, H, W); got shape {img.shape}.")
        img_chw = img if img.ndim == 3 else img[np.newaxis]
        images[sid] = Image2DModel.parse(
            img_chw,
            dims=("c", "y", "x"),
        )

        if "ecm_patches" in adata.uns:
            df = adata.uns["ecm_patches"].copy()
            df["region"] = sid
            ecm_patch_frames.append(df)

    tables: dict = {}
    if ecm_patch_frames:
        combined = pd.concat(ecm_patch_frames, ignore_index=True)
        n = len(combined)
        feat_cols = [c for c in combined.columns if c.startswith("feat_")]
        X = combined[feat_cols].values.astype(np.float32) if feat_cols else np.empty((n, 0), dtype=np.float32)
        obs_df = pd.DataFrame(
            {
                "region": pd.Categorical(combined["region"].values),
                "x": combined["x"].values,
                "y": combined["y"].values,
                "ecm_cluster": combined["ecm_cluster"].values,
                # Row id within its own region. TableModel requires an
                # instance_key whenever region_key is set, and the ids must be
                # unique per region — a global RangeIndex would not be.
                "instance_id": combined.groupby("region", sort=False).cumcount().values,
            },
            index=combined.index.astype(str),
        )
        patch_adata = ad.AnnData(X=X, obs=obs_df)
        # Patch centroids belong in obsm['spatial'], where squidpy and
        # scanpy look for coordinates — not only as obs columns.
        patch_adata.obsm["spatial"] = combined[["x", "y"]].to_numpy(dtype=np.float64)
        tables["ecm_patches"] = sd.models.TableModel.parse(
            patch_adata,
            # Only the ROIs actually represented in the table — `adatas` may
            # include ROIs that carry no ecm_patches at all.
            region=list(obs_df["region"].cat.categories),
            region_key="region",
            instance_key="instance_id",
        )

    return sd.SpatialData(images=images, tables=tables)


def read_codex(
    img: str | Path | np.ndarray | ImageContainer,
    panel: str | Path | pd.DataFrame,
    cells: str | Path | pd.DataFrame | None = None,
    **kwargs,
) -> AnnData:
    """Read CODEX data — a named alias for :func:`read_imc`.

    CODEX and IMC are read identically: both are a multiplexed image plus a
    panel and an optional cell table, and the resulting AnnData is the same.
    This alias exists only so that ``read_codex`` is discoverable; it applies no
    CODEX-specific defaults, and passes everything straight through.

    (It previously set ``cell_type_col``/``x_col``/``y_col`` explicitly, but
    every value was identical to :func:`read_imc`'s own default, which wrongly
    implied CODEX needed different handling.)

    Parameters
    ----------
    img, panel, cells
        Same as :func:`read_imc`.  ``img`` may be an
        :class:`~mantpy.im.ImageContainer` for lazy/tiled loading.
    **kwargs
        Forwarded to :func:`read_imc`.

    Returns
    -------
    AnnData
    """
    return read_imc(img, panel, cells, **kwargs)


def read_ecm_image(
    img: str | Path | np.ndarray | ImageContainer,
    *,
    marker_names: list[str] | None = None,
    sample_id: str | None = None,
) -> AnnData:
    """Create an AnnData from a standalone ECM image (no cells or panel needed).

    Use this when you have raw ECM marker images (e.g. single-channel ColIV
    TIFFs) and want to run the patch extraction → graph building pipeline
    without a full IMC/CODEX panel and cell segmentation.

    Parameters
    ----------
    img
        ``(H, W)`` or ``(C, H, W)`` numpy array, path to a TIFF file, or
        :class:`~mantpy.im.ImageContainer`.  2-D inputs are promoted to
        ``(1, H, W)``.
    marker_names
        Names for each channel.  Length must match ``C``.  If ``None``,
        defaults to ``["channel_0", "channel_1", ...]``.
    sample_id
        String identifier for this image / ROI.

    Returns
    -------
    AnnData
        Zero-observation AnnData with ``adata.var['is_ecm'] = True`` for all
        channels, ready for :func:`~mantpy.pp.extract_ecm_patches`.

    Notes
    -----
    The complete channel-first image stack is stored as NumPy arrays under
    ``adata.uns['image_container']['layers']``. The surrounding mapping is
    H5AD-safe and can be converted back to the object interface with
    :func:`mantpy.im.as_image_container`. ``adata.uns['img']`` remains the
    full stack for backward compatibility.
    """
    from mantpy.im import ImageContainer

    if isinstance(img, ImageContainer):
        ic = img
    else:
        ic = ImageContainer(_load_img_array(img))

    img_array = ic.to_array()  # (C, H, W)
    C, H, W = img_array.shape

    sid = sample_id if sample_id is not None else "sample_0"

    if marker_names is None:
        marker_names = [f"channel_{i}" for i in range(C)]
    if len(marker_names) != C:
        raise ValueError(f"marker_names has {len(marker_names)} entries but image has {C} channels.")

    var = pd.DataFrame(index=pd.Index(marker_names, dtype=str))
    var["marker_name"] = marker_names
    var["is_ecm"] = True

    obs = pd.DataFrame(index=pd.Index([], dtype=str))
    X = np.empty((0, C), dtype=np.float32)
    spatial = np.empty((0, 2), dtype=np.float32)

    adata = AnnData(X=X, obs=obs, var=var)
    adata.obsm[SPATIAL_KEY] = spatial

    hires = img_array.max(axis=0).astype(np.float32)
    mx = hires.max()
    if mx > 0:
        hires = hires / mx

    adata.uns["spatial"] = {sid: {"images": {"hires": hires}, "scalefactors": {"tissue_hires_scalef": 1.0}}}
    adata.uns["img"] = img_array

    # Store only AnnData-native values. Consumers recover the ImageContainer
    # interface with mantpy.im.as_image_container().
    adata.uns[IMAGE_CONTAINER_KEY] = ic.to_serializable()

    adata.uns[MANTPY_UNS_KEY] = {
        "io": {
            "read_ecm_image": {
                "n_channels": C,
                "height": H,
                "width": W,
                "marker_names": marker_names,
                "sample_id": sid,
            }
        }
    }

    return adata


def cell_features_from_mask(
    mask: np.ndarray,
    img: np.ndarray | str | Path,
    channel_names: Sequence[str],
    *,
    is_ecm: Sequence[bool] | str | Sequence[str] | None = None,
    channel_indices: Sequence[int] | None = None,
    normalize: Literal["arcsinh_clip", "none"] = "arcsinh_clip",
    cofactor: float = 1.0,
    clip_percentile: float = 99.5,
    max_workers: int = 2,
    sample_id: str | None = None,
    condition: str | None = None,
) -> AnnData:
    """Aggregate per-cell mean expression from a labelled segmentation mask.

    Bridges :func:`mantpy.pp.segment_cells` / :func:`mantpy.pp.segment_cells_tiled`
    (which produce a label image) and the rest of the mantpy / scverse API
    (which expects an :class:`~anndata.AnnData`). Computes per-cell means via
    row-chunked :func:`numpy.bincount` so a float64 copy of the full image
    (which :func:`scipy.ndimage.mean` would make) is never held in RAM. When
    ``img`` is a path, channels are streamed one at a time with a thread pool.

    Parameters
    ----------
    mask
        ``(H, W)`` integer label image (``0`` = background).
    img
        Either an in-memory ``(C, H, W)`` array or a path to a multi-page
        TIFF (one page per channel).
    channel_names
        Names for ``adata.var_names``. Length must match the channel count.
    is_ecm
        Either a boolean iterable of length ``len(channel_names)`` or a name
        / list of names that should be flagged True. Defaults to ``False``
        for every channel (caller can set later).
    channel_indices
        When ``img`` is a path, optionally restrict to these page indices.
        Ignored otherwise.
    normalize
        ``"arcsinh_clip"`` (default) applies ``arcsinh(x / cofactor)`` per
        channel and then clips at the per-channel ``clip_percentile`` and
        rescales to ``[0, 1]`` — the IMC/CODEX standard. ``"none"`` keeps
        raw means.
    cofactor, clip_percentile
        Forwarded to the normalisation step.
    max_workers
        Threads used to stream channels when ``img`` is a path.
    sample_id, condition
        Optional values written to every row of ``adata.obs``.

    Returns
    -------
    AnnData
        Cells × markers AnnData with:

        - ``X`` — normalised per-cell mean expression (float32)
        - ``layers['raw']`` — un-normalised means
        - ``obs['centroid_x', 'centroid_y', 'area']``
        - ``obsm['spatial']`` — ``(N, 2)`` pixel coordinates
        - ``var['is_ecm']`` — boolean

    Examples
    --------
    >>> # in-memory: mask + (C, H, W) array
    >>> adata = mt.io.cell_features_from_mask(mask, img_stack, channel_names)  # doctest: +SKIP

    >>> # streaming from disk: mask + multi-page TIFF
    >>> adata = mt.io.cell_features_from_mask(  # doctest: +SKIP
    ...     mask,
    ...     "slide.ome.tif",
    ...     channel_names=channel_names,
    ...     is_ecm=["Col4A1", "Col1A1"],
    ... )
    """
    from concurrent.futures import ThreadPoolExecutor

    from scipy.ndimage import center_of_mass, find_objects

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W); got {mask.shape}.")
    H, W = mask.shape
    n_cells = int(mask.max())

    var_names = list(channel_names)
    n_vars = len(var_names)

    # Resolve is_ecm flag
    if is_ecm is None:
        ecm_flags = [False] * n_vars
    elif isinstance(is_ecm, str):
        ecm_flags = [n == is_ecm for n in var_names]
    elif isinstance(is_ecm, list | tuple) and len(is_ecm) > 0 and isinstance(is_ecm[0], str):
        ecm_set = set(is_ecm)
        ecm_flags = [n in ecm_set for n in var_names]
    else:
        ecm_flags = [bool(b) for b in is_ecm]
        if len(ecm_flags) != n_vars:
            raise ValueError(f"is_ecm length ({len(ecm_flags)}) must match channel_names length ({n_vars}).")

    # Empty-mask early return (caller convenience for batch loops)
    if n_cells == 0:
        return _empty_cell_features_anndata(var_names, ecm_flags, sample_id, condition)

    # Centroids and areas
    slices = find_objects(mask)
    coms = center_of_mass(mask > 0, mask, range(1, n_cells + 1))
    centroids = np.array([[c[1], c[0]] for c in coms], dtype=np.float32)
    areas = np.array(
        [int((mask[s] == (i + 1)).sum()) if s is not None else 0 for i, s in enumerate(slices)], dtype=np.int32
    )

    # Drop phantom labels (area == 0). Tile-based segmentation can produce gaps
    # in the label sequence after stitching — those labels have no pixels, so
    # center_of_mass returns NaN. Keeping them poisons downstream Delaunay/k-NN.
    valid = areas > 0
    if not valid.all():
        keep_idx = np.where(valid)[0]
        centroids = centroids[keep_idx]
        areas = areas[keep_idx]
        slices = [slices[i] for i in keep_idx]
        # Rewrite the mask so labels are dense 1..N_valid (so per-channel mean
        # extraction below stays correct when it indexes by label value).
        relabel = np.zeros(n_cells + 1, dtype=np.int32)
        relabel[keep_idx + 1] = np.arange(1, len(keep_idx) + 1, dtype=np.int32)
        mask = relabel[mask]
        n_cells = int(len(keep_idx))
        # No print statement here — the caller logs cell counts. Phantom drops
        # are surfaced indirectly: AnnData will report n_obs < mask.max() pre-fix.

    # Per-cell mean — dispatch on img type
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = img[np.newaxis]
        if img.ndim != 3:
            raise ValueError(f"img array must be (C, H, W); got {img.shape}.")
        if img.shape[0] != n_vars:
            raise ValueError(f"img has {img.shape[0]} channels but channel_names has {n_vars}.")
        rows = [_cell_mean_chunked(img[c].astype(np.float32, copy=False), mask, n_cells) for c in range(n_vars)]
    else:
        # path: stream channels lazily via tifffile + threads
        import tifffile

        idx_list = list(channel_indices) if channel_indices is not None else list(range(n_vars))
        if len(idx_list) != n_vars:
            raise ValueError(f"channel_indices length ({len(idx_list)}) must match channel_names ({n_vars}).")
        path_str = str(img)

        def _load(ch_idx: int) -> np.ndarray:
            with tifffile.TiffFile(path_str) as tif:
                arr = (tif.pages[ch_idx].asarray() if len(tif.pages) > 1 else tif.asarray()[ch_idx]).astype(np.float32)
            return arr[:H, :W] if (arr.shape[0] > H or arr.shape[1] > W) else arr

        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            rows = list(
                pool.map(
                    lambda ch: _cell_mean_chunked(_load(ch), mask, n_cells),
                    idx_list,
                )
            )

    X_raw = np.stack(rows, axis=1).astype(np.float32)
    np.nan_to_num(X_raw, copy=False, nan=0.0)

    # Build AnnData
    obs = pd.DataFrame(
        {
            "centroid_x": centroids[:, 0],
            "centroid_y": centroids[:, 1],
            "area": areas,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    if sample_id is not None:
        obs[SAMPLE_ID_COL] = sample_id
    if condition is not None:
        obs[CONDITION_COL] = condition

    var = pd.DataFrame({"is_ecm": ecm_flags}, index=var_names)

    adata = ad.AnnData(X=X_raw.copy(), obs=obs, var=var)
    adata.obsm[SPATIAL_KEY] = centroids
    adata.layers[RAW_LAYER] = X_raw

    if normalize == "arcsinh_clip":
        adata.X = _arcsinh_clip_norm(adata.X, cofactor=cofactor, clip_percentile=clip_percentile)
    elif normalize != "none":
        raise ValueError(f"normalize must be 'arcsinh_clip' or 'none', got {normalize!r}.")

    adata.uns[MANTPY_UNS_KEY] = {
        "io": {
            "cell_features_from_mask": {
                "n_cells": n_cells,
                "n_channels": n_vars,
                "normalize": normalize,
                "from_path": not isinstance(img, np.ndarray),
            }
        }
    }

    return adata


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _cell_mean_chunked(
    img: np.ndarray,
    mask: np.ndarray,
    n_cells: int,
    chunk: int = 1024,
) -> np.ndarray:
    """Per-cell mean intensity via row-chunked bincount (memory-bounded)."""
    sums = np.zeros(n_cells + 1, dtype=np.float64)
    counts = np.zeros(n_cells + 1, dtype=np.int64)
    for r0 in range(0, img.shape[0], chunk):
        m = mask[r0 : r0 + chunk].ravel()
        fg = m > 0
        if not fg.any():
            continue
        sums += np.bincount(
            m[fg],
            weights=img[r0 : r0 + chunk].ravel()[fg].astype(np.float64),
            minlength=n_cells + 1,
        )
        counts += np.bincount(m[fg], minlength=n_cells + 1)
    return np.where(counts[1:] > 0, sums[1:] / counts[1:], 0.0).astype(np.float32)


def _arcsinh_clip_norm(
    X: np.ndarray,
    *,
    cofactor: float = 1.0,
    clip_percentile: float = 99.5,
) -> np.ndarray:
    """Per-column arcsinh + percentile-clip rescale to [0, 1] (IMC/CODEX standard)."""
    X = np.arcsinh(X / cofactor).astype(np.float32)
    for j in range(X.shape[1]):
        hi = float(np.percentile(X[:, j], clip_percentile))
        if hi > 0:
            X[:, j] = np.clip(X[:, j] / hi, 0.0, 1.0)
    return X


def _empty_cell_features_anndata(
    var_names: list[str],
    ecm_flags: list[bool],
    sample_id: str | None,
    condition: str | None,
) -> AnnData:
    """Empty-but-typed AnnData for masks with zero foreground cells."""
    n = 0
    obs_cols = {
        "centroid_x": np.zeros(n, dtype=np.float32),
        "centroid_y": np.zeros(n, dtype=np.float32),
        "area": np.zeros(n, dtype=np.int32),
    }
    if sample_id is not None:
        obs_cols[SAMPLE_ID_COL] = np.array([], dtype=object)
    if condition is not None:
        obs_cols[CONDITION_COL] = np.array([], dtype=object)
    obs = pd.DataFrame(obs_cols)
    var = pd.DataFrame({"is_ecm": ecm_flags}, index=var_names)
    X = np.zeros((0, len(var_names)), dtype=np.float32)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm[SPATIAL_KEY] = np.zeros((0, 2), dtype=np.float32)
    adata.layers[RAW_LAYER] = X.copy()
    return adata


# ---------------------------------------------------------------------------
# Internal helpers (unchanged)
# ---------------------------------------------------------------------------


def _load_img_array(img: str | Path | np.ndarray) -> np.ndarray:
    """Load img path or ndarray to a (C, H, W) float32 array."""
    if isinstance(img, np.ndarray):
        arr = img
    else:
        import tifffile

        arr = tifffile.imread(str(img))

    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"Image must be (C, H, W) but got shape {arr.shape}.")
    return arr.astype(np.float32)


def _load_dataframe(src: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.copy()
    return pd.read_csv(str(src))


def _validate_panel(panel: pd.DataFrame) -> None:
    for col in ("name", "ecm"):
        if col not in panel.columns:
            raise ValueError(f"Panel must have columns 'name' and 'ecm', missing: '{col}'.")


def _load_cells(
    cells: str | Path | pd.DataFrame,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    df = _load_dataframe(cells)
    for col in (x_col, y_col):
        if col not in df.columns:
            raise ValueError(f"cells DataFrame must have column '{col}'. Available columns: {list(df.columns)}.")
    return df.reset_index(drop=True)


def _extract_cell_expression(
    img: np.ndarray, cells: pd.DataFrame, x_col: str, y_col: str, radius: int = 3
) -> np.ndarray:
    """Mean expression in a square window around each cell centroid."""
    C, H, W = img.shape
    n_cells = len(cells)
    X = np.zeros((n_cells, C), dtype=np.float32)
    xs = cells[x_col].values.astype(int)
    ys = cells[y_col].values.astype(int)

    for i, (x, y) in enumerate(zip(xs, ys, strict=False)):
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        patch = img[:, y0:y1, x0:x1]
        X[i] = patch.mean(axis=(1, 2))

    return X


def _normalize_expression(X: np.ndarray, method: Literal["min-max", "znorm", "none"]) -> np.ndarray:
    if method == "none":
        return X.copy()
    X = X.copy().astype(np.float32)
    if method == "min-max":
        mn = X.min(axis=0, keepdims=True)
        mx = X.max(axis=0, keepdims=True)
        denom = np.where(mx - mn == 0, 1.0, mx - mn)
        return (X - mn) / denom
    if method == "znorm":
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma = np.where(sigma == 0, 1.0, sigma)
        return (X - mu) / sigma
    raise ValueError(f"Unknown normalize method '{method}'. Use 'min-max', 'znorm', or 'none'.")
