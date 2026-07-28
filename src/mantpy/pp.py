"""Preprocessing functions for Mantpy."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from mantpy._constants import (
    CELL_ECM_GRAPH_KEY,
    ECM_GRAPH_KEY,
    ECM_IMAGE_KEY,
    ECM_PATCHES_KEY,
    IMAGE_CONTAINER_KEY,
    RAW_LAYER,
)
from mantpy._core._patching import (
    build_ecm_image,
    build_ecm_image_from_coords,
    build_ecm_patch_dataframe,
    cluster_patches,
    compute_features,
    extract_patches,
    remove_background,
    resolve_extractors,
)
from mantpy._utils import log_params as _log_params

if TYPE_CHECKING:
    from pathlib import Path

    from mantpy.im import ImageContainer

__all__ = [
    "BackgroundRemovalSummary",
    "ClusterCountSelection",
    "CellSegmentationSummary",
    "ECMClusteringResult",
    "ECMLabelOverlaySummary",
    "ECMLeidenResolutionSelection",
    "ECMPatchSummary",
    "HEPreprocessingResult",
    "HEECMPatchSummary",
    "PatchComparison",
    "annotate_structure",
    "attach_ecm_patches",
    "apply_ecm_label_overlay",
    "cell_segmentation_summary",
    "cluster_ecm_patches",
    "compare_ecm_patches",
    "ecm_patches_from_images",
    "ecm_patch_summary",
    "ecm_label_overlay_summary",
    "extract_ecm_patches",
    "extract_ecm_patches_cohort",
    "extract_structure_ecm",
    "he_ecm_patches",
    "he_ecm_patch_summary",
    "image_ecm_patches",
    "normalize",
    "preprocess_ecm",
    "preprocess_he",
    "remove_background_patches",
    "segment_cells",
    "segment_cells_tiled",
    "select_ecm_cluster_count",
    "select_ecm_leiden_resolution",
    "split_structures",
]


_SQUIDPY_IMAGE_FEATURE_FAMILIES = ("summary", "texture", "histogram")


def _package_version(distribution: str) -> str:
    """Return an installed distribution version or a stable fallback."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version

    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _block_mean_2d(array: np.ndarray, size: int) -> np.ndarray:
    """Mean-reduce complete ``size`` square blocks from a 2-D array."""
    height, width = array.shape
    height = (height // size) * size
    width = (width // size) * size
    return array[:height, :width].reshape(height // size, size, width // size, size).mean((1, 3))


def _two_means_threshold(values: np.ndarray, *, random_state: int) -> float:
    """Midpoint between two one-dimensional KMeans centres."""
    from sklearn.cluster import KMeans

    # float64 makes one-dimensional KMeans invariant to OpenMP reduction
    # order; float32 centres can move enough to flip boundary patches.
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if not values.size or float(values.max()) <= 0:
        return float("inf")
    if np.unique(values).size == 1:
        return float(values[0])
    if values.size > 200_000:
        values = np.random.default_rng(random_state).choice(values, 200_000, replace=False)
    model = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit(values[:, None])
    return float(np.sort(model.cluster_centers_.ravel()).mean())


def _grid8_connectivities(active: np.ndarray) -> sp.csr_matrix:
    """Eight-connected adjacency over row-major foreground patches."""
    ys, xs = np.nonzero(active)
    node_id = np.full(active.shape, -1, dtype=int)
    node_id[ys, xs] = np.arange(len(ys))
    edges: list[np.ndarray] = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        shifted = np.roll(active, (-dy, -dx), axis=(0, 1))
        valid = active & shifted
        if dy:
            valid[-1, :] = False
        if dx == 1:
            valid[:, -1] = False
        elif dx == -1:
            valid[:, 0] = False
        rows, cols = np.nonzero(valid)
        if rows.size:
            edges.append(np.column_stack((node_id[rows, cols], node_id[rows + dy, cols + dx])))
    if not edges:
        return sp.csr_matrix((len(ys), len(ys)), dtype=np.float32)
    pairs = np.concatenate(edges)
    graph = sp.csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (pairs[:, 0], pairs[:, 1])),
        shape=(len(ys), len(ys)),
    )
    return graph.maximum(graph.T)


def _squidpy_image_features(
    patches: np.ndarray,
    *,
    patch_ids: Sequence[str],
    channel_names: Sequence[str],
    feature_families: Sequence[str],
    feature_kwargs: Mapping[str, Mapping[str, Any]],
    n_jobs: int,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    """Calculate public Squidpy image features on patch-local image tiles."""
    import squidpy as sq
    from joblib import Parallel, delayed

    # Squidpy represents independent images along its z dimension.  Treating
    # retained patches as z planes lets its public feature methods operate on
    # the exact tiles stored in ``obsm['image_patches']`` without fabricating
    # Visium ``uns['spatial']`` metadata or recropping the source image.
    patch_stack = np.transpose(patches, (2, 3, 0, 1))  # y, x, patch, channel
    container = sq.im.ImageContainer(
        patch_stack,
        layer="patches",
        dims=("y", "x", "z", "channels"),
        library_id=list(patch_ids),
        lazy=False,
        copy=False,
    )

    def calculate_one(patch_id: str) -> dict[str, dict[str, float]]:
        return {
            family: dict(
                getattr(container, f"features_{family}")(
                    layer="patches",
                    library_id=patch_id,
                    **feature_kwargs.get(family, {}),
                )
            )
            for family in feature_families
        }

    first = calculate_one(str(patch_ids[0]))
    # Squidpy groups keys by family and then channel.  Expose a stable,
    # marker-major schema so each marker's requested families remain adjacent.
    specifications: list[tuple[str, int, str]] = []
    for channel in range(len(channel_names)):
        channel_token = f"_ch-{channel}_"
        for family in feature_families:
            keys = [key for key in first[family] if channel_token in key]
            if not keys:
                raise RuntimeError(f"Squidpy feature family {family!r} returned no values for channel {channel}.")
            specifications.extend((family, channel, key) for key in keys)

    def flatten(values: Mapping[str, Mapping[str, float]]) -> np.ndarray:
        return np.asarray([values[family][key] for family, _channel, key in specifications], dtype=np.float32)

    def calculate_and_flatten(patch_id: str) -> np.ndarray:
        return flatten(calculate_one(str(patch_id)))

    rows = [flatten(first)]
    if len(patch_ids) > 1:
        rows.extend(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(calculate_and_flatten)(patch_id) for patch_id in patch_ids[1:]
            )
        )
    features = np.stack(rows).astype(np.float32, copy=False)

    records: list[dict[str, Any]] = []
    feature_ids: list[str] = []
    for family, channel, squidpy_key in specifications:
        marker = str(channel_names[channel])
        local_feature = squidpy_key.replace(f"_ch-{channel}_", "_", 1)
        feature_id = f"{marker}_{local_feature}"
        feature_ids.append(feature_id)
        records.append(
            {
                "marker": marker,
                "feature": local_feature,
                "feature_family": family,
                "feature_source": "squidpy",
                "squidpy_feature": squidpy_key,
                "squidpy_channel": channel,
            }
        )
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError(
            "Squidpy produced duplicate feature identifiers. Use distinct feature_name values in feature_kwargs."
        )
    var = pd.DataFrame(records, index=pd.Index(feature_ids, name="feature_id"))
    return features, var, str(sq.__version__)


def image_ecm_patches(
    image: np.ndarray,
    *,
    channel_names: Sequence[str] | None = None,
    is_ecm: Sequence[bool] | np.ndarray | None = None,
    patch_size: int = 32,
    clip_percentile: float = 99.0,
    foreground_mode: Literal["any", "all", "mean"] = "any",
    foreground_mask: np.ndarray | None = None,
    pixel_thresholds: Sequence[float] | None = None,
    patch_thresholds: Sequence[float] | None = None,
    feature_families: Sequence[Literal["summary", "texture", "histogram"]] = _SQUIDPY_IMAGE_FEATURE_FAMILIES,
    feature_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    sample_id: str | None = None,
    random_state: int = 0,
    n_jobs: int = 1,
) -> AnnData:
    """Convert one or more raw ECM image channels into patch-level AnnData.

    Parameters
    ----------
    image
        One raw ECM image ``(H, W)`` or a channel-first stack ``(C, H, W)``.
        A stack is treated as ECM-only unless ``is_ecm`` selects a subset.
    channel_names
        Names for the input channels. Defaults to ``ECM`` for a two-dimensional
        image and ``ECM_0``, ``ECM_1``, ... for a stack.
    is_ecm
        Boolean mask over input channels. Use it when ``image`` is a full
        multiplexed acquisition rather than an ECM-only stack.
    patch_size
        Width and height of each non-overlapping graph node in pixels.
    clip_percentile
        Per-channel percentile of nonzero pixels used to clip and scale raw
        intensities to ``[0, 1]``.
    foreground_mode
        How channel-specific two-means patch masks are combined: union
        (``"any"``), intersection (``"all"``), or a new two-means split on
        the across-channel mean (``"mean"``).
    foreground_mask
        Optional upstream boolean mask, either image-sized ``(H, W)`` or
        patch-grid-sized ``(H // patch_size, W // patch_size)``. When supplied
        it defines retained patches instead of ``foreground_mode``.
    pixel_thresholds, patch_thresholds
        Optional per-selected-channel thresholds on the scaled ``[0, 1]``
        images. They make a previously locked preprocessing recipe exactly
        reproducible across library versions; when omitted, deterministic
        two-means thresholds are estimated from the current image.
    feature_families
        Squidpy image-feature families calculated on each retained node tile.
        Defaults to ``("summary", "texture", "histogram")``, which produces
        35 descriptors per selected channel with Squidpy's defaults.
    feature_kwargs
        Optional family-specific keyword arguments forwarded to
        :meth:`squidpy.im.ImageContainer.features_summary`,
        :meth:`~squidpy.im.ImageContainer.features_texture`, or
        :meth:`~squidpy.im.ImageContainer.features_histogram`.
    sample_id
        Value stored in ``obs['sample_id']``.
    random_state
        Seed for deterministic two-means threshold estimation.
    n_jobs
        Threads used for per-patch Squidpy image features. ``1`` is the
        conservative default, a positive integer uses that many workers, and
        ``-1`` uses every available CPU. Results are deterministic across
        worker counts; performance depends on patch count and hardware.

    Returns
    -------
    AnnData
        One observation per retained ECM patch. Image features are in ``X``;
        channel-aware metadata are in ``var``; graph coordinates are in
        ``obsm['spatial']``; compact scaled image tiles are in
        ``obsm['image_patches']``; and the 8-connected patch lattice is in
        ``obsp['grid_connectivities']``.

    Notes
    -----
    This complements :func:`ecm_patches_from_images`: that function preserves
    the legacy carrier-AnnData workflow for multiplexed cohorts, whereas this
    function creates observation-native AnnData for Scanpy, Squidpy, and graph
    learning directly from raw ECM channels.
    """
    raw = np.asarray(image)
    if raw.ndim == 2:
        raw = raw[None, ...]
    elif raw.ndim != 3:
        raise ValueError(f"image must have shape (H, W) or (C, H, W); got {raw.shape}.")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"image must contain numeric intensities; got dtype {raw.dtype}.")
    if patch_size < 1:
        raise ValueError("patch_size must be at least 1.")
    if not 0 < clip_percentile <= 100:
        raise ValueError("clip_percentile must lie in (0, 100].")
    if foreground_mode not in {"any", "all", "mean"}:
        raise ValueError("foreground_mode must be 'any', 'all', or 'mean'.")
    if isinstance(feature_families, str):
        feature_families = (feature_families,)
    feature_families = tuple(str(family) for family in feature_families)
    if not feature_families:
        raise ValueError("feature_families must contain at least one Squidpy feature family.")
    if len(set(feature_families)) != len(feature_families):
        raise ValueError("feature_families must not contain duplicates.")
    unknown_families = set(feature_families).difference(_SQUIDPY_IMAGE_FEATURE_FAMILIES)
    if unknown_families:
        raise ValueError(
            f"feature_families supports only {_SQUIDPY_IMAGE_FEATURE_FAMILIES}; got {sorted(unknown_families)}."
        )
    resolved_feature_kwargs = {str(family): dict(kwargs) for family, kwargs in (feature_kwargs or {}).items()}
    unused_kwargs = set(resolved_feature_kwargs).difference(feature_families)
    if unused_kwargs:
        raise ValueError(f"feature_kwargs contains unselected or unknown families: {sorted(unused_kwargs)}.")
    reserved_kwargs = {"layer", "library_id", "channels"}
    for family, kwargs in resolved_feature_kwargs.items():
        reserved = reserved_kwargs.intersection(kwargs)
        if reserved:
            raise ValueError(
                f"feature_kwargs[{family!r}] cannot override Mantpy-managed arguments: {sorted(reserved)}."
            )
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0.")

    n_input = raw.shape[0]
    if channel_names is None:
        names = ["ECM"] if n_input == 1 else [f"ECM_{i}" for i in range(n_input)]
    else:
        names = [str(name) for name in channel_names]
        if len(names) != n_input:
            raise ValueError(f"channel_names has {len(names)} entries but image has {n_input} channels.")
    if len(set(names)) != len(names):
        raise ValueError("channel_names must be unique.")
    if is_ecm is None:
        selector = np.ones(n_input, dtype=bool)
    else:
        selector = np.asarray(is_ecm)
        if selector.dtype != bool or selector.shape != (n_input,):
            raise ValueError(f"is_ecm must be a boolean mask of length {n_input}; got shape {selector.shape}.")
        if not selector.any():
            raise ValueError("is_ecm selects no channels.")
    selected_indices = np.flatnonzero(selector)
    raw = raw[selected_indices].astype(np.float32, copy=False)
    names = [names[i] for i in selected_indices]
    supplied_pixel_thresholds = None if pixel_thresholds is None else np.asarray(pixel_thresholds, dtype=float)
    supplied_patch_thresholds = None if patch_thresholds is None else np.asarray(patch_thresholds, dtype=float)
    for label, supplied in (
        ("pixel_thresholds", supplied_pixel_thresholds),
        ("patch_thresholds", supplied_patch_thresholds),
    ):
        if supplied is not None and supplied.shape != (len(names),):
            raise ValueError(
                f"{label} must have one value per selected ECM channel ({len(names)}); got {supplied.shape}."
            )

    scaled = np.empty_like(raw, dtype=np.float32)
    clip_values: list[float] = []
    pixel_thresholds: list[float] = []
    pixel_masks = np.zeros_like(raw, dtype=bool)
    for channel in range(raw.shape[0]):
        nonzero = raw[channel][raw[channel] > 0]
        high = float(np.percentile(nonzero, clip_percentile)) if nonzero.size else 1.0
        high = high if high > 0 else 1.0
        clip_values.append(high)
        scaled[channel] = np.clip(raw[channel], 0, high) / high
        threshold = (
            float(supplied_pixel_thresholds[channel])
            if supplied_pixel_thresholds is not None
            else _two_means_threshold(scaled[channel][scaled[channel] > 0], random_state=random_state)
        )
        pixel_thresholds.append(threshold)
        pixel_masks[channel] = scaled[channel] >= threshold

    height, width = scaled.shape[1:]
    grid_height, grid_width = height // patch_size, width // patch_size
    if grid_height == 0 or grid_width == 0:
        raise ValueError(f"patch_size={patch_size} is larger than image shape {(height, width)}.")
    patch_means = np.stack([_block_mean_2d(channel, patch_size) for channel in scaled])
    resolved_patch_thresholds = (
        supplied_patch_thresholds
        if supplied_patch_thresholds is not None
        else np.asarray(
            [_two_means_threshold(values, random_state=random_state) for values in patch_means], dtype=float
        )
    )
    channel_active = patch_means >= resolved_patch_thresholds[:, None, None]
    mean_foreground_threshold: float | None = None
    if foreground_mask is not None:
        supplied = np.asarray(foreground_mask, dtype=bool)
        if supplied.shape == (height, width):
            active = _block_mean_2d(supplied.astype(np.float32), patch_size) > 0
        elif supplied.shape == (grid_height, grid_width):
            active = supplied.copy()
        else:
            raise ValueError(
                "foreground_mask must match the image or patch grid; "
                f"got {supplied.shape}, expected {(height, width)} or {(grid_height, grid_width)}."
            )
        foreground_source = "explicit_mask"
    elif foreground_mode == "any":
        active = channel_active.any(axis=0)
        foreground_source = "channel_union"
    elif foreground_mode == "all":
        active = channel_active.all(axis=0)
        foreground_source = "channel_intersection"
    else:
        mean_image = patch_means.mean(axis=0)
        mean_threshold = _two_means_threshold(mean_image, random_state=random_state)
        active = mean_image >= mean_threshold
        mean_foreground_threshold = float(mean_threshold)
        foreground_source = "mean_patch_intensity"
    grid_y, grid_x = np.nonzero(active)
    if not len(grid_y):
        raise ValueError("Foreground selection retained no ECM patches.")

    patches = np.empty((len(grid_y), len(names), patch_size, patch_size), dtype=np.float32)
    for patch_index, (row, col) in enumerate(zip(grid_y, grid_x, strict=True)):
        patches[patch_index] = scaled[
            :,
            row * patch_size : (row + 1) * patch_size,
            col * patch_size : (col + 1) * patch_size,
        ]

    patch_ids = [f"patch_y{row:04d}_x{col:04d}" for row, col in zip(grid_y, grid_x, strict=True)]
    features, var, squidpy_version = _squidpy_image_features(
        patches,
        patch_ids=patch_ids,
        channel_names=names,
        feature_families=feature_families,
        feature_kwargs=resolved_feature_kwargs,
        n_jobs=n_jobs,
    )
    obs = pd.DataFrame(
        {
            "sample_id": sample_id or "sample",
            "grid_y": grid_y.astype(int),
            "grid_x": grid_x.astype(int),
        },
        index=pd.Index(patch_ids, name="patch_id"),
    )
    for channel, name in enumerate(names):
        obs[f"{name}_signal_fraction"] = _block_mean_2d(pixel_masks[channel].astype(np.float32), patch_size)[
            grid_y, grid_x
        ]

    adata = AnnData(X=features, obs=obs, var=var)
    adata.obsm["spatial"] = np.column_stack(((grid_x + 0.5) * patch_size, (grid_y + 0.5) * patch_size)).astype(
        np.float32
    )
    adata.obsm["image_patches"] = patches
    adata.obsm["raw_patch_mean"] = np.stack(
        [_block_mean_2d(channel, patch_size)[grid_y, grid_x] for channel in raw], axis=1
    ).astype(np.float32)
    adata.obsp["grid_connectivities"] = _grid8_connectivities(active)
    adata.uns["image_ecm_patches"] = {
        "image_shape": np.asarray((height, width), dtype=int),
        "patch_grid_shape": np.asarray((grid_height, grid_width), dtype=int),
        "patch_size": int(patch_size),
        "channel_names": np.asarray(names, dtype=str),
        "selected_channel_indices": selected_indices.astype(int),
        "clip_percentile": float(clip_percentile),
        "clip_values": np.asarray(clip_values, dtype=np.float32),
        "pixel_thresholds": np.asarray(pixel_thresholds, dtype=np.float32),
        "patch_thresholds": resolved_patch_thresholds.astype(np.float32),
        "threshold_source": (
            "supplied"
            if supplied_pixel_thresholds is not None and supplied_patch_thresholds is not None
            else "two_means"
            if supplied_pixel_thresholds is None and supplied_patch_thresholds is None
            else "mixed"
        ),
        "pixel_threshold_source": "supplied" if supplied_pixel_thresholds is not None else "two_means",
        "patch_threshold_source": "supplied" if supplied_patch_thresholds is not None else "two_means",
        "foreground_mode": foreground_mode,
        "foreground_source": foreground_source,
        "mean_foreground_threshold": mean_foreground_threshold,
        "patch_foreground": active.astype(bool),
        "channel_patch_foreground": channel_active.astype(bool),
        "feature_backend": "squidpy",
        "squidpy_version": squidpy_version,
        "feature_families": np.asarray(feature_families, dtype=str),
        "feature_kwargs_json": json.dumps(resolved_feature_kwargs, sort_keys=True),
        "features_per_channel": int(features.shape[1] // len(names)),
        "feature_n_jobs": int(n_jobs),
        "feature_parallel_backend": "joblib threads",
    }
    return adata


def normalize(
    adata: AnnData,
    *,
    method: Literal["min-max", "znorm"] = "min-max",
    raw_layer: str = RAW_LAYER,
    inplace: bool = True,
) -> AnnData | None:
    """Normalise ``adata.X`` per feature from a stable raw layer.

    Parameters
    ----------
    adata
        AnnData object.
    method
        ``"min-max"`` rescales each channel to [0, 1].
        ``"znorm"`` standardises each channel with
        :class:`sklearn.preprocessing.StandardScaler` (mean=0, std=1).
    raw_layer
        Layer containing the untransformed feature matrix. If it does not
        exist, the current ``X`` is copied there before normalisation. Later
        calls continue to use this layer, so changing methods never compounds
        transformations already written to ``X``.
    inplace
        If ``True`` modifies ``adata`` in place and returns ``None``.
        If ``False`` returns a copy.

    Returns
    -------
    ``None`` if ``inplace=True``, otherwise a modified copy.
    """
    if method not in {"min-max", "znorm"}:
        raise ValueError(f"Unknown method '{method}'. Use 'min-max' or 'znorm'.")
    if not isinstance(raw_layer, str):
        raise TypeError(f"raw_layer must be a string, got {type(raw_layer).__name__}.")
    if not raw_layer:
        raise ValueError("raw_layer must be a non-empty string.")
    if not inplace:
        adata = adata.copy()

    raw_layer_created = raw_layer not in adata.layers
    source = adata.X if raw_layer_created else adata.layers[raw_layer]
    if raw_layer_created:
        adata.layers[raw_layer] = source.copy()
    X = source.toarray() if sp.issparse(source) else np.asarray(source)

    provenance: dict[str, Any] = {
        "requested_method": method,
        "fit_scope": "all observations in this AnnData",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "raw_layer": raw_layer,
        "raw_layer_created": bool(raw_layer_created),
        "output": "X",
    }
    if method == "min-max":
        mn = np.min(X, axis=0, keepdims=True)
        mx = np.max(X, axis=0, keepdims=True)
        data_range = mx - mn
        denom = np.where(data_range == 0, 1.0, data_range)
        adata.X = ((X - mn) / denom).astype(np.float32)
        provenance.update(
            {
                "method": "min-max",
                "feature_min": [float(value) for value in np.ravel(mn)],
                "feature_max": [float(value) for value in np.ravel(mx)],
                "feature_range": [0.0, 1.0],
                "data_range": [float(value) for value in np.ravel(data_range)],
            }
        )
    else:
        import sklearn
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
        adata.X = scaler.fit_transform(X).astype(np.float32)
        provenance.update(
            {
                "method": "sklearn.preprocessing.StandardScaler",
                "with_mean": bool(scaler.with_mean),
                "with_std": bool(scaler.with_std),
                "mean": [float(value) for value in scaler.mean_],
                "scale": [float(value) for value in scaler.scale_],
                "sklearn_version": str(sklearn.__version__),
            }
        )

    adata.uns["feature_normalization"] = provenance

    _log_params(adata, "pp", {"normalize": method})

    if not inplace:
        return adata
    return None


@dataclass(frozen=True)
class HEPreprocessingResult:
    """Compact preview and global stain reference for a whole-slide H&E image."""

    rgb: np.ndarray
    hematoxylin: np.ndarray
    eosin: np.ndarray
    tissue_mask: np.ndarray
    hematoxylin_range: tuple[float, float]
    eosin_range: tuple[float, float]
    params: Mapping[str, Any]

    @property
    def H(self) -> np.ndarray:
        """Alias for the normalised hematoxylin preview."""
        return self.hematoxylin

    @property
    def E(self) -> np.ndarray:
        """Alias for the normalised eosin preview."""
        return self.eosin

    @property
    def tissue(self) -> np.ndarray:
        """Alias for :attr:`tissue_mask`."""
        return self.tissue_mask

    def __repr__(self) -> str:
        factor = self.params.get("downsample", 1)
        tissue_pct = 100.0 * float(self.tissue_mask.mean()) if self.tissue_mask.size else 0.0
        return (
            f"H&E preprocessing · preview {self.rgb.shape[1]}×{self.rgb.shape[0]} (downsample={factor})\n"
            f"    tissue {tissue_pct:.1f}% · H {self.hematoxylin_range!r} · E {self.eosin_range!r}"
        )


@dataclass(frozen=True)
class CellSegmentationSummary:
    """Compact summary of coordinate-native cell segmentation output."""

    n_cells: int
    n_variables: int
    spatial_keys: tuple[str, ...]
    backend: str
    stain: str
    diameter_um: float
    downsample: int
    device: str
    gpu_used: bool

    def __repr__(self) -> str:
        accelerator = "GPU" if self.gpu_used else "CPU"
        return "\n".join(
            [
                "Cell segmentation",
                f"  cells       {self.n_cells:,} × {self.n_variables} variables",
                f"  input       {self.stain} · {self.diameter_um:g}-µm diameter · downsample={self.downsample}",
                f"  backend     {self.backend} · {accelerator} ({self.device})",
                f"  coordinates {', '.join(self.spatial_keys)}",
            ]
        )


def cell_segmentation_summary(cells: AnnData) -> CellSegmentationSummary:
    """Summarise an AnnData returned by :func:`segment_cells_tiled`."""
    params = cells.uns.get("segmentation")
    if not isinstance(params, Mapping):
        raise KeyError("cells.uns['segmentation'] is missing; pass coordinate-native segmentation output.")
    spatial_keys = tuple(key for key in ("spatial", "spatial_um") if key in cells.obsm)
    if not spatial_keys:
        raise KeyError("cells must contain obsm['spatial'] or obsm['spatial_um'] coordinates.")
    return CellSegmentationSummary(
        n_cells=int(cells.n_obs),
        n_variables=int(cells.n_vars),
        spatial_keys=spatial_keys,
        backend=str(params.get("backend", "unknown")),
        stain=str(params.get("stain", "unknown")),
        diameter_um=float(params.get("diameter_um", np.nan)),
        downsample=int(params.get("downsample", 1)),
        device=str(params.get("device", "unrecorded")),
        gpu_used=bool(params.get("gpu_used", False)),
    )


@dataclass(frozen=True)
class HEECMPatchSummary:
    """Compact audit summary of an observation-native H&E ECM patch table."""

    n_patches_total: int
    n_tissue: int
    n_background: int
    n_cellular: int
    n_ecm: int
    n_observations: int
    patch_size_px: int
    patch_size_um: float
    eosin_threshold: float
    eosin_threshold_rule: str
    nuclear_density_threshold_mm2: float
    nuclear_density_threshold_rule: str
    storage: str
    coordinate_keys: tuple[str, ...]
    coordinate_units: tuple[str, ...]

    def __repr__(self) -> str:
        coordinates = ", ".join(
            f"obsm[{key!r}] ({unit})" for key, unit in zip(self.coordinate_keys, self.coordinate_units, strict=True)
        )
        return "\n".join(
            [
                "H&E ECM patches",
                (
                    f"  grid       {self.n_patches_total:,} total · {self.n_tissue:,} tissue · "
                    f"{self.n_background:,} background"
                ),
                f"  retained   {self.n_ecm:,} ECM · {self.n_cellular:,} cellular",
                f"  patch      {self.patch_size_px}px · {self.patch_size_um:g} µm",
                (
                    f"  thresholds eosin ({self.eosin_threshold_rule}) ≥ {self.eosin_threshold:.4g} · "
                    f"nuclei ({self.nuclear_density_threshold_rule}) ≤ "
                    f"{self.nuclear_density_threshold_mm2:.4g}/mm²"
                ),
                f"  storage    {self.n_observations:,} {self.storage}; coordinates: {coordinates}",
            ]
        )


def _he_to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    """Normalise an ``(H, W, C)`` RGB tile to floating point ``[0, 1]``."""
    arr = np.asarray(rgb)[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"H&E input must have three RGB channels, got shape {arr.shape}.")
    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_value
    else:
        arr = arr.astype(np.float32, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size and float(finite.max()) > 1.0:
            # Ordinary RGB float rasters conventionally retain the uint8 range.
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _he_tissue_mask(rgb: np.ndarray, *, min_size_px: int = 256) -> np.ndarray:
    """Fixed HSV tissue rule used by the prostate H&E workflow."""
    from skimage.color import rgb2hsv
    from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects

    hsv = rgb2hsv(_he_to_float_rgb(rgb))
    saturation, value = hsv[..., 1], hsv[..., 2]
    background = (value > 0.90) | ((value > 0.80) & (saturation < 0.12))
    tissue = closing(~background, disk(2))
    # skimage 0.26 renamed the strict ``area_threshold/min_size`` cutoffs to
    # inclusive ``max_size``. Subtracting one preserves the historical mask
    # exactly while avoiding deprecation noise; retain a fallback for Mantpy's
    # supported skimage 0.21--0.25 range.
    hole_threshold = max(64, int(min_size_px))
    object_threshold = max(1, int(min_size_px))
    try:
        tissue = remove_small_holes(tissue, max_size=hole_threshold - 1)
        return remove_small_objects(tissue, max_size=object_threshold - 1)
    except TypeError:
        tissue = remove_small_holes(tissue, area_threshold=hole_threshold)
        return remove_small_objects(tissue, min_size=object_threshold)


def _he_scale_channel(channel: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    lo, hi = (float(value_range[0]), float(value_range[1]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"Stain percentile range must be finite, got {value_range!r}.")
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((np.asarray(channel, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _materialize_image_array(array) -> np.ndarray:
    if hasattr(array, "compute"):
        array = array.compute()
    return np.asarray(array)


def _he_channel_first_source(img, *, layer: str | None = None):
    """Resolve an RGB input to a channel-first, possibly lazy array-like."""
    from pathlib import Path as _Path

    from mantpy.im import ImageContainer

    container: ImageContainer | None = None
    if isinstance(img, ImageContainer):
        container = img
        name = layer if layer is not None else img._default_layer
        source = img.get_layer(name, compute=False)
    elif isinstance(img, str | _Path):
        container = ImageContainer(img, channel_axis=-1, lazy=True)
        source = container.get_layer(container._default_layer, compute=False)
    else:
        source = np.asarray(img)
        if source.ndim != 3:
            raise ValueError(f"H&E input must be a three-channel image, got shape {source.shape}.")
        if source.shape[0] in (3, 4):
            pass
        elif source.shape[-1] in (3, 4):
            source = np.moveaxis(source, -1, 0)
        else:
            raise ValueError(f"Could not identify the RGB channel axis in shape {source.shape}.")
    if source.ndim != 3 or int(source.shape[0]) < 3:
        raise ValueError(f"H&E input must provide RGB channels, got shape {source.shape}.")
    return source[:3], container


def _he_ranges(stain_reference: HEPreprocessingResult | Mapping[str, object]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Read H/E reference ranges from a preprocessing result or mapping."""
    if isinstance(stain_reference, HEPreprocessingResult):
        return stain_reference.hematoxylin_range, stain_reference.eosin_range
    if isinstance(stain_reference, Mapping):
        h_range = stain_reference.get("hematoxylin_range", stain_reference.get("H_range"))
        e_range = stain_reference.get("eosin_range", stain_reference.get("E_range"))
        if h_range is not None and e_range is not None:
            return (float(h_range[0]), float(h_range[1])), (float(e_range[0]), float(e_range[1]))
    raise TypeError("stain_reference must be the result of mt.pp.preprocess_he().")


def _he_deconvolve_tile(
    rgb: np.ndarray,
    hematoxylin_range: tuple[float, float],
    eosin_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    from skimage.color import rgb2hed

    hed = rgb2hed(_he_to_float_rgb(rgb))
    return (
        _he_scale_channel(hed[..., 0], hematoxylin_range),
        _he_scale_channel(hed[..., 1], eosin_range),
    )


def preprocess_he(
    img: np.ndarray | ImageContainer | str | Path,
    *,
    downsample: int = 4,
    min_size_px: int = 256,
    percentiles: tuple[float, float] = (1.0, 99.0),
    layer: str | None = None,
) -> HEPreprocessingResult:
    """Build a fixed-mask H&E preview and one slide-global stain reference.

    Only the downsampled preview is materialised.  Raw HED percentiles are
    estimated within the fixed HSV tissue mask, so every later segmentation
    tile and ECM strip can use the same 1st–99th percentile reference rather
    than acquiring tile-specific contrast.
    """
    from skimage.color import rgb2hed

    if not isinstance(downsample, int) or downsample < 1:
        raise ValueError("downsample must be a positive integer.")
    if len(percentiles) != 2 or not (0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError("percentiles must be an increasing pair within [0, 100].")

    source, _ = _he_channel_first_source(img, layer=layer)
    preview_cf = _materialize_image_array(source[:, ::downsample, ::downsample])
    rgb = _he_to_float_rgb(np.moveaxis(preview_cf, 0, -1))
    tissue = _he_tissue_mask(rgb, min_size_px=min_size_px)
    hed = rgb2hed(rgb)
    sample_mask = tissue if tissue.any() else np.ones(tissue.shape, dtype=bool)

    def _range(channel: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(channel[sample_mask], percentiles)
        lo, hi = float(lo), float(hi)
        return (lo, hi if hi > lo else lo + 1e-6)

    hematoxylin_range = _range(hed[..., 0])
    eosin_range = _range(hed[..., 1])
    params = {
        "downsample": downsample,
        "percentiles": tuple(float(v) for v in percentiles),
        "tissue_rule": "not ((value > 0.90) or (value > 0.80 and saturation < 0.12))",
        "min_size_px": int(min_size_px),
        "fit_scope": "tissue pixels in whole-slide preview",
    }
    return HEPreprocessingResult(
        rgb=rgb,
        hematoxylin=_he_scale_channel(hed[..., 0], hematoxylin_range),
        eosin=_he_scale_channel(hed[..., 1], eosin_range),
        tissue_mask=tissue,
        hematoxylin_range=hematoxylin_range,
        eosin_range=eosin_range,
        params=params,
    )


def he_ecm_patches(
    img: np.ndarray | ImageContainer | str | Path,
    cells: AnnData | np.ndarray,
    *,
    stain_reference: HEPreprocessingResult | Mapping[str, object],
    patch_size_um: float = 32.0,
    image_mpp: float | None = None,
    tissue_fraction: float = 0.30,
    eosin_threshold: Literal["otsu"] | float = "otsu",
    nuclear_density_threshold: Literal["median"] | float = "median",
    strip_patches: int = 4,
    min_size_px: int = 256,
    layer: str | None = None,
) -> AnnData:
    """Stream H&E strips into an observation-native AnnData of ECM patches.

    By default, eosin is thresholded with Otsu over retained tissue patches and
    nuclear density at its median. Either rule can instead receive a fixed,
    finite numeric threshold. Nuclear-density values are nuclei/mm². Only
    patches satisfying both criteria are returned as observations;
    background/cellular counts remain in ``uns['he_ecm_patches']`` for
    auditability.
    """
    from skimage.filters import threshold_otsu

    if patch_size_um <= 0:
        raise ValueError("patch_size_um must be positive.")
    if not (0.0 <= tissue_fraction <= 1.0):
        raise ValueError("tissue_fraction must be within [0, 1].")
    if not isinstance(strip_patches, int) or strip_patches < 1:
        raise ValueError("strip_patches must be a positive integer.")

    def _threshold_spec(value, automatic: str, name: str) -> tuple[str, float | None]:
        if isinstance(value, str):
            if value != automatic:
                raise ValueError(f"{name} must be {automatic!r} or a finite numeric value.")
            return automatic, None
        if isinstance(value, bool):
            raise ValueError(f"{name} must be {automatic!r} or a finite numeric value.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be {automatic!r} or a finite numeric value.") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"{name} numeric values must be finite, got {value!r}.")
        return "fixed", numeric

    eosin_threshold_rule, fixed_eosin_threshold = _threshold_spec(
        eosin_threshold,
        "otsu",
        "eosin_threshold",
    )
    nuclear_density_threshold_rule, fixed_density_threshold = _threshold_spec(
        nuclear_density_threshold,
        "median",
        "nuclear_density_threshold",
    )

    source, container = _he_channel_first_source(img, layer=layer)
    if image_mpp is None:
        if container is None:
            raise ValueError("image_mpp is required for array inputs.")
        image_mpp = float(container.scale)
    image_mpp = float(image_mpp)
    if image_mpp <= 0:
        raise ValueError("image_mpp must be positive.")

    h_range, e_range = _he_ranges(stain_reference)
    height, width = int(source.shape[1]), int(source.shape[2])
    patch_px = max(1, int(round(float(patch_size_um) / image_mpp)))
    ny, nx = height // patch_px, width // patch_px
    if ny == 0 or nx == 0:
        raise ValueError(
            f"Image shape {(height, width)} is smaller than one {patch_px}-pixel patch."
        )

    eosin_grid = np.empty((ny, nx), dtype=np.float32)
    hematoxylin_grid = np.empty((ny, nx), dtype=np.float32)
    tissue_grid = np.empty((ny, nx), dtype=np.float32)
    rows_per_strip = strip_patches * patch_px
    usable_width = nx * patch_px
    for y0 in range(0, ny * patch_px, rows_per_strip):
        patch_rows = min(strip_patches, ny - y0 // patch_px)
        y1 = y0 + patch_rows * patch_px
        rgb_cf = _materialize_image_array(source[:, y0:y1, :usable_width])
        rgb = np.moveaxis(rgb_cf, 0, -1)
        hematoxylin, eosin = _he_deconvolve_tile(rgb, h_range, e_range)
        tissue = _he_tissue_mask(rgb, min_size_px=min_size_px).astype(np.float32)
        grid_row = y0 // patch_px
        shape = (patch_rows, patch_px, nx, patch_px)
        eosin_grid[grid_row : grid_row + patch_rows] = eosin.reshape(shape).mean(axis=(1, 3))
        hematoxylin_grid[grid_row : grid_row + patch_rows] = hematoxylin.reshape(shape).mean(axis=(1, 3))
        tissue_grid[grid_row : grid_row + patch_rows] = tissue.reshape(shape).mean(axis=(1, 3))

    if isinstance(cells, AnnData):
        if "spatial" not in cells.obsm:
            raise KeyError("cells.obsm['spatial'] is required.")
        cell_xy = np.asarray(cells.obsm["spatial"], dtype=float)
    else:
        cell_xy = np.asarray(cells, dtype=float)
    if cell_xy.ndim != 2 or cell_xy.shape[1] != 2:
        raise ValueError(f"cells must provide (x, y) coordinates with shape (N, 2), got {cell_xy.shape}.")
    counts = np.zeros((ny, nx), dtype=np.int32)
    finite = np.isfinite(cell_xy).all(axis=1)
    cell_xy = cell_xy[finite]
    if len(cell_xy):
        grid_x = np.floor(cell_xy[:, 0] / patch_px).astype(int)
        grid_y = np.floor(cell_xy[:, 1] / patch_px).astype(int)
        inside = (grid_x >= 0) & (grid_x < nx) & (grid_y >= 0) & (grid_y < ny)
        np.add.at(counts, (grid_y[inside], grid_x[inside]), 1)

    tissue_keep = tissue_grid >= float(tissue_fraction)
    retained_eosin = eosin_grid[tissue_keep]
    patch_area_mm2 = (patch_px * image_mpp / 1000.0) ** 2
    nuclear_density = counts.astype(np.float32) / np.float32(patch_area_mm2)
    retained_density = nuclear_density[tissue_keep]
    if fixed_eosin_threshold is not None:
        resolved_eosin_threshold = fixed_eosin_threshold
    elif retained_eosin.size:
        resolved_eosin_threshold = (
            float(threshold_otsu(retained_eosin))
            if float(np.ptp(retained_eosin)) > 1e-6
            else float(retained_eosin.mean())
        )
    else:
        resolved_eosin_threshold = float("nan")
    if fixed_density_threshold is not None:
        resolved_density_threshold = fixed_density_threshold
    elif retained_density.size:
        resolved_density_threshold = float(np.median(retained_density))
    else:
        resolved_density_threshold = float("nan")
    is_ecm = tissue_keep & (eosin_grid >= resolved_eosin_threshold) & (
        nuclear_density <= resolved_density_threshold
    )
    grid_y, grid_x = np.nonzero(is_ecm)
    x0, y0 = grid_x * patch_px, grid_y * patch_px
    x1, y1 = x0 + patch_px, y0 + patch_px
    spatial = np.column_stack(((x0 + x1) / 2.0, (y0 + y1) / 2.0)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "grid_row": grid_y.astype(np.int32),
            "grid_col": grid_x.astype(np.int32),
            "x0": x0.astype(np.int32),
            "y0": y0.astype(np.int32),
            "x1": x1.astype(np.int32),
            "y1": y1.astype(np.int32),
            "eosin": eosin_grid[is_ecm],
            "hematoxylin": hematoxylin_grid[is_ecm],
            "n_nuclei": counts[is_ecm],
            "nuclear_density_mm2": nuclear_density[is_ecm],
            "tissue_fraction": tissue_grid[is_ecm],
        },
        index=pd.Index([f"ecm_patch_{i:07d}" for i in range(len(grid_x))], dtype=str),
    )
    result = AnnData(X=sp.csr_matrix((len(obs), 0), dtype=np.float32), obs=obs)
    result.obsm["spatial"] = spatial
    result.obsm["spatial_um"] = spatial * np.float32(image_mpp)
    n_total = int(ny * nx)
    n_tissue = int(tissue_keep.sum())
    n_ecm = int(is_ecm.sum())
    result.uns["he_ecm_patches"] = {
        "n_patches_total": n_total,
        "n_background": n_total - n_tissue,
        "n_cellular": n_tissue - n_ecm,
        "n_ecm": n_ecm,
        "eosin_threshold_rule": eosin_threshold_rule,
        "eosin_threshold": resolved_eosin_threshold,
        "nuclear_density_threshold_rule": nuclear_density_threshold_rule,
        "nuclear_density_threshold": resolved_density_threshold,
        "nuclear_density_threshold_mm2": resolved_density_threshold,
        "tissue_fraction_threshold": float(tissue_fraction),
        "patch_size_px": patch_px,
        "patch_size_um": float(patch_px * image_mpp),
        "requested_patch_size_um": float(patch_size_um),
        "image_mpp": image_mpp,
        "hematoxylin_range": list(h_range),
        "eosin_range": list(e_range),
        "streaming_strip_patches": strip_patches,
        "storage": "ECM patches as AnnData observations",
        "coordinate_keys": ["spatial", "spatial_um"],
        "coordinate_units": ["full-resolution pixels", "micrometres"],
    }
    return result


def he_ecm_patch_summary(ecm: AnnData) -> HEECMPatchSummary:
    """Summarise the grid, thresholds, and storage of :func:`he_ecm_patches` output."""
    if "he_ecm_patches" not in ecm.uns:
        raise KeyError("ecm.uns['he_ecm_patches'] is missing. Run mt.pp.he_ecm_patches first.")
    metadata = ecm.uns["he_ecm_patches"]
    required = {
        "n_patches_total",
        "n_background",
        "n_cellular",
        "n_ecm",
        "patch_size_px",
        "patch_size_um",
        "eosin_threshold",
        "nuclear_density_threshold_mm2",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise KeyError(f"ecm.uns['he_ecm_patches'] is missing keys: {missing}.")
    n_total = int(metadata["n_patches_total"])
    n_background = int(metadata["n_background"])
    coordinate_keys = tuple(
        str(value)
        for value in metadata.get(
            "coordinate_keys",
            ["spatial", "spatial_um"],
        )
    )
    coordinate_units = tuple(
        str(value)
        for value in metadata.get(
            "coordinate_units",
            ["full-resolution pixels", "micrometres"],
        )
    )
    if len(coordinate_keys) != len(coordinate_units):
        raise ValueError("H&E ECM coordinate_keys and coordinate_units must have matching lengths.")
    return HEECMPatchSummary(
        n_patches_total=n_total,
        n_tissue=n_total - n_background,
        n_background=n_background,
        n_cellular=int(metadata["n_cellular"]),
        n_ecm=int(metadata["n_ecm"]),
        n_observations=int(ecm.n_obs),
        patch_size_px=int(metadata["patch_size_px"]),
        patch_size_um=float(metadata["patch_size_um"]),
        eosin_threshold=float(metadata["eosin_threshold"]),
        eosin_threshold_rule=str(metadata.get("eosin_threshold_rule", "otsu")),
        nuclear_density_threshold_mm2=float(metadata["nuclear_density_threshold_mm2"]),
        nuclear_density_threshold_rule=str(metadata.get("nuclear_density_threshold_rule", "median")),
        storage=str(metadata.get("storage", "ECM patches as AnnData observations")),
        coordinate_keys=coordinate_keys,
        coordinate_units=coordinate_units,
    )


def preprocess_ecm(
    img: np.ndarray | ImageContainer | AnnData,
    *,
    method: Literal["tophat", "frangi", "sato", "none"] = "tophat",
    disk_radius: int = 5,
    sigmas: tuple[int, int] = (1, 3),
    channel_indices: list[int] | None = None,
    layer_key: str = "preprocessed",
) -> np.ndarray | None:
    """Apply a structural preprocessing filter to an ECM image.

    Modifies pixel *values* to emphasise ECM structure before thresholding or
    patch extraction.  The output has the same ``(C, H, W)`` shape and dtype as
    the input.

    Parameters
    ----------
    img
        One of:

        - ``(C, H, W)`` numpy array — returns the preprocessed array.
        - :class:`~mantpy.im.ImageContainer` — preprocesses the default layer
          and stores the result as a new layer named ``layer_key``.  Returns
          the preprocessed array.
        - :class:`~anndata.AnnData` — reads from the attached
          ``ImageContainer`` (``adata.uns['image_container']``), preprocesses,
          and writes the result as a new layer.  Returns ``None`` (in-place).
    method
        Preprocessing filter:

        - ``"tophat"``  — white top-hat (removes slow background, keeps fine ECM
          structure; standard first step for CODEX collagen channels).
          Controlled by ``disk_radius``.
        - ``"frangi"``  — Frangi ridge/vessel filter; emphasises collagen
          fibres.  Controlled by ``sigmas``.
        - ``"sato"``    — Sato plate-like structure filter.  Controlled by
          ``sigmas``.
        - ``"none"``    — identity, returns a copy of the input.
    disk_radius
        Radius of the structuring element for ``"tophat"``.
    sigmas
        ``(min_sigma, max_sigma)`` range (inclusive) for ``"frangi"`` and
        ``"sato"``.
    channel_indices
        Subset of channel indices to process.  ``None`` = all channels.
        Other channels are copied unchanged.
    layer_key
        Name of the layer to write in the ``ImageContainer`` when ``img`` is
        an ``ImageContainer`` or ``AnnData``.  Default ``"preprocessed"``.

    Returns
    -------
    np.ndarray (shape ``(C, H, W)``, dtype float32) when ``img`` is an array
    or ImageContainer.  ``None`` when ``img`` is an AnnData (in-place write).

    Notes
    -----
    Typical usage (AnnData-centric, recommended)::

        adata = mt.io.read_ecm_image("collagen.tif")
        mt.pp.preprocess_ecm(adata, method="frangi")
        mt.pp.extract_ecm_patches(adata, features=["mean", "coherence"])
        mt.pl.ecm_graph_overlay(adata)  # auto-reads preprocessed layer

    Legacy usage (explicit arrays, still supported)::

        preprocessed = mt.pp.preprocess_ecm(img, method="tophat")
        mt.pp.extract_ecm_patches(adata, preprocessed, features=["mean", "coherence"])
    """
    from mantpy.im import ImageContainer, as_image_container

    # Resolve input to an array + optional container to write back to
    container: ImageContainer | None = None
    adata_mode = False

    if isinstance(img, AnnData):
        adata_mode = True
        if IMAGE_CONTAINER_KEY not in img.uns:
            raise ValueError(
                "No ImageContainer attached to adata. "
                "Use `mt.io.read_imc()` or `mt.io.read_ecm_image()` to load data, "
                "or pass an image array/ImageContainer directly."
            )
        container = as_image_container(img.uns[IMAGE_CONTAINER_KEY])
        arr = container.to_array(dtype=np.float32)
    elif isinstance(img, ImageContainer):
        container = img
        arr = img.to_array(dtype=np.float32)
    else:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        if arr.ndim != 3:
            raise ValueError(f"img must be (C, H, W) but got shape {arr.shape}.")

    if method == "none":
        out = arr.copy()
    else:
        try:
            from skimage import filters as skf
            from skimage.morphology import disk, white_tophat
        except ImportError as exc:
            raise ImportError("preprocess_ecm requires scikit-image. Install with: pip install scikit-image") from exc

        C = arr.shape[0]
        indices = channel_indices if channel_indices is not None else list(range(C))
        out = arr.copy()

        sigma_range = range(sigmas[0], sigmas[1] + 1)

        for c in indices:
            ch = arr[c]
            if method == "tophat":
                out[c] = white_tophat(ch, disk(disk_radius)).astype(np.float32)
            elif method == "frangi":
                out[c] = skf.frangi(ch, sigmas=sigma_range, black_ridges=False).astype(np.float32)
            elif method == "sato":
                out[c] = skf.sato(ch, sigmas=sigma_range).astype(np.float32)
            else:
                raise ValueError(f"Unknown method '{method}'. Choose from 'tophat', 'frangi', 'sato', 'none'.")

    # Write back to container if available
    if container is not None:
        container.add_layer(layer_key, out)

    if adata_mode:
        return None
    return out


def extract_ecm_patches(
    adata: AnnData,
    img: np.ndarray | ImageContainer | None = None,
    *,
    patch_size: int = 5,
    ecm_K: int | Literal["auto"] | None = None,
    features: list[str | Callable],
    background_quantile: float = 0.1,
    min_signal_fraction: float | None = None,
    threshold_method: str = "li",
    threshold_value: float | None = None,
    mask: np.ndarray | None = None,
    ecm_channel: str | None = None,
    layer: str | None = None,
    tiled: bool = False,
    tile_size: int = 2048,
    tile_overlap: int = 0,
    inplace: bool = True,
) -> AnnData | None:
    """Extract ECM patches from the image and cluster them.

    Divides the ECM-channel image into non-overlapping ``patch_size × patch_size``
    patches, computes per-patch feature vectors, removes background, and clusters
    remaining patches with K-means.

    Parameters
    ----------
    adata
        AnnData object.  Must have ``adata.var['is_ecm']`` (produced by
        :func:`~mantpy.io.read_imc`), unless ``ecm_channel`` is provided
        together with a 2-D image (see below).
    img
        Either a multi-channel ``(C, H, W)`` numpy array / :class:`~mantpy.im.ImageContainer`
        whose channel count matches ``adata.n_vars`` (standard mode), or a
        single-channel ``(H, W)`` numpy array (single-channel mode).

        Single-channel mode (set ``ecm_channel`` to the variable in
        ``adata.var_names`` that this image represents) is the right call for
        gigapixel CODEX/multiplex slides where holding the full multi-channel
        stack in memory is infeasible — pass just the collagen channel and the
        function flags it in ``adata.var['is_ecm']`` for you.

        If ``None`` (default), reads from the ``ImageContainer`` attached to
        ``adata.uns['image_container']``.  Uses the layer specified by
        ``layer`` (default: ``"preprocessed"`` if it exists, else ``"image"``).
    patch_size
        Side length (pixels) of each patch.
    ecm_K
        Number of ECM clusters.  ``None`` (default) skips clustering entirely —
        all patches receive label ``0``, which is the right choice when working
        with a single ECM marker.  ``"auto"`` selects K via silhouette score
        (K=2..8).  Pass an integer for a fixed number of clusters.
    features
        List of feature extractors to apply to each patch.  Each item is either:

        - A string alias: ``"mean"``, ``"std"``, ``"entropy"``, ``"coherence"``
        - A callable with signature ``(patch: np.ndarray) -> np.ndarray`` where
          ``patch`` is ``(C, patch_size, patch_size)`` float32 and the return is
          a 1-D float32 array of any length.

        Outputs from all extractors are concatenated to form the feature vector.
        Example: ``features=["mean", "coherence"]``
    background_quantile
        Patches whose mean intensity falls below this quantile are discarded as
        background.  Only used when ``min_signal_fraction`` is ``None``.
    min_signal_fraction
        If set, uses adaptive thresholding (Li or Otsu) on the image to
        determine a signal cutoff, then discards patches where fewer than this
        fraction of pixels exceed the threshold.  This is much more effective
        than ``background_quantile`` for images with large dark regions.
        Typical value: ``0.1``.  If ``None`` (default), falls back to the
        quantile-based method for backward compatibility.
    threshold_method
        Threshold algorithm when ``min_signal_fraction`` is set: ``"li"``
        (default) or ``"otsu"``.
    mask
        Optional caller-supplied ``(C, H, W)`` or ``(H, W)`` boolean foreground
        mask. Patches with zero foreground pixels are excluded.
        If ``None`` no masking is applied beyond ``background_quantile``.
    ecm_channel
        Required when ``img`` is a 2-D ``(H, W)`` array; ignored otherwise.
        Names the variable in ``adata.var_names`` that the single-channel image
        represents.  When set, ``adata.var['is_ecm']`` is overwritten so that
        only ``ecm_channel`` is flagged as ECM.
    layer
        Which layer to read from the attached ``ImageContainer`` when ``img``
        is ``None``.  Default behaviour: uses ``"preprocessed"`` if available,
        otherwise ``"image"``.
    tiled
        Process the image tile-by-tile.  Useful for images that do not fit in
        RAM.  Requires ``img`` to be an ``ImageContainer``.
    tile_size
        Tile side length in pixels when ``tiled=True``.
    tile_overlap
        Overlap in pixels added around each tile when ``tiled=True``.
    inplace
        Modify ``adata`` in place (returns ``None``), or return a modified copy.

    Writes
    ------
    ``adata.uns['ecm_patches']``
        DataFrame with columns: ``x``, ``y``, ``ecm_cluster`` (``0`` when
        ``ecm_K=None``), ``feat_*``.
    ``adata.uns['ecm_image']``
        (H, W) integer image of cluster labels (-1 = background).
    ``adata.uns['ecm_feature_names']``
        List of dicts ``{"extractor": str, "protein": str}`` — one entry per
        ``feat_*`` column in order, preserving the identity of each
        per-protein feature column.
    ``adata.uns['mantpy']['pp']``
        Params log.

    Returns
    -------
    ``None`` if ``inplace=True``, otherwise a modified copy.
    """
    from mantpy.im import ImageContainer, as_image_container

    # ---- detect single-channel mode -------------------------------------
    # A 2-D ndarray + explicit `ecm_channel` lets callers pass just the ECM
    # channel without holding the full (C, H, W) stack in memory.
    single_channel_mode = isinstance(img, np.ndarray) and img.ndim == 2
    if single_channel_mode:
        if ecm_channel is None:
            raise ValueError(
                "When `img` is a 2-D (H, W) array, set `ecm_channel` to the "
                "variable in adata.var_names that this image represents."
            )
        if ecm_channel not in adata.var_names:
            raise ValueError(f"ecm_channel='{ecm_channel}' is not in adata.var_names.")

    if "is_ecm" not in adata.var.columns and not single_channel_mode:
        raise ValueError("Column 'is_ecm' not found in adata.var. Run `mt.read_imc(...)` first.")

    extractors = resolve_extractors(features)

    if not inplace:
        adata = adata.copy()

    # In single-channel mode, force adata.var['is_ecm'] to flag only this
    # channel.  Done after the inplace-copy so external state stays clean
    # when the caller asks for a copy.
    if single_channel_mode:
        adata.var["is_ecm"] = False
        adata.var.loc[ecm_channel, "is_ecm"] = True

    # ---- normalise img argument ------------------------------------------
    if img is None:
        if IMAGE_CONTAINER_KEY not in adata.uns:
            raise ValueError(
                "No image provided and no ImageContainer attached to adata. "
                "Either pass `img` explicitly or use `mt.io.read_imc()` / "
                "`mt.io.read_ecm_image()` to attach an ImageContainer."
            )
        ic = as_image_container(adata.uns[IMAGE_CONTAINER_KEY])
        # Select layer: explicit > "preprocessed" > default.
        # Carry `scale` across the rewrap — selecting a layer does not resample
        # the image, so silently resetting it to the 1.0 default would misreport
        # the pixel size of every container that had a non-default scale.
        if layer is not None:
            if not ic.has_layer(layer):
                raise KeyError(f"Layer '{layer}' not found in ImageContainer. Available: {ic.layers}")
            ic = ImageContainer(ic.get_layer(layer), scale=ic.scale)
        elif ic.has_layer("preprocessed"):
            ic = ImageContainer(ic.get_layer("preprocessed"), scale=ic.scale)
        else:
            ic = ImageContainer(ic.to_array(), scale=ic.scale)
    elif isinstance(img, ImageContainer):
        ic = img
    else:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        if arr.ndim != 3:
            raise ValueError(f"img must be (C, H, W), got shape {arr.shape}.")
        ic = ImageContainer(arr)

    H, W = ic.height, ic.width
    n_channels = ic.n_channels

    if single_channel_mode:
        # `ic` is (1, H, W); the only ECM channel is at array index 0.
        # adata.n_vars can be much larger; we don't enforce the channel-count
        # match in this mode.
        ecm_channel_indices = [0]
    else:
        if n_channels != adata.n_vars:
            raise ValueError(f"img has {n_channels} channels but adata has {adata.n_vars} variables. They must match.")
        ecm_channel_indices = np.where(adata.var["is_ecm"].values.astype(bool))[0].tolist()

    if len(ecm_channel_indices) == 0:
        raise ValueError("No ECM channels found (adata.var['is_ecm'] is all False). Check your panel definition.")

    # ---- normalise mask argument -----------------------------------------
    spatial_mask: np.ndarray | None = None
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 3:
            # collapse channels: foreground if any ECM channel is foreground
            ecm_mask = m[ecm_channel_indices]
            spatial_mask = ecm_mask.any(axis=0)  # (H, W)
        elif m.ndim == 2:
            spatial_mask = m
        else:
            raise ValueError(f"mask must be (H, W) or (C, H, W), got shape {m.shape}.")

    # ---- dispatch ---------------------------------------------------------
    if tiled:
        patch_df, labels_img = _extract_tiled(
            ic,
            ecm_channel_indices,
            patch_size,
            ecm_K,
            extractors,
            background_quantile,
            spatial_mask,
            tile_size,
            tile_overlap,
            H,
            W,
        )
    else:
        ecm_arr = ic[ecm_channel_indices].to_array()  # (C_ecm, H, W)
        patch_df, labels_img = _extract_whole(
            ecm_arr,
            patch_size,
            ecm_K,
            extractors,
            background_quantile,
            spatial_mask,
            H,
            W,
            min_signal_fraction=min_signal_fraction,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
        )

    adata.uns[ECM_PATCHES_KEY] = patch_df
    adata.uns[ECM_IMAGE_KEY] = labels_img

    # Build feature name → (extractor, ecm_protein) metadata so downstream
    # analyses can recover per-protein columns.
    if single_channel_mode:
        ecm_proteins = [ecm_channel]
    else:
        ecm_proteins = adata.var_names[ecm_channel_indices].tolist()
    feat_col_meta: list[dict] = []
    for feat_spec in features:
        extractor_name = feat_spec.__name__ if callable(feat_spec) else feat_spec
        for protein in ecm_proteins:
            feat_col_meta.append({"extractor": extractor_name, "protein": protein})
    adata.uns["ecm_feature_names"] = json.dumps(feat_col_meta)

    k_used = int(patch_df["ecm_cluster"].nunique())
    feature_names = [f.__name__ if callable(f) else f for f in features]
    _log_params(
        adata,
        "pp",
        {
            "extract_ecm_patches": {
                "patch_size": patch_size,
                "ecm_K": k_used,
                "features": feature_names,
                "background_quantile": background_quantile,
                "masked": spatial_mask is not None,
                "tiled": tiled,
                "n_patches_foreground": len(patch_df),
            }
        },
    )

    if not inplace:
        return adata
    return None


@dataclass(frozen=True)
class PatchComparison:
    """Row-aligned comparison of two cohort ECM-patch collections."""

    n_patches: int
    n_features: int
    coordinate_matches: int
    feature_matches: int
    label_matches: int | None

    def __repr__(self) -> str:
        lines = [
            "ECM patch comparison",
            f"  coordinates     {self.coordinate_matches:,} / {self.n_patches:,} patches match",
            f"  feature rows    {self.feature_matches:,} / {self.n_patches:,} patches match ({self.n_features} features)",
        ]
        if self.label_matches is not None:
            lines.append(f"  cluster labels  {self.label_matches:,} / {self.n_patches:,} patches match")
        return "\n".join(lines)


@dataclass(frozen=True)
class ECMPatchSummary:
    """Compact cohort-level summary of extracted ECM patches."""

    n_samples: int
    n_patches: int
    n_features: int
    n_background: int | None
    signal_counts: tuple[tuple[int, int], ...]
    feature_backend: str | None = None
    feature_families: tuple[str, ...] = ()
    channel_names: tuple[str, ...] = ()
    image_patch_shape: tuple[int, ...] | None = None

    @property
    def n_signal(self) -> int | None:
        """Number of non-background patches when labels are available."""
        if self.n_background is None:
            return None
        return self.n_patches - self.n_background

    def __repr__(self) -> str:
        lines = [
            "ECM patch cohort",
            f"  samples       {self.n_samples}",
            f"  patches       {self.n_patches:,}",
            f"  features      {self.n_features} per patch",
        ]
        if self.feature_backend is not None:
            families = ", ".join(self.feature_families)
            lines.append(f"  feature source {self.feature_backend}" + (f" ({families})" if families else ""))
        if self.channel_names:
            lines.append(f"  ECM channels  {', '.join(self.channel_names)}")
        if self.image_patch_shape is not None:
            shape = " × ".join(str(value) for value in self.image_patch_shape)
            lines.append(f"  image patches {shape}")
        if self.n_background is not None:
            lines.append(f"  background    {self.n_background:,}")
            lines.extend(f"  ECM {cluster:<8} {count:,}" for cluster, count in self.signal_counts)
        return "\n".join(lines)


@dataclass(frozen=True)
class ClusterCountSelection:
    """Candidate-background and signal-only K diagnostics."""

    candidate_scan: pd.DataFrame
    signal_scan: pd.DataFrame
    selected_total_k: int
    selected_signal_k: int
    background_component: int

    def __repr__(self) -> str:
        candidate = self.candidate_scan.set_index("signal_k")["silhouette"]
        signal_scan = self.signal_scan.set_index("K")["silhouette"]
        return "\n".join(
            [
                "ECM cluster-count selection",
                f"  selected total components   {self.selected_total_k}",
                f"  selected signal niches      {self.selected_signal_k}",
                f"  candidate-aware silhouette  {candidate.loc[self.selected_total_k - 1]:.3f}",
                f"  signal-only silhouette       {signal_scan.loc[self.selected_signal_k]:.3f}",
                "",
                "candidate_scan and signal_scan contain the full diagnostics.",
            ]
        )


@dataclass(frozen=True)
class ECMLeidenResolutionSelection:
    """Calinski-Harabasz diagnostics for a shared Leiden neighbour graph."""

    table: pd.DataFrame
    selected_resolution: float
    selected_n_clusters: int
    subset: Literal["all", "signal"]
    n_neighbors: int
    effective_n_neighbors: int
    flavor: Literal["leidenalg", "igraph"]
    feature_columns: tuple[str, ...]
    versions: Mapping[str, str]

    def __repr__(self) -> str:
        selected = self.table.loc[
            np.isclose(self.table["resolution"], self.selected_resolution)
        ].iloc[0]
        return "\n".join(
            [
                "ECM Leiden resolution selection",
                f"  selected resolution   {self.selected_resolution:g}",
                f"  observed clusters     {self.selected_n_clusters}",
                f"  Calinski-Harabasz     {float(selected['calinski_harabasz']):.3f}",
                f"  neighbour graph       k={self.effective_n_neighbors}; {self.subset} patches",
                "",
                "table contains the full resolution diagnostics.",
            ]
        )


@dataclass(frozen=True)
class ECMClusteringResult:
    """Fitted cohort-wide ECM clustering and its component summaries."""

    n_clusters: int
    cluster_key: str
    feature_columns: tuple[str, ...]
    component_means: tuple[float, ...]
    component_sizes: tuple[int, ...]
    scaler: Any
    model: Any
    method: Literal["kmeans", "leiden"] = "kmeans"
    subset: Literal["all", "signal"] = "all"
    cluster_counts: tuple[tuple[int, int], ...] = ()
    provenance: Mapping[str, Any] | None = None

    @property
    def counts(self) -> dict[int, int]:
        """Final label counts, including ``-1`` background when present."""
        if self.cluster_counts:
            return dict(self.cluster_counts)
        return dict(enumerate(self.component_sizes))

    def __repr__(self) -> str:
        lines = [
            "ECM cohort clustering",
            f"  method       {self.method}",
            f"  subset       {self.subset}",
            f"  clusters     {self.n_clusters}",
            f"  features     {len(self.feature_columns)}",
            f"  counts       {self.counts}",
        ]
        if self.method == "kmeans" and self.component_means:
            lines.append(f"  dimmest      component {int(np.argmin(self.component_means))}")
        return "\n".join(lines)


@dataclass(frozen=True)
class BackgroundRemovalSummary:
    """Summary of background removal and final signal-label ordering."""

    background_component: int
    background_patches: int
    signal_counts: tuple[tuple[int, int], ...]

    def __repr__(self) -> str:
        lines = [
            "ECM background removal",
            f"  component {self.background_component} -> background (-1): {self.background_patches:,} patches",
        ]
        lines.extend(f"  ECM {cluster}: {count:,} patches" for cluster, count in self.signal_counts)
        return "\n".join(lines)


@dataclass(frozen=True)
class ECMLabelOverlaySummary:
    """Compact validation summary for an ECM corruption overlay."""

    n_samples: int
    n_patches: int
    n_signal: int
    n_artifacts: int
    reference_mismatches: int

    def __repr__(self) -> str:
        return "\n".join(
            [
                "ECM corruption benchmark",
                f"  samples                {self.n_samples}",
                f"  ECM patches            {self.n_patches:,}",
                f"  signal patches         {self.n_signal:,}",
                f"  replacement labels     {self.n_artifacts:,}",
                f"  reference mismatches   {self.reference_mismatches:,}",
            ]
        )


def ecm_patch_summary(
    adatas: Mapping[str, AnnData] | AnnData,
    *,
    feature_prefix: str = "feat_",
    cluster_key: str | None = None,
) -> ECMPatchSummary:
    """Return a displayable summary of an ECM patch cohort.

    Pass ``cluster_key='ecm_cluster'`` after clustering to include background
    and signal counts. The extraction-stage default reports only patches and
    features because ``extract_ecm_patches(..., ecm_K=None)`` uses a temporary
    all-zero label column.
    """
    if isinstance(adatas, AnnData):
        sample = str(adatas.obs["sample_id"].iloc[0]) if adatas.n_obs and "sample_id" in adatas.obs else "sample"
        adatas = {sample: adatas}
    if not adatas:
        raise ValueError("adatas is empty; expected at least one ECM sample.")
    n_patches = 0
    n_features: int | None = None
    labels: list[np.ndarray] = []
    labels_available = True
    native_metadata: list[tuple[str, tuple[str, ...], tuple[str, ...], tuple[int, ...]]] = []
    for _sample, adata in adatas.items():
        if ECM_PATCHES_KEY in adata.uns:
            patches = adata.uns[ECM_PATCHES_KEY]
            columns = [column for column in patches if column.startswith(feature_prefix)]
            sample_labels = patches[cluster_key].to_numpy(dtype=int) if cluster_key in patches else None
        else:
            patches = adata.obs
            columns = list(adata.var_names)
            sample_labels = adata.obs[cluster_key].to_numpy(dtype=int) if cluster_key in adata.obs else None
            metadata = adata.uns.get("image_ecm_patches", {})
            if metadata and "image_patches" in adata.obsm:
                native_metadata.append(
                    (
                        str(metadata.get("feature_backend", "unknown")),
                        tuple(str(value) for value in metadata.get("feature_families", ())),
                        tuple(str(value) for value in metadata.get("channel_names", ())),
                        tuple(int(value) for value in adata.obsm["image_patches"].shape[1:]),
                    )
                )
        if n_features is None:
            n_features = len(columns)
        elif len(columns) != n_features:
            raise ValueError("ECM samples have different patch-feature counts.")
        n_patches += len(patches)
        if cluster_key is not None and sample_labels is not None:
            labels.append(sample_labels)
        else:
            labels_available = False
    if labels_available:
        pooled_labels = np.concatenate(labels)
        n_background: int | None = int((pooled_labels < 0).sum())
        signal_counts = tuple(
            (int(cluster), int((pooled_labels == cluster).sum()))
            for cluster in sorted(np.unique(pooled_labels[pooled_labels >= 0]))
        )
    else:
        n_background = None
        signal_counts = ()
    shared_metadata = (
        native_metadata[0] if native_metadata and all(item == native_metadata[0] for item in native_metadata) else None
    )
    return ECMPatchSummary(
        n_samples=len(adatas),
        n_patches=n_patches,
        n_features=n_features or 0,
        n_background=n_background,
        signal_counts=signal_counts,
        feature_backend=None if shared_metadata is None else shared_metadata[0],
        feature_families=() if shared_metadata is None else shared_metadata[1],
        channel_names=() if shared_metadata is None else shared_metadata[2],
        image_patch_shape=None if shared_metadata is None else shared_metadata[3],
    )


def ecm_patches_from_images(
    images: Mapping[str, np.ndarray | ImageContainer],
    panel: str | Path | pd.DataFrame,
    *,
    keep_col: str = "keep",
    normalize: Literal["pooled_zscore", "pooled_arcsinh_percentile", "none"] = "pooled_zscore",
    cofactor: float = 1.0,
    percentile: float = 99.5,
    normalization_chunk_size: int = 100_000,
    **extract_kwargs,
) -> dict[str, AnnData]:
    """Create ECM AnnData carriers and extract patches from image stacks.

    This is the shortest path from the public Mantpy input contract to a
    cohort of ECM patch tables. ``panel`` may be an acquisition panel with a
    ``keep`` column or the already-filtered channel panel. Each image must be
    channel-first and match the retained panel rows in count and order.
    ``normalize='pooled_arcsinh_percentile'`` reproduces the pooled ECM-pixel
    transform used by the Schistosoma workflow.
    """
    if isinstance(panel, pd.DataFrame):
        panel_df = panel.copy()
    else:
        panel_df = pd.read_csv(panel)
    panel_df.columns = [str(column).strip() for column in panel_df.columns]
    if keep_col in panel_df:
        panel_df = panel_df.loc[panel_df[keep_col].eq(1)].reset_index(drop=True)
    else:
        panel_df = panel_df.reset_index(drop=True)

    from mantpy.io import read_imc

    carriers = {
        sample: read_imc(image, panel_df, cells=None, normalize="none", sample_id=str(sample))
        for sample, image in images.items()
    }
    result = extract_ecm_patches_cohort(
        carriers,
        images,
        normalize=normalize,
        cofactor=cofactor,
        percentile=percentile,
        normalization_chunk_size=normalization_chunk_size,
        inplace=False,
        **extract_kwargs,
    )
    if result is None:  # pragma: no cover - inplace=False guarantees a result
        raise RuntimeError("ECM patch extraction returned no cohort.")
    return result


def attach_ecm_patches(
    cells_by_sample: Mapping[str, AnnData],
    ecm_by_sample: Mapping[str, AnnData],
    *,
    inplace: bool = True,
) -> dict[str, AnnData] | None:
    """Attach each ECM patch table to its matching typed-cell AnnData."""
    if set(cells_by_sample) != set(ecm_by_sample):
        raise ValueError("Cell and ECM cohorts must contain the same sample keys.")
    targets = cells_by_sample if inplace else {sample: adata.copy() for sample, adata in cells_by_sample.items()}
    for sample, adata in targets.items():
        if ECM_PATCHES_KEY not in ecm_by_sample[sample].uns:
            raise KeyError(f"ECM sample {sample!r} has no uns[{ECM_PATCHES_KEY!r}].")
        adata.uns[ECM_PATCHES_KEY] = ecm_by_sample[sample].uns[ECM_PATCHES_KEY].copy()
        adata.uns.pop(ECM_GRAPH_KEY, None)
        adata.uns.pop(CELL_ECM_GRAPH_KEY, None)
    if inplace:
        return None
    return dict(targets)


def apply_ecm_label_overlay(
    ecm_by_sample: Mapping[str, AnnData],
    overlays: Mapping[str, Mapping[str, Any]],
    *,
    cluster_key: str = "ecm_cluster",
    truth_key: str = "is_artifact",
    replacement_key: str = "ecm_cluster_artifact",
    reference_key: str = "ecm_cluster_pristine",
    max_reference_mismatch_fraction: float = 0.01,
) -> dict[str, AnnData]:
    """Apply a row-aligned label-corruption definition to ECM patches.

    Only rows flagged by ``truth_key`` receive the overlay's replacement
    label. Every other label and every patch feature remains from the input
    cohort. The optional reference labels are used only to reject an overlay
    that is not aligned with the requested clustering.
    """
    if set(ecm_by_sample) != set(overlays):
        raise ValueError("ECM samples and label overlays must contain the same sample keys.")
    if not 0 <= max_reference_mismatch_fraction <= 1:
        raise ValueError("max_reference_mismatch_fraction must lie in [0, 1].")

    targets = {sample: adata.copy() for sample, adata in ecm_by_sample.items()}
    for sample, target in targets.items():
        patches = target.uns.get(ECM_PATCHES_KEY)
        if patches is None or cluster_key not in patches:
            raise KeyError(f"Sample {sample!r} has no patch labels in {cluster_key!r}.")
        overlay = overlays[sample]
        missing = [key for key in (truth_key, replacement_key) if key not in overlay]
        if missing:
            raise KeyError(f"Overlay {sample!r} is missing keys {missing}.")
        mask = np.asarray(overlay[truth_key], dtype=bool)
        replacement = np.asarray(overlay[replacement_key], dtype=int)
        if len(mask) != len(patches) or len(replacement) != len(patches):
            raise ValueError(f"Overlay {sample!r} has {len(mask)} rows but ECM patches have {len(patches)}.")

        labels = patches[cluster_key].to_numpy(dtype=int).copy()
        reference_mismatches = 0
        if reference_key in overlay:
            reference = np.asarray(overlay[reference_key], dtype=int)
            if len(reference) != len(labels):
                raise ValueError(f"Overlay reference {sample!r} has the wrong row count.")
            reference_mismatches = int((labels != reference).sum())
            mismatch_fraction = reference_mismatches / max(len(labels), 1)
            if mismatch_fraction > max_reference_mismatch_fraction:
                raise ValueError(
                    f"Overlay {sample!r} disagrees with {reference_mismatches:,} of "
                    f"{len(labels):,} input labels ({mismatch_fraction:.2%}); "
                    "the overlay and clustering are not aligned."
                )
        labels[mask] = replacement[mask]
        updated = patches.copy()
        updated[cluster_key] = labels
        updated[truth_key] = mask
        target.uns[ECM_PATCHES_KEY] = updated
        target.uns["ecm_label_overlay"] = {
            "n_replaced": int(mask.sum()),
            "reference_mismatches": reference_mismatches,
        }
        boxes = overlay.get("artifact_boxes")
        if boxes:
            target.uns["artifact_boxes"] = list(boxes)
    return targets


def ecm_label_overlay_summary(
    adatas: Mapping[str, AnnData],
    *,
    cluster_key: str = "ecm_cluster",
    truth_key: str = "is_artifact",
) -> ECMLabelOverlaySummary:
    """Summarize counts and alignment metadata after a label overlay."""
    if not adatas:
        raise ValueError("adatas is empty; expected an overlaid ECM cohort.")
    n_patches = n_signal = n_artifacts = reference_mismatches = 0
    for sample, adata in adatas.items():
        patches = adata.uns.get(ECM_PATCHES_KEY)
        if patches is None or cluster_key not in patches or truth_key not in patches:
            raise KeyError(f"Sample {sample!r} must contain {cluster_key!r} and {truth_key!r} patch columns.")
        n_patches += len(patches)
        n_signal += int((patches[cluster_key].to_numpy(dtype=int) >= 0).sum())
        n_artifacts += int(patches[truth_key].to_numpy(dtype=bool).sum())
        metadata = adata.uns.get("ecm_label_overlay", {})
        reference_mismatches += int(metadata.get("reference_mismatches", 0))
    return ECMLabelOverlaySummary(
        n_samples=len(adatas),
        n_patches=n_patches,
        n_signal=n_signal,
        n_artifacts=n_artifacts,
        reference_mismatches=reference_mismatches,
    )


def extract_ecm_patches_cohort(
    adatas: Mapping[str, AnnData],
    images: Mapping[str, np.ndarray | ImageContainer],
    *,
    normalize: Literal["pooled_zscore", "pooled_arcsinh_percentile", "none"] = "pooled_zscore",
    cofactor: float = 1.0,
    percentile: float = 99.5,
    normalization_chunk_size: int = 100_000,
    inplace: bool = False,
    **extract_kwargs,
) -> dict[str, AnnData] | None:
    """Extract ECM patches across a cohort with one shared image normalisation.

    ``normalize='pooled_zscore'`` incrementally fits one scikit-learn
    ``StandardScaler`` across the pooled pixels of all images. The same scaler
    is applied to every ROI. ``normalize='pooled_arcsinh_percentile'`` pools
    one ECM marker at a time, computes a single percentile bound across every
    ROI pixel in arcsinh space, then applies those shared bounds to each ROI.
    Normalisation provenance is written to
    ``uns['ecm_pixel_normalization']``.
    """
    if not adatas:
        raise ValueError("adatas is empty; expected at least one ECM sample.")
    if set(adatas) != set(images):
        missing_images = sorted(set(adatas) - set(images))
        missing_adatas = sorted(set(images) - set(adatas))
        raise ValueError(f"Cohort keys differ; missing images={missing_images}, missing AnnData={missing_adatas}.")
    if normalize not in {"pooled_zscore", "pooled_arcsinh_percentile", "none"}:
        raise ValueError(
            "normalize must be 'pooled_zscore', 'pooled_arcsinh_percentile', or 'none'."
        )
    if "features" not in extract_kwargs:
        raise TypeError("extract_ecm_patches_cohort requires features=[...].")
    if normalization_chunk_size < 1:
        raise ValueError("normalization_chunk_size must be at least 1.")
    if normalize == "pooled_arcsinh_percentile":
        if not np.isfinite(cofactor) or cofactor <= 0:
            raise ValueError("cofactor must be a finite positive number.")
        if not np.isfinite(percentile) or not 0 < percentile <= 100:
            raise ValueError("percentile must lie in (0, 100].")

    from mantpy.im import ImageContainer

    def _shape(image: np.ndarray | ImageContainer) -> tuple[int, int, int]:
        shape = image.shape if isinstance(image, ImageContainer) else np.asarray(image).shape
        if len(shape) != 3:
            raise ValueError(f"Every image must be channel-first (C, H, W); got shape {shape}.")
        return tuple(int(value) for value in shape)

    def _array(image: np.ndarray | ImageContainer) -> np.ndarray:
        return image.to_array() if isinstance(image, ImageContainer) else np.asarray(image)

    def _channel_float32(image: np.ndarray | ImageContainer, channel: int) -> np.ndarray:
        if isinstance(image, ImageContainer):
            selected = image[channel]
            if not isinstance(selected, ImageContainer):
                raise TypeError("ImageContainer channel indexing returned an unexpected value.")
            return selected.to_array(dtype=np.float32)[0]
        return np.asarray(image[channel], dtype=np.float32)

    image_shapes = {sample: _shape(image) for sample, image in images.items()}
    channel_counts = {shape[0] for shape in image_shapes.values()}
    if len(channel_counts) != 1:
        raise ValueError("All cohort images must have the same channel count and order.")
    n_channels = next(iter(channel_counts))
    for sample, adata in adatas.items():
        if adata.n_vars != n_channels:
            raise ValueError(
                f"Sample {sample!r} has {adata.n_vars} variables but its image has {n_channels} channels."
            )

    scaler: Any | None = None
    if normalize == "pooled_zscore":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        for image in images.values():
            array = _array(image)
            pixels = np.asarray(array).reshape(array.shape[0], -1).T
            for start in range(0, len(pixels), normalization_chunk_size):
                scaler.partial_fit(pixels[start : start + normalization_chunk_size])

    ecm_indices: tuple[int, ...] = ()
    marker_order: tuple[str, ...] = ()
    bounds: np.ndarray | None = None
    if normalize == "pooled_arcsinh_percentile":
        reference = next(iter(adatas.values()))
        if "is_ecm" not in reference.var:
            raise ValueError("AnnData is missing var['is_ecm']; create the cohort with mt.io.read_imc().")
        reference_names = tuple(map(str, reference.var_names))
        reference_ecm = reference.var["is_ecm"].to_numpy(dtype=bool)
        ecm_indices = tuple(int(index) for index in np.flatnonzero(reference_ecm))
        if not ecm_indices:
            raise ValueError("No ECM channels found (adata.var['is_ecm'] is all False).")
        marker_order = tuple(reference_names[index] for index in ecm_indices)
        for sample, adata in adatas.items():
            names = tuple(map(str, adata.var_names))
            if names != reference_names:
                raise ValueError(f"Sample {sample!r} has a different marker order.")
            if "is_ecm" not in adata.var or not np.array_equal(
                adata.var["is_ecm"].to_numpy(dtype=bool), reference_ecm
            ):
                raise ValueError(f"Sample {sample!r} has a different ECM-marker mask.")

        total_pixels = sum(shape[1] * shape[2] for shape in image_shapes.values())
        bound_values: list[np.floating] = []
        for channel in ecm_indices:
            pooled_arcsinh = np.empty(total_pixels, dtype=np.float32)
            offset = 0
            for image in images.values():
                values = _channel_float32(image, channel).reshape(-1)
                stop = offset + len(values)
                pooled_arcsinh[offset:stop] = np.arcsinh(values / cofactor)
                offset = stop
            bound_values.append(np.percentile(pooled_arcsinh, percentile))
        bounds = np.asarray(bound_values, dtype=np.float32)

    targets = adatas if inplace else {key: adata.copy() for key, adata in adatas.items()}
    for key, target in targets.items():
        image = images[key]
        if normalize == "none":
            prepared = _array(image)
        elif scaler is not None:
            array = _array(image)
            pixels = np.asarray(array).reshape(array.shape[0], -1).T
            transformed = np.empty(pixels.shape, dtype=np.float32)
            for start in range(0, len(pixels), normalization_chunk_size):
                stop = min(start + normalization_chunk_size, len(pixels))
                transformed[start:stop] = scaler.transform(pixels[start:stop]).astype(np.float32)
            prepared = transformed.T.reshape(array.shape)
        else:
            if bounds is None:
                raise RuntimeError("Pooled percentile bounds were not computed.")
            prepared = np.empty(image_shapes[key], dtype=np.float32)
            bound_by_channel = dict(zip(ecm_indices, bounds, strict=True))
            for channel in range(n_channels):
                transformed_channel = np.arcsinh(_channel_float32(image, channel) / cofactor)
                bound = bound_by_channel.get(channel)
                if bound is not None and bound > 0:
                    transformed_channel = np.clip(transformed_channel / bound, 0.0, 1.0)
                prepared[channel] = transformed_channel
        target.uns.pop(ECM_PATCHES_KEY, None)
        extract_ecm_patches(target, prepared, **extract_kwargs)
        if scaler is not None:
            target.uns["ecm_pixel_scaler"] = {
                "mean": scaler.mean_.copy(),
                "scale": scaler.scale_.copy(),
                "n_samples_seen": int(scaler.n_samples_seen_),
            }
        else:
            target.uns.pop("ecm_pixel_scaler", None)
        normalization_provenance: dict[str, Any] = {"method": normalize}
        if bounds is not None:
            normalization_provenance.update(
                {
                    "cofactor": float(cofactor),
                    "percentile": float(percentile),
                    "bounds": bounds.copy(),
                    "marker_order": list(marker_order),
                    "bounds_dtype": str(bounds.dtype),
                }
            )
        elif scaler is not None:
            normalization_provenance.update(
                {
                    "mean": scaler.mean_.copy(),
                    "scale": scaler.scale_.copy(),
                    "marker_order": list(map(str, target.var_names)),
                }
            )
        target.uns["ecm_pixel_normalization"] = normalization_provenance
        _log_params(
            target,
            "pp",
            {"extract_ecm_patches_cohort": normalization_provenance.copy()},
        )

    if inplace:
        return None
    return dict(targets)


def _pooled_patch_features(
    adatas: Mapping[str, AnnData],
    feature_prefix: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
    """Validate and pool a cohort's row-aligned ECM patch features."""
    if not adatas:
        raise ValueError("adatas is empty; expected at least one ECM sample.")
    feature_columns: tuple[str, ...] | None = None
    matrices: list[np.ndarray] = []
    lengths: list[int] = []
    for sample, adata in adatas.items():
        patches = adata.uns.get(ECM_PATCHES_KEY)
        if patches is None:
            raise KeyError(f"Sample {sample!r} has no uns[{ECM_PATCHES_KEY!r}].")
        columns = tuple(column for column in patches if column.startswith(feature_prefix))
        if not columns:
            raise ValueError(f"Sample {sample!r} has no patch features beginning with {feature_prefix!r}.")
        if feature_columns is None:
            feature_columns = columns
        elif columns != feature_columns:
            raise ValueError(f"Sample {sample!r} has a different patch-feature schema.")
        # Preserve the historical Schistosoma workflow: feature blocks were
        # pooled as float32 before StandardScaler. The dtype is scientifically
        # relevant for patches on a K-means decision boundary.
        matrix = patches.loc[:, columns].to_numpy(dtype=np.float32)
        matrices.append(matrix)
        lengths.append(len(matrix))
    if not sum(lengths):
        raise ValueError("The ECM cohort contains no patches.")
    return np.vstack(matrices), feature_columns or (), tuple(lengths)


def select_ecm_cluster_count(
    adatas: Mapping[str, AnnData],
    *,
    signal_k_range: range | list[int] | tuple[int, ...] = range(2, 7),
    feature_prefix: str = "feat_",
    background: Literal["lowest_mean"] = "lowest_mean",
    random_state: int = 0,
    n_seeds: int = 3,
    n_init: int = 10,
    sample_size: int = 8000,
) -> ClusterCountSelection:
    """Select total ECM components, then reproduce the signal-only K scan.

    The candidate diagnostic fits ``signal K + 1`` components and treats the
    dimmest component as provisional background before calculating silhouette
    on the remaining labels. After selecting a total component count, a second
    diagnostic fixes that background partition and refits signal-only K-means.
    The latter provides a conditional signal-only diagnostic after the
    background partition has been selected.
    """
    if background != "lowest_mean":
        raise ValueError("background must be 'lowest_mean'.")
    candidate_signal_k = tuple(int(k) for k in signal_k_range)
    if not candidate_signal_k or min(candidate_signal_k) < 2:
        raise ValueError("signal_k_range must contain integers of at least 2.")
    if n_seeds < 0 or n_init < 1 or sample_size < 2:
        raise ValueError("n_seeds must be non-negative, n_init positive, and sample_size at least 2.")

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    pooled, _, _ = _pooled_patch_features(adatas, feature_prefix)
    standardized = StandardScaler().fit_transform(pooled)
    candidate_rows: list[dict[str, int | float]] = []
    candidate_models: dict[int, Any] = {}
    for signal_k in candidate_signal_k:
        total_k = signal_k + 1
        if total_k >= len(standardized):
            raise ValueError(f"Candidate total K={total_k} requires more than {total_k} patches.")
        model = KMeans(total_k, n_init=n_init, random_state=random_state).fit(standardized)
        labels = model.labels_.astype(int)
        background_component = int(np.argmin(model.cluster_centers_.mean(axis=1)))
        signal_mask = labels != background_component
        signal_indices = np.flatnonzero(signal_mask)
        rng = np.random.default_rng(random_state)
        sampled = rng.choice(
            signal_indices,
            size=min(sample_size, len(signal_indices)),
            replace=False,
        )
        silhouette = float(silhouette_score(standardized[sampled], labels[sampled]))
        aris = [
            float(
                adjusted_rand_score(
                    labels,
                    KMeans(total_k, n_init=n_init, random_state=random_state + seed).fit_predict(standardized),
                )
            )
            for seed in range(1, n_seeds + 1)
        ]
        candidate_rows.append(
            {
                "signal_k": signal_k,
                "total_k": total_k,
                "silhouette": silhouette,
                "mean_cross_seed_ari": float(np.mean(aris)) if aris else float("nan"),
                "background_patches": int((~signal_mask).sum()),
            }
        )
        candidate_models[total_k] = model

    candidate_scan = pd.DataFrame(candidate_rows)
    selected_row = candidate_scan.loc[candidate_scan["silhouette"].idxmax()]
    selected_total_k = int(selected_row["total_k"])
    selected_model = candidate_models[selected_total_k]
    background_component = int(np.argmin(selected_model.cluster_centers_.mean(axis=1)))
    selected_signal = selected_model.labels_ != background_component
    signal_features = StandardScaler().fit_transform(pooled[selected_signal])
    rng = np.random.default_rng(random_state)
    sampled = rng.choice(
        len(signal_features),
        size=min(sample_size, len(signal_features)),
        replace=False,
    )

    signal_rows: list[dict[str, int | float]] = []
    for signal_k in candidate_signal_k:
        reference = KMeans(signal_k, n_init=n_init, random_state=random_state).fit_predict(signal_features)
        silhouette = float(silhouette_score(signal_features[sampled], reference[sampled]))
        aris = [
            float(
                adjusted_rand_score(
                    reference,
                    KMeans(
                        signal_k,
                        n_init=n_init,
                        random_state=random_state + seed,
                    ).fit_predict(signal_features),
                )
            )
            for seed in range(1, n_seeds + 1)
        ]
        signal_rows.append(
            {
                "K": signal_k,
                "silhouette": silhouette,
                "mean_cross_seed_ari": float(np.mean(aris)) if aris else float("nan"),
            }
        )
    signal_scan = pd.DataFrame(signal_rows)
    selected_signal_k = int(signal_scan.loc[signal_scan["silhouette"].idxmax(), "K"])
    return ClusterCountSelection(
        candidate_scan=candidate_scan,
        signal_scan=signal_scan,
        selected_total_k=selected_total_k,
        selected_signal_k=selected_signal_k,
        background_component=background_component,
    )


def select_ecm_leiden_resolution(
    adatas: Mapping[str, AnnData],
    resolutions: list[float] | tuple[float, ...] | np.ndarray,
    *,
    subset: Literal["all", "signal"] = "signal",
    n_neighbors: int = 15,
    random_state: int = 0,
    flavor: Literal["leidenalg", "igraph"] = "leidenalg",
    feature_prefix: str = "feat_",
    cluster_key: str = "ecm_cluster",
) -> ECMLeidenResolutionSelection:
    """Select a Leiden resolution on one shared cohort neighbour graph."""
    candidate_resolutions = tuple(float(value) for value in resolutions)
    if not candidate_resolutions:
        raise ValueError("resolutions must contain at least one candidate.")
    if len(set(candidate_resolutions)) != len(candidate_resolutions):
        raise ValueError("resolutions must not contain duplicate values.")
    if any(not np.isfinite(value) or value <= 0 for value in candidate_resolutions):
        raise ValueError("Every resolution must be a finite positive number.")
    if subset not in {"all", "signal"}:
        raise ValueError("subset must be 'all' or 'signal'.")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2.")
    if flavor not in {"leidenalg", "igraph"}:
        raise ValueError("flavor must be 'leidenalg' or 'igraph'.")

    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("Leiden resolution selection requires scanpy and a Leiden backend.") from exc
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.preprocessing import StandardScaler

    pooled, feature_columns, _ = _pooled_patch_features(adatas, feature_prefix)
    if not np.isfinite(pooled).all():
        pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
    standardized = StandardScaler().fit_transform(pooled)
    if standardized.dtype != np.float32:
        standardized = standardized.astype(np.float32)

    if subset == "all":
        active = np.ones(len(pooled), dtype=bool)
    else:
        label_blocks: list[np.ndarray] = []
        for sample, adata in adatas.items():
            patches = adata.uns[ECM_PATCHES_KEY]
            if cluster_key not in patches:
                raise KeyError(f"Sample {sample!r} has no patch labels in {cluster_key!r}.")
            label_blocks.append(patches[cluster_key].to_numpy(dtype=int))
        active = np.concatenate(label_blocks) >= 0
    selected_features = standardized[active].astype(np.float32, copy=False)
    if len(selected_features) < 3:
        raise ValueError("Leiden resolution selection requires at least three selected patches.")
    effective_n_neighbors = min(int(n_neighbors), len(selected_features) - 1)

    leiden_adata = AnnData(X=selected_features)
    sc.pp.neighbors(
        leiden_adata,
        n_neighbors=effective_n_neighbors,
        use_rep="X",
        random_state=random_state,
    )
    leiden_kwargs: dict[str, Any] = {"flavor": flavor}
    if flavor == "igraph":
        leiden_kwargs.update({"directed": False, "n_iterations": 2})

    rows: list[dict[str, int | float]] = []
    for index, resolution in enumerate(candidate_resolutions):
        key_added = f"_mantpy_leiden_resolution_{index}"
        with warnings.catch_warnings():
            if flavor == "leidenalg":
                warnings.filterwarnings(
                    "ignore",
                    message="In the future, the default backend for leiden will be igraph.*",
                    category=FutureWarning,
                )
            sc.tl.leiden(
                leiden_adata,
                resolution=resolution,
                random_state=random_state,
                key_added=key_added,
                **leiden_kwargs,
            )
        labels = leiden_adata.obs[key_added].astype(int).to_numpy()
        observed_n_clusters = int(np.unique(labels).size)
        score = (
            float(calinski_harabasz_score(selected_features, labels))
            if 1 < observed_n_clusters < len(labels)
            else float("nan")
        )
        rows.append(
            {
                "resolution": resolution,
                "n_clusters": observed_n_clusters,
                "calinski_harabasz": score,
            }
        )

    table = pd.DataFrame(rows)
    valid = table.loc[np.isfinite(table["calinski_harabasz"])].copy()
    if valid.empty:
        raise ValueError(
            "No candidate resolution produced a partition with a finite Calinski-Harabasz score."
        )
    selected = valid.sort_values(
        ["calinski_harabasz", "resolution"], ascending=[False, True]
    ).iloc[0]

    from mantpy._version import __version__

    versions = {
        "mantpy": __version__,
        "numpy": np.__version__,
        "scikit-learn": _package_version("scikit-learn"),
        "scanpy": _package_version("scanpy"),
        "igraph": _package_version("igraph"),
        "leidenalg": _package_version("leidenalg"),
    }
    return ECMLeidenResolutionSelection(
        table=table,
        selected_resolution=float(selected["resolution"]),
        selected_n_clusters=int(selected["n_clusters"]),
        subset=subset,
        n_neighbors=int(n_neighbors),
        effective_n_neighbors=effective_n_neighbors,
        flavor=flavor,
        feature_columns=feature_columns,
        versions=versions,
    )


def cluster_ecm_patches(
    adatas: Mapping[str, AnnData],
    *,
    n_clusters: int | None = None,
    method: Literal["kmeans", "leiden"] = "kmeans",
    subset: Literal["all", "signal"] = "all",
    feature_prefix: str = "feat_",
    random_state: int = 0,
    n_init: int = 10,
    n_neighbors: int = 15,
    resolution: float = 0.3,
    flavor: Literal["leidenalg", "igraph"] = "leidenalg",
    cluster_key: str = "ecm_cluster",
) -> ECMClusteringResult:
    """Cluster pooled ECM patches and write cohort-consistent labels."""
    if method not in {"kmeans", "leiden"}:
        raise ValueError("method must be 'kmeans' or 'leiden'.")
    if subset not in {"all", "signal"}:
        raise ValueError("subset must be 'all' or 'signal'.")
    if method == "kmeans":
        if n_clusters is None or n_clusters < 1:
            raise ValueError("n_clusters must be at least 1 when method='kmeans'.")
        if n_init < 1:
            raise ValueError("n_init must be at least 1.")
    else:
        if n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2.")
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be a finite positive number.")
        if flavor not in {"leidenalg", "igraph"}:
            raise ValueError("flavor must be 'leidenalg' or 'igraph'.")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    pooled, feature_columns, lengths = _pooled_patch_features(adatas, feature_prefix)
    if not np.isfinite(pooled).all():
        pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(pooled)
    if standardized.dtype != np.float32:
        standardized = standardized.astype(np.float32)

    if subset == "all":
        active = np.ones(len(pooled), dtype=bool)
    else:
        existing_blocks: list[np.ndarray] = []
        for sample, adata in adatas.items():
            patches = adata.uns[ECM_PATCHES_KEY]
            if cluster_key not in patches:
                raise KeyError(f"Sample {sample!r} has no patch labels in {cluster_key!r}.")
            existing_blocks.append(patches[cluster_key].to_numpy(dtype=int))
        active = np.concatenate(existing_blocks) >= 0
    n_active = int(active.sum())
    if n_active == 0:
        raise ValueError(f"No patches are available for subset={subset!r}.")

    feature_metadata_raw = next(iter(adatas.values())).uns.get("ecm_feature_names")
    if isinstance(feature_metadata_raw, str):
        try:
            feature_metadata = json.loads(feature_metadata_raw)
        except (TypeError, ValueError):
            feature_metadata = None
    elif isinstance(feature_metadata_raw, list):
        feature_metadata = feature_metadata_raw
    else:
        feature_metadata = None
    mean_feature_indices: tuple[int, ...] = ()
    if isinstance(feature_metadata, list) and len(feature_metadata) == len(feature_columns):
        mean_feature_indices = tuple(
            index
            for index, metadata in enumerate(feature_metadata)
            if isinstance(metadata, dict) and metadata.get("extractor") == "mean"
        )

    model: Any | None
    if method == "kmeans":
        if n_clusters is None:
            raise RuntimeError("n_clusters validation failed.")
        if n_clusters > n_active:
            raise ValueError(f"n_clusters={n_clusters} exceeds the {n_active} selected patches.")
        model = KMeans(n_clusters, n_init=n_init, random_state=random_state).fit(standardized[active])
        selected_labels = model.labels_.astype(np.int32)
        observed_n_clusters = int(n_clusters)
        if mean_feature_indices:
            raw_means = pooled[active][:, mean_feature_indices]
            component_means = tuple(
                float(raw_means[selected_labels == component].mean())
                for component in range(observed_n_clusters)
            )
        else:
            component_means = tuple(float(value) for value in model.cluster_centers_.mean(axis=1))
    else:
        if n_active < 2:
            raise ValueError("Leiden clustering requires at least two selected patches.")
        try:
            import scanpy as sc
        except ImportError as exc:
            raise ImportError("method='leiden' requires scanpy and a Leiden backend.") from exc

        leiden_adata = AnnData(X=standardized[active].astype(np.float32, copy=False))
        sc.pp.neighbors(
            leiden_adata,
            n_neighbors=n_neighbors,
            use_rep="X",
            random_state=random_state,
        )
        leiden_kwargs: dict[str, Any] = {"flavor": flavor}
        if flavor == "igraph":
            leiden_kwargs.update({"directed": False, "n_iterations": 2})
        with warnings.catch_warnings():
            if flavor == "leidenalg":
                warnings.filterwarnings(
                    "ignore",
                    message="In the future, the default backend for leiden will be igraph.*",
                    category=FutureWarning,
                )
            sc.tl.leiden(
                leiden_adata,
                resolution=resolution,
                random_state=random_state,
                key_added="_mantpy_leiden",
                **leiden_kwargs,
            )
        raw_labels = leiden_adata.obs["_mantpy_leiden"].astype(int).to_numpy()
        raw_values, raw_counts = np.unique(raw_labels, return_counts=True)
        ordered = sorted(
            zip(raw_values.tolist(), raw_counts.tolist(), strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        relabel = {int(raw_label): label for label, (raw_label, _) in enumerate(ordered)}
        selected_labels = np.fromiter(
            (relabel[int(label)] for label in raw_labels),
            dtype=np.int32,
            count=len(raw_labels),
        )
        observed_n_clusters = len(ordered)
        raw_for_means = pooled[active]
        if mean_feature_indices:
            raw_for_means = raw_for_means[:, mean_feature_indices]
        component_means = tuple(
            float(raw_for_means[selected_labels == component].mean())
            for component in range(observed_n_clusters)
        )
        model = None

    labels = np.full(len(pooled), -1, dtype=np.int32)
    labels[active] = selected_labels
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    cluster_counts = tuple(
        (int(label), int(count))
        for label, count in zip(unique_labels, label_counts, strict=True)
    )
    counts_by_label = dict(cluster_counts)
    component_sizes = tuple(
        counts_by_label.get(component, 0) for component in range(observed_n_clusters)
    )

    from mantpy._version import __version__

    versions = {
        "mantpy": __version__,
        "numpy": np.__version__,
        "scikit-learn": _package_version("scikit-learn"),
    }
    if method == "leiden":
        versions.update(
            {
                "scanpy": _package_version("scanpy"),
                "igraph": _package_version("igraph"),
                "leidenalg": _package_version("leidenalg"),
            }
        )
    provenance: dict[str, Any] = {
        "provenance_version": 1,
        "mantpy_version": __version__,
        "method": method,
        "subset": subset,
        "cluster_key": cluster_key,
        "feature_columns": list(feature_columns),
        "random_state": int(random_state),
        "n_clusters": observed_n_clusters,
        "cluster_counts": {str(label): count for label, count in cluster_counts},
        "standardization": {
            "method": "StandardScaler",
            "fit_subset": "all",
            "input_dtype": str(pooled.dtype),
        },
        "versions": versions,
    }
    if method == "kmeans":
        provenance.update({"n_init": int(n_init), "requested_n_clusters": int(observed_n_clusters)})
    else:
        provenance.update(
            {
                "n_neighbors": int(n_neighbors),
                "resolution": float(resolution),
                "flavor": flavor,
                "label_order": "descending_size_then_original_label",
            }
        )

    offset = 0
    for adata, length in zip(adatas.values(), lengths, strict=True):
        patches = adata.uns[ECM_PATCHES_KEY].copy()
        patches[cluster_key] = labels[offset : offset + length]
        adata.uns[ECM_PATCHES_KEY] = patches
        adata.uns["ecm_clustering"] = {
            **provenance,
            "feature_columns": provenance["feature_columns"].copy(),
            "cluster_counts": provenance["cluster_counts"].copy(),
            "standardization": provenance["standardization"].copy(),
            "versions": provenance["versions"].copy(),
        }
        _log_params(
            adata,
            "pp",
            {"cluster_ecm_patches": adata.uns["ecm_clustering"].copy()},
        )
        offset += length
    return ECMClusteringResult(
        n_clusters=observed_n_clusters,
        cluster_key=cluster_key,
        feature_columns=feature_columns,
        component_means=component_means,
        component_sizes=component_sizes,
        scaler=scaler,
        model=model,
        method=method,
        subset=subset,
        cluster_counts=cluster_counts,
        provenance=provenance,
    )


def remove_background_patches(
    adatas: Mapping[str, AnnData],
    clustering: ECMClusteringResult,
    *,
    background: Literal["lowest_mean"] = "lowest_mean",
    order_signal_by: Literal["size", "label"] = "size",
    cluster_key: str | None = None,
) -> BackgroundRemovalSummary:
    """Relabel the dimmest component as ``-1`` and order signal labels."""
    if clustering.method != "kmeans" or clustering.subset != "all":
        raise ValueError("Background removal requires an all-patch K-means clustering result.")
    if background != "lowest_mean":
        raise ValueError("background must be 'lowest_mean'.")
    if order_signal_by not in {"size", "label"}:
        raise ValueError("order_signal_by must be 'size' or 'label'.")
    key = cluster_key or clustering.cluster_key
    background_component = int(np.argmin(clustering.component_means))
    signal_components = [component for component in range(clustering.n_clusters) if component != background_component]
    if order_signal_by == "size":
        signal_components.sort(key=lambda component: (-clustering.component_sizes[component], component))
    mapping = {component: label for label, component in enumerate(signal_components)}
    mapping[background_component] = -1
    final_counts = dict.fromkeys(range(len(signal_components)), 0)
    background_patches = 0
    for sample, adata in adatas.items():
        patches = adata.uns.get(ECM_PATCHES_KEY)
        if patches is None or key not in patches:
            raise KeyError(f"Sample {sample!r} has no raw component labels in {key!r}.")
        raw = patches[key].to_numpy(dtype=int)
        unexpected = sorted(set(np.unique(raw)) - set(mapping))
        if unexpected:
            raise ValueError(f"Sample {sample!r} contains unexpected component labels {unexpected}.")
        relabelled = np.fromiter((mapping[int(label)] for label in raw), dtype=np.int32)
        updated = patches.copy()
        updated[key] = relabelled
        adata.uns[ECM_PATCHES_KEY] = updated
        background_patches += int((relabelled < 0).sum())
        for label in final_counts:
            final_counts[label] += int((relabelled == label).sum())
    return BackgroundRemovalSummary(
        background_component=background_component,
        background_patches=background_patches,
        signal_counts=tuple(final_counts.items()),
    )


def compare_ecm_patches(
    observed: Mapping[str, AnnData],
    reference: Mapping[str, AnnData],
    *,
    feature_prefix: str = "feat_",
    cluster_key: str | None = "ecm_cluster",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> PatchComparison:
    """Compare row-aligned patch coordinates, features, and optional labels."""
    if set(observed) != set(reference):
        raise ValueError("Observed and reference cohorts must contain the same ROI keys.")
    n_patches = 0
    n_features: int | None = None
    coordinate_matches = 0
    feature_matches = 0
    label_matches = 0
    labels_checked = cluster_key is not None
    for key in observed:
        left = observed[key].uns.get(ECM_PATCHES_KEY)
        right = reference[key].uns.get(ECM_PATCHES_KEY)
        if left is None or right is None:
            raise ValueError(f"ROI {key!r} is missing uns['{ECM_PATCHES_KEY}'] in one cohort.")
        if len(left) != len(right):
            raise ValueError(f"ROI {key!r} has {len(left)} observed and {len(right)} reference patches.")
        cols = [c for c in left.columns if c.startswith(feature_prefix)]
        if cols != [c for c in right.columns if c.startswith(feature_prefix)]:
            raise ValueError(f"ROI {key!r} has mismatched feature columns.")
        n_features = len(cols) if n_features is None else n_features
        if len(cols) != n_features:
            raise ValueError(f"ROI {key!r} has a different number of patch features.")
        n_patches += len(left)
        coord_equal = np.isclose(
            left[["x", "y"]].to_numpy(dtype=float),
            right[["x", "y"]].to_numpy(dtype=float),
            atol=atol,
            rtol=rtol,
        ).all(axis=1)
        feat_equal = np.isclose(
            left[cols].to_numpy(dtype=float),
            right[cols].to_numpy(dtype=float),
            atol=atol,
            rtol=rtol,
        ).all(axis=1)
        coordinate_matches += int(coord_equal.sum())
        feature_matches += int(feat_equal.sum())
        if cluster_key is not None and cluster_key in left and cluster_key in right:
            label_matches += int(
                (left[cluster_key].to_numpy(dtype=int) == right[cluster_key].to_numpy(dtype=int)).sum()
            )
        elif cluster_key is not None:
            labels_checked = False
    return PatchComparison(
        n_patches=n_patches,
        n_features=n_features or 0,
        coordinate_matches=coordinate_matches,
        feature_matches=feature_matches,
        label_matches=label_matches if labels_checked else None,
    )


# ---------------------------------------------------------------------------
# Extraction paths
# ---------------------------------------------------------------------------


def _extract_whole(
    ecm_arr: np.ndarray,
    patch_size: int,
    ecm_K: int | Literal["auto"],
    extractors: list,
    background_quantile: float,
    spatial_mask: np.ndarray | None,
    H: int,
    W: int,
    *,
    min_signal_fraction: float | None = None,
    threshold_method: str = "li",
    threshold_value: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Standard single-pass extraction on a fully in-memory array."""
    patches, cx, cy = extract_patches(ecm_arr, patch_size)
    feats = compute_features(patches, extractors)

    if min_signal_fraction is not None:
        from mantpy._core._patching import remove_background_adaptive

        bg_mask = remove_background_adaptive(
            patches,
            min_signal_fraction=min_signal_fraction,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
        )
        if spatial_mask is not None:
            spatial_keep = _spatial_mask_for_patches(cx, cy, spatial_mask)
            bg_mask = bg_mask & spatial_keep
    elif spatial_mask is not None:
        spatial_keep = _spatial_mask_for_patches(cx, cy, spatial_mask)
        # Compute background threshold within spatially-foreground patches only,
        # so the quantile is relative to ECM signal (not empty background tiles).
        intensity_keep = np.zeros(len(feats), dtype=bool)
        if spatial_keep.any():
            fg_idx = np.where(spatial_keep)[0]
            intensity_keep_fg = remove_background(feats[fg_idx], background_quantile)
            intensity_keep[fg_idx] = intensity_keep_fg
        bg_mask = intensity_keep
    else:
        bg_mask = remove_background(feats, background_quantile)

    fg_features = feats[bg_mask]
    fg_cx = cx[bg_mask]
    fg_cy = cy[bg_mask]

    if len(fg_features) == 0:
        raise ValueError(
            "All patches were removed as background. Try lowering `background_quantile` or adjusting the mask."
        )

    if ecm_K is None:
        labels = np.zeros(len(fg_features), dtype=np.int32)
    else:
        labels, _ = cluster_patches(fg_features, ecm_K)

    patch_df = build_ecm_patch_dataframe(fg_cx, fg_cy, fg_features, labels)

    n_patches_y = H // patch_size
    n_patches_x = W // patch_size
    labels_img = build_ecm_image(labels, bg_mask, n_patches_y, n_patches_x, patch_size, H, W)
    return patch_df, labels_img


def _extract_tiled(
    ic: ImageContainer,
    ecm_channel_indices: list[int],
    patch_size: int,
    ecm_K: int | Literal["auto"],
    extractors: list,
    background_quantile: float,
    spatial_mask: np.ndarray | None,
    tile_size: int,
    tile_overlap: int,
    H: int,
    W: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Tile-by-tile extraction — never loads the full image into RAM."""
    all_features: list[np.ndarray] = []
    all_cx: list[np.ndarray] = []
    all_cy: list[np.ndarray] = []

    for tile_ic, info in ic.tile_iter(tile_size=tile_size, overlap=tile_overlap):
        ecm_tile = tile_ic[ecm_channel_indices].to_array()  # (C_ecm, th, tw)

        patches, cx_local, cy_local = extract_patches(ecm_tile, patch_size)
        if len(patches) == 0:
            continue
        feats = compute_features(patches, extractors)

        if spatial_mask is not None:
            gx = cx_local + info["xc0"]
            gy = cy_local + info["yc0"]
            spatial_keep = _spatial_mask_for_patches(gx, gy, spatial_mask)
            bg_mask = np.zeros(len(feats), dtype=bool)
            if spatial_keep.any():
                fg_idx = np.where(spatial_keep)[0]
                intensity_keep_fg = remove_background(feats[fg_idx], background_quantile)
                bg_mask[fg_idx] = intensity_keep_fg
        else:
            bg_mask = remove_background(feats, background_quantile)

        if bg_mask.sum() == 0:
            continue

        fg_features = feats[bg_mask]
        fg_cx = cx_local[bg_mask] + info["xc0"]
        fg_cy = cy_local[bg_mask] + info["yc0"]

        if tile_overlap > 0:
            inner_x0 = float(info["x0"] - info["xc0"])
            inner_y0 = float(info["y0"] - info["yc0"])
            inner_x1 = inner_x0 + float(info["x1"] - info["x0"])
            inner_y1 = inner_y0 + float(info["y1"] - info["y0"])
            keep = (
                (cx_local[bg_mask] >= inner_x0)
                & (cx_local[bg_mask] < inner_x1)
                & (cy_local[bg_mask] >= inner_y0)
                & (cy_local[bg_mask] < inner_y1)
            )
            fg_features = fg_features[keep]
            fg_cx = fg_cx[keep]
            fg_cy = fg_cy[keep]

        if len(fg_features) == 0:
            continue

        all_features.append(fg_features)
        all_cx.append(fg_cx)
        all_cy.append(fg_cy)

    if not all_features:
        raise ValueError(
            "All patches were removed as background across all tiles. "
            "Try lowering `background_quantile` or adjusting the mask."
        )

    features_all = np.concatenate(all_features, axis=0)
    cx_all = np.concatenate(all_cx, axis=0)
    cy_all = np.concatenate(all_cy, axis=0)

    if ecm_K is None:
        labels = np.zeros(len(features_all), dtype=np.int32)
    else:
        labels, _ = cluster_patches(features_all, ecm_K)

    patch_df = build_ecm_patch_dataframe(cx_all, cy_all, features_all, labels)
    labels_img = build_ecm_image_from_coords(cx_all, cy_all, labels, patch_size, H, W)
    return patch_df, labels_img


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spatial_mask_for_patches(
    cx: np.ndarray,
    cy: np.ndarray,
    spatial_mask: np.ndarray,
) -> np.ndarray:
    """Return boolean array: True if the patch centroid is foreground in the mask."""
    H, W = spatial_mask.shape
    xi = np.clip(cx.astype(int), 0, W - 1)
    yi = np.clip(cy.astype(int), 0, H - 1)
    return spatial_mask[yi, xi]


# ---------------------------------------------------------------------------
# Structure annotation & splitting
# ---------------------------------------------------------------------------


def annotate_structure(
    adata: AnnData,
    *,
    name: str,
    markers: list[str],
    cluster_key: str = "leiden",
    n_top_clusters: int = 1,
    min_size: int = 5,
    max_dist: float = 50.0,
    roi_key: str | None = "roi_id",
    key_added: str = "structure",
    inplace: bool = True,
    verbose: bool = False,
) -> AnnData | None:
    """Annotate cells belonging to a biological structure based on marker expression.

    Identifies structures (e.g. islets, ducts, tumour nests) by scoring Leiden
    clusters on a set of defining markers and then spatially filtering isolated
    cells.

    Parameters
    ----------
    adata
        AnnData with clustered cells. Must have ``adata.obs[cluster_key]`` and
        ``adata.obsm['spatial']``.
    name
        Name of the structure (e.g. ``"islet"``, ``"duct"``). Cells belonging
        to the structure are labelled with this value.
    markers
        Marker names (or partial patterns) that define the structure.
        Fuzzy-matched against ``adata.var_names`` (e.g. ``"C-PEP"`` matches
        ``"Nd145_C-peptide"``).
    cluster_key
        Column in ``adata.obs`` containing cluster labels.
    n_top_clusters
        Number of top-scoring clusters to label as the structure.
    min_size
        Minimum number of spatially connected cells to retain a component.
        Smaller components are demoted to ``"other"``.
    max_dist
        Maximum distance (pixels) between cells to be considered spatially
        connected when filtering isolated components.
    roi_key
        Column in ``adata.obs`` identifying ROIs. Spatial filtering is applied
        per ROI. If ``None``, all cells are treated as one ROI.
    key_added
        Column name added to ``adata.obs`` for the annotation.
    inplace
        If ``True``, modify ``adata`` in place. Otherwise return a copy.
    verbose
        When ``True``, prints a summary: matched markers, top clusters and
        their scores, cells initially labelled, cells demoted by spatial
        filtering, and the final count. Default ``False`` (silent).

    Returns
    -------
    ``None`` if ``inplace=True``, otherwise the annotated AnnData.

    Examples
    --------
    >>> mt.pp.annotate_structure(adata, name="islet", markers=["C-PEP", "GCG", "SST"])
    >>> mt.pp.annotate_structure(adata, name="duct", markers=["CK19", "SOX9"], verbose=True)
    """
    import re
    from collections import Counter

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if not inplace:
        adata = adata.copy()

    # --- Step 1: fuzzy-match markers against var_names ---
    matched_markers = []
    for pattern in markers:
        for var_name in adata.var_names:
            if re.search(re.escape(pattern), var_name, re.IGNORECASE):
                matched_markers.append(var_name)
                break

    if not matched_markers:
        raise ValueError(
            f"None of the markers {markers} matched any variable in "
            f"adata.var_names. Available: {list(adata.var_names[:10])}..."
        )

    if verbose:
        # verbose=True writes a user-facing summary to stdout (notebook contract; capsys-tested).
        print(f"[annotate_structure] name='{name}', matched markers: {matched_markers}")  # noqa: T201

    # --- Step 2: score each cell ---
    marker_idx = [adata.var_names.get_loc(m) for m in matched_markers]
    cell_scores = np.asarray(adata.X[:, marker_idx].sum(axis=1)).flatten()

    # --- Step 3: score clusters, pick top N ---
    clusters = adata.obs[cluster_key].values
    cluster_ids = np.unique(clusters)
    cluster_means = {}
    for cid in cluster_ids:
        cluster_means[cid] = cell_scores[clusters == cid].mean()

    sorted_clusters = sorted(cluster_means, key=cluster_means.get, reverse=True)
    top_clusters = set(sorted_clusters[:n_top_clusters])

    if verbose:
        top_str = ", ".join(f"{c}={cluster_means[c]:.3f}" for c in sorted_clusters[:n_top_clusters])
        print(f"[annotate_structure] top {n_top_clusters} cluster(s) by mean score: {top_str}")  # noqa: T201

    # --- Step 4: initial labelling ---
    labels = np.where(np.isin(clusters, list(top_clusters)), name, "other").astype(object)
    n_initial = int((labels == name).sum())

    # --- Step 5: spatial filtering per ROI ---
    coords = adata.obsm["spatial"]

    if roi_key is not None and roi_key in adata.obs.columns:
        roi_values = adata.obs[roi_key].values
        unique_rois = np.unique(roi_values)
    else:
        roi_values = np.zeros(len(adata), dtype=int)
        unique_rois = [0]

    for roi in unique_rois:
        roi_idx = np.where(roi_values == roi)[0]
        struct_local = np.where(labels[roi_idx] == name)[0]
        if len(struct_local) < min_size:
            labels[roi_idx[struct_local]] = "other"
            continue

        struct_global = roi_idx[struct_local]
        struct_coords = coords[struct_global]

        tree = cKDTree(struct_coords)
        pairs = list(tree.query_pairs(r=max_dist))

        if not pairs:
            labels[struct_global] = "other"
            continue

        rows_p, cols_p = zip(*pairs, strict=False)
        n = len(struct_local)
        adj = csr_matrix(
            (np.ones(len(rows_p) * 2), (list(rows_p) + list(cols_p), list(cols_p) + list(rows_p))),
            shape=(n, n),
        )
        _, comp_labels = connected_components(adj, directed=False)
        comp_sizes = Counter(comp_labels)

        for i, lbl in enumerate(comp_labels):
            if comp_sizes[lbl] < min_size:
                labels[struct_global[i]] = "other"

    # --- Step 6: store ---
    adata.obs[key_added] = pd.Categorical(labels, categories=[name, "other"])

    if verbose:
        n_final = int((labels == name).sum())
        n_demoted = n_initial - n_final
        print(  # noqa: T201  user-facing stdout summary (notebook contract; capsys-tested)
            f"[annotate_structure] cells labelled '{name}': "
            f"{n_initial} initial -> {n_final} final "
            f"(demoted {n_demoted} by spatial filter; "
            f"{n_final / max(len(labels), 1):.1%} of {len(labels)} total)"
        )

    if not inplace:
        return adata
    return None


def split_structures(
    adata: AnnData,
    *,
    structure_key: str = "structure",
    structure_value: str | None = None,
    max_dist: float = 50.0,
    min_cells: int = 5,
    roi_key: str | None = "roi_id",
    key_added: str | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Split annotated structure cells into individual spatial instances.

    Finds connected components among cells labelled as the target structure,
    returning one entry per physical instance (e.g. one islet, one duct).

    Parameters
    ----------
    adata
        AnnData with ``adata.obs[structure_key]`` and ``adata.obsm['spatial']``.
    structure_key
        Column in ``adata.obs`` containing structure annotations (from
        :func:`annotate_structure`).
    structure_value
        Which category to split. If ``None``, uses the first non-``"other"``
        category.
    max_dist
        Maximum pixel distance for two cells to be spatially connected.
    min_cells
        Minimum cells per component to keep.
    roi_key
        Column in ``adata.obs`` identifying ROIs. If ``None``, all cells are
        treated as one ROI.
    key_added
        When set, also writes the per-cell structure UID to
        ``adata.obs[key_added]`` as a :class:`~pandas.Categorical` (cells not in
        any retained component become ``NaN``). This saves callers from
        writing their own ``iloc`` loop.

    Returns
    -------
    List of ``(structure_uid, obs_indices)`` tuples. ``structure_uid`` has the
    format ``"{roi}_{structure_value}{N:02d}"``. ``obs_indices`` is an integer
    array of row positions in ``adata``.

    Examples
    --------
    >>> structures = mt.pp.split_structures(adata, structure_value="islet")
    >>> # ... or have it write the column for you:
    >>> structures = mt.pp.split_structures(  # doctest: +SKIP
    ...     adata,
    ...     structure_value="islet",
    ...     key_added="structure_id",
    ... )
    """
    from collections import Counter

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if structure_key not in adata.obs.columns:
        raise KeyError(f"'{structure_key}' not found in adata.obs. Run mt.pp.annotate_structure() first.")

    categories = (
        adata.obs[structure_key].cat.categories.tolist()
        if hasattr(adata.obs[structure_key], "cat")
        else list(adata.obs[structure_key].unique())
    )

    if structure_value is None:
        non_other = [c for c in categories if c != "other"]
        if not non_other:
            raise ValueError("No non-'other' categories found in structure_key.")
        structure_value = non_other[0]

    coords = adata.obsm["spatial"]
    labels = adata.obs[structure_key].values

    if roi_key is not None and roi_key in adata.obs.columns:
        roi_values = adata.obs[roi_key].values
        unique_rois = np.unique(roi_values)
    else:
        roi_values = np.full(len(adata), "all")
        unique_rois = ["all"]

    results: list[tuple[str, np.ndarray]] = []

    for roi in unique_rois:
        roi_idx = np.where(roi_values == roi)[0]
        struct_local = np.where(labels[roi_idx] == structure_value)[0]
        if len(struct_local) == 0:
            continue

        struct_global = roi_idx[struct_local]
        struct_coords = coords[struct_global]

        if len(struct_local) < min_cells:
            continue

        tree = cKDTree(struct_coords)
        pairs = list(tree.query_pairs(r=max_dist))

        if pairs:
            rows_p, cols_p = zip(*pairs, strict=False)
            n = len(struct_local)
            adj = csr_matrix(
                (np.ones(len(rows_p) * 2), (list(rows_p) + list(cols_p), list(cols_p) + list(rows_p))),
                shape=(n, n),
            )
            _, comp_labels = connected_components(adj, directed=False)
        else:
            comp_labels = np.arange(len(struct_local))

        comp_sizes = Counter(comp_labels)
        comp_idx = 0
        for comp_id in sorted(comp_sizes.keys()):
            if comp_sizes[comp_id] < min_cells:
                continue
            local_mask = np.where(comp_labels == comp_id)[0]
            global_indices = struct_global[local_mask]
            uid = f"{roi}_{structure_value}{comp_idx:02d}"
            results.append((uid, global_indices))
            comp_idx += 1

    # Optionally write the structure UID per cell to adata.obs[key_added].
    # Cells not in any retained component get NaN (preserved as the missing
    # category by pandas.Categorical).
    if key_added is not None:
        uid_arr = np.full(adata.n_obs, None, dtype=object)
        for uid, idx in results:
            uid_arr[idx] = uid
        adata.obs[key_added] = pd.Categorical(uid_arr, categories=[uid for uid, _ in results])

    return results


def extract_structure_ecm(
    adata_roi: AnnData,
    *,
    structure_indices: np.ndarray,
    structure_uid: str,
    ecm_channel_img: np.ndarray,
    structure_mask: np.ndarray | None = None,
    cell_ecm_graph_key: str = "cell_ecm_graph",
    ecm_patches_key: str = "ecm_patches",
    spatial_key: str = "spatial",
    min_ecm_nodes: int = 5,
    precomputed_sauvola_binary: np.ndarray | None = None,
) -> AnnData | None:
    """Extract a per-structure ECM AnnData from a ROI-level cell-ECM graph.

    Selects the ECM patches that are connected (via cell-ECM edges) to the
    cells of one individual structure (e.g. an islet), then attaches
    structure-relative polar coordinates and Sauvola texture for downstream
    structure-level graph analysis.

    Parameters
    ----------
    adata_roi
        Per-ROI AnnData. Must already have the following in ``uns``:

        - ``adata_roi.uns[cell_ecm_graph_key]`` — a NetworkX graph with cell
          and ECM nodes named ``cell_{i}`` / ``ecm_{j}`` (as produced by
          :func:`~mantpy.gr.build_cell_ecm_graph`).
        - ``adata_roi.uns[ecm_patches_key]`` — the ECM patch DataFrame from
          :func:`extract_ecm_patches`.
    structure_indices
        obs indices (into ``adata_roi``) for cells belonging to this structure,
        as returned by :func:`split_structures`.
    structure_uid
        Identifier stored on the output AnnData (``uns['structure_uid']``).
    ecm_channel_img
        ``(H, W)`` ECM channel image (e.g. hot-pixel-filtered collagen) used to
        compute ``sauvola_pos`` and — together with ``structure_mask`` — the
        11-D graph feature vector.
    structure_mask
        Optional ``(H, W)`` binary mask of the structure's footprint (for
        regionprops-based area/circularity/solidity). If ``None``, a convex
        hull of the structure's cell centroids is used.
    cell_ecm_graph_key
        Key of the cell-ECM graph in ``adata_roi.uns``.
    ecm_patches_key
        Key of the ECM patch DataFrame in ``adata_roi.uns``.
    spatial_key
        Key of the ``(N, 2)`` cell coordinates in ``adata_roi.obsm``.
    min_ecm_nodes
        If fewer than this many ECM patches are connected to the structure,
        returns ``None``.

    Returns
    -------
    Per-structure ``AnnData`` with ``X`` = ECM feature matrix and obs columns
    ``x``, ``y``, ``sin_theta``, ``cos_theta``, ``radial_dist``,
    ``sauvola_pos``, ``ecm_cluster``. Also attaches
    ``uns['islet_graph_feat']`` (11-D), ``uns['structure_uid']``,
    ``uns['structure_centroid']``. Returns ``None`` when fewer than
    ``min_ecm_nodes`` ECM patches are connected.
    """
    import numpy as np
    import pandas as pd
    from anndata import AnnData as _AnnData

    if cell_ecm_graph_key not in adata_roi.uns:
        raise ValueError(f"adata_roi.uns[{cell_ecm_graph_key!r}] not found. Run mt.gr.build_cell_ecm_graph first.")
    if ecm_patches_key not in adata_roi.uns:
        raise ValueError(f"adata_roi.uns[{ecm_patches_key!r}] not found. Run mt.pp.extract_ecm_patches first.")

    G = adata_roi.uns[cell_ecm_graph_key]
    patches_df = adata_roi.uns[ecm_patches_key]

    # Collect ECM node ids connected to structure cells via cell-ecm edges
    struct_cell_ids = {f"cell_{int(i)}" for i in structure_indices}
    ecm_hits: set[int] = set()
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") != "cell-ecm":
            continue
        if u in struct_cell_ids and str(v).startswith("ecm_"):
            ecm_hits.add(int(str(v).split("_")[1]))
        elif v in struct_cell_ids and str(u).startswith("ecm_"):
            ecm_hits.add(int(str(u).split("_")[1]))

    if len(ecm_hits) < min_ecm_nodes:
        return None

    ecm_idx = np.array(sorted(ecm_hits), dtype=int)
    sel = patches_df.iloc[ecm_idx].reset_index(drop=True)

    # Structure centroid in image coordinates (patch x/y is image x/y)
    coords = np.asarray(adata_roi.obsm[spatial_key], dtype=float)[structure_indices]
    cx = float(coords[:, 0].mean())
    cy = float(coords[:, 1].mean())

    px = sel["x"].values.astype(float)
    py = sel["y"].values.astype(float)
    dx = px - cx
    dy = py - cy
    theta = np.arctan2(dy, dx)
    radial = np.sqrt(dx * dx + dy * dy)

    # Sauvola binary at each patch centroid. Callers iterating many structures
    # over the same slide should compute this once and pass via
    # `precomputed_sauvola_binary` -- threshold_sauvola on a 30kx30k image takes
    # ~2 min and dominates cohort augment cost otherwise.
    if precomputed_sauvola_binary is not None:
        col_binary = precomputed_sauvola_binary
    else:
        try:
            from skimage.filters import threshold_sauvola

            s_thresh = threshold_sauvola(ecm_channel_img, window_size=51, k=0.2)
            col_binary = (ecm_channel_img > s_thresh).astype(np.uint8)
        except (ImportError, ValueError):
            # scikit-image absent or degenerate intensity; fall back to a mean threshold.
            col_binary = (ecm_channel_img > float(ecm_channel_img.mean())).astype(np.uint8)

    H, W = ecm_channel_img.shape
    pr = np.clip(py.astype(int), 0, H - 1)
    pc = np.clip(px.astype(int), 0, W - 1)
    sauvola_pos = col_binary[pr, pc].astype(np.float32)

    # Feature matrix from the ecm_patches DataFrame
    feat_cols = [c for c in patches_df.columns if c.startswith("feat_")]
    X = sel[feat_cols].to_numpy(dtype=np.float32)
    var_names = [c[len("feat_") :] if c.startswith("feat_") else c for c in feat_cols]

    n = len(sel)
    obs = pd.DataFrame(
        {
            "x": px.astype(np.float32),
            "y": py.astype(np.float32),
            "sin_theta": np.sin(theta).astype(np.float32),
            "cos_theta": np.cos(theta).astype(np.float32),
            "radial_dist": radial.astype(np.float32),
            "sauvola_pos": sauvola_pos,
            "ecm_cluster": sel["ecm_cluster"].values.astype(int)
            if "ecm_cluster" in sel.columns
            else np.zeros(n, dtype=int),
        }
    )
    var = pd.DataFrame(index=var_names) if var_names else pd.DataFrame(index=[])
    out = _AnnData(X=X, obs=obs, var=var)

    # Build a footprint mask for regionprops-based graph features
    if structure_mask is None:
        footprint = _hull_mask_from_points(coords, H, W)
    else:
        footprint = (structure_mask > 0).astype(np.uint8)

    graph_feat = _compute_islet_graph_feat_11d(
        footprint,
        patch_rows=py.astype(np.float32),
        patch_cols=px.astype(np.float32),
        is_peri_flags=np.ones(n, dtype=np.int8),
        sauvola_vals=sauvola_pos,
    )

    out.uns["islet_graph_feat"] = graph_feat
    out.uns["structure_uid"] = structure_uid
    out.uns["structure_centroid"] = (cx, cy)
    out.uns["islet_id"] = structure_uid
    out.uns["centroid_r"] = cy
    out.uns["centroid_c"] = cx
    return out


def _hull_mask_from_points(points_xy: np.ndarray, H: int, W: int) -> np.ndarray:
    """Binary convex-hull mask from (N, 2) (x, y) points."""
    from skimage.draw import polygon as _polygon

    mask = np.zeros((H, W), dtype=np.uint8)
    if len(points_xy) < 3:
        # Tiny structures: dilate the points themselves
        xs = np.clip(points_xy[:, 0].astype(int), 0, W - 1)
        ys = np.clip(points_xy[:, 1].astype(int), 0, H - 1)
        mask[ys, xs] = 1
        try:
            from skimage.morphology import dilation, disk

            mask = dilation(mask, disk(3)).astype(np.uint8)
        except ImportError:
            pass
        return mask
    try:
        from scipy.spatial import ConvexHull, QhullError

        hull = ConvexHull(points_xy)
        xs = points_xy[hull.vertices, 0]
        ys = points_xy[hull.vertices, 1]
        rr, cc = _polygon(ys, xs, shape=(H, W))
        mask[rr, cc] = 1
    except (ImportError, QhullError):
        # Degenerate (collinear) points have no convex hull; rasterise the raw points.
        xs = np.clip(points_xy[:, 0].astype(int), 0, W - 1)
        ys = np.clip(points_xy[:, 1].astype(int), 0, H - 1)
        mask[ys, xs] = 1
    return mask


def segment_cells(
    img: np.ndarray | ImageContainer,
    *,
    nuclear_channel: int | str = 0,
    membrane_channel: int | str | None = None,
    backend: Literal["cellpose"] = "cellpose",
    image_mpp: float = 1.0,
    compartment: Literal["nuclear", "whole-cell", "both"] = "nuclear",
    gpu: bool = True,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    store_in: ImageContainer | None = None,
    layer_name: str = "cell_mask",
) -> np.ndarray:
    """Segment cells from a spatial omics image using deep learning.

    Uses Cellpose with nuclear and optional membrane channels.
    Returns an integer label mask where 0 = background and 1..N = individual cells.

    Parameters
    ----------
    img
        ``(C, H, W)`` numpy array, ``(H, W)`` single-channel array, or
        :class:`~mantpy.im.ImageContainer`.
    nuclear_channel
        Channel index (int) or name (str, requires ImageContainer with
        ``channel_names``) for the nuclear signal (e.g. Ir191_DNA for IMC,
        DAPI for CODEX).
    membrane_channel
        Optional membrane/cytoplasm channel for whole-cell segmentation.
        If ``None``, only the nuclear signal is used.
    backend
        Segmentation backend. Currently only ``'cellpose'`` is supported.
    image_mpp
        Microns per pixel. IMC = 1.0, CODEX typically ~0.377.
    compartment
        Segmentation target: ``'nuclear'``, ``'whole-cell'``, or ``'both'``.
    gpu
        Use GPU if available (Cellpose only).
    cellprob_threshold
        Cell probability threshold (Cellpose only). Lower = more cells.
    flow_threshold
        Flow error threshold (Cellpose only). Higher = more permissive.
    store_in
        If provided, stores the mask as a named layer in this
        :class:`~mantpy.im.ImageContainer`.
    layer_name
        Layer name when storing in an ImageContainer.

    Returns
    -------
    np.ndarray
        Integer label mask, shape ``(H, W)``, dtype int32.

    Examples
    --------
    >>> import mantpy as mt
    >>> ic = mt.im.ImageContainer("roi.tif", channel_names=["Ir191_DNA", ...])
    >>> mask = mt.pp.segment_cells(ic, nuclear_channel="Ir191_DNA")
    >>> mask.shape  # (H, W)
    """
    from mantpy.im import ImageContainer as _IC

    # Resolve input to array
    if isinstance(img, _IC):
        arr = img.to_array(dtype=np.float32)
        channel_names = img.channel_names
    else:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        if arr.ndim != 3:
            raise ValueError(f"img must be (C, H, W) or (H, W), got shape {arr.shape}.")
        channel_names = None

    # Resolve nuclear channel
    if isinstance(nuclear_channel, str):
        if channel_names is None:
            raise ValueError("String channel names require an ImageContainer with channel_names set.")
        try:
            nuclear_channel = channel_names.index(nuclear_channel)
        except ValueError:
            raise ValueError(f"Nuclear channel '{nuclear_channel}' not found. Available: {channel_names}") from None
    nuclear_img = arr[nuclear_channel]

    # Resolve membrane channel
    membrane_img = None
    if membrane_channel is not None:
        if isinstance(membrane_channel, str):
            if channel_names is None:
                raise ValueError("String channel names require an ImageContainer with channel_names set.")
            try:
                membrane_channel = channel_names.index(membrane_channel)
            except ValueError:
                raise ValueError(
                    f"Membrane channel '{membrane_channel}' not found. Available: {channel_names}"
                ) from None
        membrane_img = arr[membrane_channel]

    # Dispatch to backend
    if backend == "cellpose":
        from mantpy._core._segmentation import run_cellpose

        mask = run_cellpose(
            nuclear_img,
            membrane_img,
            image_mpp=image_mpp,
            compartment=compartment,
            gpu=gpu,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Supported: 'cellpose'.")

    # Store in ImageContainer if requested
    if store_in is not None:
        store_in.add_layer(layer_name, mask.astype(np.float32))

    return mask


def _segment_cells_tiled_legacy(
    img: str | Path | np.ndarray,
    *,
    nuclear_channel: int = 0,
    diameter_um: float = 8.0,
    image_mpp: float = 1.0,
    tile_size: int = 2048,
    overlap: int = 128,
    batch_size: int | None = None,
    gpu: bool = True,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    cache: str | Path | None = None,
) -> np.ndarray:
    """Tile-based Cellpose segmentation for gigapixel multiplex images.

    Streams the nuclear channel from disk (or accepts an in-memory array),
    runs Cellpose on ``tile_size``×``tile_size`` tiles with ``overlap`` border,
    and stitches per-tile labels into globally-unique IDs.

    Why this exists: ``mt.pp.segment_cells`` loads the full image into memory,
    which is fine for IMC ROIs (~1K × 1K) but not for CODEX/Vectra slides
    (~30K × 30K). ``squidpy.im.segment(method=callable, chunks=N, lazy=True)``
    handles chunked dispatch but does *not* stitch global labels — so even
    with squidpy you'd need this last step. We do both here in one call.

    Parameters
    ----------
    img
        Path to a multi-page TIFF (one page per channel), an in-memory
        ``(C, H, W)`` array, or a 2-D nuclear image already extracted.
    nuclear_channel
        Page / channel index for DAPI / Hoechst / Ir191. Ignored when ``img``
        is already 2-D.
    diameter_um
        Approximate cell diameter in microns (Cellpose's ``diameter`` is the
        nuclear-pixel diameter, computed as ``diameter_um / image_mpp``).
    image_mpp
        Microns per pixel of the input image. CODEX ≈ 0.377, IMC = 1.0.
    tile_size, overlap
        Tiling geometry. Tiles overlap by ``overlap`` pixels; only the inner
        ``tile_size - overlap`` portion of each tile contributes to the final
        mask, which avoids labels being cut at tile boundaries.
    batch_size
        Tiles per Cellpose batch. ``None`` (default) auto-picks based on free
        VRAM (assumes ~16 MB raw × ~4× Cellpose overhead per 2048² float32
        tile, with 50% safety). Falls back to 4 when CUDA is unavailable.
    gpu
        Whether to enable GPU mode in Cellpose.
    cellprob_threshold, flow_threshold
        Forwarded to ``CellposeModel.eval``.
    cache
        Optional ``.npz`` path. If the file exists, it is loaded (mask is
        returned without re-segmenting — useful when iterating on graph
        parameters downstream). After successful segmentation, the result
        is saved there.

    Returns
    -------
    np.ndarray
        Integer label image of shape ``(H, W)``, dtype int32. ``0`` =
        background; ``1..N`` = unique cells.

    Examples
    --------
    >>> # Streaming from disk + cache:
    >>> mask = mt.pp.segment_cells_tiled(  # doctest: +SKIP
    ...     "slide.ome.tif",
    ...     nuclear_channel=0,
    ...     image_mpp=0.377,
    ...     cache="slide_mask.npz",
    ... )
    """
    from pathlib import Path as _Path

    import tifffile

    cache_path = _Path(cache) if cache is not None else None

    # ------- 1) Get the 2-D nuclear plane into RAM ----------------------
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            dapi = img.astype(np.float32, copy=False)
        elif img.ndim == 3:
            dapi = img[nuclear_channel].astype(np.float32, copy=False)
        else:
            raise ValueError(f"img array must be 2-D or 3-D; got {img.shape}.")
    else:
        # Path: read only the nuclear page off disk.
        #
        # This used to try sq.im.ImageContainer(lazy=True) first, "preferring
        # squidpy for the lazy/disk-aware open". It was neither: `.values`
        # materialises every channel into RAM only to index one out of it, so on
        # a 39-channel stack it cost ~39x the memory of the page read below for
        # an identical result. tifffile addresses the single page directly.
        with tifffile.TiffFile(str(img)) as tif:
            if len(tif.pages) > 1:
                dapi = tif.pages[nuclear_channel].asarray().astype(np.float32)
            else:
                a = tif.asarray()
                dapi = (a[nuclear_channel] if a.ndim == 3 else a).astype(np.float32)

    H, W = dapi.shape

    # ------- 2) Cache hit? skip cellpose -----------------------------------
    if cache_path is not None and cache_path.exists():
        m = np.load(str(cache_path))["masks"]
        if m.shape == (H, W):
            return m.astype(np.int32, copy=False)
        # else: shape mismatch — re-segment

    # ------- 3) Cellpose ---------------------------------------------------
    from cellpose.models import CellposeModel

    model = CellposeModel(gpu=gpu)

    if batch_size is None:
        try:
            import torch

            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                batch_size = max(1, int(free * 0.5 / ((tile_size**2) * 4 * 4)))
            else:
                batch_size = 4
        except (ImportError, RuntimeError):
            # No torch, or a CUDA runtime hiccup querying free memory.
            batch_size = 4

    diameter_px = diameter_um / image_mpp

    full = np.zeros((H, W), dtype=np.int32)
    max_label = 0
    step = tile_size - overlap
    specs = [(y, x, min(H, y + tile_size), min(W, x + tile_size)) for y in range(0, H, step) for x in range(0, W, step)]

    for s0 in range(0, len(specs), batch_size):
        chunk_specs = specs[s0 : s0 + batch_size]
        # Skip near-empty tiles (saves Cellpose calls on background)
        kept_meta = []
        kept_tiles = []
        for y0, x0, y1, x1 in chunk_specs:
            tile = dapi[y0:y1, x0:x1]
            if tile.size == 0 or tile.mean() < tile.max() * 0.01:
                kept_meta.append((y0, x0, y1, x1, None))
                continue
            kept_meta.append((y0, x0, y1, x1, len(kept_tiles)))
            kept_tiles.append(tile)

        if kept_tiles:
            masks, _, _ = model.eval(
                kept_tiles,
                diameter=diameter_px,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
            )
        else:
            masks = []

        for y0, x0, _y1, _x1, mi in kept_meta:
            if mi is None:
                continue
            tm = np.asarray(masks[mi], dtype=np.int32)
            iy0 = overlap // 2 if y0 > 0 else 0
            ix0 = overlap // 2 if x0 > 0 else 0
            iy1 = tm.shape[0] - (overlap // 2 if y0 + tile_size < H else 0)
            ix1 = tm.shape[1] - (overlap // 2 if x0 + tile_size < W else 0)
            inner = tm[iy0:iy1, ix0:ix1]
            inner_relabeled = np.where(inner > 0, inner + max_label, 0)
            full[y0 + iy0 : y0 + iy0 + inner.shape[0], x0 + ix0 : x0 + ix0 + inner.shape[1]] = inner_relabeled
            if inner_relabeled.max() > 0:
                max_label = int(full.max())

    # ------- 4) Cache save -------------------------------------------------
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(cache_path), masks=full)

    return full


def segment_cells_tiled(
    img: str | Path | np.ndarray | ImageContainer,
    *,
    nuclear_channel: int = 0,
    stain: Literal["channel", "hematoxylin"] = "channel",
    stain_reference: HEPreprocessingResult | Mapping[str, object] | None = None,
    diameter_um: float = 8.0,
    image_mpp: float = 1.0,
    downsample: int = 1,
    tile_size: int = 2048,
    overlap: int = 128,
    batch_size: int | None = None,
    gpu: bool = True,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    cache: str | Path | None = None,
    output: Literal["mask", "anndata"] = "mask",
) -> np.ndarray | AnnData:
    """Tile-based Cellpose segmentation with an H&E/observation-native path.

    The historical channel/mask call remains unchanged.  With
    ``stain='hematoxylin'`` RGB tiles are colour-deconvolved against one
    slide-global :func:`preprocess_he` reference. ``output='anndata'`` emits
    cell centroids and areas directly while tiles stream through the model;
    it never allocates a whole-slide label mask.

    Parameters
    ----------
    img
        Channel-first image stack, 2-D nuclear image, RGB
        :class:`~mantpy.im.ImageContainer`, or raster path.
    stain
        ``"channel"`` preserves the legacy nuclear-channel path;
        ``"hematoxylin"`` derives the nuclear signal from RGB.
    stain_reference
        Slide-global H/E reference returned by :func:`preprocess_he`. If
        omitted for H&E, a downsampled reference is fitted automatically.
    downsample
        Integer spatial downsampling applied before segmentation. AnnData
        coordinates and areas are converted back to full-resolution pixels.
    output
        ``"mask"`` returns the legacy full-resolution label image.
        ``"anndata"`` returns zero-variable AnnData with
        ``obsm['spatial']`` (full-resolution pixels) and
        ``obsm['spatial_um']``.
    """
    from pathlib import Path as _Path

    from mantpy.im import ImageContainer as _IC

    if stain not in {"channel", "hematoxylin"}:
        raise ValueError("stain must be 'channel' or 'hematoxylin'.")
    if output not in {"mask", "anndata"}:
        raise ValueError("output must be 'mask' or 'anndata'.")
    if not isinstance(downsample, int) or downsample < 1:
        raise ValueError("downsample must be a positive integer.")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size.")
    if image_mpp <= 0 or diameter_um <= 0:
        raise ValueError("image_mpp and diameter_um must be positive.")

    # Keep the exact established path (including .npz cache semantics) for
    # legacy calls; all new behavior is opt-in through one of the new kwargs.
    if (
        stain == "channel"
        and output == "mask"
        and downsample == 1
        and not isinstance(img, _IC)
    ):
        return _segment_cells_tiled_legacy(
            img,
            nuclear_channel=nuclear_channel,
            diameter_um=diameter_um,
            image_mpp=image_mpp,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            gpu=gpu,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            cache=cache,
        )

    cache_path = _Path(cache) if cache is not None else None
    if output == "anndata" and cache_path is not None:
        if cache_path.suffix.lower() != ".h5ad":
            raise ValueError("output='anndata' requires an .h5ad cache path.")
        if cache_path.exists():
            import anndata as ad

            return ad.read_h5ad(cache_path)

    container: _IC | None = None
    if stain == "hematoxylin":
        source, container = _he_channel_first_source(img)
        full_height, full_width = int(source.shape[1]), int(source.shape[2])
        if stain_reference is None:
            stain_reference = preprocess_he(container if container is not None else img)
        hematoxylin_range, eosin_range = _he_ranges(stain_reference)
        segmented_source = source[:, ::downsample, ::downsample]

        def read_tile(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
            rgb_cf = _materialize_image_array(segmented_source[:, y0:y1, x0:x1])
            rgb = np.moveaxis(rgb_cf, 0, -1)
            hematoxylin, _ = _he_deconvolve_tile(rgb, hematoxylin_range, eosin_range)
            return hematoxylin

        seg_height, seg_width = int(segmented_source.shape[1]), int(segmented_source.shape[2])
    else:
        if isinstance(img, _IC):
            container = img
            stack = img.get_layer(img._default_layer, compute=False)
            if stack.ndim != 3:
                raise ValueError(f"ImageContainer layer must be 3-D, got {stack.shape}.")
            plane = stack[nuclear_channel]
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                plane = img
            elif img.ndim == 3:
                plane = img[nuclear_channel]
            else:
                raise ValueError(f"img array must be 2-D or 3-D; got {img.shape}.")
        else:
            import tifffile

            with tifffile.TiffFile(str(img)) as tif:
                if len(tif.pages) > 1:
                    plane = tif.pages[nuclear_channel].asarray()
                else:
                    array = tif.asarray()
                    plane = array[nuclear_channel] if array.ndim == 3 else array
        full_height, full_width = int(plane.shape[0]), int(plane.shape[1])
        segmented_source = plane[::downsample, ::downsample]
        seg_height, seg_width = int(segmented_source.shape[0]), int(segmented_source.shape[1])

        def read_tile(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
            return _materialize_image_array(segmented_source[y0:y1, x0:x1]).astype(np.float32, copy=False)

    if output == "mask" and cache_path is not None and cache_path.exists():
        with np.load(cache_path) as cached:
            mask = cached["masks"]
        if mask.shape == (full_height, full_width):
            return mask.astype(np.int32, copy=False)

    from cellpose.models import CellposeModel

    model = CellposeModel(gpu=gpu)
    raw_device = getattr(model, "device", None)
    if raw_device is None:
        raw_device = getattr(getattr(model, "net", None), "device", None)
    model_gpu = getattr(model, "gpu", None)
    if raw_device is not None:
        resolved_device = str(raw_device).lower()
        device_type = str(getattr(raw_device, "type", resolved_device.split(":", 1)[0])).lower()
    elif model_gpu is not None:
        resolved_device = "gpu" if bool(model_gpu) else "cpu"
        device_type = resolved_device
    else:
        # A test double (or older backend) may expose neither attribute. CPU
        # is known when explicitly requested; a requested accelerator cannot
        # be claimed as used without evidence from the instantiated model.
        resolved_device = "unknown" if gpu else "cpu"
        device_type = resolved_device
    gpu_used = bool(model_gpu) if model_gpu is not None else device_type in {"cuda", "gpu", "mps"}
    if batch_size is None:
        try:
            import torch

            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()
                batch_size = max(1, int(free * 0.5 / ((tile_size**2) * 4 * 4)))
            else:
                batch_size = 4
        except (ImportError, RuntimeError):
            batch_size = 4
    batch_size = max(1, int(batch_size))

    effective_mpp = image_mpp * downsample
    diameter_px = diameter_um / effective_mpp
    step = tile_size - overlap
    specs = [
        (y, x, min(seg_height, y + tile_size), min(seg_width, x + tile_size))
        for y in range(0, seg_height, step)
        for x in range(0, seg_width, step)
    ]
    full = np.zeros((seg_height, seg_width), dtype=np.int32) if output == "mask" else None
    max_label = 0
    centroids: list[np.ndarray] = []
    areas: list[np.ndarray] = []

    for start in range(0, len(specs), batch_size):
        chunk_specs = specs[start : start + batch_size]
        kept_meta: list[tuple[int, int, int, int, int | None]] = []
        kept_tiles: list[np.ndarray] = []
        for y0, x0, y1, x1 in chunk_specs:
            tile = read_tile(y0, x0, y1, x1)
            tile_max = float(tile.max()) if tile.size else 0.0
            if tile.size == 0 or tile_max <= 0 or float(tile.mean()) < tile_max * 0.01:
                kept_meta.append((y0, x0, y1, x1, None))
                continue
            kept_meta.append((y0, x0, y1, x1, len(kept_tiles)))
            kept_tiles.append(tile)

        if kept_tiles:
            masks, _, _ = model.eval(
                kept_tiles,
                diameter=diameter_px,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
            )
        else:
            masks = []

        for y0, x0, _y1, _x1, mask_index in kept_meta:
            if mask_index is None:
                continue
            tile_mask = np.asarray(masks[mask_index], dtype=np.int32)
            inner_y0 = overlap // 2 if y0 > 0 else 0
            inner_x0 = overlap // 2 if x0 > 0 else 0
            inner_y1 = tile_mask.shape[0] - (overlap // 2 if y0 + tile_size < seg_height else 0)
            inner_x1 = tile_mask.shape[1] - (overlap // 2 if x0 + tile_size < seg_width else 0)
            if output == "mask":
                inner = tile_mask[inner_y0:inner_y1, inner_x0:inner_x1]
                relabeled = np.where(inner > 0, inner + max_label, 0).astype(np.int32, copy=False)
                assert full is not None
                full[
                    y0 + inner_y0 : y0 + inner_y0 + inner.shape[0],
                    x0 + inner_x0 : x0 + inner_x0 + inner.shape[1],
                ] = relabeled
                if relabeled.size:
                    max_label = max(max_label, int(relabeled.max()))
            else:
                from skimage.measure import regionprops_table

                props = regionprops_table(tile_mask, properties=("area", "centroid"))
                cy = np.asarray(props["centroid-0"], dtype=float)
                cx = np.asarray(props["centroid-1"], dtype=float)
                area = np.asarray(props["area"], dtype=float)
                owned = (
                    (cy >= inner_y0)
                    & (cy < inner_y1)
                    & (cx >= inner_x0)
                    & (cx < inner_x1)
                )
                if owned.any():
                    xy = np.column_stack((x0 + cx[owned], y0 + cy[owned])) * float(downsample)
                    centroids.append(xy.astype(np.float32))
                    areas.append((area[owned] * downsample**2).astype(np.float32))

    if output == "mask":
        assert full is not None
        if downsample > 1:
            full = np.repeat(np.repeat(full, downsample, axis=0), downsample, axis=1)
            full = full[:full_height, :full_width]
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, masks=full)
        return full

    spatial = np.concatenate(centroids, axis=0) if centroids else np.empty((0, 2), dtype=np.float32)
    area_px2 = np.concatenate(areas) if areas else np.empty((0,), dtype=np.float32)
    obs = pd.DataFrame(
        {
            "x": spatial[:, 0],
            "y": spatial[:, 1],
            "area_px2": area_px2,
        },
        index=pd.Index([f"cell_{i:08d}" for i in range(len(spatial))], dtype=str),
    )
    result = AnnData(X=sp.csr_matrix((len(obs), 0), dtype=np.float32), obs=obs)
    result.obsm["spatial"] = spatial
    result.obsm["spatial_um"] = spatial * np.float32(image_mpp)
    result.uns["segmentation"] = {
        "backend": "cellpose",
        "stain": stain,
        "gpu_requested": bool(gpu),
        "device": resolved_device,
        "gpu_used": gpu_used,
        "diameter_um": float(diameter_um),
        "image_mpp": float(image_mpp),
        "downsample": downsample,
        "tile_size": tile_size,
        "overlap": overlap,
        "cellprob_threshold": float(cellprob_threshold),
        "flow_threshold": float(flow_threshold),
        "fullres_shape": [full_height, full_width],
    }
    if stain == "hematoxylin":
        result.uns["segmentation"]["hematoxylin_range"] = list(hematoxylin_range)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_h5ad(cache_path)
    return result


def _compute_islet_graph_feat_11d(
    islet_single: np.ndarray,
    patch_rows: np.ndarray,
    patch_cols: np.ndarray,
    is_peri_flags: np.ndarray,
    sauvola_vals: np.ndarray,
) -> np.ndarray:
    """Compute the 11-D graph-level feature vector for one islet.

    Feature order:
    [log_area_norm, max_gap, peri_sauvola_cov, peri_frac, circularity, solidity,
     n_components_norm, largest_cc_frac, degree_entropy, assortativity_sauvola,
     angular_gap_entropy]
    """
    from collections import Counter

    from skimage.measure import regionprops

    props = regionprops(islet_single.astype(np.int32))
    if props:
        rp = props[0]
        cy, cx = rp.centroid
        islet_area = rp.area
        circ = float(4 * np.pi * rp.area / (rp.perimeter**2 + 1e-6))
        solid = float(rp.solidity)
    else:
        cy = cx = 0.0
        islet_area = 1
        circ = solid = 0.0

    log_area_norm = float(np.log1p(islet_area) / np.log1p(1e6))

    peri_mask_bool = is_peri_flags > 0
    n_peri = int(peri_mask_bool.sum())
    n_total = len(patch_rows)
    peri_frac = n_peri / max(n_total, 1)

    peri_sauvola_cov = float(sauvola_vals[peri_mask_bool].mean()) if n_peri > 0 else 0.0

    # Angular features on peri nodes
    if n_peri >= 2:
        peri_r = patch_rows[peri_mask_bool].astype(float)
        peri_c = patch_cols[peri_mask_bool].astype(float)
        peri_theta = np.arctan2(peri_r - cy, peri_c - cx)
        sorted_theta = np.sort(peri_theta)
        gaps = np.diff(sorted_theta)
        wrap_gap = 2 * np.pi - sorted_theta[-1] + sorted_theta[0]
        max_gap = float(max(gaps.max(), wrap_gap)) / (2 * np.pi)

        sector_bins = np.linspace(-np.pi, np.pi, 19)
        sector_counts, _ = np.histogram(peri_theta, bins=sector_bins)
        n_empty = int((sector_counts == 0).sum())
        n_filled = 18 - n_empty
        if n_empty > 0 and n_filled > 0:
            p_e = n_empty / 18
            p_f = n_filled / 18
            angular_gap_entropy = float(-(p_e * np.log(p_e) + p_f * np.log(p_f)) / np.log(2))
        else:
            angular_gap_entropy = 0.0
    else:
        max_gap = 1.0
        angular_gap_entropy = 0.0

    # Connectivity features: simple chain on peri ring (approximate via KD-tree)
    if n_peri >= 2:
        from scipy.spatial import cKDTree

        peri_coords = np.stack(
            [
                patch_rows[peri_mask_bool].astype(float),
                patch_cols[peri_mask_bool].astype(float),
            ],
            axis=1,
        )
        tree = cKDTree(peri_coords)
        k_nn = min(4, n_peri)
        _, indices = tree.query(peri_coords, k=k_nn)

        edge_set: set[tuple[int, int]] = set()
        for i, nbrs in enumerate(indices):
            for j in nbrs[1:]:
                edge_set.add((min(i, j), max(i, j)))

        # Connected components via union-find
        parent = list(range(n_peri))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in edge_set:
            pi, pj = _find(i), _find(j)
            if pi != pj:
                parent[pi] = pj

        roots = [_find(i) for i in range(n_peri)]
        cc_sizes = Counter(roots)
        n_components = len(cc_sizes)
        n_components_norm = float(n_components) / n_peri
        largest_cc_frac = float(max(cc_sizes.values())) / n_peri

        # Degree entropy
        deg_vals = np.array([sum(1 for e in edge_set if i in e) for i in range(n_peri)], dtype=float)
        deg_sum = deg_vals.sum()
        if deg_sum > 0:
            p = deg_vals / deg_sum
            p = p[p > 0]
            degree_entropy = float(-np.sum(p * np.log(p)) / np.log(max(len(p), 2)))
        else:
            degree_entropy = 0.0

        # Sauvola assortativity
        src_s, dst_s = [], []
        peri_sauvola = sauvola_vals[peri_mask_bool]
        for i, j in edge_set:
            src_s.append(float(peri_sauvola[i]))
            dst_s.append(float(peri_sauvola[j]))
        if len(src_s) >= 4 and np.std(src_s) > 0 and np.std(dst_s) > 0:
            assortativity_sauvola = float(np.corrcoef(src_s, dst_s)[0, 1])
        else:
            assortativity_sauvola = 0.0
    else:
        n_components_norm = 1.0
        largest_cc_frac = 1.0 if n_peri == 1 else 0.0
        degree_entropy = 0.0
        assortativity_sauvola = 0.0

    return np.array(
        [
            log_area_norm,
            max_gap,
            peri_sauvola_cov,
            peri_frac,
            circ,
            solid,
            n_components_norm,
            largest_cc_frac,
            degree_entropy,
            assortativity_sauvola,
            angular_gap_entropy,
        ],
        dtype=np.float32,
    )
