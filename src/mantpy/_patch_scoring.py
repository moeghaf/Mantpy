"""Pixel-weighted scoring of image-patch labels against a pixel annotation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

_PATCH_METADATA_KEY = "image_ecm_patches"
_BLOCK_PIXELS = 1 << 21


@dataclass(frozen=True)
class PatchLabelScore:
    """Output of :func:`score_patch_labels`.

    Attributes
    ----------
    ari
        Adjusted Rand index between the labels and the annotation, counted over
        the annotated pixels inside the analysed patches.
    contingency
        Pixel counts, one row per annotated class and one column per cluster.
    matching
        The annotated class matched one-to-one to each cluster (index: cluster).
        Clusters without a match, or sharing no pixel with their match, are
        absent.
    matched_labels
        One value per patch: the class matched to the patch's cluster, as a
        categorical series over the annotated classes. Missing for patches in an
        unmatched cluster. Pass it to :func:`mantpy.pl.patch_map` to colour every
        partition by the same classes.
    recovery
        For each annotated class, the fraction of its scored pixels that fall in
        its matched cluster. Zero for a class that no cluster is matched to;
        missing for a class without scored pixels.
    reference
        One value per patch: the annotated class covering most of its annotated
        pixels. Missing for patches without annotated pixels.
    coverage
        Fraction of all annotated pixels that lie inside the analysed patches.
    """

    ari: float
    contingency: pd.DataFrame
    matching: pd.Series
    matched_labels: pd.Series
    recovery: pd.Series
    reference: pd.Series
    coverage: float

    def __repr__(self) -> str:
        n_classes, n_clusters = self.contingency.shape
        return (
            f"PatchLabelScore(ari={self.ari:.3f}, n_classes={n_classes}, n_clusters={n_clusters}, "
            f"n_patches={self.matched_labels.size}, coverage={self.coverage:.3f})"
        )


def _contingency_ari(table: np.ndarray) -> float:
    """Adjusted Rand index from a contingency table of counts."""
    table = np.asarray(table, dtype=np.int64)
    n = table.sum()
    if n < 2:
        return float("nan")

    def pairs(values: np.ndarray) -> float:
        return float((values.astype(np.float64) * (values - 1) / 2).sum())

    observed, rows, columns = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    expected = rows * columns / (n * (n - 1) / 2)
    denominator = (rows + columns) / 2 - expected
    return 1.0 if denominator == 0 else float((observed - expected) / denominator)


def _integer_image(image: Any, name: str) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == bool:
        image = image.astype(np.uint8)
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"{name} must be a 2-D integer image; got shape {image.shape} and dtype {image.dtype}.")
    return image


def _value_counts(image: np.ndarray) -> np.ndarray:
    """Count every non-negative value of an integer image, in row blocks to bound memory."""
    maximum = int(image.max()) if image.size else 0
    counts = np.zeros(max(maximum, 0) + 1, dtype=np.int64)
    rows = max(1, _BLOCK_PIXELS // max(1, image.shape[1]))
    for start in range(0, image.shape[0], rows):
        block = image[start : start + rows].ravel()
        block = block[block >= 0] if np.issubdtype(block.dtype, np.signedinteger) else block
        counts += np.bincount(block, minlength=counts.size)
    return counts


def _patch_blocks(
    images: Sequence[np.ndarray], top: np.ndarray, left: np.ndarray, size: int
) -> Iterator[tuple[slice, list[np.ndarray]]]:
    """Yield ``(rows, pixels)`` for consecutive patches; each ``pixels`` entry has shape ``(n, size * size)``."""
    offsets = np.arange(size)
    step = max(1, _BLOCK_PIXELS // (size * size))
    for start in range(0, top.size, step):
        part = slice(start, start + step)
        ys = (top[part, None] + offsets)[:, :, None]
        xs = (left[part, None] + offsets)[:, None, :]
        yield part, [image[ys, xs].reshape(ys.shape[0], -1) for image in images]


def _lookup(values: np.ndarray, size: int) -> np.ndarray:
    """Array mapping an image value to its position in ``values`` (-1 when absent)."""
    table = np.full(size, -1, dtype=np.int64)
    table[values] = np.arange(values.size)
    return table


def _classes(totals: np.ndarray, class_names: Sequence[Any] | Mapping[int, Any] | None) -> tuple[np.ndarray, pd.Index]:
    present = np.flatnonzero(totals)
    present = present[present > 0]
    if present.size == 0:
        raise ValueError("annotation has no annotated pixels (every value is 0 or negative).")
    if class_names is None:
        return present, pd.Index(present)
    if isinstance(class_names, Mapping):
        values = np.asarray(sorted(int(value) for value in class_names), dtype=np.int64)
        names = pd.Index([class_names[value] for value in values])
    else:
        names = pd.Index(list(class_names))
        values = np.arange(1, len(names) + 1, dtype=np.int64)
    if values.size == 0 or values.min() < 1:
        raise ValueError("class_names must name at least one positive annotation value.")
    unnamed = np.setdiff1d(present, values)
    if unnamed.size:
        raise ValueError(f"annotation contains values without a class name: {unnamed.tolist()}.")
    if not names.is_unique:
        raise ValueError("class_names must be unique.")
    return values, names


def _patch_labels(adata: AnnData, labels: Any) -> np.ndarray:
    if isinstance(labels, str):
        if labels not in adata.obs:
            raise KeyError(f"labels={labels!r} is not a column of adata.obs.")
        return adata.obs[labels].to_numpy()
    if isinstance(labels, pd.Series) and not labels.index.equals(adata.obs_names):
        if labels.index.is_unique and adata.obs_names.isin(labels.index).all():
            return labels.reindex(adata.obs_names).to_numpy()
    return np.asarray(labels)


def score_patch_labels(
    adata: AnnData,
    labels: str | np.ndarray | pd.Series,
    annotation: np.ndarray,
    *,
    class_names: Sequence[Any] | Mapping[int, Any] | None = None,
    patch_size: int | None = None,
) -> PatchLabelScore:
    """Score image-patch labels against a pixel annotation, weighting by pixels.

    Patch labels are compared with a pixel annotation of the same image. Every
    annotated pixel inside an analysed patch counts once, so a patch that
    straddles two annotated classes contributes to both, and partly annotated
    patches count in proportion to their annotated area. This gives the pixel
    contingency table, its adjusted Rand index, and a one-to-one matching of
    clusters to annotated classes for display.

    ``labels`` may also be a pixel label image (for example a pixel-intensity
    clustering). It is then scored on exactly the same pixels as patch labels,
    so pixel-level and patch-level methods can be compared directly.

    Parameters
    ----------
    adata
        Patch table from :func:`mantpy.pp.image_ecm_patches`, with ``grid_y``
        and ``grid_x`` in ``adata.obs``.
    labels
        One of: the name of a column of ``adata.obs``; one label per patch, in
        ``adata`` row order (a series indexed by patch name is aligned); or a
        2-D image of non-negative integer labels with the annotation's shape.
    annotation
        2-D integer image with the shape of the analysed image. Positive values
        are annotated classes; ``0`` and negative values are not annotated.
    class_names
        Names of the annotated classes. A sequence names the values
        ``1, 2, ...`` in order; a mapping names its keys. Named classes that are
        absent from the annotation are kept, with zero counts. By default the
        classes are the positive values present in the annotation.
    patch_size
        Patch edge length in pixels. Read from
        ``adata.uns["image_ecm_patches"]`` by default.

    Returns
    -------
    PatchLabelScore
        ``ari``, ``contingency``, ``matching``, ``matched_labels``,
        ``recovery``, ``reference`` and ``coverage``.

    Notes
    -----
    Clusters are matched to classes by maximising the matched pixel count
    (Hungarian assignment). For a pixel label image, ``matched_labels`` uses the
    most frequent label of each patch.

    Examples
    --------
    >>> patches = mt.pp.image_ecm_patches(image, channel_names=["ColIV"], patch_size=32)  # doctest: +SKIP
    >>> score = mt.tl.score_patch_labels(patches, clusters, annotation, class_names=layers)  # doctest: +SKIP
    >>> score.ari  # doctest: +SKIP
    >>> mt.pl.patch_map(patches, score.matched_labels, palette=layer_colours)  # doctest: +SKIP

    """
    from scipy.optimize import linear_sum_assignment

    annotation = _integer_image(annotation, "annotation")
    metadata = adata.uns.get(_PATCH_METADATA_KEY, {})
    size = patch_size if patch_size is not None else metadata.get("patch_size")
    if size is None:
        raise ValueError("patch_size is not stored in adata.uns['image_ecm_patches']; pass patch_size=.")
    size = int(size)
    if size < 1:
        raise ValueError(f"patch_size must be a positive integer; got {size}.")
    missing = [column for column in ("grid_y", "grid_x") if column not in adata.obs]
    if missing:
        raise KeyError(f"adata.obs is missing {missing}; expected a patch table from mt.pp.image_ecm_patches.")
    image_shape = metadata.get("image_shape")
    if image_shape is not None and tuple(int(v) for v in image_shape) != annotation.shape:
        raise ValueError(
            f"annotation has shape {annotation.shape} but the patches were cut from an image of shape "
            f"{tuple(int(v) for v in image_shape)}."
        )
    top = adata.obs["grid_y"].to_numpy(dtype=np.int64) * size
    left = adata.obs["grid_x"].to_numpy(dtype=np.int64) * size
    height, width = annotation.shape
    if adata.n_obs and (top.min() < 0 or left.min() < 0 or top.max() + size > height or left.max() + size > width):
        raise ValueError("Some patches lie outside the annotation image; check patch_size and the annotation shape.")

    totals = _value_counts(annotation)
    class_values, classes = _classes(totals, class_names)
    class_totals = np.zeros(class_values.size, dtype=np.int64)
    known = class_values < totals.size
    class_totals[known] = totals[class_values[known]]
    class_of = _lookup(class_values[known], totals.size)
    class_of[class_of >= 0] = np.flatnonzero(known)
    n_classes = class_values.size
    signed = np.issubdtype(annotation.dtype, np.signedinteger)

    values = _patch_labels(adata, labels)
    pixel_labels = values.ndim == 2
    if pixel_labels:
        values = _integer_image(values, "A pixel label image")
        if values.shape != annotation.shape:
            raise ValueError(f"The pixel label image has shape {values.shape}; expected {annotation.shape}.")
        if values.size and int(values.min()) < 0:
            raise ValueError("A pixel label image must hold non-negative integer labels.")
        label_counts = _value_counts(values)
        clusters = np.flatnonzero(label_counts)
        cluster_of = _lookup(clusters, label_counts.size)
    else:
        if values.shape != (adata.n_obs,):
            raise ValueError(
                f"labels must give one label per patch ({adata.n_obs}) or a pixel image of shape {annotation.shape}; "
                f"got shape {values.shape}."
            )
        if pd.isna(values).any():
            raise ValueError("labels contains missing values; every analysed patch needs a label.")
        codes, clusters = pd.factorize(values, sort=True)
    n_clusters = len(clusters)

    counts = np.zeros((adata.n_obs, n_classes), dtype=np.int64)
    votes = np.zeros((adata.n_obs, n_clusters), dtype=np.int64) if pixel_labels else None
    contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
    images = [annotation, values] if pixel_labels else [annotation]
    for part, (block, *label_block) in _patch_blocks(images, top, left, size):
        class_index = class_of[np.maximum(block, 0) if signed else block]
        annotated = class_index >= 0
        patch = np.broadcast_to(np.arange(block.shape[0])[:, None], block.shape)
        counts[part] = np.bincount(
            patch[annotated] * n_classes + class_index[annotated], minlength=block.shape[0] * n_classes
        ).reshape(block.shape[0], n_classes)
        if pixel_labels:
            predicted = cluster_of[label_block[0]]
            votes[part] = np.bincount(
                (patch * n_clusters + predicted).ravel(), minlength=block.shape[0] * n_clusters
            ).reshape(block.shape[0], n_clusters)
            contingency += np.bincount(
                class_index[annotated] * n_clusters + predicted[annotated], minlength=n_classes * n_clusters
            ).reshape(n_classes, n_clusters)

    if pixel_labels:
        used = votes.sum(0) > 0
        clusters, votes, contingency = clusters[used], votes[:, used], contingency[:, used]
        n_clusters = int(used.sum())
        codes = votes.argmax(1) if n_clusters else np.zeros(adata.n_obs, dtype=np.int64)
    else:
        np.add.at(contingency.T, codes, counts)

    cluster_index = pd.Index(clusters, name="cluster")
    class_index = classes.rename("class")
    matched_class, matched_cluster = linear_sum_assignment(-contingency)
    shared = contingency[matched_class, matched_cluster] > 0  # a match without shared pixels is arbitrary
    matched_class, matched_cluster = matched_class[shared], matched_cluster[shared]
    class_of_cluster = np.full(n_clusters, -1, dtype=np.int64)
    class_of_cluster[matched_cluster] = matched_class
    order = np.argsort(matched_cluster)
    matching = pd.Series(
        classes[matched_class[order]].to_numpy(), index=cluster_index[matched_cluster[order]], name="class"
    )
    scored = contingency.sum(1).astype(np.float64)
    recovery = np.where(scored > 0, 0.0, np.nan)
    recovery[matched_class] = contingency[matched_class, matched_cluster] / scored[matched_class]
    patch_class = class_of_cluster[codes] if n_clusters else np.full(adata.n_obs, -1, dtype=np.int64)
    majority = np.where(counts.sum(1) > 0, counts.argmax(1), -1)

    def per_patch(class_codes: np.ndarray, name: str) -> pd.Series:
        return pd.Series(pd.Categorical.from_codes(class_codes, categories=classes), index=adata.obs_names, name=name)

    total = int(class_totals.sum())
    return PatchLabelScore(
        ari=_contingency_ari(contingency),
        contingency=pd.DataFrame(contingency, index=class_index, columns=cluster_index),
        matching=matching,
        matched_labels=per_patch(patch_class, "matched_class"),
        recovery=pd.Series(recovery, index=class_index, name="recovery"),
        reference=per_patch(majority, "annotated_class"),
        coverage=float(counts.sum() / total) if total else float("nan"),
    )
