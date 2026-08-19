"""ECM patch extraction, feature computation, and K-means clustering.

Internal module — not part of the public API.

Feature extractors operate on **batched** patches with shape
``(N, C, P, P)`` and return ``(N, C)``. Custom extractors must follow the
same contract; see :data:`FEATURE_REGISTRY` for built-in examples.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Feature extractors — all take (N, C, P, P) and return (N, C) float32
# ---------------------------------------------------------------------------


def feat_mean(patches: np.ndarray) -> np.ndarray:
    """Per-channel mean across the spatial dimensions → (N, C)."""
    return patches.mean(axis=(2, 3)).astype(np.float32)


def feat_std(patches: np.ndarray) -> np.ndarray:
    """Per-channel std across the spatial dimensions → (N, C)."""
    return patches.std(axis=(2, 3)).astype(np.float32)


def feat_max(patches: np.ndarray) -> np.ndarray:
    """Per-channel max across the spatial dimensions → (N, C)."""
    return patches.max(axis=(2, 3)).astype(np.float32)


def feat_signal_fraction(patches: np.ndarray) -> np.ndarray:
    """Per-channel fraction of pixels above the per-patch channel mean → (N, C)."""
    mu = patches.mean(axis=(2, 3), keepdims=True)
    return (patches > mu).mean(axis=(2, 3)).astype(np.float32)


def feat_entropy(patches: np.ndarray) -> np.ndarray:
    """Per-channel Shannon entropy from a 32-bin per-patch histogram → (N, C)."""
    N, C, P, _ = patches.shape
    flat = patches.reshape(N, C, P * P).astype(np.float32)
    mn = flat.min(axis=2, keepdims=True)
    mx = flat.max(axis=2, keepdims=True)
    rng = np.where(mx > mn, mx - mn, np.float32(1.0))
    # Bin index in [0, 31]; constant patches collapse to bin 0.
    q = np.clip(((flat - mn) / rng * 32).astype(np.int64), 0, 31)
    # Offset each (n, c) row into its own 32-slot bucket so one bincount suffices.
    offsets = (np.arange(N * C, dtype=np.int64).reshape(N, C) * 32)[:, :, None]
    counts = np.bincount((q + offsets).ravel(), minlength=N * C * 32).reshape(N, C, 32).astype(np.float32)
    p = counts / (counts.sum(axis=2, keepdims=True) + _EPS)
    return (-(p * np.log(p + _EPS)).sum(axis=2)).astype(np.float32)


def feat_gradient_magnitude(patches: np.ndarray) -> np.ndarray:
    """Per-channel mean gradient magnitude → (N, C)."""
    p = patches.astype(np.float32, copy=False)
    gy = np.gradient(p, axis=2)
    gx = np.gradient(p, axis=3)
    return np.sqrt(gx * gx + gy * gy).mean(axis=(2, 3)).astype(np.float32)


def feat_coherence(patches: np.ndarray) -> np.ndarray:
    """Per-channel fibre coherence via the structure tensor → (N, C).

    coherence = (λ1 − λ2) / (λ1 + λ2 + ε)  where λ are eigenvalues of the
    2×2 structure tensor J = [[Ixx, Ixy],[Ixy, Iyy]].  Values in [0, 1]:
    1 = perfectly oriented, 0 = isotropic.
    """
    p = patches.astype(np.float32, copy=False)
    gy = np.gradient(p, axis=2)
    gx = np.gradient(p, axis=3)
    Jxx = (gx * gx).mean(axis=(2, 3))
    Jxy = (gx * gy).mean(axis=(2, 3))
    Jyy = (gy * gy).mean(axis=(2, 3))
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    disc = np.sqrt(np.maximum(0.0, (trace / 2) ** 2 - det))
    lam1 = trace / 2 + disc
    lam2 = trace / 2 - disc
    return ((lam1 - lam2) / (lam1 + lam2 + _EPS)).astype(np.float32)


# ---------------------------------------------------------------------------
# GLCM (Haralick) texture features
# ---------------------------------------------------------------------------

_GLCM_LEVELS = 16
_GLCM_DISTANCES = (1,)
_GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def _patch_to_uint8(patch: np.ndarray) -> np.ndarray:
    """Rescale to [0, _GLCM_LEVELS) and cast to uint8 for graycomatrix."""
    p = patch.astype(np.float32)
    mn, mx = p.min(), p.max()
    if mx - mn < 1e-8:
        return np.zeros_like(p, dtype=np.uint8)
    q = ((p - mn) / (mx - mn)) * (_GLCM_LEVELS - 1)
    return np.clip(q, 0, _GLCM_LEVELS - 1).astype(np.uint8)


def _glcm_prop(patches: np.ndarray, prop: str) -> np.ndarray:
    """Mean Haralick ``prop`` over 4 angles at distance 1 → (N, C).

    ``prop`` is one of ``"contrast"``, ``"homogeneity"``, ``"energy"``,
    ``"correlation"``. Loops internally over (N, C) because
    ``skimage.feature.graycomatrix`` does not accept a batch dimension.
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError as exc:  # pragma: no cover — skimage is a hard dep elsewhere
        raise ImportError("GLCM features require scikit-image. Install with `pip install scikit-image`.") from exc

    N, C = patches.shape[:2]
    out = np.empty((N, C), dtype=np.float32)
    distances = list(_GLCM_DISTANCES)
    angles = list(_GLCM_ANGLES)
    for n in range(N):
        for c in range(C):
            q = _patch_to_uint8(patches[n, c])
            glcm = graycomatrix(
                q,
                distances=distances,
                angles=angles,
                levels=_GLCM_LEVELS,
                symmetric=True,
                normed=True,
            )
            out[n, c] = float(graycoprops(glcm, prop).mean())
    return out


def feat_glcm_contrast(patches: np.ndarray) -> np.ndarray:
    """Mean GLCM contrast over 4 angles → (N, C)."""
    return _glcm_prop(patches, "contrast")


def feat_glcm_homogeneity(patches: np.ndarray) -> np.ndarray:
    """Mean GLCM homogeneity (inverse difference moment) over 4 angles → (N, C)."""
    return _glcm_prop(patches, "homogeneity")


def feat_glcm_energy(patches: np.ndarray) -> np.ndarray:
    """Mean GLCM energy (angular second moment) over 4 angles → (N, C)."""
    return _glcm_prop(patches, "energy")


def feat_glcm_correlation(patches: np.ndarray) -> np.ndarray:
    """Mean GLCM correlation over 4 angles → (N, C)."""
    return _glcm_prop(patches, "correlation")


# ---------------------------------------------------------------------------
# Skeleton-morphology features (fibre structure on the thresholded patch)
# ---------------------------------------------------------------------------


def _skeletonise(patch_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (skeleton, foreground mask). Uses the channel mean as threshold."""
    from skimage.morphology import skeletonize

    ch = patch_channel.astype(np.float32)
    mu = float(ch.mean())
    mask = ch > mu
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool), mask
    return skeletonize(mask).astype(bool), mask


def feat_skeleton_density(patches: np.ndarray) -> np.ndarray:
    """Fraction of foreground pixels on the morphological skeleton → (N, C).

    Proxy for total fibre length per unit area: dense reticular networks
    (e.g. lymph-node stroma) have higher skeleton density than sparse linear
    bundles (e.g. intestinal basement membrane).
    """
    N, C = patches.shape[:2]
    out = np.empty((N, C), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            sk, _mask = _skeletonise(patches[n, c])
            out[n, c] = float(sk.sum()) / float(sk.size)
    return out


def _count_branch_points(skel: np.ndarray) -> int:
    """Pixels on the skeleton with >= 3 skeleton-neighbours (8-connectivity)."""
    if not skel.any():
        return 0
    from scipy.ndimage import convolve

    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbours = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    # neighbours counts self; branch point = skeleton pixel with >=4 (incl. self).
    return int(((skel) & (neighbours >= 4)).sum())


def feat_branchpoint_density(patches: np.ndarray) -> np.ndarray:
    """Skeleton branch-point count normalised by patch area → (N, C)."""
    N, C = patches.shape[:2]
    out = np.empty((N, C), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            sk, _ = _skeletonise(patches[n, c])
            out[n, c] = float(_count_branch_points(sk)) / float(sk.size)
    return out


def feat_mean_branch_length(patches: np.ndarray) -> np.ndarray:
    """Mean length (in pixels) of skeleton segments between branch points → (N, C).

    Computed as ``total_skeleton_pixels / (branch_points + endpoint_segments + 1)``;
    approximates a typical fibre-segment length scale per patch.
    """
    N, C = patches.shape[:2]
    out = np.empty((N, C), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            sk, _ = _skeletonise(patches[n, c])
            n_px = int(sk.sum())
            if n_px == 0:
                out[n, c] = 0.0
            else:
                n_bp = _count_branch_points(sk)
                out[n, c] = float(n_px) / float(n_bp + 1)
    return out


#: Named string aliases for built-in feature extractors.
FEATURE_REGISTRY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": feat_mean,
    "std": feat_std,
    "entropy": feat_entropy,
    "coherence": feat_coherence,
    "max": feat_max,
    "signal_fraction": feat_signal_fraction,
    "gradient_magnitude": feat_gradient_magnitude,
    # Haralick texture features
    "glcm_contrast": feat_glcm_contrast,
    "glcm_homogeneity": feat_glcm_homogeneity,
    "glcm_energy": feat_glcm_energy,
    "glcm_correlation": feat_glcm_correlation,
    # Skeleton-morphology (fibre structure)
    "skeleton_density": feat_skeleton_density,
    "branchpoint_density": feat_branchpoint_density,
    "mean_branch_length": feat_mean_branch_length,
}


def resolve_extractors(
    features: list[str | Callable],
) -> list[Callable[[np.ndarray], np.ndarray]]:
    """Resolve a mixed list of string aliases and callables to a callable list."""
    resolved = []
    for f in features:
        if callable(f):
            resolved.append(f)
        elif isinstance(f, str):
            if f not in FEATURE_REGISTRY:
                raise ValueError(
                    f"Unknown feature name '{f}'. "
                    f"Available: {list(FEATURE_REGISTRY)}. "
                    "Pass a callable for custom features."
                )
            resolved.append(FEATURE_REGISTRY[f])
        else:
            raise TypeError(f"features items must be str or callable, got {type(f)}.")
    return resolved


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def extract_patches(
    img: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide a (C, H, W) image into non-overlapping square patches.

    Returns
    -------
    patches : np.ndarray, shape (N, C, patch_size, patch_size)
    cx : np.ndarray, shape (N,)  — patch x-centroids (pixel coords)
    cy : np.ndarray, shape (N,)  — patch y-centroids (pixel coords)
    """
    C, H, W = img.shape
    n_y = H // patch_size
    n_x = W // patch_size

    patches, cx, cy = [], [], []
    for iy in range(n_y):
        for ix in range(n_x):
            y0, y1 = iy * patch_size, (iy + 1) * patch_size
            x0, x1 = ix * patch_size, (ix + 1) * patch_size
            patches.append(img[:, y0:y1, x0:x1])
            cx.append((x0 + x1) / 2.0)
            cy.append((y0 + y1) / 2.0)

    if not patches:
        return np.empty((0, C, patch_size, patch_size), dtype=np.float32), np.array([]), np.array([])

    return np.array(patches, dtype=np.float32), np.array(cx), np.array(cy)


def compute_features(
    patches: np.ndarray,
    extractors: list[Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
    """Apply each batched extractor to the patch stack and concatenate outputs.

    Parameters
    ----------
    patches
        Shape ``(N, C, P, P)``.
    extractors
        List of callables, each accepting ``(N, C, P, P)`` and returning
        a ``(N, F_i)`` array.

    Returns
    -------
    np.ndarray, shape ``(N, F)`` where F = sum of extractor output widths.
    """
    if len(patches) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate([fn(patches) for fn in extractors], axis=1).astype(np.float32)


def remove_background(
    features: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """Return boolean mask (True = keep) for patches above background threshold.

    Background is defined as patches whose mean feature value is below the
    ``quantile``-th percentile across all patches.
    """
    row_means = features.mean(axis=1)
    threshold = np.quantile(row_means, quantile)
    return row_means > threshold


def remove_background_adaptive(
    patches: np.ndarray,
    min_signal_fraction: float = 0.1,
    threshold_method: str = "li",
    threshold_value: float | None = None,
) -> np.ndarray:
    """Adaptive background removal using intensity thresholding.

    Applies Li (or Otsu / Triangle / Yen) thresholding to determine a data-
    adaptive intensity cutoff per image, then discards patches where fewer
    than ``min_signal_fraction`` of pixels exceed that threshold.

    Parameters
    ----------
    patches
        Shape (N, C, P, P) — raw patch intensities.
    min_signal_fraction
        Minimum fraction of above-threshold pixels for a patch to be
        considered foreground.
    threshold_method
        ``"li"`` / ``"otsu"`` / ``"triangle"`` / ``"yen"``.  Ignored when
        ``threshold_value`` is provided.
    threshold_value
        If not ``None``, skip per-image threshold computation and use this
        fixed value for every channel.  Used for cohort-wide / global
        thresholding (e.g. after histogram matching).

    Returns
    -------
    np.ndarray of bool, shape (N,) — True = foreground (keep).
    """
    from skimage import filters as ski_filters

    N, C = patches.shape[0], patches.shape[1]
    fg = np.zeros(N, dtype=bool)

    for c in range(C):
        if threshold_value is not None:
            thresh = float(threshold_value)
        else:
            channel_flat = patches[:, c, :, :].ravel()
            pos = channel_flat[channel_flat > 0]
            if len(pos) < 10:
                continue

            _TH_FNS = {
                "li": ski_filters.threshold_li,
                "otsu": ski_filters.threshold_otsu,
                "triangle": ski_filters.threshold_triangle,
                "yen": ski_filters.threshold_yen,
            }
            try:
                thresh = _TH_FNS.get(threshold_method, ski_filters.threshold_otsu)(pos)
            except ValueError:
                # Degenerate intensity (e.g. all-equal pixels) breaks the histogram threshold.
                thresh = float(np.median(pos))

        # Vectorised: (N, P, P) > scalar → mean over spatial dims → (N,)
        fracs = (patches[:, c, :, :] > thresh).reshape(N, -1).mean(axis=1)
        fg |= fracs > min_signal_fraction

    return fg


def cluster_patches(
    features: np.ndarray,
    K: int | Literal["auto"],
) -> tuple[np.ndarray, int]:
    """K-means cluster patch feature vectors.

    Parameters
    ----------
    features
        Shape (N, F).
    K
        Number of clusters, or ``"auto"`` to select via silhouette score (K=2..8).

    Returns
    -------
    labels : np.ndarray, shape (N,)  — integer cluster labels
    k_used : int                     — K that was actually used
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    n_samples = len(X)

    if K == "auto":
        K = _select_k_silhouette(X, k_range=range(2, min(9, n_samples)))

    K = min(K, n_samples)
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels.astype(np.int32), int(K)


def _select_k_silhouette(X: np.ndarray, k_range: range) -> int:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k, best_score = 2, -1.0
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


# ---------------------------------------------------------------------------
# DataFrame / image assembly
# ---------------------------------------------------------------------------


def build_ecm_patch_dataframe(
    cx: np.ndarray,
    cy: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Assemble the ``adata.uns['ecm_patches']`` DataFrame."""
    n_feat = features.shape[1]
    feat_cols = {f"feat_{i}": features[:, i] for i in range(n_feat)}
    df = pd.DataFrame({"x": cx, "y": cy, "ecm_cluster": labels, **feat_cols})
    df.index = df.index.astype(str)
    return df


def build_ecm_image(
    labels_full: np.ndarray,
    mask: np.ndarray,
    n_patches_y: int,
    n_patches_x: int,
    patch_size: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Build a (H, W) cluster-label image for visualisation.

    Background patches are labelled -1.
    """
    label_image = np.full((n_patches_y, n_patches_x), -1, dtype=np.int32)
    patch_idx = 0
    foreground_idx = 0
    for iy in range(n_patches_y):
        for ix in range(n_patches_x):
            if mask[patch_idx]:
                label_image[iy, ix] = labels_full[foreground_idx]
                foreground_idx += 1
            patch_idx += 1

    full = np.repeat(np.repeat(label_image, patch_size, axis=0), patch_size, axis=1)
    return full[:H, :W]


def build_ecm_image_from_coords(
    cx: np.ndarray,
    cy: np.ndarray,
    labels: np.ndarray,
    patch_size: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Build a (H, W) label image from patch centroid coordinates.

    Used when patches were collected tile-by-tile (tiled extraction path)
    and we only have the foreground centroids and their cluster labels —
    not the full mask array that ``build_ecm_image`` requires.
    """
    label_image = np.full((H, W), -1, dtype=np.int32)
    for x, y, lbl in zip(cx, cy, labels, strict=False):
        x0 = max(0, int(x) - patch_size // 2)
        y0 = max(0, int(y) - patch_size // 2)
        x1 = min(W, x0 + patch_size)
        y1 = min(H, y0 + patch_size)
        label_image[y0:y1, x0:x1] = int(lbl)
    return label_image
