"""Synthetic ROI generator for documentation and quick experimentation.

Every other loader in :mod:`mantpy.datasets` downloads tens to hundreds of
megabytes from Zenodo, which makes them a poor fit for a first example. This
module builds an equivalent ROI in memory in well under a second and ships zero
bytes in the wheel.

The image is *structured*, not random noise: ECM channels carry a diagonal
gradient plus a fibre-like band, and cell-marker channels carry Gaussian blobs
centred on the cell coordinates. That matters for teaching — ECM patch
clustering on uniform noise produces arbitrary labels, so an example built on it
would demonstrate the call signatures while quietly misrepresenting what the
output means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mantpy.ds import Bunch

__all__ = ["toy_ecm_roi"]

#: Channel panel returned by :func:`toy_ecm_roi`. Two ECM markers, two cell
#: lineage markers, one nuclear counterstain — the minimum that exercises the
#: cell/ECM split the rest of the package is built around.
_PANEL = (
    ("ColIV", 1),
    ("FN", 1),
    ("CD20", 0),
    ("EpCAM", 0),
    ("DAPI", 0),
)


def _ecm_channel(size: int, rng: np.random.Generator, *, angle: float) -> np.ndarray:
    """A diagonal intensity gradient crossed by a soft fibre-like band."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    gradient = (np.cos(angle) * xx + np.sin(angle) * yy) / size
    band = np.exp(-(((yy - xx) / (0.18 * size)) ** 2))
    field = 0.55 * gradient + 0.45 * band
    field += rng.normal(0.0, 0.04, size=field.shape)
    return field


def _cell_channel(
    size: int, centres: np.ndarray, rng: np.random.Generator, *, radius: float
) -> np.ndarray:
    """Gaussian blobs centred on the given cell coordinates."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    field = np.zeros((size, size), dtype=np.float64)
    for cx, cy in centres:
        field += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * radius**2)))
    field += rng.normal(0.0, 0.03, size=field.shape)
    return field


def _to_uint16(field: np.ndarray, ceiling: int) -> np.ndarray:
    lo, hi = float(field.min()), float(field.max())
    scaled = (field - lo) / (hi - lo) if hi > lo else np.zeros_like(field)
    return (scaled * ceiling).astype(np.uint16)


def toy_ecm_roi(*, size: int = 96, n_cells: int = 40, seed: int = 0) -> Bunch:
    """Build a small synthetic ROI with no download.

    The return value matches the shape of the real loaders, so the same code
    works against :func:`mantpy.datasets.coliv_intestine` and friends by
    swapping the first line.

    Parameters
    ----------
    size
        Side length of the square ROI in pixels.
    n_cells
        Number of cells to place. They are drawn in two spatial groups so
        neighbourhood statistics have something to find.
    seed
        Seed for the underlying :class:`numpy.random.Generator`, so a given
        ``(size, n_cells, seed)`` always yields byte-identical output.

    Returns
    -------
    mantpy.ds.Bunch
        ``image`` — ``(5, size, size)`` uint16 stack matching ``panel`` row
        order; ``panel`` — DataFrame with ``name`` and ``ecm`` columns;
        ``cells`` — DataFrame with ``celltype``, ``centroid-0`` and
        ``centroid-1``.

    Examples
    --------
    >>> import mantpy as mt
    >>> roi = mt.datasets.toy_ecm_roi()
    >>> adata = mt.io.read_imc(
    ...     roi.image, panel=roi.panel, cells=roi.cells,
    ...     sample_id="toy", condition="ctrl",
    ... )
    >>> adata.n_obs
    40

    Notes
    -----
    The data are synthetic. Cluster labels, enrichment scores and topology
    statistics computed on this ROI describe the generator, not biology — use
    it to learn the API, then move to the real datasets in this module.
    """
    if size < 16:
        raise ValueError(f"size must be at least 16 pixels; got {size}.")
    if n_cells < 2:
        raise ValueError(f"n_cells must be at least 2; got {n_cells}.")

    rng = np.random.default_rng(seed)

    # Two loose spatial groups, so neighbourhood composition is not uniform.
    half = n_cells // 2
    margin = 0.15 * size
    group_a = rng.normal(loc=0.30 * size, scale=0.11 * size, size=(half, 2))
    group_b = rng.normal(loc=0.70 * size, scale=0.13 * size, size=(n_cells - half, 2))
    centres = np.clip(np.vstack([group_a, group_b]), margin, size - 1 - margin)

    lineage = np.array(["B"] * half + ["EpCAM+"] * (n_cells - half))

    channels = [
        _to_uint16(_ecm_channel(size, rng, angle=0.0), 900),  # ColIV
        _to_uint16(_ecm_channel(size, rng, angle=np.pi / 2), 700),  # FN
        _to_uint16(
            _cell_channel(size, centres[lineage == "B"], rng, radius=2.4), 800
        ),  # CD20
        _to_uint16(
            _cell_channel(size, centres[lineage == "EpCAM+"], rng, radius=2.4), 800
        ),  # EpCAM
        _to_uint16(_cell_channel(size, centres, rng, radius=1.8), 1000),  # DAPI
    ]

    panel = pd.DataFrame(
        {"name": [n for n, _ in _PANEL], "ecm": [e for _, e in _PANEL]}
    )
    cells = pd.DataFrame(
        {
            "celltype": lineage,
            # read_imc reads y from centroid-0 and x from centroid-1.
            "centroid-0": centres[:, 1],
            "centroid-1": centres[:, 0],
        }
    )

    return Bunch(
        image=np.stack(channels).astype(np.uint16),
        panel=panel,
        cells=cells,
    )
