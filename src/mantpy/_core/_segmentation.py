"""Cell segmentation backends for spatial proteomics images.

Internal module — use :func:`mantpy.pp.segment_cells` instead.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def run_cellpose(
    nuclear: np.ndarray,
    membrane: np.ndarray | None = None,
    *,
    image_mpp: float = 1.0,
    compartment: Literal["nuclear", "whole-cell", "both"] = "nuclear",
    gpu: bool = True,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
) -> np.ndarray:
    """Run Cellpose segmentation.

    Parameters
    ----------
    nuclear
        (H, W) float32 nuclear channel image.
    membrane
        Optional (H, W) float32 membrane channel for whole-cell segmentation.
    image_mpp
        Microns per pixel. Used to compute ``diameter`` in pixels.
        Nuclear diameter assumed ~6-10 µm; whole-cell ~12-15 µm.
    compartment
        ``'nuclear'`` segments nuclei only.
        ``'whole-cell'`` or ``'both'`` segments full cells using membrane
        + nuclear channels.
    gpu
        Use GPU if available.
    cellprob_threshold
        Cell probability threshold (lower = more cells detected).
    flow_threshold
        Flow error threshold (higher = more permissive).

    Returns
    -------
    (H, W) int32 label mask (0 = background, 1..N = cells).
    """
    try:
        from cellpose.models import CellposeModel
    except ImportError as exc:
        raise ImportError(
            "segment_cells with backend='cellpose' requires Cellpose. Install with: pip install cellpose"
        ) from exc

    # Estimate cell diameter in pixels from microns-per-pixel
    if compartment == "nuclear":
        diameter_um = 8.0
    else:
        diameter_um = 15.0

    diameter_px = diameter_um / image_mpp

    model = CellposeModel(gpu=gpu)

    if compartment == "nuclear" or membrane is None:
        img = nuclear
    else:
        # Stack membrane (primary) + nuclear (secondary) for whole-cell
        img = np.stack([membrane, nuclear], axis=-1)  # (H, W, 2)

    masks, _, _ = model.eval(
        img,
        diameter=diameter_px,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
    )

    return masks.astype(np.int32)
