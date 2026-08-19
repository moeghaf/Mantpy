"""Shared utilities for Mantpy modules."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from mantpy._constants import MANTPY_UNS_KEY


def log_params(adata: AnnData, module: str, params: dict) -> None:
    """Write params into ``adata.uns['mantpy'][module]``."""
    if MANTPY_UNS_KEY not in adata.uns:
        adata.uns[MANTPY_UNS_KEY] = {}
    if module not in adata.uns[MANTPY_UNS_KEY]:
        adata.uns[MANTPY_UNS_KEY][module] = {}
    adata.uns[MANTPY_UNS_KEY][module].update(params)


def bh_fdr_correction(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, vectorised.

    Implements the standard Benjamini-Hochberg false-discovery-rate
    procedure: sort p-values, multiply each by ``n / rank``, then enforce
    monotone non-increasing in the reverse-sorted order.  Result is
    clipped to ``[0, 1]``.

    Parameters
    ----------
    p_values
        1-D iterable of p-values to adjust.

    Returns
    -------
    np.ndarray
        Adjusted p-values in the input order, same length as ``p_values``.

    Examples
    --------
    >>> import numpy as np
    >>> bh_fdr_correction([0.001, 0.01, 0.04, 0.04, 0.5])
    array([0.005, 0.025, 0.05 , 0.05 , 0.5  ])
    """
    from scipy.stats import false_discovery_control

    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p.copy()
    # Delegate to SciPy's vectorised implementation of the standard procedure.
    return false_discovery_control(p, method="bh")


def contrast_text_color(bg_hex: str, threshold: float = 140.0) -> str:
    """Return ``"black"`` or ``"white"`` for legible text on ``bg_hex``.

    Uses the ITU-R BT.601 perceived-luminance formula
    ``Y = 0.299 R + 0.587 G + 0.114 B`` (8-bit channels).  ``threshold`` is
    the Y value at which the function switches from ``"white"`` (on darker
    backgrounds) to ``"black"`` (on lighter ones); the default 140 was
    tuned so mid-luminance Tol-bright colours like amber ``#DDAA33``
    receive black text and deep violet ``#7A4794`` receives white.

    Parameters
    ----------
    bg_hex
        Hex colour string, with or without leading ``#`` (e.g. ``"#7A4794"``
        or ``"7A4794"``).
    threshold
        Luminance cut-off in 0-255 range.

    Returns
    -------
    str
        Either ``"black"`` or ``"white"``.

    Examples
    --------
    >>> contrast_text_color("#0072B2")
    'white'
    >>> contrast_text_color("#DDAA33")
    'black'
    """
    h = bg_hex.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-digit hex colour (with or without '#'); got {bg_hex!r}.")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if y > threshold else "white"
