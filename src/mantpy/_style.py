"""Opt-in Matplotlib styling for Mantpy figures.

Importing Mantpy does not change ``matplotlib.rcParams``. Call
:func:`apply_publication_style` explicitly when consistent, export-friendly
figure defaults are desired.
"""

from __future__ import annotations

from typing import Any

# Kept at module scope so tests and downstream code can inspect the exact
# values without applying them.
_PUBLICATION_STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.4,
    "lines.linewidth": 0.4,
    "patch.linewidth": 0.4,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
}


def apply_publication_style(**overrides: Any) -> dict[str, Any]:
    """Apply Mantpy's export-friendly ``rcParams`` preset.

    The preset uses an Arial-first sans-serif stack, embeddable Type-42 fonts,
    SVG text kept as text, and thin axes, line, patch, and tick widths suited to
    compact multi-panel figures.

    Parameters
    ----------
    **overrides
        Additional ``rcParams`` keys to set after the preset.

    Returns
    -------
    dict
        The exact ``{key: value}`` map that was applied, including overrides.

    Examples
    --------
    Apply the preset at the top of a plotting notebook::

        import mantpy as mt

        mt.style.apply_publication_style()

    Override one value without re-declaring the preset::

        mt.style.apply_publication_style(**{"axes.linewidth": 0.6})
    """
    import matplotlib as mpl

    applied = {**_PUBLICATION_STYLE, **overrides}
    for key, value in applied.items():
        mpl.rcParams[key] = value
    return applied


__all__ = ["apply_publication_style"]
