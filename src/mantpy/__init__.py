"""
Mantpy — a framework for extracellular matrix analysis in spatial proteomics.

Graph-based analysis of the extracellular matrix and cell--ECM interactions in
multiplexed imaging and spatial transcriptomics.

Usage
-----
    import mantpy as mt

    ic = mt.im.ImageContainer("codex.tif", lazy=True)  # large-image support
    adata = mt.read_imc(ic, panel=panel, cells=cells)
    mt.pp.extract_ecm_patches(adata, ic, tiled=True, tile_size=2048)
    mt.gr.build_cell_graph(adata)
    mt.gr.build_ecm_graph(adata)
    mt.gr.build_cell_ecm_graph(adata)
    mt.tl.interaction_test(adata)
    mt.pl.cell_ecm_graph(adata)
"""

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType

import mantpy.datasets as datasets
import mantpy.ds as ds
import mantpy.fetch as fetch
import mantpy.gr as gr
import mantpy.im as im
import mantpy.io as io
import mantpy.palette as palette
import mantpy.pp as pp
import mantpy.style as style
import mantpy.tl as tl
from mantpy._version import __version__
from mantpy.ds import MantpyDataset
from mantpy.im import ImageContainer
from mantpy.io import read_codex, read_ecm_image, read_imc, read_imc_folder


def __getattr__(name: str) -> _ModuleType:
    """Load optional and plotting namespaces only when first requested.

    Plotting and optional neural dependencies can import
    ``matplotlib.pyplot``, which resolves Matplotlib's backend and changes
    ``rcParams["backend"]``. Keeping them lazy makes the base package import
    free of plotting-state changes while preserving ``mt.pl`` and ``mt.nn``.
    """
    if name in {"nn", "pl"}:
        module = _import_module(f"mantpy.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "io",
    "im",
    "pp",
    "gr",
    "tl",
    "pl",
    "ds",
    "datasets",
    "fetch",
    "nn",
    "palette",
    "style",
    "read_imc",
    "read_imc_folder",
    "read_codex",
    "read_ecm_image",
    "ImageContainer",
    "MantpyDataset",
]


def __dir__() -> list[str]:
    """List public lazy exports alongside the populated module namespace."""
    return sorted(set(globals()) | set(__all__))
