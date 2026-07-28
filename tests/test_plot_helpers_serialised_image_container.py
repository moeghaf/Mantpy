"""Plot helpers must accept the H5AD-safe image-container payload.

`adata.uns["image_container"]` holds a plain mapping, not a live
:class:`~mantpy.im.ImageContainer` — that is what makes a Mantpy object
writable to h5ad. Helpers that read the image must therefore route through
:func:`~mantpy.im.as_image_container` rather than calling ``.to_array()`` on
whatever is in ``uns``.

Three helpers called ``.to_array()`` directly and raised
``AttributeError: 'dict' object has no attribute 'to_array'`` for every caller.
It surfaced in the Schistosoma reproduction notebook, not in the suite, because
the existing tests construct their own containers rather than using the object
`read_imc` actually produces.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import mantpy as mt
from mantpy._constants import IMAGE_CONTAINER_KEY


@pytest.fixture
def roi_adata():
    roi = mt.datasets.toy_ecm_roi(size=64, n_cells=12)
    adata = mt.io.read_imc(
        roi.image, panel=roi.panel, cells=roi.cells, sample_id="toy", condition="ctrl"
    )
    mt.pp.extract_ecm_patches(adata, roi.image, patch_size=8, ecm_K=2, features=["mean"])
    return adata


def test_read_imc_stores_a_mapping_not_a_live_container(roi_adata):
    """The premise of this module: uns holds the serialised form."""
    payload = roi_adata.uns[IMAGE_CONTAINER_KEY]
    assert not hasattr(payload, "to_array")
    assert "layers" in payload


def test_marker_otsu_composite_accepts_the_serialised_payload(roi_adata):
    _, ax = plt.subplots()
    mt.pl.plot_marker_otsu_composite(ax, roi_adata, ["ColIV", "FN"], show_legend=False)
    plt.close("all")


def test_marker_otsu_composite_accepts_a_live_container(roi_adata):
    """as_image_container is idempotent, so both forms must work."""
    roi_adata.uns[IMAGE_CONTAINER_KEY] = mt.im.as_image_container(
        roi_adata.uns[IMAGE_CONTAINER_KEY]
    )
    _, ax = plt.subplots()
    mt.pl.plot_marker_otsu_composite(ax, roi_adata, ["ColIV", "FN"], show_legend=False)
    plt.close("all")


def test_no_helper_calls_to_array_on_a_raw_uns_lookup():
    """Guard the whole class of bug, not just the three sites that had it."""
    import pathlib
    import re

    source = (
        pathlib.Path(mt.__file__).parent / "_plot_helpers.py"
    ).read_text(encoding="utf-8")
    offenders = re.findall(r"uns\[[^]]+\]\.to_array\(\)", source)
    assert offenders == [], (
        "route image-container access through mantpy.im.as_image_container; "
        f"found raw lookups: {offenders}"
    )
