"""Tests for ``mt.pl.plot_marker_otsu_composite``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mantpy as mt


def test_plot_marker_otsu_composite_draws_thresholded_rgb_image() -> None:
    image = np.zeros((2, 8, 8), dtype=np.float32)
    image[0, :6, :6] = 2.0
    image[1, 2:, 2:] = 4.0
    adata = ad.AnnData(
        X=np.zeros((1, 2), dtype=np.float32),
        var=pd.DataFrame(index=["marker_a", "marker_b"]),
    )
    adata.uns["image_container"] = mt.im.ImageContainer(
        image,
        channel_names=["marker_a", "marker_b"],
    )

    fig, ax = plt.subplots()
    result = mt.pl.plot_marker_otsu_composite(
        ax,
        adata,
        ["marker_a", "marker_b"],
        colors=["#ff0000", "#00ff00"],
    )

    assert result is ax
    assert len(ax.images) == 1
    assert np.asarray(ax.images[0].get_array()).shape == (8, 8, 3)
    assert len(ax.get_xticks()) == 0
    assert len(ax.get_yticks()) == 0
    plt.close(fig)
