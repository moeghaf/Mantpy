"""Regression checks for requesting the complete, unfiltered patch grid."""

from __future__ import annotations

import numpy as np
import pytest

import mantpy as mt
from mantpy._core._patching import remove_background


@pytest.mark.parametrize(
    "values",
    ([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 3.0], [1.0, 1.0, 1.0, 1.0]),
)
def test_zero_quantile_retains_minima_and_ties(values):
    features = np.asarray(values, dtype=np.float32)[:, None]

    keep = remove_background(features, quantile=0.0)

    np.testing.assert_array_equal(keep, np.ones(len(values), dtype=bool))


def test_zero_quantile_retains_the_complete_public_patch_grid():
    image = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
    adata = mt.read_ecm_image(image, marker_names=["ColIV"])

    mt.pp.extract_ecm_patches(
        adata,
        image,
        patch_size=2,
        ecm_K=None,
        features=["mean"],
        background_quantile=0,
    )

    patches = adata.uns["ecm_patches"]
    assert len(patches) == 16
    np.testing.assert_allclose(
        np.sort(patches["feat_0"].to_numpy()),
        [4.5, 6.5, 8.5, 10.5, 20.5, 22.5, 24.5, 26.5, 36.5, 38.5, 40.5, 42.5, 52.5, 54.5, 56.5, 58.5],
    )


def test_empty_patch_set_has_an_empty_boolean_mask():
    keep = remove_background(np.empty((0, 2), dtype=np.float32), quantile=0)

    assert keep.shape == (0,)
    assert keep.dtype == np.bool_


def test_positive_quantile_still_excludes_values_at_the_threshold():
    features = np.asarray([0.0, 1.0, 1.0, 3.0], dtype=np.float32)[:, None]

    np.testing.assert_array_equal(
        remove_background(features, quantile=0.5),
        [False, False, False, True],
    )
