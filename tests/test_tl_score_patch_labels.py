"""Tests for pixel-weighted scoring of patch labels against a pixel annotation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

import mantpy as mt

PATCH = 4


def _patches(height=24, width=32, drop=()):
    rng = np.random.default_rng(0)
    image = rng.integers(1, 200, size=(height, width)).astype(np.uint16)
    foreground = np.ones((height // PATCH, width // PATCH), dtype=bool)
    for y, x in drop:
        foreground[y, x] = False
    return mt.pp.image_ecm_patches(image, channel_names=["ColIV"], patch_size=PATCH, foreground_mask=foreground)


def _annotation(height=24, width=32):
    annotation = np.zeros((height, width), dtype=np.uint8)
    annotation[:, :10] = 1
    annotation[:, 10:22] = 2
    annotation[:, 22:] = 3
    annotation[:3] = 0  # an unannotated strip, so the first patch row is partly annotated
    return annotation


def _expanded(adata, labels):
    """Paint one label per patch into a pixel image; pixels outside the patches are -1."""
    image = np.full(tuple(adata.uns["image_ecm_patches"]["image_shape"]), -1, dtype=np.int64)
    for (y, x), label in zip(adata.obs[["grid_y", "grid_x"]].to_numpy(int), labels, strict=True):
        image[y * PATCH : (y + 1) * PATCH, x * PATCH : (x + 1) * PATCH] = label
    return image


def test_ari_matches_pixel_level_adjusted_rand_index():
    adata = _patches(drop=[(0, 0), (3, 5)])
    annotation = _annotation()
    labels = np.random.default_rng(1).integers(0, 3, adata.n_obs)

    score = mt.tl.score_patch_labels(adata, labels, annotation)

    predicted = _expanded(adata, labels)
    scored = (annotation > 0) & (predicted >= 0)
    assert score.ari == pytest.approx(adjusted_rand_score(annotation[scored], predicted[scored]), abs=1e-12)
    expected = pd.crosstab(annotation[scored], predicted[scored]).to_numpy()
    np.testing.assert_array_equal(score.contingency.to_numpy(), expected)
    assert score.coverage == pytest.approx(scored.sum() / (annotation > 0).sum())
    assert list(score.contingency.index) == [1, 2, 3]
    assert list(score.contingency.columns) == [0, 1, 2]


def test_pixel_label_image_is_scored_on_the_same_pixels():
    adata = _patches(drop=[(1, 1)])
    annotation = _annotation()
    labels = np.random.default_rng(2).integers(0, 3, adata.n_obs)
    pixels = np.clip(_expanded(adata, labels), 0, None).astype(np.uint8)

    by_patch = mt.tl.score_patch_labels(adata, labels, annotation)
    by_pixel = mt.tl.score_patch_labels(adata, pixels, annotation)

    assert by_pixel.ari == pytest.approx(by_patch.ari, abs=1e-12)
    np.testing.assert_array_equal(by_pixel.contingency.to_numpy(), by_patch.contingency.to_numpy())
    pd.testing.assert_series_equal(by_pixel.matched_labels, by_patch.matched_labels)
    pd.testing.assert_series_equal(by_pixel.reference, by_patch.reference)


def test_pixel_label_image_counts_every_pixel_inside_a_patch():
    adata = _patches()
    annotation = _annotation()
    pixels = np.random.default_rng(4).integers(0, 4, annotation.shape).astype(np.int32)

    score = mt.tl.score_patch_labels(adata, pixels, annotation)

    scored = annotation > 0
    assert score.ari == pytest.approx(adjusted_rand_score(annotation[scored], pixels[scored]), abs=1e-12)
    assert score.coverage == pytest.approx(1.0)
    assert score.matched_labels.shape == (adata.n_obs,)


def test_matching_names_patches_by_their_class_and_reports_recovery():
    adata = _patches()
    annotation = np.zeros((24, 32), dtype=np.uint8)
    annotation[:, :16] = 1
    annotation[:, 16:] = 2
    grid_x = adata.obs["grid_x"].to_numpy()
    adata.obs["domain"] = np.where(grid_x < 4, "b", "a")  # cluster "b" covers class 1 exactly

    score = mt.tl.score_patch_labels(adata, "domain", annotation, class_names=["Left", "Right"])

    assert score.ari == pytest.approx(1.0)
    assert score.matching.to_dict() == {"a": "Right", "b": "Left"}
    assert list(score.matched_labels.cat.categories) == ["Left", "Right"]
    assert (score.matched_labels.to_numpy() == np.where(grid_x < 4, "Left", "Right")).all()
    assert (score.reference.to_numpy() == score.matched_labels.to_numpy()).all()
    assert score.recovery.to_dict() == {"Left": 1.0, "Right": 1.0}
    assert score.matched_labels.index.equals(adata.obs_names)
    assert "ari=1.000" in repr(score)


def test_extra_clusters_and_unannotated_patches_are_left_unmatched():
    adata = _patches()
    annotation = np.zeros((24, 32), dtype=np.int16)
    annotation[:, :16] = 1
    annotation[:, 16:28] = 2
    annotation[:, 28:] = -1  # negative values are not annotated
    grid_x = adata.obs["grid_x"].to_numpy()
    labels = np.select([grid_x < 4, grid_x < 6], [0, 1], default=2)

    score = mt.tl.score_patch_labels(adata, labels, annotation, class_names={1: "A", 2: "B", 3: "C"})

    assert list(score.contingency.index) == ["A", "B", "C"]
    assert score.contingency.loc["C"].sum() == 0
    assert score.matching.to_dict() == {0: "A", 1: "B"}  # cluster 2 shares no pixel with the remaining class
    assert score.matched_labels.isna().to_numpy().tolist() == (grid_x >= 6).tolist()
    assert score.reference.isna().to_numpy().tolist() == (grid_x == 7).tolist()
    assert score.recovery["A"] == pytest.approx(1.0)
    assert score.recovery["B"] == pytest.approx(8 / 12)
    assert np.isnan(score.recovery["C"])


def test_series_labels_are_aligned_by_patch_name():
    adata = _patches()
    annotation = _annotation()
    labels = pd.Series(np.random.default_rng(3).integers(0, 3, adata.n_obs), index=adata.obs_names)

    shuffled = labels.sample(frac=1, random_state=0)

    assert mt.tl.score_patch_labels(adata, shuffled, annotation).ari == pytest.approx(
        mt.tl.score_patch_labels(adata, labels.to_numpy(), annotation).ari
    )


def test_invalid_inputs_are_rejected():
    adata = _patches()
    annotation = _annotation()
    labels = np.zeros(adata.n_obs, dtype=int)

    with pytest.raises(ValueError, match="shape"):
        mt.tl.score_patch_labels(adata, labels, annotation[:, :-4])
    with pytest.raises(ValueError, match="one label per patch"):
        mt.tl.score_patch_labels(adata, labels[:-1], annotation)
    with pytest.raises(ValueError, match="without a class name"):
        mt.tl.score_patch_labels(adata, labels, annotation, class_names=["only one"])
    with pytest.raises(ValueError, match="no annotated pixels"):
        mt.tl.score_patch_labels(adata, labels, np.zeros_like(annotation))
    with pytest.raises(ValueError, match="integer image"):
        mt.tl.score_patch_labels(adata, labels, annotation.astype(float))
    with pytest.raises(ValueError, match="missing values"):
        mt.tl.score_patch_labels(adata, np.full(adata.n_obs, np.nan), annotation)
    with pytest.raises(KeyError, match="not a column"):
        mt.tl.score_patch_labels(adata, "absent", annotation)


def test_recovery_is_zero_for_an_annotated_class_without_a_matched_cluster():
    adata = _patches()
    annotation = np.zeros((24, 32), dtype=np.int16)
    annotation[:, :16] = 1
    annotation[:, 16:] = 2
    labels = np.zeros(adata.n_obs, dtype=int)  # one cluster for two annotated classes

    score = mt.tl.score_patch_labels(adata, labels, annotation, class_names={1: "A", 2: "B"})

    assert len(score.matching) == 1
    matched = score.matching.iloc[0]
    other = "B" if matched == "A" else "A"
    assert score.recovery[matched] == pytest.approx(1.0)
    assert score.recovery[other] == 0.0
