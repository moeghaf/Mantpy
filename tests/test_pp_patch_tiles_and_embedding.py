"""Patch tiles for the patch encoder and embedding-based ECM clustering."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import mantpy as mt

# ---------------------------------------------------------------------------
# mt.pp.ecm_patch_tiles
# ---------------------------------------------------------------------------


def _ecm_panel(n_channels: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [f"matrix_{index}" for index in range(n_channels)],
            "keep": [1] * n_channels,
            "ecm": [1] * n_channels,
        }
    )


def _mixed_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["matrix_a", "cell_marker", "matrix_b"],
            "keep": [1, 1, 1],
            "ecm": [1, 0, 1],
        }
    )


def _images(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "roi_a": rng.random((3, 20, 30), dtype=np.float32),
        "roi_b": rng.random((3, 20, 30), dtype=np.float32),
    }


def _tile_cohort(images: dict[str, np.ndarray], panel: pd.DataFrame | None = None, **kwargs) -> dict[str, ad.AnnData]:
    options = {"patch_size": 5, "features": ["mean"], "ecm_K": None, "background_quantile": 0.0}
    options.update(kwargs)
    return mt.pp.ecm_patches_from_images(images, _ecm_panel() if panel is None else panel, normalize="none", **options)


def _grid_tiles(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Row-major patch blocks of a full lattice, written as a plain reshape."""
    channels, height, width = image.shape
    rows, columns = height // patch_size, width // patch_size
    return (
        image.reshape(channels, rows, patch_size, columns, patch_size)
        .transpose(1, 3, 0, 2, 4)
        .reshape(-1, channels, patch_size, patch_size)
    )


def _feature_matrix(cohort: dict[str, ad.AnnData]) -> np.ndarray:
    return np.vstack(
        [adata.uns["ecm_patches"].filter(like="feat_").to_numpy(dtype=np.float32) for adata in cohort.values()]
    )


def test_ecm_patch_tiles_matches_grid_reshape() -> None:
    images = _images()
    cohort = _tile_cohort(images)
    tables_before = {sample: adata.uns["ecm_patches"].copy() for sample, adata in cohort.items()}

    tiles = mt.pp.ecm_patch_tiles(cohort, images)

    expected = np.concatenate([images[sample].reshape(3, 4, 5, 6, 5).transpose(1, 3, 0, 2, 4) for sample in cohort])
    expected = expected.reshape(-1, 3, 5, 5)
    assert tiles.dtype == np.float32
    assert tiles.shape == (48, 3, 5, 5)
    assert np.array_equal(tiles, expected)
    np.testing.assert_allclose(tiles.mean((2, 3)), _feature_matrix(cohort), rtol=1e-6)
    # Nothing is written.
    for sample, adata in cohort.items():
        pd.testing.assert_frame_equal(adata.uns["ecm_patches"], tables_before[sample])
        assert list(adata.uns["mantpy"]["pp"]) == ["extract_ecm_patches", "extract_ecm_patches_cohort"]


def test_ecm_patch_tiles_follows_adatas_order_not_images_order() -> None:
    images = _images()
    cohort = _tile_cohort(images)
    reversed_images = dict(reversed(list(images.items())))
    reversed_cohort = dict(reversed(list(cohort.items())))

    assert np.array_equal(mt.pp.ecm_patch_tiles(cohort, reversed_images), mt.pp.ecm_patch_tiles(cohort, images))
    assert np.array_equal(
        mt.pp.ecm_patch_tiles(reversed_cohort, images),
        np.concatenate([_grid_tiles(images["roi_b"], 5), _grid_tiles(images["roi_a"], 5)]),
    )


def test_ecm_patch_tiles_signal_rows_align_with_cluster_subset() -> None:
    images = _images(1)
    cohort = _tile_cohort(images)
    rng = np.random.default_rng(3)
    masks = {}
    for sample, adata in cohort.items():
        table = adata.uns["ecm_patches"].copy()
        table["ecm_cluster"] = rng.integers(-1, 3, size=len(table)).astype(np.int32)
        adata.uns["ecm_patches"] = table
        masks[sample] = table["ecm_cluster"].to_numpy() >= 0
    all_tiles = mt.pp.ecm_patch_tiles(cohort, images)

    signal_tiles = mt.pp.ecm_patch_tiles(cohort, images, subset="signal")

    pooled_mask = np.concatenate([masks[sample] for sample in cohort])
    assert 0 < pooled_mask.sum() < len(pooled_mask)
    assert np.array_equal(signal_tiles, all_tiles[pooled_mask])
    assert np.array_equal(
        signal_tiles,
        np.concatenate([_grid_tiles(images[sample], 5)[masks[sample]] for sample in cohort]),
    )

    clustering = mt.pp.cluster_ecm_patches(cohort, n_clusters=2, subset="signal")
    assert len(signal_tiles) == sum(clustering.component_sizes) == int(pooled_mask.sum())
    # One feature row per tile is exactly what the embedding path expects.
    features = signal_tiles.reshape(len(signal_tiles), -1)
    embedded = mt.pp.cluster_ecm_patches(cohort, n_clusters=2, subset="signal", embedding=features)
    assert sum(embedded.component_sizes) == len(signal_tiles)
    for sample, adata in cohort.items():
        assert np.array_equal(adata.uns["ecm_patches"]["ecm_cluster"].to_numpy() >= 0, masks[sample])


def test_ecm_patch_tiles_signal_reads_cluster_key() -> None:
    images = _images(2)
    cohort = _tile_cohort(images)
    for adata in cohort.values():
        table = adata.uns["ecm_patches"].copy()
        table["stage_two"] = np.where(np.arange(len(table)) % 3 == 0, -1, 1)
        adata.uns["ecm_patches"] = table

    # The extraction-stage column is all zero, so every row is signal there.
    assert len(mt.pp.ecm_patch_tiles(cohort, images, subset="signal")) == 48
    custom = mt.pp.ecm_patch_tiles(cohort, images, subset="signal", cluster_key="stage_two")
    keep = np.arange(24) % 3 != 0
    assert np.array_equal(custom, np.concatenate([_grid_tiles(images[sample], 5)[keep] for sample in cohort]))


def test_ecm_patch_tiles_empty_selection_keeps_trailing_shape() -> None:
    images = _images()
    cohort = _tile_cohort(images)
    for adata in cohort.values():
        table = adata.uns["ecm_patches"].copy()
        table["ecm_cluster"] = -1
        adata.uns["ecm_patches"] = table

    tiles = mt.pp.ecm_patch_tiles(cohort, images, subset="signal")

    assert tiles.shape == (0, 3, 5, 5)
    assert tiles.dtype == np.float32


def test_ecm_patch_tiles_background_filtered_table() -> None:
    images = _images(4)
    cohort = _tile_cohort(images, background_quantile=0.5)
    n_rows = sum(len(adata.uns["ecm_patches"]) for adata in cohort.values())
    assert 0 < n_rows < 48

    tiles = mt.pp.ecm_patch_tiles(cohort, images)

    assert tiles.shape == (n_rows, 3, 5, 5)
    np.testing.assert_allclose(tiles.mean((2, 3)), _feature_matrix(cohort), rtol=1e-6)
    # Each tile is the lattice block at the row's own centre.
    offset = 0
    for sample, adata in cohort.items():
        table = adata.uns["ecm_patches"]
        grid_index = ((table["y"] - 2.5) / 5 * 6 + (table["x"] - 2.5) / 5).to_numpy(dtype=int)
        assert np.array_equal(tiles[offset : offset + len(table)], _grid_tiles(images[sample], 5)[grid_index])
        offset += len(table)


def test_ecm_patch_tiles_incomplete_lattice_and_even_patch_size() -> None:
    rng = np.random.default_rng(5)
    images = {"roi_a": rng.random((3, 9, 14), dtype=np.float32)}
    cohort = _tile_cohort(images, patch_size=4)

    tiles = mt.pp.ecm_patch_tiles(cohort, images)

    # 9 x 14 pixels hold a 2 x 3 lattice of 4-pixel patches; the remainder is unused.
    assert np.array_equal(tiles, _grid_tiles(images["roi_a"][:, :8, :12], 4))


def test_ecm_patch_tiles_selects_ecm_channels() -> None:
    images = _images(6)
    cohort = _tile_cohort(images, _mixed_panel())
    assert [bool(flag) for flag in cohort["roi_a"].var["is_ecm"]] == [True, False, True]

    full_panel = mt.pp.ecm_patch_tiles(cohort, images)
    ecm_only = mt.pp.ecm_patch_tiles(cohort, {sample: image[[0, 2]] for sample, image in images.items()})

    assert full_panel.shape == (48, 2, 5, 5)
    assert np.array_equal(
        full_panel,
        np.concatenate([_grid_tiles(images[sample][[0, 2]], 5) for sample in cohort]),
    )
    assert np.array_equal(ecm_only, full_panel)
    np.testing.assert_allclose(full_panel.mean((2, 3)), _feature_matrix(cohort), rtol=1e-6)


def test_ecm_patch_tiles_accepts_image_containers_and_other_dtypes() -> None:
    images = _images(7)
    cohort = _tile_cohort(images, _mixed_panel())
    expected = mt.pp.ecm_patch_tiles(cohort, images)

    containers = {sample: mt.im.ImageContainer(image) for sample, image in images.items()}
    assert np.array_equal(mt.pp.ecm_patch_tiles(cohort, containers), expected)
    ecm_containers = {sample: mt.im.ImageContainer(image[[0, 2]]) for sample, image in images.items()}
    assert np.array_equal(mt.pp.ecm_patch_tiles(cohort, ecm_containers), expected)

    doubles = {sample: image.astype(np.float64) for sample, image in images.items()}
    from_doubles = mt.pp.ecm_patch_tiles(cohort, doubles)
    assert from_doubles.dtype == np.float32
    assert np.array_equal(from_doubles, expected)


def test_ecm_patch_tiles_two_dimensional_single_channel_image() -> None:
    rng = np.random.default_rng(8)
    image = rng.random((12, 8), dtype=np.float32)
    carrier = mt.read_imc(np.zeros((3, 12, 8), dtype=np.float32), _mixed_panel(), cells=None, normalize="none")
    mt.pp.extract_ecm_patches(
        carrier,
        image,
        ecm_channel="matrix_b",
        patch_size=4,
        features=["mean"],
        ecm_K=None,
        background_quantile=0.0,
    )

    tiles = mt.pp.ecm_patch_tiles({"roi": carrier}, {"roi": image})

    assert tiles.shape == (6, 1, 4, 4)
    assert np.array_equal(tiles, _grid_tiles(image[np.newaxis], 4))


def test_ecm_patch_tiles_cuts_tiled_extraction_from_row_coordinates() -> None:
    rng = np.random.default_rng(9)
    image = rng.random((3, 24, 36), dtype=np.float32)
    carrier = mt.read_imc(image, _ecm_panel(), cells=None, normalize="none")
    mt.pp.extract_ecm_patches(
        carrier,
        mt.im.ImageContainer(image),
        patch_size=4,
        features=["mean"],
        ecm_K=None,
        background_quantile=0.0,
        tiled=True,
        tile_size=10,
        tile_overlap=3,
    )
    table = carrier.uns["ecm_patches"]
    # Tiles start every 10 pixels, so their patches leave the image-wide 4-pixel lattice.
    assert (((table["x"] - 2) % 4) != 0).any()

    tiles = mt.pp.ecm_patch_tiles({"roi": carrier}, {"roi": image})

    assert tiles.shape == (len(table), 3, 4, 4)
    np.testing.assert_allclose(tiles.mean((2, 3)), table.filter(like="feat_").to_numpy(dtype=np.float32), rtol=1e-6)


def test_ecm_patch_tiles_patch_size_argument_and_log() -> None:
    images = _images()
    cohort = _tile_cohort(images)
    expected = mt.pp.ecm_patch_tiles(cohort, images)

    for adata in cohort.values():
        del adata.uns["mantpy"]["pp"]["extract_ecm_patches"]
    with pytest.raises(ValueError, match=r"Sample 'roi_a': patch_size is not logged; pass patch_size=\."):
        mt.pp.ecm_patch_tiles(cohort, images)
    assert np.array_equal(mt.pp.ecm_patch_tiles(cohort, images, patch_size=5), expected)

    # An explicit patch size takes precedence over the logged one.
    for adata in cohort.values():
        adata.uns["mantpy"]["pp"]["extract_ecm_patches"] = {"patch_size": 7}
    assert np.array_equal(mt.pp.ecm_patch_tiles(cohort, images, patch_size=5), expected)

    cohort["roi_b"].uns["mantpy"]["pp"]["extract_ecm_patches"] = {"patch_size": 5}
    with pytest.raises(ValueError, match=r"Samples were extracted with different patch sizes; pass patch_size=\."):
        mt.pp.ecm_patch_tiles(cohort, images)
    with pytest.raises(ValueError, match=r"patch_size must be at least 1\."):
        mt.pp.ecm_patch_tiles(cohort, images, patch_size=0)


def test_ecm_patch_tiles_errors() -> None:
    images = _images()
    cohort = _tile_cohort(images, _mixed_panel())

    with pytest.raises(ValueError, match=r"adatas is empty; expected at least one ECM sample\."):
        mt.pp.ecm_patch_tiles({}, {})

    # Mismatched keys.
    with pytest.raises(
        ValueError, match=r"Cohort keys differ; missing images=\['roi_b'\], missing AnnData=\['roi_c'\]\."
    ):
        mt.pp.ecm_patch_tiles(cohort, {"roi_a": images["roi_a"], "roi_c": images["roi_b"]})

    # A bad subset value.
    with pytest.raises(ValueError, match=r"subset must be 'all' or 'signal'\."):
        mt.pp.ecm_patch_tiles(cohort, images, subset="foreground")

    # Wrong channel count: neither the full panel (3) nor the ECM channels (2).
    wrong_channels = {sample: np.concatenate([image, image[:1]]) for sample, image in images.items()}
    with pytest.raises(
        ValueError,
        match=r"Sample 'roi_a': image has 4 channels; expected 3 \(full panel\) or 2 \(ECM only\)\.",
    ):
        mt.pp.ecm_patch_tiles(cohort, wrong_channels)
    with pytest.raises(
        ValueError, match=r"Every image must be channel-first \(C, H, W\); got shape \(1, 3, 20, 30\)\."
    ):
        mt.pp.ecm_patch_tiles(cohort, {sample: image[np.newaxis] for sample, image in images.items()})

    # A window out of bounds: the image is smaller than the extracted one ...
    cropped = {sample: image[:, :, :29] for sample, image in images.items()}
    with pytest.raises(
        ValueError, match=r"Sample 'roi_a': a patch window falls outside the image; check images and patch_size\."
    ):
        mt.pp.ecm_patch_tiles(cohort, cropped)
    # ... or the patch size is larger than the extracted one.
    with pytest.raises(ValueError, match=r"Sample 'roi_a': a patch window falls outside the image"):
        mt.pp.ecm_patch_tiles(cohort, images, patch_size=7)

    # A patch size of the other parity cannot have produced these centres.
    with pytest.raises(ValueError, match=r"Sample 'roi_a': patch centres are not on a 4-pixel lattice\."):
        mt.pp.ecm_patch_tiles(cohort, images, patch_size=4)

    # A missing cluster_key with subset='signal'.
    with pytest.raises(KeyError, match=r"Sample 'roi_a' has no patch labels in 'stage_two'\."):
        mt.pp.ecm_patch_tiles(cohort, images, subset="signal", cluster_key="stage_two")


def test_ecm_patch_tiles_errors_from_sample_state() -> None:
    images = _images()

    # A deleted patch_size log.
    cohort = _tile_cohort(images)
    del cohort["roi_b"].uns["mantpy"]
    with pytest.raises(ValueError, match=r"Sample 'roi_b': patch_size is not logged; pass patch_size=\."):
        mt.pp.ecm_patch_tiles(cohort, images)

    # A centre moved off the pixel lattice.
    cohort = _tile_cohort(images)
    table = cohort["roi_b"].uns["ecm_patches"].copy()
    table.loc[table.index[4], "x"] += 0.5
    cohort["roi_b"].uns["ecm_patches"] = table
    with pytest.raises(ValueError, match=r"Sample 'roi_b': patch centres are not on a 5-pixel lattice\."):
        mt.pp.ecm_patch_tiles(cohort, images)
    table.loc[table.index[4], "x"] = np.nan
    with pytest.raises(ValueError, match=r"Sample 'roi_b': patch centres are not on a 5-pixel lattice\."):
        mt.pp.ecm_patch_tiles(cohort, images)

    # A centre moved by whole pixels until its window leaves the image.
    cohort = _tile_cohort(images)
    table = cohort["roi_a"].uns["ecm_patches"].copy()
    table.loc[table.index[5], "x"] += 1.0
    cohort["roi_a"].uns["ecm_patches"] = table
    with pytest.raises(ValueError, match=r"Sample 'roi_a': a patch window falls outside the image"):
        mt.pp.ecm_patch_tiles(cohort, images)
    table.loc[table.index[5], "x"] = -2.5
    with pytest.raises(ValueError, match=r"Sample 'roi_a': a patch window falls outside the image"):
        mt.pp.ecm_patch_tiles(cohort, images)

    # A sample without a patch table, or without the ECM flag.
    cohort = _tile_cohort(images)
    del cohort["roi_b"].uns["ecm_patches"]
    with pytest.raises(KeyError, match=r"Sample 'roi_b' has no uns\['ecm_patches'\]\."):
        mt.pp.ecm_patch_tiles(cohort, images)
    cohort = _tile_cohort(images)
    del cohort["roi_a"].var["is_ecm"]
    with pytest.raises(
        ValueError, match=r"Column 'is_ecm' not found in adata\.var\. Run `mt\.read_imc\(\.\.\.\)` first\."
    ):
        mt.pp.ecm_patch_tiles(cohort, images)

    # Samples whose ECM channel counts differ cannot be stacked.
    cohort = _tile_cohort(images)
    cohort["roi_b"].var["is_ecm"] = [True, False, True]
    with pytest.raises(ValueError, match=r"Sample 'roi_b' has 2 ECM channels but earlier samples have 3\."):
        mt.pp.ecm_patch_tiles(cohort, images)


def test_ecm_patch_tiles_validates_before_copying_pixels() -> None:
    class _Unreadable(mt.im.ImageContainer):
        def to_array(self, *args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("pixels were read before the cohort was validated")

    images = _images()
    cohort = _tile_cohort(images)
    table = cohort["roi_b"].uns["ecm_patches"].copy()
    table.loc[table.index[0], "y"] += 0.25
    cohort["roi_b"].uns["ecm_patches"] = table

    with pytest.raises(ValueError, match="not on a 5-pixel lattice"):
        mt.pp.ecm_patch_tiles(cohort, {sample: _Unreadable(image) for sample, image in images.items()})


# ---------------------------------------------------------------------------
# mt.pp.cluster_ecm_patches: default keywords are unchanged
# ---------------------------------------------------------------------------


def _feature_cohort(*, mean_metadata: bool = True) -> dict[str, ad.AnnData]:
    """Three well-separated feature blobs of 24, 18 and 12 patches in two samples."""
    rng = np.random.default_rng(11)
    clouds = [
        rng.normal((-6.0, -6.0), 0.1, size=(24, 2)),
        rng.normal((0.0, 6.0), 0.1, size=(18, 2)),
        rng.normal((6.0, -2.0), 0.1, size=(12, 2)),
    ]
    pooled = np.vstack(clouds).astype(np.float32)
    pooled = pooled[np.random.default_rng(5).permutation(len(pooled))]
    cohort: dict[str, ad.AnnData] = {}
    for sample, rows in {"roi_a": pooled[:30], "roi_b": pooled[30:]}.items():
        carrier = ad.AnnData(X=np.empty((0, 0), dtype=np.float32))
        carrier.uns["ecm_patches"] = pd.DataFrame(
            {
                "x": np.arange(len(rows), dtype=float),
                "y": np.zeros(len(rows), dtype=float),
                "ecm_cluster": np.zeros(len(rows), dtype=np.int32),
                "feat_0": rows[:, 0],
                "feat_1": rows[:, 1],
            }
        )
        if mean_metadata:
            carrier.uns["ecm_feature_names"] = json.dumps(
                [{"extractor": "mean", "protein": "Fibronectin"}, {"extractor": "mean", "protein": "CollagenI"}]
            )
        cohort[sample] = carrier
    return cohort


def _labels(cohort: dict[str, ad.AnnData], key: str = "ecm_cluster") -> np.ndarray:
    return np.concatenate([adata.uns["ecm_patches"][key].to_numpy() for adata in cohort.values()])


def _set_labels(cohort: dict[str, ad.AnnData], labels: np.ndarray, key: str = "ecm_cluster") -> None:
    offset = 0
    for adata in cohort.values():
        table = adata.uns["ecm_patches"].copy()
        table[key] = np.asarray(labels[offset : offset + len(table)], dtype=np.int32)
        adata.uns["ecm_patches"] = table
        offset += len(table)


# Outputs of release 1.1.1 on ``_feature_cohort()``, recorded before the
# ``embedding`` and ``order_by`` keywords were added (scikit-learn 1.7.2,
# scanpy 1.11.5, leidenalg 0.10.2).
_RELEASED_KMEANS_ALL = {
    "labels": [
        *[1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        *[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    ],
    "component_sizes": (24, 30),
    "component_means": (-6.0124382972717285, 2.6125307083129883),
    "cluster_counts": ((0, 24), (1, 30)),
}
_RELEASED_KMEANS_SIGNAL = {
    "labels": [
        *[0, 0, -1, -1, 0, 1, 0, -1, -1, 0, 0, 0, -1, 1, 1, -1, 0, -1, 0, 1, 1, 0, 0, -1, -1, 0, 1, 1, -1, -1],
        *[0, -1, -1, 1, 0, 0, 0, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 0, 1, 0],
    ],
    "component_sizes": (18, 12),
    "component_means": (3.0191187858581543, 2.002647876739502),
    "cluster_counts": ((-1, 24), (0, 18), (1, 12)),
}
_RELEASED_LEIDEN_ALL = {
    "labels": [
        *[1, 1, 0, 0, 1, 2, 1, 0, 0, 1, 1, 1, 0, 2, 2, 0, 1, 0, 1, 2, 2, 1, 1, 0, 0, 1, 2, 2, 0, 0],
        *[1, 0, 0, 2, 1, 1, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 2, 1],
    ],
    "component_sizes": (24, 18, 12),
    "component_means": (-6.0124382972717285, 3.0191187858581543, 2.002647876739502),
    "cluster_counts": ((0, 24), (1, 18), (2, 12)),
}


def _assert_released_outputs(cohort, result, released, provenance) -> None:
    labels = _labels(cohort)
    assert labels.dtype == np.int32
    assert labels.tolist() == released["labels"]
    assert result.n_clusters == len(released["component_sizes"])
    assert result.component_sizes == released["component_sizes"]
    assert result.component_means == pytest.approx(released["component_means"], rel=1e-6)
    assert result.cluster_counts == released["cluster_counts"]
    assert result.feature_columns == ("feat_0", "feat_1")
    assert result.cluster_key == "ecm_cluster"
    assert result.provenance == provenance
    for adata in cohort.values():
        stored = adata.uns["ecm_clustering"]
        assert stored == provenance
        # Same keys in the same order: nothing is added at default keywords.
        assert list(stored) == list(provenance)
        assert list(stored["standardization"]) == list(provenance["standardization"])
        assert adata.uns["mantpy"]["pp"] == {"cluster_ecm_patches": provenance}
        assert list(adata.uns["mantpy"]["pp"]["cluster_ecm_patches"]) == list(provenance)


def _released_kmeans_provenance(*, subset: str, random_state: int, n_init: int, cluster_counts: dict) -> dict:
    return {
        "provenance_version": 1,
        "mantpy_version": mt.__version__,
        "method": "kmeans",
        "subset": subset,
        "cluster_key": "ecm_cluster",
        "feature_columns": ["feat_0", "feat_1"],
        "random_state": random_state,
        "n_clusters": 2,
        "cluster_counts": cluster_counts,
        "standardization": {"method": "StandardScaler", "fit_subset": "all", "input_dtype": "float32"},
        "versions": {"mantpy": mt.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "n_init": n_init,
        "requested_n_clusters": 2,
    }


def test_cluster_ecm_patches_default_unchanged() -> None:
    cohort = _feature_cohort()

    stage_one = mt.pp.cluster_ecm_patches(cohort, n_clusters=2)

    assert stage_one.method == "kmeans"
    assert stage_one.subset == "all"
    _assert_released_outputs(
        cohort,
        stage_one,
        _RELEASED_KMEANS_ALL,
        _released_kmeans_provenance(subset="all", random_state=0, n_init=10, cluster_counts={"0": 24, "1": 30}),
    )
    assert np.array_equal(stage_one.model.labels_, _RELEASED_KMEANS_ALL["labels"])
    assert stage_one.scaler.n_samples_seen_ == 54

    removal = mt.pp.remove_background_patches(cohort, stage_one)
    assert removal.background_component == 0
    assert removal.background_patches == 24
    assert removal.signal_counts == ((0, 30),)

    stage_two = mt.pp.cluster_ecm_patches(cohort, n_clusters=2, subset="signal", random_state=3, n_init=4)

    assert stage_two.subset == "signal"
    _assert_released_outputs(
        cohort,
        stage_two,
        _RELEASED_KMEANS_SIGNAL,
        _released_kmeans_provenance(
            subset="signal", random_state=3, n_init=4, cluster_counts={"-1": 24, "0": 18, "1": 12}
        ),
    )
    # The table path keeps fitting the scaler on every row, background included.
    assert stage_two.scaler.n_samples_seen_ == 54


def test_cluster_ecm_patches_default_unchanged_leiden() -> None:
    pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")
    from importlib.metadata import version

    cohort = _feature_cohort()

    result = mt.pp.cluster_ecm_patches(cohort, method="leiden", n_neighbors=5, resolution=0.1)

    assert result.method == "leiden"
    assert result.model is None
    provenance = {
        "provenance_version": 1,
        "mantpy_version": mt.__version__,
        "method": "leiden",
        "subset": "all",
        "cluster_key": "ecm_cluster",
        "feature_columns": ["feat_0", "feat_1"],
        "random_state": 0,
        "n_clusters": 3,
        "cluster_counts": {"0": 24, "1": 18, "2": 12},
        "standardization": {"method": "StandardScaler", "fit_subset": "all", "input_dtype": "float32"},
        "versions": {
            "mantpy": mt.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "scanpy": version("scanpy"),
            "igraph": version("igraph"),
            "leidenalg": version("leidenalg"),
        },
        "n_neighbors": 5,
        "resolution": 0.1,
        "flavor": "leidenalg",
        "label_order": "descending_size_then_original_label",
    }
    _assert_released_outputs(cohort, result, _RELEASED_LEIDEN_ALL, provenance)


@pytest.mark.parametrize("subset", ["all", "signal"])
@pytest.mark.parametrize("mean_metadata", [True, False])
def test_cluster_ecm_patches_default_matches_released_recipe(subset: str, mean_metadata: bool) -> None:
    """The released K-means path, written out with scikit-learn, in any library version."""
    cohort = _feature_cohort(mean_metadata=mean_metadata)
    existing = np.random.default_rng(2).integers(-1, 2, size=54)
    _set_labels(cohort, existing)
    pooled = _feature_matrix(cohort)
    active = np.ones(54, dtype=bool) if subset == "all" else existing >= 0

    result = mt.pp.cluster_ecm_patches(cohort, n_clusters=3, subset=subset, random_state=1, n_init=5)

    scaler = StandardScaler()
    standardized = scaler.fit_transform(pooled)
    assert standardized.dtype == np.float32
    model = KMeans(3, n_init=5, random_state=1).fit(standardized[active])
    expected = np.full(54, -1, dtype=np.int32)
    expected[active] = model.labels_
    assert np.array_equal(_labels(cohort), expected)
    assert np.array_equal(result.model.labels_, model.labels_)
    assert np.array_equal(result.model.cluster_centers_, model.cluster_centers_)
    assert np.array_equal(result.scaler.mean_, scaler.mean_)
    assert np.array_equal(result.scaler.scale_, scaler.scale_)
    assert result.component_sizes == tuple(np.bincount(model.labels_, minlength=3).tolist())
    if mean_metadata:
        means = tuple(float(pooled[active][model.labels_ == component].mean()) for component in range(3))
    else:
        means = tuple(float(value) for value in model.cluster_centers_.mean(axis=1))
    assert result.component_means == means
    for adata in cohort.values():
        assert "feature_source" not in adata.uns["ecm_clustering"]
        assert "label_order" not in adata.uns["ecm_clustering"]
        assert "label_mapping" not in adata.uns["ecm_clustering"]


# ---------------------------------------------------------------------------
# mt.pp.cluster_ecm_patches: embedding= and order_by=
# ---------------------------------------------------------------------------


def _embedding_cohort() -> tuple[dict[str, ad.AnnData], np.ndarray, np.ndarray]:
    """Three samples with background rows, and a three-blob embedding of the signal rows.

    The blobs hold 50, 30 and 10 patches. The patch-table features are noise,
    so only the embedding can recover them.
    """
    rng = np.random.default_rng(21)
    lengths = {"roi_a": 40, "roi_b": 45, "roi_c": 35}
    n_patches = sum(lengths.values())
    signal = np.ones(n_patches, dtype=bool)
    signal[rng.choice(n_patches, size=30, replace=False)] = False
    blob = rng.permutation(np.repeat([0, 1, 2], [10, 50, 30]))
    centres = np.asarray([[4.0, 0.0, -3.0, 1.0], [-4.0, 3.0, 2.0, 0.0], [0.0, -5.0, 4.0, -2.0]])
    embedding = (centres[blob] + rng.normal(0.0, 0.2, size=(90, 4))).astype(np.float32)
    features = rng.normal(size=(n_patches, 2)).astype(np.float32)
    cohort: dict[str, ad.AnnData] = {}
    offset = 0
    for sample, length in lengths.items():
        rows = slice(offset, offset + length)
        carrier = ad.AnnData(X=np.empty((0, 0), dtype=np.float32))
        carrier.uns["ecm_patches"] = pd.DataFrame(
            {
                "x": np.arange(length, dtype=float),
                "y": np.zeros(length, dtype=float),
                "ecm_cluster": np.where(signal[rows], 0, -1).astype(np.int32),
                "feat_0": features[rows, 0],
                "feat_1": features[rows, 1],
            }
        )
        carrier.uns["ecm_feature_names"] = json.dumps(
            [{"extractor": "mean", "protein": "Fibronectin"}, {"extractor": "std", "protein": "Fibronectin"}]
        )
        cohort[sample] = carrier
        offset += length
    return cohort, embedding, signal


def _notebook_recipe(embedding: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """The hand-written clustering that ``embedding=`` and ``order_by='size'`` replace."""
    standardised = StandardScaler().fit_transform(embedding).astype(np.float32)
    raw_labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(standardised)
    order = np.argsort(-np.bincount(raw_labels, minlength=n_clusters), kind="stable")
    rank = np.empty(n_clusters, dtype=int)
    rank[order] = np.arange(n_clusters)
    return rank[raw_labels], raw_labels


def test_cluster_ecm_patches_embedding_matches_sklearn_recipe() -> None:
    cohort, embedding, signal = _embedding_cohort()
    features_before = _feature_matrix(cohort)
    expected_signal, raw_labels = _notebook_recipe(embedding, 3)

    result = mt.pp.cluster_ecm_patches(
        cohort, n_clusters=3, subset="signal", embedding=embedding, random_state=0, n_init=10, order_by="size"
    )

    expected = np.full(len(signal), -1, dtype=np.int32)
    expected[signal] = expected_signal
    labels = _labels(cohort)
    assert labels.dtype == np.int32
    assert np.array_equal(labels, expected)
    assert result.component_sizes == tuple(np.bincount(expected_signal, minlength=3).tolist()) == (50, 30, 10)
    assert result.cluster_counts == ((-1, 30), (0, 50), (1, 30), (2, 10))
    assert result.n_clusters == 3
    assert result.method == "kmeans"
    assert result.subset == "signal"
    assert result.feature_columns == ("embedding_0", "embedding_1", "embedding_2", "embedding_3")
    # The model keeps its own numbering; the scaler saw the selected rows only.
    assert np.array_equal(result.model.labels_, raw_labels)
    assert result.scaler.n_samples_seen_ == 90
    np.testing.assert_allclose(result.scaler.mean_, embedding.mean(axis=0, dtype=np.float64), rtol=1e-6)
    # Patch features are untouched and still give the component means.
    assert np.array_equal(_feature_matrix(cohort), features_before)
    means = tuple(float(features_before[signal][expected_signal == component, 0].mean()) for component in range(3))
    assert result.component_means == means


def test_cluster_ecm_patches_embedding_provenance() -> None:
    cohort, embedding, _ = _embedding_cohort()

    result = mt.pp.cluster_ecm_patches(cohort, n_clusters=3, subset="signal", embedding=embedding, order_by="size")

    model_sizes = np.bincount(result.model.labels_, minlength=3)
    order = sorted(range(3), key=lambda component: (-model_sizes[component], component))
    provenance = {
        "provenance_version": 1,
        "mantpy_version": mt.__version__,
        "method": "kmeans",
        "subset": "signal",
        "cluster_key": "ecm_cluster",
        "feature_columns": ["embedding_0", "embedding_1", "embedding_2", "embedding_3"],
        "random_state": 0,
        "n_clusters": 3,
        "cluster_counts": {"-1": 30, "0": 50, "1": 30, "2": 10},
        "standardization": {"method": "StandardScaler", "fit_subset": "signal", "input_dtype": "float32"},
        "versions": {"mantpy": mt.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "n_init": 10,
        "requested_n_clusters": 3,
        "feature_source": "embedding",
        "label_order": "descending_size_then_original_label",
        "label_mapping": {str(model_label): label for label, model_label in enumerate(order)},
    }
    assert result.provenance == provenance
    stored = [adata.uns["ecm_clustering"] for adata in cohort.values()]
    for adata, entry in zip(cohort.values(), stored, strict=True):
        assert entry == provenance
        assert adata.uns["mantpy"]["pp"]["cluster_ecm_patches"] == provenance
    # Each sample owns its copy of the mutable entries.
    stored[0]["label_mapping"]["0"] = 99
    stored[0]["feature_columns"].append("extra")
    assert stored[1] == provenance
    assert result.provenance == provenance


def test_cluster_ecm_patches_embedding_mapping_equals_array() -> None:
    cohort, embedding, signal = _embedding_cohort()
    by_array = {sample: adata.copy() for sample, adata in cohort.items()}
    by_mapping = {sample: adata.copy() for sample, adata in cohort.items()}
    blocks = {}
    offset = 0
    start = 0
    for sample, adata in cohort.items():
        n_selected = int(signal[start : start + len(adata.uns["ecm_patches"])].sum())
        blocks[sample] = embedding[offset : offset + n_selected]
        offset += n_selected
        start += len(adata.uns["ecm_patches"])
    # The mapping is read by key, in adatas order.
    shuffled = {sample: blocks[sample] for sample in ("roi_c", "roi_a", "roi_b")}

    array_result = mt.pp.cluster_ecm_patches(
        by_array, n_clusters=3, subset="signal", embedding=embedding, order_by="size"
    )
    mapping_result = mt.pp.cluster_ecm_patches(
        by_mapping, n_clusters=3, subset="signal", embedding=shuffled, order_by="size"
    )

    assert np.array_equal(_labels(by_mapping), _labels(by_array))
    assert mapping_result.component_sizes == array_result.component_sizes
    assert mapping_result.component_means == array_result.component_means
    assert mapping_result.provenance == array_result.provenance
    assert np.array_equal(mapping_result.model.cluster_centers_, array_result.model.cluster_centers_)


def test_cluster_ecm_patches_embedding_subset_all_and_double_precision() -> None:
    cohort, _, _ = _embedding_cohort()
    rng = np.random.default_rng(31)
    blob = rng.permutation(np.repeat([0, 1, 2], [70, 30, 20]))
    embedding = np.asarray([[5.0, 0.0], [-5.0, 4.0], [0.0, -6.0]])[blob] + rng.normal(0.0, 0.2, size=(120, 2))
    assert embedding.dtype == np.float64
    single = {sample: adata.copy() for sample, adata in cohort.items()}

    result = mt.pp.cluster_ecm_patches(cohort, n_clusters=3, embedding=embedding, order_by="size")
    single_result = mt.pp.cluster_ecm_patches(
        single, n_clusters=3, embedding=embedding.astype(np.float32), order_by="size"
    )

    labels = _labels(cohort)
    assert (labels >= 0).all()
    assert np.array_equal(labels, _notebook_recipe(embedding.astype(np.float32), 3)[0])
    assert result.component_sizes == (70, 30, 20)
    assert result.provenance["standardization"] == {
        "method": "StandardScaler",
        "fit_subset": "all",
        "input_dtype": "float32",
    }
    assert result.scaler.n_samples_seen_ == 120
    assert result.scaler.mean_.dtype == single_result.scaler.mean_.dtype
    assert np.array_equal(labels, _labels(single))
    assert result.provenance == single_result.provenance


def test_cluster_ecm_patches_order_by_size_tie_keeps_lower_label() -> None:
    cohort = _feature_cohort()
    rng = np.random.default_rng(41)
    blob = rng.permutation(np.repeat([0, 1, 2], [20, 20, 14]))
    embedding = (np.asarray([[5.0, 0.0], [-5.0, 4.0], [0.0, -6.0]])[blob] + rng.normal(0.0, 0.1, size=(54, 2))).astype(
        np.float32
    )

    result = mt.pp.cluster_ecm_patches(cohort, n_clusters=3, embedding=embedding, order_by="size")

    model_labels = result.model.labels_
    model_sizes = np.bincount(model_labels, minlength=3)
    assert sorted(model_sizes.tolist()) == [14, 20, 20]
    tied = np.flatnonzero(model_sizes == 20)
    smallest = int(np.flatnonzero(model_sizes == 14)[0])
    mapping = {int(tied[0]): 0, int(tied[1]): 1, smallest: 2}
    assert tied[0] < tied[1]
    assert result.provenance["label_mapping"] == {str(key): mapping[key] for key in sorted(mapping)}
    assert np.array_equal(_labels(cohort), [mapping[int(label)] for label in model_labels])
    assert result.component_sizes == (20, 20, 14)


def test_cluster_ecm_patches_order_by_size_all_equal_keeps_model_labels() -> None:
    reference = _feature_cohort()
    cohort = _feature_cohort()
    rng = np.random.default_rng(43)
    blob = rng.permutation(np.repeat([0, 1], [27, 27]))
    embedding = (np.asarray([[3.0, 0.0], [-3.0, 1.0]])[blob] + rng.normal(0.0, 0.1, size=(54, 2))).astype(np.float32)

    unordered = mt.pp.cluster_ecm_patches(reference, n_clusters=2, embedding=embedding)
    ordered = mt.pp.cluster_ecm_patches(cohort, n_clusters=2, embedding=embedding, order_by="size")

    assert ordered.component_sizes == (27, 27)
    assert ordered.provenance["label_mapping"] == {"0": 0, "1": 1}
    assert np.array_equal(_labels(cohort), _labels(reference))
    assert np.array_equal(_labels(cohort), unordered.model.labels_)


@pytest.mark.parametrize("order_by", [None, "label", "size"])
@pytest.mark.parametrize("use_embedding", [False, True])
def test_cluster_ecm_patches_label_mapping(order_by: str | None, use_embedding: bool) -> None:
    cohort, embedding, signal = _embedding_cohort()
    if not use_embedding:
        # Put the blobs in the patch table so that the table path finds them too.
        offset = 0
        selected = 0
        for adata in cohort.values():
            table = adata.uns["ecm_patches"].copy()
            rows = signal[offset : offset + len(table)]
            table.loc[rows, ["feat_0", "feat_1"]] = embedding[selected : selected + int(rows.sum()), :2]
            adata.uns["ecm_patches"] = table
            offset += len(table)
            selected += int(rows.sum())
    kwargs = {"embedding": embedding} if use_embedding else {}

    result = mt.pp.cluster_ecm_patches(cohort, n_clusters=3, subset="signal", order_by=order_by, **kwargs)

    written = _labels(cohort)[signal]
    if not use_embedding and order_by is None:
        assert "label_mapping" not in result.provenance
        assert "label_order" not in result.provenance
        assert "feature_source" not in result.provenance
        assert np.array_equal(written, result.model.labels_)
        return
    mapping = result.provenance["label_mapping"]
    assert sorted(mapping) == ["0", "1", "2"]
    assert sorted(mapping.values()) == [0, 1, 2]
    lookup = np.asarray([mapping[str(model_label)] for model_label in range(3)])
    assert np.array_equal(lookup[result.model.labels_], written)
    assert result.provenance["feature_source"] == ("embedding" if use_embedding else "patch_table")
    assert result.provenance["standardization"]["fit_subset"] == ("signal" if use_embedding else "all")
    assert result.feature_columns[0] == ("embedding_0" if use_embedding else "feat_0")
    if order_by == "size":
        assert result.provenance["label_order"] == "descending_size_then_original_label"
        assert list(result.component_sizes) == sorted(result.component_sizes, reverse=True)
        assert result.component_sizes == (50, 30, 10)
    else:
        assert result.provenance["label_order"] == "model"
        assert mapping == {"0": 0, "1": 1, "2": 2}
        assert np.array_equal(written, result.model.labels_)
    assert result.component_sizes == tuple(np.bincount(written, minlength=3).tolist())
    # Component means follow the written labels.
    raw_means = _feature_matrix(cohort)[signal][:, 0]
    assert result.component_means == tuple(float(raw_means[written == component].mean()) for component in range(3))


def test_cluster_ecm_patches_order_by_size_permutes_centre_means() -> None:
    renumbered = 0
    # The model numbering depends on the seed; several seeds make sure that
    # at least one fit is not already in size order.
    for random_state in range(6):
        cohort = _feature_cohort(mean_metadata=False)
        reference = _feature_cohort(mean_metadata=False)

        unordered = mt.pp.cluster_ecm_patches(reference, n_clusters=3, random_state=random_state)
        ordered = mt.pp.cluster_ecm_patches(cohort, n_clusters=3, random_state=random_state, order_by="size")

        centre_means = ordered.model.cluster_centers_.mean(axis=1)
        assert unordered.component_means == tuple(float(value) for value in centre_means)
        mapping = ordered.provenance["label_mapping"]
        assert ordered.component_sizes == (24, 18, 12)
        for model_label in range(3):
            assert ordered.component_means[mapping[str(model_label)]] == float(centre_means[model_label])
        assert np.array_equal(ordered.model.labels_, unordered.model.labels_)
        assert np.array_equal(ordered.model.cluster_centers_, unordered.model.cluster_centers_)
        assert ordered.provenance["feature_source"] == "patch_table"
        assert ordered.provenance["standardization"] == unordered.provenance["standardization"]
        renumbered += mapping != {"0": 0, "1": 1, "2": 2}
    assert renumbered > 0


def test_cluster_ecm_patches_order_by_size_then_remove_background() -> None:
    default = _feature_cohort()
    ordered = _feature_cohort()

    default_result = mt.pp.cluster_ecm_patches(default, n_clusters=3)
    ordered_result = mt.pp.cluster_ecm_patches(ordered, n_clusters=3, order_by="size")
    assert ordered_result.component_sizes == (24, 18, 12)
    # The dimmest blob is also the largest one, so size order puts it first.
    assert int(np.argmin(ordered_result.component_means)) == 0
    assert sorted(ordered_result.component_means) == sorted(default_result.component_means)

    default_removal = mt.pp.remove_background_patches(default, default_result)
    ordered_removal = mt.pp.remove_background_patches(ordered, ordered_result)

    assert ordered_removal.background_component == 0
    assert ordered_removal.background_patches == default_removal.background_patches == 24
    assert ordered_removal.signal_counts == default_removal.signal_counts == ((0, 18), (1, 12))
    assert np.array_equal(_labels(ordered), _labels(default))
    dimmest = _feature_matrix(ordered).mean(axis=1) < -5
    assert np.array_equal(_labels(ordered) == -1, dimmest)


def test_cluster_ecm_patches_leiden_embedding_and_order_by() -> None:
    pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")
    cohort, embedding, signal = _embedding_cohort()
    plain = {sample: adata.copy() for sample, adata in cohort.items()}
    sized = {sample: adata.copy() for sample, adata in cohort.items()}

    result = mt.pp.cluster_ecm_patches(
        cohort, method="leiden", subset="signal", embedding=embedding, n_neighbors=8, resolution=0.1
    )

    labels = _labels(cohort)
    assert np.array_equal(labels >= 0, signal)
    assert result.model is None
    assert result.component_sizes == (50, 30, 10)
    assert result.feature_columns == ("embedding_0", "embedding_1", "embedding_2", "embedding_3")
    assert result.provenance["feature_source"] == "embedding"
    assert result.provenance["label_order"] == "descending_size_then_original_label"
    assert sorted(result.provenance["label_mapping"].values()) == [0, 1, 2]
    assert result.provenance["standardization"]["fit_subset"] == "signal"
    assert list(result.provenance)[-3:] == ["label_order", "feature_source", "label_mapping"]

    # Leiden is always size-ordered, so order_by='size' changes the provenance only.
    plain_result = mt.pp.cluster_ecm_patches(plain, method="leiden", subset="signal", n_neighbors=8, resolution=0.5)
    sized_result = mt.pp.cluster_ecm_patches(
        sized, method="leiden", subset="signal", n_neighbors=8, resolution=0.5, order_by="size"
    )
    assert np.array_equal(_labels(sized), _labels(plain))
    assert sized_result.component_sizes == plain_result.component_sizes
    assert sized_result.component_means == plain_result.component_means
    added = {key: sized_result.provenance[key] for key in ("feature_source", "label_mapping")}
    assert added["feature_source"] == "patch_table"
    assert sized_result.provenance == {**plain_result.provenance, **added}
    assert "label_mapping" not in plain_result.provenance


def test_cluster_ecm_patches_embedding_errors() -> None:
    cohort, embedding, signal = _embedding_cohort()
    labels_before = _labels(cohort).copy()

    def cluster(**kwargs):
        options = {"n_clusters": 3, "subset": "signal", "embedding": embedding}
        options.update(kwargs)
        return mt.pp.cluster_ecm_patches(cohort, **options)

    # Wrong rows.
    with pytest.raises(ValueError, match=r"embedding has 89 rows but subset='signal' selects 90 patches\."):
        cluster(embedding=embedding[:-1])
    with pytest.raises(ValueError, match=r"embedding has 90 rows but subset='all' selects 120 patches\."):
        cluster(subset="all")
    # 1-D and 3-D input.
    with pytest.raises(ValueError, match=r"embedding must be a 2-D patch-by-feature array\."):
        cluster(embedding=embedding[:, 0])
    with pytest.raises(ValueError, match=r"embedding must be a 2-D patch-by-feature array\."):
        cluster(embedding=embedding[:, :, np.newaxis])
    # Non-finite values are an error, not zero-filled as in the patch table.
    for value in (np.nan, np.inf):
        broken = embedding.copy()
        broken[7, 2] = value
        with pytest.raises(ValueError, match=r"embedding contains NaN or infinite values\."):
            cluster(embedding=broken)
    # order_by.
    with pytest.raises(
        ValueError,
        match=r"order_by='label' is not available for method='leiden', which is always ordered by size\.",
    ):
        cluster(method="leiden", order_by="label")
    with pytest.raises(ValueError, match=r"order_by must be None, 'size' or 'label'\."):
        cluster(order_by="count")
    with pytest.raises(ValueError, match=r"order_by must be None, 'size' or 'label'\."):
        mt.pp.cluster_ecm_patches(cohort, n_clusters=3, order_by="count")
    # The existing checks still apply to the selected rows.
    with pytest.raises(ValueError, match=r"n_clusters=91 exceeds the 90 selected patches\."):
        cluster(n_clusters=91)

    # Mapping input.
    counts = []
    offset = 0
    for adata in cohort.values():
        counts.append(int(signal[offset : offset + len(adata.uns["ecm_patches"])].sum()))
        offset += len(adata.uns["ecm_patches"])
    bounds = np.cumsum([0, *counts])
    blocks = {sample: embedding[bounds[index] : bounds[index + 1]] for index, sample in enumerate(cohort)}
    with pytest.raises(
        ValueError, match=r"embedding keys differ from adatas; missing=\['roi_c'\], extra=\['roi_d'\]\."
    ):
        cluster(embedding={"roi_a": blocks["roi_a"], "roi_b": blocks["roi_b"], "roi_d": blocks["roi_c"]})
    with pytest.raises(
        ValueError,
        match=rf"Sample 'roi_b': embedding has {counts[1] - 1} rows but subset='signal' selects {counts[1]} patches\.",
    ):
        cluster(embedding={**blocks, "roi_b": blocks["roi_b"][1:]})
    with pytest.raises(ValueError, match=r"embedding must be a 2-D patch-by-feature array\."):
        cluster(embedding={**blocks, "roi_a": blocks["roi_a"][:, 0]})
    with pytest.raises(ValueError, match=r"embedding arrays must have the same number of columns\."):
        cluster(embedding={**blocks, "roi_c": blocks["roi_c"][:, :3]})

    # A rejected call writes nothing.
    assert np.array_equal(_labels(cohort), labels_before)
    assert all("ecm_clustering" not in adata.uns for adata in cohort.values())


def test_cluster_ecm_patches_embedding_provenance_survives_h5ad(tmp_path) -> None:
    cohort, embedding, _ = _embedding_cohort()
    result = mt.pp.cluster_ecm_patches(cohort, n_clusters=3, subset="signal", embedding=embedding, order_by="size")
    carrier = cohort["roi_a"]
    table = carrier.uns.pop("ecm_patches")
    path = tmp_path / "roi_a.h5ad"

    carrier.write_h5ad(path)
    restored = ad.read_h5ad(path).uns["ecm_clustering"]

    assert restored["feature_source"] == "embedding"
    assert restored["label_order"] == "descending_size_then_original_label"
    assert {key: int(value) for key, value in restored["label_mapping"].items()} == result.provenance["label_mapping"]
    assert list(restored["feature_columns"]) == list(result.feature_columns)
    assert len(table) == 40


def test_tiles_to_embedding_clustering_round_trip() -> None:
    """Tiles of the signal patches feed an embedding that is clustered and written back in place."""
    rng = np.random.default_rng(51)
    levels = np.asarray([0.0, 1.0, 4.0, 9.0], dtype=np.float32)
    images = {}
    lattice = {}
    for sample in ("roi_a", "roi_b"):
        lattice[sample] = rng.permutation(np.repeat([0, 1, 2, 3], [9, 7, 5, 3])).reshape(4, 6)
        blocks = np.kron(levels[lattice[sample]], np.ones((5, 5), dtype=np.float32))
        images[sample] = (blocks[np.newaxis] + rng.normal(0.0, 0.01, size=(3, 20, 30))).astype(np.float32)
    cohort = _tile_cohort(images)

    two_groups = mt.pp.cluster_ecm_patches(cohort, n_clusters=2)
    mt.pp.remove_background_patches(cohort, two_groups)
    signal = _labels(cohort) >= 0
    tiles = mt.pp.ecm_patch_tiles(cohort, images, subset="signal")
    assert len(tiles) == int(signal.sum())
    embedding = tiles.mean(axis=(2, 3))

    result = mt.pp.cluster_ecm_patches(cohort, n_clusters=2, subset="signal", embedding=embedding, order_by="size")

    labels = _labels(cohort)
    assert np.array_equal(labels >= 0, signal)
    pooled_lattice = np.concatenate([lattice[sample].ravel() for sample in cohort])
    for label in range(result.n_clusters):
        assert len(set(pooled_lattice[labels == label].tolist())) == 1
    assert list(result.component_sizes) == sorted(result.component_sizes, reverse=True)
    assert sum(result.component_sizes) == int(signal.sum())
