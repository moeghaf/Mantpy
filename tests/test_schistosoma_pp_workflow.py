from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantpy as mt


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Fibronectin", "cell_marker", "CollagenI"],
            "keep": [1, 1, 1],
            "ecm": [1, 0, 1],
        }
    )


def test_pooled_arcsinh_percentile_matches_historical_float32_path() -> None:
    images = {
        "roi_a": np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4) + np.float32(1),
        "roi_b": np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4) + np.float32(51),
    }
    cofactor = 2.0
    percentile = 75.0

    cohort = mt.pp.ecm_patches_from_images(
        images,
        _panel(),
        normalize="pooled_arcsinh_percentile",
        cofactor=cofactor,
        percentile=percentile,
        patch_size=2,
        ecm_K=None,
        features=["mean"],
        background_quantile=0.0,
    )

    ecm_indices = np.array([0, 2])
    pooled = np.concatenate(
        [image[ecm_indices].reshape(len(ecm_indices), -1) for image in images.values()],
        axis=1,
    ).astype(np.float32)
    historical_pooled = np.arcsinh(pooled / cofactor)
    expected_bounds = np.asarray(
        np.percentile(historical_pooled, percentile, axis=1),
        dtype=np.float32,
    )
    assert expected_bounds.dtype == np.float32

    for sample, adata in cohort.items():
        provenance = adata.uns["ecm_pixel_normalization"]
        assert provenance["method"] == "pooled_arcsinh_percentile"
        assert provenance["cofactor"] == cofactor
        assert provenance["percentile"] == percentile
        assert list(provenance["marker_order"]) == ["Fibronectin", "CollagenI"]
        np.testing.assert_array_equal(provenance["bounds"], expected_bounds)

        # Reproduce the historical full-stack transform, then compare each
        # retained patch's mean features bit-for-bit at its recorded centroid.
        expected_image = np.arcsinh(images[sample] / cofactor)
        for local, channel in enumerate(ecm_indices):
            if expected_bounds[local] > 0:
                expected_image[channel] = np.clip(
                    expected_image[channel] / expected_bounds[local],
                    0.0,
                    1.0,
                )
        patches = adata.uns["ecm_patches"]
        expected_features = []
        for row in patches.itertuples(index=False):
            x0 = int(row.x - 1)
            y0 = int(row.y - 1)
            expected_features.append(expected_image[ecm_indices, y0 : y0 + 2, x0 : x0 + 2].mean(axis=(1, 2)))
        np.testing.assert_array_equal(
            patches[["feat_0", "feat_1"]].to_numpy(dtype=np.float32),
            np.asarray(expected_features, dtype=np.float32),
        )


def _staged_cluster_cohort() -> tuple[dict[str, ad.AnnData], np.ndarray]:
    rng = np.random.default_rng(7)
    clouds = [
        rng.normal((-10.0, -10.0), 0.08, size=(60, 2)),
        rng.normal((-2.0, 7.0), 0.08, size=(90, 2)),
        rng.normal((7.0, 5.0), 0.08, size=(60, 2)),
        rng.normal((8.0, -5.0), 0.08, size=(30, 2)),
    ]
    pooled = np.vstack(clouds).astype(np.float32)
    cohort: dict[str, ad.AnnData] = {}
    for sample, rows in {"roi_a": pooled[:120], "roi_b": pooled[120:]}.items():
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
        carrier.uns["ecm_feature_names"] = json.dumps(
            [
                {"extractor": "mean", "protein": "Fibronectin"},
                {"extractor": "mean", "protein": "CollagenI"},
            ]
        )
        cohort[sample] = carrier
    return cohort, pooled


def test_staged_kmeans_background_then_leiden_preserves_minus_one() -> None:
    pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")

    cohort, pooled = _staged_cluster_cohort()
    stage_one = mt.pp.cluster_ecm_patches(cohort, n_clusters=2, random_state=0)
    assert stage_one.method == "kmeans"
    assert stage_one.provenance is not None
    assert stage_one.provenance["standardization"]["input_dtype"] == "float32"
    np.testing.assert_allclose(stage_one.scaler.mean_, pooled.mean(axis=0), rtol=1e-6)

    removal = mt.pp.remove_background_patches(cohort, stage_one)
    assert removal.background_patches == 60
    background_before = {
        sample: adata.uns["ecm_patches"]["ecm_cluster"].to_numpy(dtype=int) < 0 for sample, adata in cohort.items()
    }

    duplicate = {sample: adata.copy() for sample, adata in cohort.items()}
    leiden = mt.pp.cluster_ecm_patches(
        cohort,
        method="leiden",
        subset="signal",
        n_neighbors=10,
        resolution=0.1,
        random_state=0,
    )
    leiden_duplicate = mt.pp.cluster_ecm_patches(
        duplicate,
        method="leiden",
        subset="signal",
        n_neighbors=10,
        resolution=0.1,
        random_state=0,
    )

    assert leiden.method == "leiden"
    assert leiden.subset == "signal"
    assert leiden.n_clusters == 3
    assert leiden.counts[-1] == 60
    assert list(leiden.component_sizes) == sorted(leiden.component_sizes, reverse=True)
    assert sum(leiden.component_sizes) == 180
    assert leiden.counts == leiden_duplicate.counts

    for sample, adata in cohort.items():
        labels = adata.uns["ecm_patches"]["ecm_cluster"].to_numpy(dtype=int)
        duplicate_labels = duplicate[sample].uns["ecm_patches"]["ecm_cluster"].to_numpy(dtype=int)
        assert np.array_equal(labels, duplicate_labels)
        assert np.array_equal(labels < 0, background_before[sample])
        assert np.all(labels[background_before[sample]] == -1)

        provenance = adata.uns["ecm_clustering"]
        assert provenance["method"] == "leiden"
        assert provenance["subset"] == "signal"
        assert provenance["n_neighbors"] == 10
        assert provenance["resolution"] == 0.1
        assert provenance["flavor"] == "leidenalg"
        assert provenance["provenance_version"] == 1
        assert provenance["mantpy_version"] == mt.__version__
        assert provenance["versions"]["mantpy"] == mt.__version__
        assert provenance["standardization"] == {
            "method": "StandardScaler",
            "fit_subset": "all",
            "input_dtype": "float32",
        }


def test_select_leiden_resolution_scores_one_shared_signal_graph() -> None:
    pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")

    cohort, _ = _staged_cluster_cohort()
    stage_one = mt.pp.cluster_ecm_patches(cohort, n_clusters=2, random_state=0)
    mt.pp.remove_background_patches(cohort, stage_one)
    labels_before = {
        sample: adata.uns["ecm_patches"]["ecm_cluster"].to_numpy(dtype=int).copy() for sample, adata in cohort.items()
    }

    selection = mt.pp.select_ecm_leiden_resolution(
        cohort,
        resolutions=(0.05, 0.1, 0.5),
        subset="signal",
        n_neighbors=10,
        random_state=0,
        flavor="leidenalg",
    )

    assert list(selection.table) == ["resolution", "n_clusters", "calinski_harabasz"]
    assert list(selection.table["resolution"]) == [0.05, 0.1, 0.5]
    assert selection.selected_resolution in {0.05, 0.1, 0.5}
    selected_row = selection.table.loc[selection.table["resolution"].eq(selection.selected_resolution)].iloc[0]
    assert selected_row["calinski_harabasz"] == pytest.approx(selection.table["calinski_harabasz"].max())
    assert selection.selected_n_clusters == int(selected_row["n_clusters"])
    assert selection.effective_n_neighbors == 10
    assert selection.flavor == "leidenalg"
    assert selection.versions["mantpy"] == mt.__version__
    assert all(
        np.array_equal(
            labels_before[sample],
            adata.uns["ecm_patches"]["ecm_cluster"].to_numpy(dtype=int),
        )
        for sample, adata in cohort.items()
    )


def test_patch_comparison_can_explicitly_skip_deposited_labels() -> None:
    cohort, _ = _staged_cluster_cohort()

    comparison = mt.pp.compare_ecm_patches(
        cohort,
        {sample: adata.copy() for sample, adata in cohort.items()},
        cluster_key=None,
        atol=0.0,
        rtol=0.0,
    )

    assert comparison.label_matches is None
    assert "cluster labels" not in repr(comparison)
