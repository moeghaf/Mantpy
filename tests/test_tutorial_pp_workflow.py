from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantpy as mt


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["matrix_a", "cell_marker", "matrix_b"],
            "keep": [1, 1, 1],
            "ecm": [1, 0, 1],
        }
    )


def test_ecm_patches_from_differently_shaped_images_uses_shared_scaler() -> None:
    images = {
        "r1": np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4),
        "r2": np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6),
    }
    cohort = mt.pp.ecm_patches_from_images(
        images,
        _panel(),
        patch_size=2,
        ecm_K=None,
        features=["mean"],
        background_quantile=0.0,
    )
    summary = mt.pp.ecm_patch_summary(cohort)
    assert summary.n_samples == 2
    assert summary.n_patches == 8
    assert summary.n_features == 2
    assert summary.n_background is None
    assert all("ecm_pixel_scaler" in adata.uns for adata in cohort.values())
    assert all(adata.uns["ecm_pixel_scaler"]["n_samples_seen"] == 40 for adata in cohort.values())


def _cluster_cohort() -> dict[str, ad.AnnData]:
    rng = np.random.default_rng(4)
    clouds = [
        rng.normal((-8, -8), 0.15, size=(120, 2)),
        rng.normal((-5, 7), 0.15, size=(80, 2)),
        rng.normal((7, 5), 0.15, size=(80, 2)),
        rng.normal((8, -6), 0.15, size=(80, 2)),
    ]
    pooled = np.vstack(clouds).astype(np.float32)
    cohort: dict[str, ad.AnnData] = {}
    for sample, rows in {"r1": pooled[:180], "r2": pooled[180:]}.items():
        carrier = ad.AnnData(X=np.empty((0, 0), dtype=np.float32))
        carrier.uns["ecm_patches"] = pd.DataFrame(
            {
                "x": np.arange(len(rows), dtype=float),
                "y": np.zeros(len(rows)),
                "ecm_cluster": np.zeros(len(rows), dtype=int),
                "feat_0": rows[:, 0],
                "feat_1": rows[:, 1],
            }
        )
        cohort[sample] = carrier
    return cohort


def test_select_cluster_remove_and_attach_are_staged() -> None:
    cohort = _cluster_cohort()
    selection = mt.pp.select_ecm_cluster_count(
        cohort,
        signal_k_range=range(2, 4),
        sample_size=300,
        random_state=0,
    )
    assert selection.selected_total_k == 4
    assert selection.selected_signal_k == 3
    assert list(selection.candidate_scan["total_k"]) == [3, 4]
    assert list(selection.signal_scan["K"]) == [2, 3]

    clustering = mt.pp.cluster_ecm_patches(
        cohort,
        n_clusters=selection.selected_total_k,
        random_state=0,
    )
    removal = mt.pp.remove_background_patches(cohort, clustering)
    summary = mt.pp.ecm_patch_summary(cohort, cluster_key="ecm_cluster")
    assert removal.background_patches == 120
    assert summary.n_background == 120
    assert [count for _, count in summary.signal_counts] == [80, 80, 80]

    cells = {
        sample: ad.AnnData(
            X=np.empty((1, 0), dtype=np.float32),
            obs=pd.DataFrame({"cell_type": ["A"]}, index=["c0"]),
        )
        for sample in cohort
    }
    attached = mt.pp.attach_ecm_patches(cells, cohort, inplace=False)
    assert attached is not None
    assert "ecm_patches" not in cells["r1"].uns
    assert len(attached["r1"].uns["ecm_patches"]) == len(cohort["r1"].uns["ecm_patches"])


def test_corruption_overlay_changes_only_flagged_labels() -> None:
    cohort = _cluster_cohort()
    clustering = mt.pp.cluster_ecm_patches(cohort, n_clusters=4, random_state=0)
    mt.pp.remove_background_patches(cohort, clustering)
    overlays = {}
    original_features = {}
    for sample, carrier in cohort.items():
        patches = carrier.uns["ecm_patches"]
        mask = np.zeros(len(patches), dtype=bool)
        mask[:3] = True
        replacement = patches["ecm_cluster"].to_numpy(dtype=int).copy()
        replacement[mask] = 2
        overlays[sample] = {
            "is_artifact": mask,
            "ecm_cluster_artifact": replacement,
            "ecm_cluster_pristine": patches["ecm_cluster"].to_numpy(dtype=int),
            "artifact_boxes": [{"center": (1, 1), "half": 1}],
        }
        original_features[sample] = patches[["feat_0", "feat_1"]].copy()

    corrupted = mt.pp.apply_ecm_label_overlay(cohort, overlays)
    overlay_summary = mt.pp.ecm_label_overlay_summary(corrupted)
    assert overlay_summary.n_artifacts == 6
    assert overlay_summary.reference_mismatches == 0
    for sample in cohort:
        original = cohort[sample].uns["ecm_patches"]
        changed = corrupted[sample].uns["ecm_patches"]
        mask = changed["is_artifact"].to_numpy(dtype=bool)
        assert np.array_equal(changed.loc[~mask, "ecm_cluster"], original.loc[~mask, "ecm_cluster"])
        pd.testing.assert_frame_equal(changed[["feat_0", "feat_1"]], original_features[sample])
        assert "is_artifact" not in original


def test_corruption_overlay_rejects_misaligned_reference() -> None:
    cohort = _cluster_cohort()
    clustering = mt.pp.cluster_ecm_patches(cohort, n_clusters=4, random_state=0)
    mt.pp.remove_background_patches(cohort, clustering)
    overlays = {}
    for sample, carrier in cohort.items():
        n = len(carrier.uns["ecm_patches"])
        overlays[sample] = {
            "is_artifact": np.zeros(n, dtype=bool),
            "ecm_cluster_artifact": np.zeros(n, dtype=int),
            "ecm_cluster_pristine": np.full(n, 99, dtype=int),
        }
    with pytest.raises(ValueError, match="not aligned"):
        mt.pp.apply_ecm_label_overlay(cohort, overlays)
