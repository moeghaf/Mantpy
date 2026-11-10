"""Tests for ``mt.pl.marker_contrast_bars`` and the per-sample points of ``mt.pl.cell_ecm_enrichment_bars``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, set before pyplot is imported

from types import SimpleNamespace

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import mantpy as mt

GROUPS = {"roi_1": "mouse_01", "roi_2": "mouse_01", "roi_3": "mouse_02"}


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close("all")


# ------------------------------------------------------------- marker_contrast_bars
def _contrast() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.5, -0.2, 0.1], [0.7, -0.4, -0.1], [0.6, -0.3, 0.3]],
        index=["roi_1", "roi_2", "roi_3"], columns=["FN", "CS", "HA"],
    )


def _bar_widths(ax) -> list[float]:
    return [round(patch.get_width(), 6) for patch in ax.patches]


def test_bars_are_the_sample_means_in_ascending_order():
    table = _contrast()
    fig, ax = plt.subplots()
    out = mt.pl.marker_contrast_bars(table, ax=ax, show=False)
    assert out is ax
    assert [label.get_text() for label in ax.get_yticklabels()] == ["CS", "HA", "FN"]
    assert _bar_widths(ax) == [-0.3, 0.1, 0.6]
    colours = [to_hex(patch.get_facecolor()) for patch in ax.patches]
    assert colours == ["#bdbdbd", "#2a9d8f", "#2a9d8f"]


def test_one_point_per_sample_and_marker():
    table = _contrast()
    fig, ax = plt.subplots()
    mt.pl.marker_contrast_bars(table, ax=ax, show=False)
    assert len(ax.collections) == len(table)
    for collection, sample in zip(ax.collections, table.index, strict=True):
        offsets = np.asarray(collection.get_offsets())
        np.testing.assert_allclose(offsets[:, 0], table.loc[sample, ["CS", "HA", "FN"]].to_numpy())
        np.testing.assert_allclose(offsets[:, 1], [0, 1, 2])
    assert ax.get_legend() is None

    fig, ax = plt.subplots()
    mt.pl.marker_contrast_bars(table, ax=ax, show_samples=False, show=False)
    assert len(ax.collections) == 0


def test_sample_groups_give_marker_shapes_and_a_legend():
    fig, ax = plt.subplots()
    mt.pl.marker_contrast_bars(_contrast(), sample_groups=GROUPS, ax=ax, show=False)
    legend = ax.get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["mouse_01", "mouse_02"]
    assert [handle.get_marker() for handle in legend.legend_handles] == ["o", "^"]

    fig, ax = plt.subplots()
    mt.pl.marker_contrast_bars(_contrast(), sample_groups=pd.Series(GROUPS), legend=False, ax=ax, show=False)
    assert ax.get_legend() is None

    with pytest.raises(ValueError, match="roi_3"):
        mt.pl.marker_contrast_bars(_contrast(), sample_groups={"roi_1": "a", "roi_2": "b"}, show=False)


def test_a_result_object_and_a_series_are_accepted():
    table = _contrast()
    result = SimpleNamespace(per_sample=table, mean=table.mean())
    fig, ax = plt.subplots()
    mt.pl.marker_contrast_bars(result, ax=ax, show=False)
    assert _bar_widths(ax) == [-0.3, 0.1, 0.6]
    assert len(ax.collections) == 3

    fig, ax = plt.subplots()
    mt.pl.marker_contrast_bars(table.mean(), ax=ax, show=False)
    assert _bar_widths(ax) == [-0.3, 0.1, 0.6]
    assert len(ax.collections) == 0

    with pytest.raises(TypeError, match="cell_ecm_marker_contrast"):
        mt.pl.marker_contrast_bars([0.1, 0.2], show=False)


def test_order_options():
    table = _contrast()

    def labels(order):
        fig, ax = plt.subplots()
        mt.pl.marker_contrast_bars(table, order=order, ax=ax, show=False)
        return [label.get_text() for label in ax.get_yticklabels()]

    assert labels("descending") == ["FN", "HA", "CS"]
    assert labels("input") == ["FN", "CS", "HA"]
    assert labels(["HA", "FN"]) == ["HA", "FN"]
    with pytest.raises(ValueError, match="order"):
        mt.pl.marker_contrast_bars(table, order="by_name", show=False)
    with pytest.raises(ValueError, match="Col-I"):
        mt.pl.marker_contrast_bars(table, order=["FN", "Col-I"], show=False)


def test_labels_title_and_colours():
    fig, ax = plt.subplots()
    mt.pl.marker_contrast_bars(
        _contrast(), color="#112233", negative_color="#445566", xlabel="difference", title="f", ax=ax, show=False
    )
    assert ax.get_xlabel() == "difference"
    assert ax.get_title(loc="left") == "f"
    assert [to_hex(patch.get_facecolor()) for patch in ax.patches] == ["#445566", "#112233", "#112233"]


# ----------------------------------------- cell_ecm_enrichment_bars with sample points
def _details() -> SimpleNamespace:
    pooled = pd.DataFrame({"cluster": [0, 1, 2], "log2_enr": [-3.0, 2.0, -1.5]})
    per_roi = pd.DataFrame(
        {
            "roi": np.repeat(["roi_1", "roi_2", "roi_3"], 3),
            "cluster": np.tile([0, 1, 2], 3),
            "n_obs": [0, 40, 5, 6, 50, 1, 9, 30, 7],
            "n_exp": [20.0, 10.0, 15.0] * 3,
            "log2_enr": [-38.0, 2.5, -1.8, -3.5, 2.6, -5.0, -4.0, 3.1, 0.7],
        }
    )
    return SimpleNamespace(pooled=pooled, per_roi=per_roi)


def _points(ax) -> np.ndarray:
    return np.vstack([np.asarray(collection.get_offsets()) for collection in ax.collections])


def test_enrichment_bars_draw_one_point_per_sample_and_cluster():
    details = _details()
    fig, ax = plt.subplots()
    out = mt.pl.cell_ecm_enrichment_bars(details, annotate="none", show_samples=True, ax=ax, show=False)
    assert out is ax
    assert [round(patch.get_height(), 6) for patch in ax.patches] == [-3.0, 2.0, -1.5]
    points = _points(ax)
    assert len(points) == 9
    # Samples are spread around the bar of their cluster, in the order of the table.
    np.testing.assert_allclose(points[:, 0], np.tile([0, 1, 2], 3) + np.repeat([-0.2, 0.0, 0.2], 3))
    np.testing.assert_allclose(points[1:, 1], details.per_roi["log2_enr"].to_numpy()[1:])
    assert len(ax.texts) == 0


def test_a_sample_without_contact_is_drawn_below_the_bars_not_at_its_ratio():
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(_details(), annotate="none", show_samples=True, ax=ax, show=False)
    low, high = ax.get_ylim()
    assert high == pytest.approx(5.0 * 1.3) and low == pytest.approx(-5.0 * 1.3)  # the largest finite ratio is 5
    no_contact = ax.collections[0]
    assert np.asarray(no_contact.get_offsets())[0, 1] == pytest.approx(-1.18 * 5.0)
    assert no_contact.get_facecolors().size == 0  # open marker
    assert to_hex(no_contact.get_edgecolors()[0]) == "#808080"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["no contact"]


def test_enrichment_bars_sample_groups_and_legend():
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(
        _details(), annotate="none", show_samples=True, sample_groups=GROUPS, ax=ax, show=False
    )
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["mouse_01", "mouse_02", "no contact"]

    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(
        _details(), annotate="none", show_samples=True, sample_groups=GROUPS, legend=False, ax=ax, show=False
    )
    assert ax.get_legend() is None

    with pytest.raises(ValueError, match="sample_groups"):
        mt.pl.cell_ecm_enrichment_bars(
            _details(), annotate="none", show_samples=True, sample_groups={"roi_1": "a"}, show=False
        )


def test_enrichment_bars_read_the_per_sample_table_from_anndata():
    details = _details()
    adata = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
    adata.uns["cell_ecm_enrichment"] = details.pooled
    with pytest.raises(ValueError, match="return_details"):
        mt.pl.cell_ecm_enrichment_bars(adata, annotate="none", show_samples=True, show=False)

    adata.uns["cell_ecm_enrichment_per_roi"] = details.per_roi
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(adata, annotate="none", show_samples=True, ax=ax, show=False)
    from_adata = _points(ax)
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(details, annotate="none", show_samples=True, ax=ax, show=False)
    np.testing.assert_allclose(from_adata, _points(ax))

    adata.uns["cell_ecm_enrichment_per_roi"] = details.per_roi.drop(columns="roi")
    with pytest.raises(ValueError, match="roi"):
        mt.pl.cell_ecm_enrichment_bars(adata, annotate="none", show_samples=True, show=False)


def test_without_show_samples_the_plot_is_as_before():
    details = _details()
    adata = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
    adata.uns["cell_ecm_enrichment"] = details.pooled
    adata.uns["cell_ecm_enrichment_per_roi"] = details.per_roi
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(adata, annotate="effect", ax=ax, show=False)
    assert len(ax.collections) == 0
    assert ax.get_legend() is None
    assert ax.get_ylim() == pytest.approx((-3.0 * 1.3, 3.0 * 1.3))
    assert len(ax.texts) == 3


def test_annotate_none_needs_no_p_values():
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(_details(), annotate="none", ax=ax, show=False)
    assert len(ax.texts) == 0
    with pytest.raises(ValueError, match="p_fdr"):
        mt.pl.cell_ecm_enrichment_bars(_details(), annotate="stars", show=False)
