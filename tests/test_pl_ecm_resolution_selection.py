"""The ECM state count must be auditable, not asserted.

`select_ecm_leiden_resolution` optimises the *resolution* by
Calinski-Harabasz; the number of ECM states is whatever Leiden emits at the
winner and is never itself optimised. This plot exists so a reader can see
the margin the selection won by.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import mantpy as mt
from mantpy.pp import ECMLeidenResolutionSelection


def _selection(n_clusters=(3, 4, 5, 7, 8), scores=(80.0, 94.0, 102.0, 110.0, 108.0)):
    resolutions = [0.1, 0.2, 0.3, 0.4, 0.5][: len(n_clusters)]
    best = int(max(range(len(scores)), key=lambda i: scores[i]))
    return ECMLeidenResolutionSelection(
        table=pd.DataFrame(
            {
                "resolution": resolutions,
                "n_clusters": list(n_clusters),
                "calinski_harabasz": list(scores),
            }
        ),
        selected_resolution=resolutions[best],
        selected_n_clusters=int(n_clusters[best]),
        subset="signal",
        n_neighbors=15,
        effective_n_neighbors=15,
        flavor="leidenalg",
        feature_columns=("feat_0", "feat_1"),
        versions={"leidenalg": "0.11.0"},
    )


def test_plots_both_series_against_resolution():
    ax = mt.pl.ecm_resolution_selection(_selection(), show=False)
    twin = next(a for a in ax.get_figure().axes if a is not ax)

    assert [tuple(x) for x in ax.lines[0].get_xydata()][0][0] == pytest.approx(0.1)
    assert ax.get_xlabel() == "Leiden resolution"
    assert "Calinski" in ax.get_ylabel()
    assert "ECM states" in twin.get_ylabel()
    plt.close("all")


def test_title_states_the_resolution_and_the_resulting_count():
    ax = mt.pl.ecm_resolution_selection(_selection(), show=False)
    title = ax.get_title()

    assert "0.4" in title
    assert "7 ECM states" in title
    plt.close("all")


def test_marks_the_selected_resolution():
    ax = mt.pl.ecm_resolution_selection(_selection(), show=False)

    verticals = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert verticals, "the selected resolution must be marked"
    plt.close("all")


def test_reports_whatever_count_leiden_produced():
    """Not hard-wired to seven — the plot reports the observed count."""
    ax = mt.pl.ecm_resolution_selection(
        _selection(n_clusters=(2, 3, 5), scores=(10.0, 40.0, 22.0)), show=False
    )

    assert "3 ECM states" in ax.get_title()
    plt.close("all")


def test_accepts_a_caller_supplied_axis():
    _, ax = plt.subplots()
    assert mt.pl.ecm_resolution_selection(_selection(), ax=ax, show=False) is ax
    plt.close("all")
