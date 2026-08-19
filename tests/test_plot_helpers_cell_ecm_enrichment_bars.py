"""Tests for ``mt.pl.cell_ecm_enrichment_bars``.

Covers the descriptive ``annotate='effect'`` mode, the default BH-FDR star
annotation and the v3 ``annotate='effect_q'`` style that prints the signed
log2 effect size + floored permutation q (e.g. ``+2.1`` / ``q<2e-4``).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import mantpy as mt


@pytest.fixture(autouse=True)
def _close_plots():
    yield
    plt.close("all")


def _adata_with_enrichment() -> ad.AnnData:
    """AnnData carrying a 3-cluster enrichment table at the default uns key."""
    df = pd.DataFrame(
        {
            "cluster": [0, 1, 2],
            "log2_enr": [-1.5, 2.1, -0.4],
            "p_fdr": [0.30, 2e-4, 0.40],  # only ECM1 significant
            "significant": [False, True, False],
        }
    )
    a = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
    a.uns["cell_ecm_enrichment"] = df
    return a


def test_default_stars_annotation():
    a = _adata_with_enrichment()
    fig, ax = plt.subplots()
    out = mt.pl.cell_ecm_enrichment_bars(a, ax=ax, show=False)
    assert out is ax
    texts = [t.get_text() for t in ax.texts]
    # ECM1 is significant at q=2e-4 -> "***"; the others "ns".
    assert "***" in texts
    assert texts.count("ns") == 2


def test_effect_q_annotation_wording():
    """effect_q prints signed log2 + floored q in the v3 wording."""
    a = _adata_with_enrichment()
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(a, ax=ax, annotate="effect_q", show=False)
    joined = "\n".join(t.get_text() for t in ax.texts)
    # Signed effect for the enriched bar.
    assert "+2.1" in joined
    # Depleted bars carry their negative signed effect, not bare stars.
    assert "-1.5" in joined
    # Floored permutation q for the significant bar (n_perm=5000 floor),
    # in the compact v3 wording.
    assert "q<2e-4" in joined
    # Non-significant bars are 'ns'.
    assert "ns" in joined
    # No stars in this style.
    assert "***" not in joined


def test_effect_annotation_needs_no_p_values():
    """effect prints signed log2 values without inferential fields."""
    a = _adata_with_enrichment()
    a.uns["cell_ecm_enrichment"] = a.uns["cell_ecm_enrichment"][["cluster", "log2_enr"]].copy()
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(a, ax=ax, annotate="effect", show=False)
    texts = [text.get_text() for text in ax.texts]
    assert texts == ["-1.5", "+2.1", "-0.4"]
    assert not any("q" in text or "ns" in text or "*" in text for text in texts)


def test_effect_q_custom_floor():
    a = _adata_with_enrichment()
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(
        a,
        ax=ax,
        annotate="effect_q",
        qfloor=1e-3,
        show=False,
    )
    joined = "\n".join(t.get_text() for t in ax.texts)
    assert "q<1e-3" in joined


def test_invalid_annotate_raises():
    a = _adata_with_enrichment()
    with pytest.raises(ValueError, match="effect_q"):
        mt.pl.cell_ecm_enrichment_bars(a, annotate="bogus", show=False)


def test_missing_key_raises():
    a = ad.AnnData(np.zeros((1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="cell_ecm_enrichment"):
        mt.pl.cell_ecm_enrichment_bars(a, show=False)


def test_default_annotate_is_stars_backwards_compatible():
    """Calling with no annotate kwarg must behave exactly as before."""
    a = _adata_with_enrichment()
    fig, ax1 = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(a, ax=ax1, show=False)
    default_texts = sorted(t.get_text() for t in ax1.texts)

    fig2, ax2 = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(a, ax=ax2, annotate="stars", show=False)
    explicit_texts = sorted(t.get_text() for t in ax2.texts)

    assert default_texts == explicit_texts


def test_cluster_labels_none_is_default_ecm_labels():
    """Without cluster_labels the x-ticks are the default ``ECM {k}`` (backward compat)."""
    a = _adata_with_enrichment()
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(a, ax=ax, show=False)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["ECM 0", "ECM 1", "ECM 2"]


def test_cluster_labels_dict_override_with_per_key_fallback():
    """A dict relabels by cluster key; a missing key falls back to ``ECM {k}``."""
    a = _adata_with_enrichment()
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(
        a, ax=ax, cluster_labels={0: "Alveolar", 1: "Bronchovascular wall"}, show=False,
    )
    # 0 and 1 are relabelled; cluster 2 is absent -> default label.
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Alveolar",
        "Bronchovascular wall",
        "ECM 2",
    ]


def test_cluster_labels_list_override_with_bounds_fallback():
    """A too-short list indexes by position; out-of-range positions fall back."""
    a = _adata_with_enrichment()
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(
        a, ax=ax, cluster_labels=["Alveolar", "Bronchovascular wall"], show=False,
    )
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "Alveolar",
        "Bronchovascular wall",
        "ECM 2",
    ]


def test_cluster_labels_preserve_per_cluster_colour():
    """Relabelled x-tick labels keep the per-cluster colour mapping."""
    a = _adata_with_enrichment()
    colours = {0: "#785ef0", 1: "#2a9d8f", 2: "#e9a820"}
    fig, ax = plt.subplots()
    mt.pl.cell_ecm_enrichment_bars(
        a,
        ax=ax,
        cluster_colors=colours,
        cluster_labels={0: "Alveolar", 1: "Bronchovascular wall", 2: "Peribronchovascular"},
        show=False,
    )
    label_colours = [matplotlib.colors.to_hex(t.get_color()) for t in ax.get_xticklabels()]
    assert label_colours == ["#785ef0", "#2a9d8f", "#e9a820"]
