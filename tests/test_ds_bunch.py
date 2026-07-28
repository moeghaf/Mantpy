"""Tests for :class:`mantpy.ds.Bunch` — sklearn-style cohort container."""

from __future__ import annotations

import pandas as pd

import mantpy as mt


def test_dict_and_attribute_access_agree():
    b = mt.ds.Bunch(adatas={"r1": object()}, best_k=7)
    assert b.adatas is b["adatas"]
    assert b.best_k == b["best_k"] == 7


def test_dict_subclass_isinstance():
    """Legacy ``isinstance(data, dict)`` checks still pass."""
    b = mt.ds.Bunch(adatas={})
    assert isinstance(b, dict)


def test_setattr_round_trips_to_dict():
    b = mt.ds.Bunch(adatas={})
    b.new_field = "hello"
    assert b["new_field"] == "hello"


def test_repr_summarises_known_keys():
    b = mt.ds.Bunch(adatas={"r1": 1, "r2": 2}, best_k=7, sample_meta=object())
    text = repr(b)
    assert "2 ROIs" in text
    assert "K=7" in text
    assert "sample_meta" in text


def test_summarize_silent_without_sample_meta(capsys):
    b = mt.ds.Bunch(adatas={"r1": object()}, best_k=5)
    b._summarize()
    out = capsys.readouterr().out
    assert "Loaded 1 ROIs" in out
    assert "K = 5" in out


def test_summarize_prints_sample_meta_when_no_ipython(monkeypatch, capsys):
    """When IPython display fails, summarize falls back to .to_string()."""
    sample_meta = pd.DataFrame({"group": ["Naive_KO", "Infected_WT"]})
    b = mt.ds.Bunch(adatas={"r1": object()}, best_k=7, sample_meta=sample_meta)

    # Simulate "no IPython" path
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("IPython"):
            raise ImportError("no IPython in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    b._summarize()
    out = capsys.readouterr().out
    assert "Loaded 1 ROIs" in out
    assert "Naive_KO" in out


def test_html_repr_contains_summary_strings():
    sample_meta = pd.DataFrame({"group": ["Naive_KO"]})
    b = mt.ds.Bunch(adatas={"r1": object()}, best_k=7, sample_meta=sample_meta)
    html = b._repr_html_()
    assert "1 ROIs" in html
    assert "K = 7" in html


def test_empty_bunch_repr_does_not_raise():
    b = mt.ds.Bunch()
    assert "empty" in repr(b) or "0 ROIs" in repr(b)


def _lung_bunch(**overrides):
    """A LungBunch shaped like ``fetch.load_balbc_pbs_lung``'s return."""
    rois = [f"m16263_slide2_00{i}" for i in range(1, 7)]
    fields = {
        "cells_by_sample": dict.fromkeys(rois, object()),
        "adatas_ecm": dict.fromkeys(rois, object()),
        "cohort_cells": dict.fromkeys(rois, object()),
        "cohort_ecm": dict.fromkeys(rois, object()),
        "roi_names": rois,
        "K_ecm": 3,
        "target_celltype": "AEC",
        # The manifest for the whole experiment, not the cohort — see LungBunch.
        "metadata": pd.DataFrame({"sample_id": ["m16263_slide1_001"], "strain": ["C57BL6J"]}),
    }
    fields.update(overrides)
    return mt.ds.LungBunch(**fields)


def test_lung_bunch_reports_its_own_schema():
    """The lung bundle names its ROI dict ``cohort_cells``, not ``adatas``."""
    b = _lung_bunch()
    assert "6 ROIs" in repr(b)
    assert "K=3" in repr(b)
    assert "6 ROIs" in b._repr_html_()
    assert "K = 3" in b._repr_html_()


def test_lung_bunch_does_not_render_the_experiment_manifest():
    """``metadata`` spans three slides; showing its head() would print
    C57BL6J rows inside a BALB/c loader."""
    b = _lung_bunch()
    assert "C57BL6J" not in b._repr_html_()


def test_lung_bunch_summarize_prints_the_repr(capsys):
    """`_summarize` is the printed form of the repr, not a second format."""
    b = _measured_bunch()
    b._summarize()
    assert capsys.readouterr().out.rstrip("\n") == repr(b)


def test_lung_bunch_falls_back_when_it_cannot_measure(capsys):
    """The `object()` placeholders here have no .n_obs; the summary must
    degrade to the plain Bunch repr rather than raise."""
    _lung_bunch()._summarize()
    assert "6 ROIs" in capsys.readouterr().out


def test_schema_spec_does_not_leak_into_bundle_data():
    """``__init__`` aliases ``__dict__`` to self, so the spec must be class-level."""
    b = _lung_bunch()
    assert len(b) == 8
    for attr in ("_ROI_KEYS", "_K_KEYS", "_META_KEYS"):
        assert attr not in b


def test_repr_elides_keys_beyond_the_cap():
    b = _lung_bunch(**{f"extra_{i}": i for i in range(10)})
    assert "more" in repr(b)


def test_data_key_named_get_does_not_shadow_lookup():
    """Lookup goes through ``dict.get``, not ``self.get``."""
    b = _lung_bunch(get="not-a-method")
    assert "6 ROIs" in repr(b)


# ---------------------------------------------------------------- summary ----


def _measured_bunch():
    """A LungBunch with real AnnData, so the summary has something to measure."""
    import anndata as ad
    import numpy as np

    rois = ["r1", "r2"]
    cells, ecm, art = {}, {}, {}
    for r in rois:
        c = ad.AnnData(np.zeros((10, 0), dtype=np.float32))
        c.obs["celltype"] = ["AEC"] * 6 + ["ATII"] * 4
        cells[r] = c
        e = ad.AnnData(np.zeros((0, 39), dtype=np.float32))
        e.uns["strain"], e.uns["condition"] = "BALB/c", "PBS"
        # 4 patches, 1 of them background (-1) -> 25%
        e.uns["ecm_patches"] = pd.DataFrame({"x": [1.0, 2, 3, 4], "y": [1.0, 2, 3, 4], "ecm_cluster": [-1, 0, 1, 2]})
        ecm[r] = e
        a = ad.AnnData(np.zeros((0, 39), dtype=np.float32))
        a.uns["ecm_patches"] = pd.DataFrame({"is_artifact": [True, False, True, False]})
        art[r] = a
    return mt.ds.LungBunch(
        cohort_cells=cells, cohort_ecm=ecm, adatas_ecm_artifact=art,
        roi_names=rois, K_ecm=3, ecm_names=[f"m{i}" for i in range(10)],
        tissue="mouse lung", technology="IMC",
        metadata=pd.DataFrame({"sample_id": rois + ["other"], "Mouse": [31, 34, 99], "Slide": [2, 2, 1]}),
    )


def test_summary_measures_the_bundle():
    text = repr(_measured_bunch())
    assert "mouse lung · IMC · BALB/c · PBS" in text
    assert "2 ROIs from 2 mice (slide 2)" in text     # 'other'/Mouse 99 excluded
    assert "cells     20" in text
    assert "2 types (obs['celltype'])" in text
    assert "8 patches" in text                        # 2 ROIs x 4
    assert "10 of 39 markers" in text
    assert "K = 3" in text
    assert "25% background" in text
    assert "4 patches flagged" in text


def test_summary_reports_no_strain_when_rois_disagree():
    """A mixed cohort must not silently report one ROI's strain as the whole."""
    b = _measured_bunch()
    b["cohort_ecm"]["r2"].uns["strain"] = "C57BL6J"
    assert "BALB/c" not in repr(b)
    assert "PBS" in repr(b)          # condition still agrees


def test_summary_survives_a_partial_bundle():
    import anndata as ad
    import numpy as np

    b = mt.ds.LungBunch(cohort_cells={"r1": ad.AnnData(np.zeros((5, 0), dtype=np.float32))})
    assert "1 ROI" in repr(b) and "1 ROIs" not in repr(b)
    assert repr(mt.ds.LungBunch())          # empty must not raise


def test_summary_never_raises_on_malformed_data():
    """A repr is not allowed to be the thing that kills a session."""
    b = mt.ds.LungBunch(cohort_cells={"r1": object()}, K_ecm=3)
    assert isinstance(repr(b), str)
