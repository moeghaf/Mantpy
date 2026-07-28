"""Public registry and fail-closed behavior for the Schistosoma loader."""

from __future__ import annotations

import pytest

from mantpy import fetch
from mantpy.datasets import _registry


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"verbose": False},
        {"include_raw": True},
    ],
)
def test_unpublished_registry_fails_closed(monkeypatch, kwargs):
    incomplete = dict(_registry.DATASET_REGISTRY)
    incomplete["record_id"] = None
    monkeypatch.setattr(_registry, "DATASET_REGISTRY", incomplete)

    with pytest.raises(RuntimeError, match="has not been published yet"):
        fetch.load_schistosoma_ecm_cohort(**kwargs)


@pytest.mark.parametrize("kwargs", [{"zenodo_dir": "missing"}, {"raw_dir": "missing"}])
def test_missing_local_override_is_reported(kwargs):
    with pytest.raises(FileNotFoundError, match="Dataset directory does not exist"):
        fetch.load_schistosoma_ecm_cohort(**kwargs)


def test_fetch_module_exposes_no_cohort_specific_remote_configuration():
    assert not hasattr(fetch, "_SCHISTOSOMA_ECM_RECORD")
    assert not hasattr(fetch, "_SCHISTOSOMA_ECM_FILES")
