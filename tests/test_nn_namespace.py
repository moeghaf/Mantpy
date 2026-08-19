"""Tests for the stable, lazy ``mantpy.nn`` public namespace."""

from __future__ import annotations

import builtins
import importlib
from types import SimpleNamespace

import pytest

import mantpy.nn as nn

EXPECTED_PUBLIC_API = [
    "NeighbourCompositionBaseline",
    "PatchEncoder",
    "encode_patches",
    "GraphMAE",
    "encode_graphmae",
    "ECMClusterGraphBundle",
    "NodeClassifier",
    "build_ohe_cluster_graphs",
]


def test_all_is_fixed_before_optional_models_are_imported() -> None:
    assert nn.__all__ == EXPECTED_PUBLIC_API
    assert set(EXPECTED_PUBLIC_API) <= set(dir(nn))


def test_importing_namespace_does_not_import_optional_model_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    optional_modules = {module_name for module_name, _extra in nn._LAZY_IMPORTS.values()}
    real_import = builtins.__import__

    def reject_optional_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in optional_modules:
            raise AssertionError(f"eagerly imported optional module {name}")
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as namespace_patch:
        for symbol in nn._LAZY_IMPORTS:
            namespace_patch.delitem(nn.__dict__, symbol, raising=False)
        namespace_patch.setattr(builtins, "__import__", reject_optional_import)
        reloaded = importlib.reload(nn)

        assert reloaded.__all__ == EXPECTED_PUBLIC_API
        assert not set(nn._LAZY_IMPORTS) & reloaded.__dict__.keys()


def test_optional_symbol_is_imported_once_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = object()
    missing = object()
    previous = nn.__dict__.pop("GraphMAE", missing)
    calls: list[str] = []

    def fake_import(module_name: str) -> SimpleNamespace:
        calls.append(module_name)
        return SimpleNamespace(GraphMAE=marker)

    monkeypatch.setattr(nn, "_import_module", fake_import)

    try:
        assert nn.GraphMAE is marker
        assert nn.GraphMAE is marker
        assert calls == ["mantpy.nn._graph_mae"]
    finally:
        nn.__dict__.pop("GraphMAE", None)
        if previous is not missing:
            nn.__dict__["GraphMAE"] = previous


@pytest.mark.parametrize(
    ("symbol", "extra"),
    [
        ("PatchEncoder", "patch"),
        ("GraphMAE", "gnn"),
        ("NodeClassifier", "gnn"),
    ],
)
def test_missing_optional_dependencies_report_install_extra(
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
    extra: str,
) -> None:
    def missing_dependency(_module_name: str) -> None:
        raise ModuleNotFoundError("simulated missing optional dependency")

    monkeypatch.delitem(nn.__dict__, symbol, raising=False)
    monkeypatch.setattr(nn, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=rf'pip install "mantpy\[{extra}\]"'):
        getattr(nn, symbol)


def test_unknown_symbol_raises_attribute_error() -> None:
    name = "not_a_model"
    with pytest.raises(AttributeError, match="has no attribute 'not_a_model'"):
        getattr(nn, name)
