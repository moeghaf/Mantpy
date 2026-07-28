"""Optional neural network models for Mantpy.

Models are imported lazily so the base package remains lightweight.

Available models
----------------
- :class:`GraphMAE` — node-level masked-feature graph autoencoder (niche detection)
- :class:`NodeClassifier` — node-level classifier
- :class:`PatchEncoder` — self-supervised CNN encoder for image patches
"""

from importlib import import_module as _import_module
from typing import Any as _Any

# Lightweight baselines use only Mantpy's required dependencies and remain
# eagerly available. Neural models are resolved below only when requested.
from mantpy.nn._baselines import NeighbourCompositionBaseline

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "PatchEncoder": ("mantpy.nn._patch_encoder", "patch"),
    "encode_patches": ("mantpy.nn._patch_encoder", "patch"),
    "GraphMAE": ("mantpy.nn._graph_mae", "gnn"),
    "encode_graphmae": ("mantpy.nn._graph_mae", "gnn"),
    "ECMClusterGraphBundle": ("mantpy.nn._node_classifier", "gnn"),
    "NodeClassifier": ("mantpy.nn._node_classifier", "gnn"),
    "build_ohe_cluster_graphs": ("mantpy.nn._node_classifier", "gnn"),
}

# Keep the public contract fixed across base, [patch], and [gnn] installs.
__all__ = [
    "NeighbourCompositionBaseline",
    *_LAZY_IMPORTS,
]


def __getattr__(name: str) -> _Any:
    """Resolve an optional neural symbol on first access."""
    try:
        module_name, extra = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    try:
        value = getattr(_import_module(module_name), name)
    except ImportError as exc:
        raise ImportError(
            f"mantpy.nn.{name} requires optional dependencies from mantpy[{extra}]. "
            f'Install them with: pip install "mantpy[{extra}]"'
        ) from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List public lazy exports alongside the populated module namespace."""
    return sorted(set(globals()) | set(__all__))
