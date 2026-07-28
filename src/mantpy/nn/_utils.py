"""Internal shared training utilities for ``mantpy.nn`` models.

Centralises seed and device handling so every model in :mod:`mantpy.nn`
behaves identically:

* :func:`seed_everything` — full Python/NumPy/PyTorch/CUDA/cuBLAS/cuDNN seed
  pin for end-to-end reproducibility across optional neural models.
* :func:`resolve_accelerator` — one string ↔ device resolver
  (``"auto"`` / ``"cpu"`` / ``"cuda"`` / ``"gpu"``) so user-facing kwargs are
  consistent across optional neural models.
"""

from __future__ import annotations

import os
import random
from importlib import metadata

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError('mantpy.nn utilities require PyTorch. Install with: pip install "mantpy[gnn]"') from e


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:  # pragma: no cover - editable edge case
        return "unknown"


def seed_everything(seed: int, *, device: str | None = None) -> None:
    """Seed Python / NumPy / PyTorch (+ CUDA + cuBLAS + cuDNN) for end-to-end reproducibility.

    Pins ``PYTHONHASHSEED``, ``CUBLAS_WORKSPACE_CONFIG`` (required for
    deterministic matmul on CUDA 10.2+), and switches cuDNN to its
    deterministic mode. Falls back to ``warn_only=True`` for
    :func:`torch.use_deterministic_algorithms` so any op without a
    deterministic implementation logs a warning instead of crashing
    training.

    Parameters
    ----------
    seed
        The integer seed.
    device
        ``"cuda"`` / ``"cpu"`` / ``"gpu"`` / ``"auto"`` / ``None``. When the
        resolved device is CUDA, the CUDA-side seeds and cuBLAS pin are
        applied too. ``None`` is treated as ``"auto"``.
    """
    resolved = resolve_accelerator(device) if device is not None else resolve_accelerator("auto")

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if resolved == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (RuntimeError, AttributeError):
        pass


def resolve_accelerator(accelerator: str | None) -> str:
    """Resolve a user-facing accelerator string to ``"cpu"`` or ``"cuda"``.

    Accepts the same vocabulary as PyTorch Lightning's ``Trainer`` flag:

    * ``"auto"`` (default), ``None`` — CUDA if available, else CPU.
    * ``"cuda"`` or ``"gpu"`` — force CUDA.
    * ``"cpu"`` — force CPU.

    Returns
    -------
    str
        Either ``"cpu"`` or ``"cuda"``. Use this directly with
        :meth:`torch.nn.Module.to`.
    """
    if accelerator is None or accelerator == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    a = accelerator.lower()
    if a in ("cuda", "gpu"):
        return "cuda"
    if a == "cpu":
        return "cpu"
    raise ValueError(f"Unknown accelerator '{accelerator}'. Expected one of: 'auto', 'cpu', 'cuda', 'gpu'.")
