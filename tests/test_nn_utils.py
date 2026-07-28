"""Tests for mantpy.nn._utils (shared seed + accelerator helpers)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from mantpy.nn._utils import resolve_accelerator, seed_everything


class TestResolveAccelerator:
    def test_auto_returns_cuda_or_cpu(self):
        out = resolve_accelerator("auto")
        assert out in ("cpu", "cuda")
        assert out == ("cuda" if torch.cuda.is_available() else "cpu")

    def test_none_is_treated_as_auto(self):
        assert resolve_accelerator(None) == resolve_accelerator("auto")

    def test_cpu_is_forced(self):
        assert resolve_accelerator("cpu") == "cpu"

    def test_gpu_alias_maps_to_cuda(self):
        assert resolve_accelerator("gpu") == "cuda"
        assert resolve_accelerator("cuda") == "cuda"

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match="Unknown accelerator"):
            resolve_accelerator("tpu")


class TestSeedEverything:
    def test_torch_rng_is_reproducible(self):
        seed_everything(7, device="cpu")
        a = torch.randn(5)
        seed_everything(7, device="cpu")
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_numpy_rng_is_reproducible(self):
        seed_everything(11, device="cpu")
        a = np.random.rand(5)
        seed_everything(11, device="cpu")
        b = np.random.rand(5)
        assert np.array_equal(a, b)

    def test_different_seeds_diverge(self):
        seed_everything(1, device="cpu")
        a = torch.randn(5)
        seed_everything(2, device="cpu")
        b = torch.randn(5)
        assert not torch.equal(a, b)

    def test_none_device_falls_back_to_auto(self):
        # Should not raise — device=None means "auto".
        seed_everything(0, device=None)
