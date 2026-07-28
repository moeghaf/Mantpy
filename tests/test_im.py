"""Tests for mantpy.im.ImageContainer."""

from __future__ import annotations

import numpy as np
import pytest

import mantpy as mt
from mantpy.im import ImageContainer


@pytest.fixture
def arr_5c():
    """(C=5, H=64, W=64) float32 array."""
    rng = np.random.default_rng(0)
    return rng.random((5, 64, 64), dtype=np.float32)


@pytest.fixture
def ic(arr_5c):
    return ImageContainer(arr_5c)


class TestInit:
    def test_from_ndarray(self, arr_5c):
        ic = ImageContainer(arr_5c)
        assert ic.shape == (5, 64, 64)

    def test_from_2d_array(self):
        arr = np.zeros((32, 32), dtype=np.float32)
        ic = ImageContainer(arr)
        assert ic.shape == (1, 32, 32)

    def test_bad_ndim_raises(self):
        with pytest.raises(ValueError, match="C, H, W"):
            ImageContainer(np.zeros((2, 3, 4, 5)))

    def test_channel_names(self, arr_5c):
        names = ["ColIV", "FN", "CD20", "EpCAM", "DAPI"]
        ic = ImageContainer(arr_5c, channel_names=names)
        assert ic.channel_names == names

    def test_channel_names_mismatch_raises(self, arr_5c):
        with pytest.raises(ValueError, match="channel_names"):
            ImageContainer(arr_5c, channel_names=["A", "B"])

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            ImageContainer("nonexistent_file.tif")

    def test_repr(self, ic):
        r = repr(ic)
        assert "ImageContainer" in r
        assert "5, 64, 64" in r


class TestProperties:
    def test_shape(self, ic):
        assert ic.shape == (5, 64, 64)

    def test_n_channels(self, ic):
        assert ic.n_channels == 5

    def test_height(self, ic):
        assert ic.height == 64

    def test_width(self, ic):
        assert ic.width == 64


class TestToArray:
    def test_returns_ndarray(self, ic):
        arr = ic.to_array()
        assert isinstance(arr, np.ndarray)

    def test_shape_preserved(self, ic):
        assert ic.to_array().shape == (5, 64, 64)

    def test_dtype(self, ic):
        assert ic.to_array(dtype=np.float32).dtype == np.float32

    def test_values_match_input(self, arr_5c):
        ic = ImageContainer(arr_5c)
        np.testing.assert_array_almost_equal(ic.to_array(), arr_5c)


class TestChannelSubset:
    def test_single_channel(self, ic):
        sub = ic[0]
        assert sub.shape == (1, 64, 64)

    def test_multi_channel(self, ic):
        sub = ic[[0, 2, 4]]
        assert sub.shape == (3, 64, 64)

    def test_returns_imagecontainer(self, ic):
        assert isinstance(ic[[0, 1]], ImageContainer)

    def test_channel_names_subset(self, arr_5c):
        names = ["A", "B", "C", "D", "E"]
        ic = ImageContainer(arr_5c, channel_names=names)
        sub = ic[[1, 3]]
        assert sub.channel_names == ["B", "D"]


class TestCrop:
    def test_basic_crop(self, ic):
        tile = ic.crop(x=10, y=10, w=20, h=20)
        assert tile.shape == (5, 20, 20)

    def test_returns_imagecontainer(self, ic):
        assert isinstance(ic.crop(x=0, y=0, w=32, h=32), ImageContainer)

    def test_crop_clamps_to_bounds(self, ic):
        # request region that extends beyond the image boundary
        tile = ic.crop(x=50, y=50, w=100, h=100)
        assert tile.width == 14  # 64 - 50
        assert tile.height == 14

    def test_full_crop_equals_original(self, ic, arr_5c):
        cropped = ic.crop(x=0, y=0, w=ic.width, h=ic.height)
        np.testing.assert_array_equal(cropped.to_array(), arr_5c)


class TestTileIter:
    def test_yields_tuples(self, ic):
        items = list(ic.tile_iter(tile_size=32))
        assert len(items) > 0
        tile, info = items[0]
        assert isinstance(tile, ImageContainer)
        assert isinstance(info, dict)

    def test_info_keys(self, ic):
        _, info = next(ic.tile_iter(tile_size=32))
        for key in ("x0", "y0", "x1", "y1", "xc0", "yc0"):
            assert key in info

    def test_tile_count_no_overlap(self, ic):
        # 64×64 image with 32×32 tiles → 4 tiles
        tiles = list(ic.tile_iter(tile_size=32))
        assert len(tiles) == 4

    def test_tile_count_uneven(self):
        ic = ImageContainer(np.zeros((1, 70, 70), dtype=np.float32))
        # 70 / 32 → 3 steps: 0,32,64 → 3×3 = 9 tiles
        tiles = list(ic.tile_iter(tile_size=32))
        assert len(tiles) == 9

    def test_overlap_enlarges_tile(self, ic):
        tile, _ = next(ic.tile_iter(tile_size=32, overlap=8))
        # first tile (0..32) padded left/top by 0 (edge), right/bottom by 8
        # → w = 32 + 8 = 40 (clamped to 64 if needed)
        assert tile.width >= 32
        assert tile.height >= 32

    def test_channels_preserved(self, ic):
        tile, _ = next(ic.tile_iter(tile_size=32))
        assert tile.n_channels == ic.n_channels


class TestMaxProjection:
    def test_returns_2d(self, ic):
        proj = ic.max_projection()
        assert proj.ndim == 2
        assert proj.shape == (ic.height, ic.width)

    def test_normalised_01(self, ic):
        proj = ic.max_projection()
        assert proj.max() <= 1.0 + 1e-6
        assert proj.min() >= 0.0


class TestShow:
    def test_returns_axes(self, ic):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = ic.show(show=False)
        from matplotlib.axes import Axes

        assert isinstance(ax, Axes)
        plt.close("all")


class TestIntegration:
    def test_accessible_as_mt_im(self, arr_5c):
        ic = mt.im.ImageContainer(arr_5c)
        assert ic.shape[0] == 5

    def test_top_level_alias(self, arr_5c):
        ic = mt.ImageContainer(arr_5c)
        assert isinstance(ic, ImageContainer)
