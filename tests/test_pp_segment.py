"""Tests for mt.pp.segment_cells — uses mocks to avoid heavy model downloads."""

from unittest.mock import patch

import numpy as np
import pytest

from mantpy.im import ImageContainer


@pytest.fixture
def synthetic_nuclear():
    """64×64 nuclear image with a bright spot."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[20:40, 20:40] = 1.0
    return img


@pytest.fixture
def synthetic_multichannel():
    """3-channel (C, H, W) image."""
    rng = np.random.default_rng(42)
    return rng.random((3, 64, 64)).astype(np.float32)


@pytest.fixture
def mock_cellpose_predict():
    """Patch Cellpose to return a synthetic mask."""
    fake_mask = np.zeros((64, 64), dtype=np.int32)
    fake_mask[20:40, 20:40] = 1
    fake_mask[45:55, 45:55] = 2

    with patch("mantpy._core._segmentation.run_cellpose") as mock_run:
        mock_run.return_value = fake_mask
        yield mock_run


class TestSegmentCellsCellpose:
    def test_output_shape(self, synthetic_multichannel, mock_cellpose_predict):
        from mantpy.pp import segment_cells

        mask = segment_cells(synthetic_multichannel, nuclear_channel=0)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.int32

    def test_single_channel_input(self, synthetic_nuclear, mock_cellpose_predict):
        from mantpy.pp import segment_cells

        mask = segment_cells(synthetic_nuclear, nuclear_channel=0)
        assert mask.shape == (64, 64)

    def test_image_container_input(self, synthetic_multichannel, mock_cellpose_predict):
        from mantpy.pp import segment_cells

        names = ["Ir191_DNA", "Ir193_DNA", "Tm169_Collagen"]
        ic = ImageContainer(synthetic_multichannel, channel_names=names)
        mask = segment_cells(ic, nuclear_channel="Ir191_DNA")
        assert mask.shape == (64, 64)

    def test_string_channel_not_found_raises(self, synthetic_multichannel, mock_cellpose_predict):
        from mantpy.pp import segment_cells

        names = ["Ir191_DNA", "Ir193_DNA", "Tm169_Collagen"]
        ic = ImageContainer(synthetic_multichannel, channel_names=names)
        with pytest.raises(ValueError, match="not found"):
            segment_cells(ic, nuclear_channel="DAPI")

    def test_string_channel_without_names_raises(self, synthetic_multichannel, mock_cellpose_predict):
        from mantpy.pp import segment_cells

        with pytest.raises(ValueError, match="channel_names"):
            segment_cells(synthetic_multichannel, nuclear_channel="Ir191_DNA")

    def test_store_in_image_container(self, synthetic_multichannel, mock_cellpose_predict):
        from mantpy.pp import segment_cells

        ic = ImageContainer(synthetic_multichannel)
        segment_cells(ic, nuclear_channel=0, store_in=ic, layer_name="cell_mask")
        assert ic.has_layer("cell_mask")
        stored = ic.get_layer("cell_mask")
        assert stored.shape[1:] == (64, 64)

    def test_invalid_backend_raises(self, synthetic_multichannel, mock_cellpose_predict):
        from mantpy.pp import segment_cells

        with pytest.raises(ValueError, match="Unknown backend"):
            segment_cells(synthetic_multichannel, nuclear_channel=0, backend="invalid")

    def test_membrane_channel_used(self, synthetic_multichannel, mock_cellpose_predict):
        from mantpy.pp import segment_cells

        names = ["Ir191_DNA", "Ir193_DNA", "CD45"]
        ic = ImageContainer(synthetic_multichannel, channel_names=names)
        mask = segment_cells(ic, nuclear_channel="Ir191_DNA", membrane_channel="CD45")
        assert mask.shape == (64, 64)

    def test_mask_has_cell_labels(self, synthetic_multichannel, mock_cellpose_predict):
        from mantpy.pp import segment_cells

        mask = segment_cells(synthetic_multichannel, nuclear_channel=0)
        assert mask.max() == 2
        assert mask.min() == 0

    def test_gpu_and_threshold_params_passed(self, synthetic_multichannel, mock_cellpose_predict):
        from mantpy.pp import segment_cells

        segment_cells(
            synthetic_multichannel,
            nuclear_channel=0,
            gpu=False,
            cellprob_threshold=-1.0,
            flow_threshold=0.8,
        )
        _, kwargs = mock_cellpose_predict.call_args
        assert kwargs["gpu"] is False
        assert kwargs["cellprob_threshold"] == -1.0
        assert kwargs["flow_threshold"] == 0.8


# -----------------------------------------------------------------------------
# segment_cells_tiled — Cellpose mock so tests don't download / run a real model.
# -----------------------------------------------------------------------------


def _make_fake_cellpose_model(label_offset_per_call: bool = True):
    """Return a callable that mimics ``CellposeModel`` for the tiled test path.

    Each call to ``model.eval(tiles, ...)`` returns one tile-local mask per
    input tile. To exercise the global-relabel logic we hand back labels
    starting at 1 in every tile.
    """
    state = {"calls": 0}

    class _FakeModel:
        def __init__(self, gpu=True):
            pass

        def eval(self, tiles, diameter=None, cellprob_threshold=0.0, flow_threshold=0.4):
            state["calls"] += 1
            masks = []
            for t in tiles:
                # 2 cells per tile, simple square labels
                m = np.zeros(t.shape, dtype=np.int32)
                h, w = m.shape
                m[h // 4 : h // 2, w // 4 : w // 2] = 1
                m[h // 2 : 3 * h // 4, w // 2 : 3 * w // 4] = 2
                masks.append(m)
            return masks, None, None

    return _FakeModel, state


class TestSegmentCellsTiled:
    @pytest.fixture(autouse=True)
    def _require_cellpose(self):
        # The tiled path patches ``cellpose.models.CellposeModel`` directly,
        # which imports cellpose; skip when the optional dep is absent.
        pytest.importorskip("cellpose.models")

    def test_returns_globally_unique_labels(self, monkeypatch):
        from mantpy.pp import segment_cells_tiled

        FakeModel, _ = _make_fake_cellpose_model()
        monkeypatch.setattr("cellpose.models.CellposeModel", FakeModel)

        # 256x256 with bright signal so background-skip doesn't drop tiles
        img = np.full((256, 256), 100.0, dtype=np.float32)
        mask = segment_cells_tiled(
            img,
            nuclear_channel=0,
            tile_size=128,
            overlap=16,
            batch_size=1,
            gpu=False,
        )
        assert mask.shape == img.shape
        assert mask.dtype == np.int32
        # > 2 distinct labels means tiles got globally relabeled, not all collapsed to {1, 2}
        assert mask.max() > 2

    def test_cache_round_trip_skips_cellpose(self, tmp_path, monkeypatch):
        from mantpy.pp import segment_cells_tiled

        FakeModel, state = _make_fake_cellpose_model()
        monkeypatch.setattr("cellpose.models.CellposeModel", FakeModel)

        img = np.full((128, 128), 100.0, dtype=np.float32)
        cache = tmp_path / "mask.npz"

        # First call: model runs, cache file is written.
        mask1 = segment_cells_tiled(
            img,
            tile_size=64,
            overlap=16,
            batch_size=1,
            gpu=False,
            cache=cache,
        )
        assert cache.exists()
        first_calls = state["calls"]
        assert first_calls > 0

        # Second call: cache hit, model NOT invoked again.
        mask2 = segment_cells_tiled(
            img,
            tile_size=64,
            overlap=16,
            batch_size=1,
            gpu=False,
            cache=cache,
        )
        np.testing.assert_array_equal(mask1, mask2)
        assert state["calls"] == first_calls  # no new calls

    def test_accepts_tiff_path(self, tmp_path, monkeypatch):
        import tifffile

        from mantpy.pp import segment_cells_tiled

        FakeModel, _ = _make_fake_cellpose_model()
        monkeypatch.setattr("cellpose.models.CellposeModel", FakeModel)

        # 2-channel TIFF; channel 0 = DAPI
        stack = np.full((2, 128, 128), 100.0, dtype=np.float32)
        path = tmp_path / "stack.tif"
        with tifffile.TiffWriter(str(path)) as w:
            for c in range(stack.shape[0]):
                w.write(stack[c])

        mask = segment_cells_tiled(
            path,
            nuclear_channel=0,
            tile_size=64,
            overlap=16,
            batch_size=1,
            gpu=False,
        )
        assert mask.shape == (128, 128)
        assert mask.max() >= 1
