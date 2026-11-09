"""Tests for mt.nn.PatchEncoder."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_LIGHTNING = (
    importlib.util.find_spec("pytorch_lightning") is not None or importlib.util.find_spec("lightning") is not None
)

pytestmark = pytest.mark.skipif(not (HAS_TORCH and HAS_LIGHTNING), reason="requires mantpy[patch]")


@pytest.fixture
def patches():
    rng = np.random.default_rng(0)
    return rng.random((64, 1, 16, 16)).astype("float32")


def test_fit_transform_shape(patches):
    from mantpy.nn import PatchEncoder

    enc = PatchEncoder(latent_dim=8).fit(patches, max_epochs=2, accelerator="cpu", random_state=0)
    emb = enc.transform(patches)
    assert emb.shape == (64, 8)
    assert np.isfinite(emb).all()


def test_reproducible_with_seed(patches):
    from mantpy.nn import PatchEncoder

    a = PatchEncoder(latent_dim=4).fit(patches, max_epochs=2, accelerator="cpu", random_state=0).transform(patches)
    b = PatchEncoder(latent_dim=4).fit(patches, max_epochs=2, accelerator="cpu", random_state=0).transform(patches)
    np.testing.assert_allclose(a, b)


def test_autoencoder_objective(patches):
    from mantpy.nn import PatchEncoder

    enc = PatchEncoder(latent_dim=4, objective="autoencoder").fit(
        patches,
        max_epochs=2,
        accelerator="cpu",
        random_state=0,
    )
    assert enc.transform(patches).shape == (64, 4)


def test_invalid_latent_dim_raises():
    from mantpy.nn import PatchEncoder

    with pytest.raises(ValueError):
        PatchEncoder(latent_dim=0)


def test_invalid_objective_raises():
    from mantpy.nn import PatchEncoder

    with pytest.raises(ValueError):
        PatchEncoder(objective="pca")


def test_repr_reports_fitted_state(patches):
    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(latent_dim=4)
    assert "status='unfitted'" in repr(encoder)
    encoder.fit(
        patches,
        max_epochs=1,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
    )
    text = repr(encoder)
    assert "status='fitted'" in text
    assert "input_shape=(1, 16, 16)" in text


def test_non_4d_patches_raise(patches):
    from mantpy.nn import PatchEncoder

    with pytest.raises(ValueError):
        PatchEncoder(latent_dim=4).fit(patches[:, 0], max_epochs=1, accelerator="cpu")  # (N, P, P), no channel axis


def test_transform_before_fit_raises(patches):
    from mantpy.nn import PatchEncoder

    with pytest.raises(RuntimeError):
        PatchEncoder().transform(patches)


def test_fits_on_multiple_ecm_channels():
    from mantpy.nn import PatchEncoder

    rng = np.random.default_rng(0)
    multichannel = rng.random((32, 3, 8, 8), dtype=np.float32)
    embedding = (
        PatchEncoder(latent_dim=5)
        .fit(multichannel, max_epochs=1, batch_size=32, accelerator="cpu", random_state=0)
        .transform(multichannel)
    )

    assert embedding.shape == (32, 5)
    assert np.isfinite(embedding).all()


def test_fit_transform_is_standardised_and_supports_single_patch_inference(patches):
    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(latent_dim=4)
    embedding = encoder.fit_transform(
        patches,
        max_epochs=1,
        batch_size=32,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
    )

    assert isinstance(embedding, np.ndarray)
    np.testing.assert_allclose(embedding.mean(axis=0), 0, atol=2e-5)
    assert encoder.transform(patches[:1]).shape == (1, 4)
    assert len(encoder.history["train_loss"]) == 1


def test_encode_patches_pools_anndata_and_records_provenance(tmp_path):
    import anndata
    from anndata import AnnData
    from packaging.version import Version

    from mantpy.nn import PatchEncoder, encode_patches

    rng = np.random.default_rng(1)
    left = AnnData(np.zeros((12, 0), dtype=np.float32))
    right = AnnData(np.zeros((10, 0), dtype=np.float32))
    left.obsm["image_patches"] = rng.random((12, 2, 8, 8), dtype=np.float32)
    right.obsm["image_patches"] = rng.random((10, 2, 8, 8), dtype=np.float32)
    left.obs["held_out_label"] = "not-read"

    encoder = encode_patches(
        [left, right],
        latent_dim=5,
        max_epochs=1,
        batch_size=16,
        accelerator="cpu",
        random_state=3,
        enable_progress_bar=False,
    )

    assert isinstance(encoder, PatchEncoder)
    assert left.obsm["X_cnn"].shape == (12, 5)
    assert right.obsm["X_cnn"].shape == (10, 5)
    params = left.uns["X_cnn_params"]
    assert params["objective"] == "contrastive"
    assert params["architecture"]["input_shape"] == [2, 8, 8]
    assert params["training"]["n_samples"] == 2
    assert params["training"]["n_patches"] == 22
    assert params["training"]["accelerator"] == "cpu"
    assert params["training"]["resolved_device"] == "cpu"
    assert params["standardization"] == "pooled StandardScaler"
    assert set(params["software"]) == {"mantpy", "torch", "pytorch_lightning", "scikit_learn"}
    assert all(isinstance(version, str) for version in params["software"].values())
    assert "held_out_label" not in str(params)
    # The project requires anndata>=0.12 for Arrow-string serialization, but
    # keep this focused test runnable in older local development environments.
    if Version(anndata.__version__) >= Version("0.12"):
        left.write_h5ad(tmp_path / "encoded.h5ad")


def test_save_load_roundtrip_preserves_features(tmp_path, patches):
    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(latent_dim=6).fit(
        patches,
        max_epochs=1,
        batch_size=32,
        accelerator="cpu",
        random_state=2,
        enable_progress_bar=False,
    )
    expected = encoder.transform(patches[:5])
    encoder.save(tmp_path / "patch_encoder")
    restored = PatchEncoder.load(tmp_path / "patch_encoder")

    assert restored.is_trained
    np.testing.assert_allclose(restored.transform(patches[:5]), expected, rtol=1e-6, atol=1e-6)


def test_pooled_fit_rejects_mismatched_patch_shapes(patches):
    from mantpy.nn import PatchEncoder

    with pytest.raises(ValueError, match="expected shared channel and spatial shape"):
        PatchEncoder().fit(
            [patches, np.ones((4, 2, 16, 16), dtype=np.float32)],
            max_epochs=1,
            accelerator="cpu",
        )


def test_fit_handles_single_patch_tail_batch():
    from mantpy.nn import PatchEncoder

    patches = np.random.default_rng(4).random((33, 1, 8, 8), dtype=np.float32)
    encoder = PatchEncoder(latent_dim=4).fit(
        patches,
        max_epochs=1,
        batch_size=32,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
    )

    assert encoder.transform(patches).shape == (33, 4)


# ---------------------------------------------------------------------------
# Shape options of the built-in encoder: conv_blocks, conv_width, pool_size,
# conv_activation and pool_type.
# ---------------------------------------------------------------------------

# Encoder selected for the liver analysis (10 x 10 patches of six ECM markers).
LIVER_OPTIONS = {
    "latent_dim": 96,
    "conv_blocks": 2,
    "conv_width": 32,
    "pool_size": 2,
    "conv_activation": "gelu",
    "pool_type": "max",
}
SHAPE_OPTION_NAMES = ("conv_blocks", "conv_width", "pool_size", "conv_activation", "pool_type")
SHAPE_OPTION_DEFAULTS = {
    "conv_blocks": None,
    "conv_width": 16,
    "pool_size": None,
    "conv_activation": "relu",
    "pool_type": "avg",
}


@pytest.fixture
def liver_patches():
    rng = np.random.default_rng(0)
    return rng.random((64, 6, 10, 10)).astype("float32")


def _original_conv_encoder(in_size: int, in_ch: int, feat: int):
    """The built-in encoder exactly as it was before the shape options existed."""
    import torch.nn as nn

    class _OriginalConvEnc(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            ch, size, depth = in_ch, in_size, 0
            while size > 4:
                out = min(16 * 2**depth, 128)
                layers += [nn.Conv2d(ch, out, 3, 2, 1), nn.BatchNorm2d(out), nn.ReLU()]
                ch, size, depth = out, (size + 1) // 2, depth + 1
            self.conv = nn.Sequential(*layers)
            self.lin = nn.Linear(ch * size * size, feat)

        def forward(self, x):
            return self.lin(self.conv(x).flatten(1))

    return _OriginalConvEnc()


def _conv_layers(encoder):
    import torch.nn as nn

    return [module for module in encoder.conv if isinstance(module, nn.Conv2d)]


def _assert_same_state(built, expected):
    import torch

    built_state, expected_state = built.state_dict(), expected.state_dict()
    assert list(built_state) == list(expected_state)
    for key, value in expected_state.items():
        assert built_state[key].shape == value.shape, key
        assert torch.equal(built_state[key], value), key


def test_shape_options_default_to_the_original_architecture():
    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder()
    assert {name: getattr(encoder, name) for name in SHAPE_OPTION_NAMES} == SHAPE_OPTION_DEFAULTS
    assert encoder._encoder_options == {
        "blocks": None,
        "width": 16,
        "pool_size": None,
        "activation": "relu",
        "pool_type": "avg",
    }


@pytest.mark.parametrize(
    ("in_size", "in_channels"),
    [(4, 2), (8, 3), (10, 6), (16, 1), (33, 2), (128, 1)],
)
def test_default_options_build_the_original_encoder(in_size, in_channels):
    import torch

    from mantpy.nn import PatchEncoder
    from mantpy.nn._patch_encoder import _ConvEnc

    torch.manual_seed(0)
    expected = _original_conv_encoder(in_size, in_channels, 8)
    torch.manual_seed(0)
    bare = _ConvEnc(in_size, in_channels, 8)
    torch.manual_seed(0)
    built = _ConvEnc(in_size, in_channels, 8, **PatchEncoder(latent_dim=8)._encoder_options)

    batch = torch.rand(5, in_channels, in_size, in_size)
    expected.eval()
    for candidate in (bare, built):
        assert candidate.pool is None
        assert [type(module) for module in candidate.conv] == [type(module) for module in expected.conv]
        # Same state-dict keys, shapes and, for a fixed seed, identical weights.
        _assert_same_state(candidate, expected)
        candidate.eval()
        with torch.no_grad():
            assert torch.equal(candidate(batch), expected(batch))


def test_default_fit_matches_the_original_encoder_as_backbone(patches):
    from mantpy.nn import PatchEncoder

    fit_kwargs = {
        "max_epochs": 2,
        "batch_size": 32,
        "accelerator": "cpu",
        "random_state": 0,
        "enable_progress_bar": False,
    }
    default = PatchEncoder(latent_dim=8).fit(patches, **fit_kwargs)
    original = PatchEncoder(latent_dim=8, backbone=lambda: _original_conv_encoder(16, 1, 8)).fit(patches, **fit_kwargs)

    _assert_same_state(default.model, original.model)
    assert default.history == original.history
    np.testing.assert_array_equal(default.transform(patches), original.transform(patches))


@pytest.mark.parametrize("blocks", [1, 2, 3, 5])
def test_conv_blocks_sets_the_number_of_convolution_blocks(blocks):
    import torch
    import torch.nn as nn

    from mantpy.nn._patch_encoder import _ConvEnc

    encoder = _ConvEnc(16, 1, 8, blocks=blocks)
    convs = _conv_layers(encoder)
    size = 16
    for _ in range(blocks):
        size = (size + 1) // 2

    assert len(convs) == blocks
    assert len(encoder.conv) == 3 * blocks
    assert sum(isinstance(module, nn.BatchNorm2d) for module in encoder.conv) == blocks
    assert all(conv.stride == (2, 2) and conv.kernel_size == (3, 3) for conv in convs)
    assert encoder.lin.in_features == convs[-1].out_channels * size * size
    assert encoder(torch.rand(3, 1, 16, 16)).shape == (3, 8)


def test_conv_blocks_none_halves_until_four_pixels():
    from mantpy.nn._patch_encoder import _ConvEnc

    # 10 -> 5 -> 3 needs two blocks by default; one block leaves a 5 x 5 map.
    assert len(_conv_layers(_ConvEnc(10, 6, 8))) == 2
    assert _ConvEnc(10, 6, 8).lin.in_features == 32 * 3 * 3
    assert _ConvEnc(10, 6, 8, blocks=1).lin.in_features == 16 * 5 * 5


@pytest.mark.parametrize(
    ("width", "blocks", "expected"),
    [
        (16, 3, [16, 32, 64]),
        (32, 2, [32, 64]),
        (32, 4, [32, 64, 128, 128]),
        (48, 3, [48, 96, 128]),
        (8, None, [8, 16]),
        (200, 1, [128]),
    ],
)
def test_conv_width_doubles_per_block_and_is_capped_at_128(width, blocks, expected):
    import torch.nn as nn

    from mantpy.nn._patch_encoder import _ConvEnc

    encoder = _ConvEnc(16, 3, 8, blocks=blocks, width=width)
    convs = _conv_layers(encoder)
    norms = [module for module in encoder.conv if isinstance(module, nn.BatchNorm2d)]

    assert [conv.out_channels for conv in convs] == expected
    assert [conv.in_channels for conv in convs] == [3, *expected[:-1]]
    assert [norm.num_features for norm in norms] == expected


@pytest.mark.parametrize("pool_size", [1, 2, 3])
@pytest.mark.parametrize("pool_type", ["avg", "max"])
def test_pool_size_and_pool_type_set_the_pooled_readout(pool_size, pool_type):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from mantpy.nn._patch_encoder import _ConvEnc

    # Two blocks leave a 3 x 3 map of 64 channels from a 10 x 10 patch.
    encoder = _ConvEnc(10, 6, 96, blocks=2, width=32, pool_size=pool_size, pool_type=pool_type).eval()
    pool_class = nn.AdaptiveMaxPool2d if pool_type == "max" else nn.AdaptiveAvgPool2d
    functional = F.adaptive_max_pool2d if pool_type == "max" else F.adaptive_avg_pool2d

    assert type(encoder.pool) is pool_class
    assert encoder.pool.output_size == pool_size
    assert encoder.lin.in_features == 64 * pool_size * pool_size
    batch = torch.rand(7, 6, 10, 10)
    with torch.no_grad():
        feature_map = encoder.conv(batch)
        assert feature_map.shape == (7, 64, 3, 3)
        expected = encoder.lin(functional(feature_map, pool_size).flatten(1))
        assert torch.equal(encoder(batch), expected)


def test_without_pool_size_the_feature_map_is_flattened_unpooled():
    from mantpy.nn._patch_encoder import _ConvEnc

    encoder = _ConvEnc(10, 6, 96, blocks=2, width=32)
    assert encoder.pool is None
    assert encoder.lin.in_features == 64 * 3 * 3
    # Pooling adds no parameters, so the state-dict keys do not depend on it.
    pooled = _ConvEnc(10, 6, 96, blocks=2, width=32, pool_size=2, pool_type="max")
    assert list(pooled.state_dict()) == list(encoder.state_dict())


@pytest.mark.parametrize("activation", ["relu", "gelu"])
def test_conv_activation_selects_relu_or_exact_gelu(activation):
    import torch.nn as nn

    from mantpy.nn._patch_encoder import _ConvEnc

    encoder = _ConvEnc(16, 1, 8, blocks=3, activation=activation)
    activations = [module for module in encoder.conv if isinstance(module, nn.ReLU | nn.GELU)]

    assert len(activations) == 3
    if activation == "gelu":
        assert all(type(module) is nn.GELU and module.approximate == "none" for module in activations)
    else:
        assert all(type(module) is nn.ReLU for module in activations)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"conv_blocks": 0}, "conv_blocks"),
        ({"conv_blocks": -1}, "conv_blocks"),
        ({"conv_width": 0}, "conv_width"),
        ({"conv_width": -16}, "conv_width"),
        ({"pool_size": 0}, "pool_size"),
        ({"pool_size": -2}, "pool_size"),
        ({"conv_activation": "tanh"}, "conv_activation"),
        ({"conv_activation": "GELU"}, "conv_activation"),
        ({"conv_activation": None}, "conv_activation"),
        ({"pool_size": 2, "pool_type": "median"}, "pool_type"),
        ({"pool_size": 2, "pool_type": "MAX"}, "pool_type"),
    ],
)
def test_invalid_shape_options_raise(options, match):
    from mantpy.nn import PatchEncoder

    with pytest.raises(ValueError, match=match):
        PatchEncoder(**options)


def test_max_pooling_without_pool_size_raises():
    from mantpy.nn import PatchEncoder

    with pytest.raises(ValueError, match="requires pool_size"):
        PatchEncoder(pool_type="max")
    with pytest.raises(ValueError, match="requires pool_size"):
        PatchEncoder(conv_blocks=2, conv_width=32, conv_activation="gelu", pool_type="max")


@pytest.mark.parametrize(
    "options",
    [
        {"conv_blocks": 2},
        {"conv_width": 32},
        {"pool_size": 2},
        {"conv_activation": "gelu"},
        {"pool_size": 2, "pool_type": "max"},
        {name: value for name, value in LIVER_OPTIONS.items() if name != "latent_dim"},
    ],
)
def test_shape_options_cannot_be_combined_with_a_custom_backbone(options):
    import torch.nn as nn

    from mantpy.nn import PatchEncoder

    with pytest.raises(ValueError, match="cannot be combined with a custom backbone"):
        PatchEncoder(backbone=nn.Flatten, **options)
    with pytest.raises(ValueError, match="cannot be combined with a custom backbone"):
        PatchEncoder(backbone=nn.Flatten(), **options)


def test_custom_backbone_with_default_options_omits_them_from_provenance():
    import torch.nn as nn

    from mantpy.nn import PatchEncoder

    custom = PatchEncoder(backbone=nn.Flatten, **SHAPE_OPTION_DEFAULTS)
    architecture = custom.provenance(patch_key="image_patches", key_added="X_cnn")["architecture"]
    assert not set(SHAPE_OPTION_NAMES) & set(architecture)

    builtin = PatchEncoder().provenance(patch_key="image_patches", key_added="X_cnn")["architecture"]
    assert {name: builtin[name] for name in SHAPE_OPTION_NAMES} == SHAPE_OPTION_DEFAULTS


def test_pool_size_larger_than_the_feature_map_raises(patches, liver_patches):
    from mantpy.nn import PatchEncoder
    from mantpy.nn._patch_encoder import _ConvEnc

    # 10 -> 5 -> 3: a 4 x 4 pool does not fit the 3 x 3 map; 3 x 3 does.
    with pytest.raises(ValueError, match=r"pool_size=4 exceeds the 3 x 3 feature map left after 2 convolution"):
        _ConvEnc(10, 6, 8, blocks=2, pool_size=4)
    assert _ConvEnc(10, 6, 8, blocks=2, pool_size=3).lin.in_features == 32 * 3 * 3

    encoder = PatchEncoder(latent_dim=8, conv_blocks=2, pool_size=4)
    with pytest.raises(ValueError, match="pool_size=4 exceeds"):
        encoder.fit(liver_patches, max_epochs=1, accelerator="cpu", enable_progress_bar=False)
    assert not encoder.is_trained

    # Default depth: 16 -> 8 -> 4 leaves a 4 x 4 map.
    with pytest.raises(ValueError, match="pool_size=5 exceeds the 4 x 4 feature map"):
        PatchEncoder(latent_dim=8, pool_size=5).fit(
            patches, max_epochs=1, accelerator="cpu", enable_progress_bar=False
        )


def test_liver_configuration_fits_on_cpu(liver_patches):
    import torch.nn as nn

    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(temperature=0.1, **LIVER_OPTIONS).fit(
        liver_patches,
        max_epochs=2,
        batch_size=32,
        lr=3e-4,
        weight_decay=1e-4,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
    )
    embedding = encoder.transform(liver_patches)

    assert embedding.shape == (64, 96)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    assert len(encoder.history["train_loss"]) == 2
    assert encoder.transform(liver_patches[:1]).shape == (1, 96)

    model = encoder.model
    assert [(conv.in_channels, conv.out_channels) for conv in _conv_layers(model)] == [(6, 32), (32, 64)]
    assert sum(type(module) is nn.GELU for module in model.conv) == 2
    assert not any(isinstance(module, nn.ReLU) for module in model.conv)
    assert type(model.pool) is nn.AdaptiveMaxPool2d
    assert model.pool.output_size == 2
    assert (model.lin.in_features, model.lin.out_features) == (64 * 2 * 2, 96)
    assert not model.training


def test_shape_options_work_with_the_autoencoder_objective(liver_patches):
    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(objective="autoencoder", **LIVER_OPTIONS).fit(
        liver_patches,
        max_epochs=1,
        batch_size=32,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
    )
    assert encoder.transform(liver_patches).shape == (64, 96)


def test_liver_configuration_is_reproducible_on_cpu(liver_patches):
    from mantpy.nn import PatchEncoder

    fit_kwargs = {
        "max_epochs": 2,
        "batch_size": 32,
        "accelerator": "cpu",
        "random_state": 0,
        "enable_progress_bar": False,
    }
    first = PatchEncoder(temperature=0.1, **LIVER_OPTIONS).fit(liver_patches, **fit_kwargs).transform(liver_patches)
    second = PatchEncoder(temperature=0.1, **LIVER_OPTIONS).fit(liver_patches, **fit_kwargs).transform(liver_patches)
    np.testing.assert_allclose(first, second)


def test_pooling_relaxes_strict_determinism_to_a_warning(patches, liver_patches):
    import torch

    from mantpy.nn import PatchEncoder

    fit_kwargs = {"max_epochs": 1, "batch_size": 32, "accelerator": "cpu", "enable_progress_bar": False}
    try:
        PatchEncoder(temperature=0.1, **LIVER_OPTIONS).fit(liver_patches, **fit_kwargs)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()

        # Explicit trainer keywords still take precedence.
        PatchEncoder(temperature=0.1, **LIVER_OPTIONS).fit(liver_patches, deterministic=True, **fit_kwargs)
        assert not torch.is_deterministic_algorithms_warn_only_enabled()

        PatchEncoder(latent_dim=4, pool_size=2).fit(patches, **fit_kwargs)
        assert torch.is_deterministic_algorithms_warn_only_enabled()

        # Without pooling the previous strict behaviour is unchanged.
        PatchEncoder(latent_dim=4, conv_blocks=2, conv_width=32, conv_activation="gelu").fit(patches, **fit_kwargs)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(True)


def test_save_load_roundtrip_preserves_shape_options(tmp_path, liver_patches):
    import json

    import torch.nn as nn

    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(temperature=0.1, **LIVER_OPTIONS).fit(
        liver_patches,
        max_epochs=2,
        batch_size=32,
        accelerator="cpu",
        random_state=2,
        enable_progress_bar=False,
    )
    expected = encoder.transform(liver_patches)
    encoder.save(tmp_path / "liver_encoder")

    config = json.loads((tmp_path / "liver_encoder" / "config.json").read_text())
    assert {name: config[name] for name in LIVER_OPTIONS} == LIVER_OPTIONS
    assert config["custom_backbone"] is False

    restored = PatchEncoder.load(tmp_path / "liver_encoder")
    assert restored.is_trained
    assert {name: getattr(restored, name) for name in LIVER_OPTIONS} == LIVER_OPTIONS
    assert restored._encoder_options == encoder._encoder_options
    assert type(restored.model.pool) is nn.AdaptiveMaxPool2d
    assert restored.model.pool.output_size == 2
    assert sum(type(module) is nn.GELU for module in restored.model.conv) == 2
    _assert_same_state(restored.model, encoder.model)
    np.testing.assert_allclose(restored.transform(liver_patches), expected, rtol=1e-6, atol=1e-6)
    architecture = restored.provenance(patch_key="image_patches", key_added="X_cnn")["architecture"]
    assert {name: architecture[name] for name in SHAPE_OPTION_NAMES} == {
        name: LIVER_OPTIONS[name] for name in SHAPE_OPTION_NAMES
    }


def test_load_reads_checkpoints_saved_before_the_shape_options(tmp_path, patches):
    import json

    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(latent_dim=6).fit(
        patches,
        max_epochs=1,
        batch_size=32,
        accelerator="cpu",
        random_state=2,
        enable_progress_bar=False,
    )
    expected = encoder.transform(patches[:5])
    encoder.save(tmp_path / "patch_encoder")
    config_path = tmp_path / "patch_encoder" / "config.json"
    config = json.loads(config_path.read_text())
    assert {name: config[name] for name in SHAPE_OPTION_NAMES} == SHAPE_OPTION_DEFAULTS
    for name in SHAPE_OPTION_NAMES:
        del config[name]
    config_path.write_text(json.dumps(config))

    restored = PatchEncoder.load(tmp_path / "patch_encoder")
    assert {name: getattr(restored, name) for name in SHAPE_OPTION_NAMES} == SHAPE_OPTION_DEFAULTS
    np.testing.assert_allclose(restored.transform(patches[:5]), expected, rtol=1e-6, atol=1e-6)


def test_encode_patches_forwards_shape_options_and_records_provenance(tmp_path):
    import anndata
    import torch.nn as nn
    from anndata import AnnData
    from packaging.version import Version

    from mantpy.nn import PatchEncoder, encode_patches

    rng = np.random.default_rng(1)
    left = AnnData(np.zeros((20, 0), dtype=np.float32))
    right = AnnData(np.zeros((14, 0), dtype=np.float32))
    left.obsm["image_patches"] = rng.random((20, 6, 10, 10), dtype=np.float32)
    right.obsm["image_patches"] = rng.random((14, 6, 10, 10), dtype=np.float32)

    encoder = encode_patches(
        [left, right],
        temperature=0.1,
        max_epochs=2,
        batch_size=16,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
        **LIVER_OPTIONS,
    )

    assert isinstance(encoder, PatchEncoder)
    assert {name: getattr(encoder, name) for name in LIVER_OPTIONS} == LIVER_OPTIONS
    assert [conv.out_channels for conv in _conv_layers(encoder.model)] == [32, 64]
    assert type(encoder.model.pool) is nn.AdaptiveMaxPool2d
    assert encoder.model.lin.in_features == 64 * 2 * 2
    assert left.obsm["X_cnn"].shape == (20, 96)
    assert right.obsm["X_cnn"].shape == (14, 96)
    for item in (left, right):
        params = item.uns["X_cnn_params"]
        assert params["temperature"] == 0.1
        assert params["architecture"] == {
            "encoder": "convolutional",
            "latent_dim": 96,
            "input_shape": [6, 10, 10],
            "conv_blocks": 2,
            "conv_width": 32,
            "pool_size": 2,
            "conv_activation": "gelu",
            "pool_type": "max",
        }
    if Version(anndata.__version__) >= Version("0.12"):
        left.write_h5ad(tmp_path / "encoded.h5ad")


def test_encode_patches_records_default_shape_options(tmp_path):
    import anndata
    from anndata import AnnData
    from packaging.version import Version

    from mantpy.nn import encode_patches

    rng = np.random.default_rng(1)
    adata = AnnData(np.zeros((12, 0), dtype=np.float32))
    adata.obsm["image_patches"] = rng.random((12, 2, 8, 8), dtype=np.float32)

    encoder = encode_patches(
        adata, latent_dim=5, max_epochs=1, batch_size=12, accelerator="cpu", enable_progress_bar=False
    )

    architecture = adata.uns["X_cnn_params"]["architecture"]
    assert {name: architecture[name] for name in SHAPE_OPTION_NAMES} == SHAPE_OPTION_DEFAULTS
    assert encoder.model.pool is None
    assert [conv.out_channels for conv in _conv_layers(encoder.model)] == [16]
    # ``None`` options must stay serialisable with the rest of the provenance.
    if Version(anndata.__version__) >= Version("0.12"):
        adata.write_h5ad(tmp_path / "encoded.h5ad")


# ---------------------------------------------------------------------------
# 32 x 32 single-channel patches with conv_blocks, conv_width and pool_size
# alone (ReLU, average pooling): the default encoder and its pooled variants.
# ---------------------------------------------------------------------------


@pytest.fixture
def wide_patches():
    rng = np.random.default_rng(0)
    return rng.random((48, 1, 32, 32)).astype("float32")


def _pooling_modules(encoder):
    import torch.nn as nn

    return [module for module in encoder.modules() if isinstance(module, nn.AdaptiveAvgPool2d | nn.AdaptiveMaxPool2d)]


def test_default_encoder_for_32_pixel_patches_is_unchanged(wide_patches):
    import torch.nn as nn

    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(latent_dim=8).fit(
        wide_patches,
        max_epochs=1,
        batch_size=24,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
    )
    model = encoder.model

    # 32 -> 16 -> 8 -> 4: three blocks, then the 64 x 4 x 4 map is flattened.
    assert [(conv.in_channels, conv.out_channels) for conv in _conv_layers(model)] == [(1, 16), (16, 32), (32, 64)]
    assert [type(module) for module in model.conv] == [nn.Conv2d, nn.BatchNorm2d, nn.ReLU] * 3
    assert model.pool is None
    assert _pooling_modules(model) == []
    assert [name for name, _ in model.named_children()] == ["conv", "lin"]
    assert (model.lin.in_features, model.lin.out_features) == (1024, 8)
    state = model.state_dict()
    assert list(state) == list(_original_conv_encoder(32, 1, 8).state_dict())
    assert all(key.startswith(("conv.", "lin.")) for key in state)
    assert encoder.transform(wide_patches).shape == (48, 8)


def test_pool_size_two_gives_a_256_feature_readout_and_fits_with_default_trainer_settings(wide_patches):
    import torch.nn as nn

    from mantpy.nn import PatchEncoder

    # No trainer keywords: the default trainer settings must not raise when pooling.
    encoder = PatchEncoder(latent_dim=8, pool_size=2).fit(wide_patches, max_epochs=2, accelerator="cpu", random_state=0)
    model = encoder.model
    embedding = encoder.transform(wide_patches)

    assert [conv.out_channels for conv in _conv_layers(model)] == [16, 32, 64]
    assert type(model.pool) is nn.AdaptiveAvgPool2d
    assert model.pool.output_size == 2
    assert (model.lin.in_features, model.lin.out_features) == (64 * 2 * 2, 8)
    # Pooling has no parameters: the keys stay those of the original encoder.
    assert list(model.state_dict()) == list(_original_conv_encoder(32, 1, 8).state_dict())
    assert embedding.shape == (48, 8)
    assert np.isfinite(embedding).all()
    assert len(encoder.history["train_loss"]) == 2


def test_pool_size_one_with_conv_width_32_gives_a_128_feature_readout(wide_patches):
    import torch.nn as nn

    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(latent_dim=8, conv_width=32, pool_size=1).fit(
        wide_patches,
        max_epochs=1,
        batch_size=24,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
    )
    model = encoder.model

    assert [conv.out_channels for conv in _conv_layers(model)] == [32, 64, 128]
    assert type(model.pool) is nn.AdaptiveAvgPool2d
    assert model.pool.output_size == 1
    assert (model.lin.in_features, model.lin.out_features) == (128, 8)
    assert encoder.transform(wide_patches).shape == (48, 8)


@pytest.mark.parametrize(
    ("conv_blocks", "channels", "readout"),
    [
        (None, [16, 32, 64], 64 * 4 * 4),
        (2, [16, 32], 32 * 8 * 8),
        (4, [16, 32, 64, 128], 128 * 2 * 2),
    ],
)
def test_conv_blocks_overrides_the_automatic_depth_for_32_pixel_patches(conv_blocks, channels, readout):
    import torch

    from mantpy.nn import PatchEncoder
    from mantpy.nn._patch_encoder import _ConvEnc

    options = PatchEncoder(latent_dim=8, conv_blocks=conv_blocks)._encoder_options
    encoder = _ConvEnc(32, 1, 8, **options)

    assert [conv.out_channels for conv in _conv_layers(encoder)] == channels
    assert encoder.pool is None
    assert encoder.lin.in_features == readout
    assert encoder(torch.rand(3, 1, 32, 32)).shape == (3, 8)


def test_save_load_roundtrip_with_pool_size_and_conv_width(tmp_path, wide_patches):
    import json

    import torch.nn as nn

    from mantpy.nn import PatchEncoder

    encoder = PatchEncoder(latent_dim=6, conv_width=32, pool_size=2).fit(
        wide_patches,
        max_epochs=2,
        batch_size=24,
        accelerator="cpu",
        random_state=2,
        enable_progress_bar=False,
    )
    expected = encoder.transform(wide_patches)
    encoder.save(tmp_path / "pooled_encoder")

    config = json.loads((tmp_path / "pooled_encoder" / "config.json").read_text())
    assert (config["conv_blocks"], config["conv_width"], config["pool_size"]) == (None, 32, 2)

    restored = PatchEncoder.load(tmp_path / "pooled_encoder")
    assert restored.is_trained
    assert (restored.conv_blocks, restored.conv_width, restored.pool_size) == (None, 32, 2)
    assert [conv.out_channels for conv in _conv_layers(restored.model)] == [32, 64, 128]
    assert type(restored.model.pool) is nn.AdaptiveAvgPool2d
    assert restored.model.lin.in_features == 128 * 2 * 2
    _assert_same_state(restored.model, encoder.model)
    np.testing.assert_allclose(restored.transform(wide_patches), expected, rtol=1e-6, atol=1e-6)


def test_encode_patches_forwards_conv_blocks_conv_width_and_pool_size():
    import torch.nn as nn
    from anndata import AnnData

    from mantpy.nn import encode_patches

    rng = np.random.default_rng(1)
    adata = AnnData(np.zeros((24, 0), dtype=np.float32))
    adata.obsm["image_patches"] = rng.random((24, 1, 32, 32), dtype=np.float32)

    # Two blocks instead of the automatic three: 32 -> 16 -> 8, pooled to 2 x 2.
    encoder = encode_patches(
        adata,
        latent_dim=5,
        conv_blocks=2,
        conv_width=32,
        pool_size=2,
        max_epochs=1,
        batch_size=12,
        accelerator="cpu",
        random_state=0,
        enable_progress_bar=False,
    )

    assert (encoder.conv_blocks, encoder.conv_width, encoder.pool_size) == (2, 32, 2)
    assert [conv.out_channels for conv in _conv_layers(encoder.model)] == [32, 64]
    assert type(encoder.model.pool) is nn.AdaptiveAvgPool2d
    assert encoder.model.lin.in_features == 64 * 2 * 2
    assert adata.obsm["X_cnn"].shape == (24, 5)
    architecture = adata.uns["X_cnn_params"]["architecture"]
    assert architecture["encoder"] == "convolutional"
    assert architecture["input_shape"] == [1, 32, 32]
    assert {name: architecture[name] for name in ("conv_blocks", "conv_width", "pool_size")} == {
        "conv_blocks": 2,
        "conv_width": 32,
        "pool_size": 2,
    }
