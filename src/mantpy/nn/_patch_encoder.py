"""Self-supervised, Lightning-powered CNN features for image patches.

The public :class:`PatchEncoder` learns node-local representations from
single- or multi-channel image patches. :func:`encode_patches` provides the
one-line AnnData workflow: train across samples, standardise the shared
embedding, and write it to ``obsm``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from anndata import AnnData
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler
except ImportError as e:  # pragma: no cover
    raise ImportError('PatchEncoder requires PyTorch. Install with: pip install "mantpy[patch]"') from e

try:
    import pytorch_lightning as pl
except ImportError:
    try:
        import lightning.pytorch as pl
    except ImportError as e:  # pragma: no cover
        raise ImportError('PatchEncoder requires PyTorch Lightning. Install with: pip install "mantpy[patch]"') from e

from mantpy.nn._utils import _package_version, seed_everything

logger = logging.getLogger(__name__)

_AUG_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "default": (0.7, 1.3, 0.1, 0.05),
    "strong": (0.5, 1.5, 0.2, 0.10),
}


def _component(spec, kwargs, default_factory):
    """Build a component after seeding, without probing its forward pass."""
    if spec is None:
        return default_factory()
    if isinstance(spec, nn.Module):
        if kwargs:
            raise ValueError("Component kwargs require a factory, not an initialized module.")
        return spec
    if not callable(spec):
        raise TypeError("A model component must be an nn.Module or a callable factory.")
    module = spec(**dict(kwargs or {}))
    if not isinstance(module, nn.Module):
        raise TypeError("A model factory must return an nn.Module.")
    return module


def _component_name(spec) -> str:
    if spec is None:
        return "default"
    return getattr(spec, "__qualname__", type(spec).__qualname__)


class _ArrayPatchDataset(Dataset):
    """A zero-copy, CPU-backed view over one NumPy patch array."""

    def __init__(self, patches: np.ndarray):
        self.patches = patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.patches[index])


class _ExhaustingRandomSampler(RandomSampler):
    """Consume complete epoch draws when a trainer stops after a known length."""

    def __iter__(self):
        return iter(list(super().__iter__()))


class _ConvEnc(nn.Module):
    """Halve spatial dimensions until at most four pixels, then project."""

    def __init__(self, in_size: int, in_ch: int, feat: int):
        super().__init__()
        layers: list[nn.Module] = []
        ch, size, depth = in_ch, in_size, 0
        while size > 4:
            out = min(16 * 2**depth, 128)
            layers += [nn.Conv2d(ch, out, 3, 2, 1), nn.BatchNorm2d(out), nn.ReLU()]
            ch, size, depth = out, (size + 1) // 2, depth + 1
        self.conv = nn.Sequential(*layers)
        self.lin = nn.Linear(ch * size * size, feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(self.conv(x).flatten(1))


class _ConvDec(nn.Module):
    """Mirror decoder used by the optional autoencoder objective."""

    def __init__(self, in_size: int, in_ch: int, feat: int):
        super().__init__()
        channels: list[int] = []
        size = in_size
        while size > 4:
            channels.append(min(16 * 2 ** len(channels), 128))
            size = (size + 1) // 2
        self.channels = channels[-1] if channels else in_ch
        self.size = size
        self.lin = nn.Linear(feat, self.channels * size * size)
        reverse = channels[::-1] + [in_ch]
        decoder: list[nn.Module] = []
        for i in range(len(channels)):
            decoder.append(nn.ConvTranspose2d(reverse[i], reverse[i + 1], 4, 2, 1))
            if i < len(channels) - 1:
                decoder.append(nn.ReLU())
        self.dec = nn.Sequential(*decoder)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(self.lin(z).view(-1, self.channels, self.size, self.size))


def _augment(x: torch.Tensor, params: tuple[float, float, float, float]) -> torch.Tensor:
    """Apply flips, transpose, intensity jitter, and Gaussian noise."""
    contrast_low, contrast_high, brightness_range, noise = params
    for dims in ((-1,), (-2,)):
        mask = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(mask[:, None, None, None], x.flip(dims), x)
    mask = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(mask[:, None, None, None], x.transpose(-1, -2), x)
    contrast = contrast_low + (contrast_high - contrast_low) * torch.rand(x.shape[0], 1, 1, 1, device=x.device)
    brightness = -brightness_range + 2 * brightness_range * torch.rand(x.shape[0], 1, 1, 1, device=x.device)
    return (x * contrast + brightness + noise * torch.randn_like(x)).clamp(0, 1)


def _nt_xent(z: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute SimCLR NT-Xent for two stacked views of every patch."""
    z = F.normalize(z, dim=1)
    n_views = z.shape[0]
    half = n_views // 2
    similarity = z @ z.T / temperature
    similarity.fill_diagonal_(-1e9)
    targets = (torch.arange(n_views, device=z.device) + half) % n_views
    return F.cross_entropy(similarity, targets)


class _PatchTrainingModule(pl.LightningModule):
    """Private Lightning module; only its fitted encoder is exposed."""

    def __init__(
        self,
        *,
        in_size: int,
        in_channels: int,
        latent_dim: int,
        objective: Literal["contrastive", "autoencoder"],
        temperature: float,
        augment: Literal["default", "strong"],
        lr: float,
        weight_decay: float,
        max_epochs: int,
        backbone=None,
        backbone_kwargs=None,
        projection=None,
        projection_kwargs=None,
        decoder=None,
        decoder_kwargs=None,
        optimizer_factory=None,
        optimizer_kwargs=None,
        scheduler="cosine",
        scheduler_factory=None,
        scheduler_kwargs=None,
        contrastive_loss=None,
        reconstruction_loss=None,
    ):
        super().__init__()
        self.encoder = _component(
            backbone, backbone_kwargs, lambda: _ConvEnc(in_size, in_channels, latent_dim)
        )
        self.latent_dim = latent_dim
        self.objective = objective
        self.temperature = temperature
        self.augment = augment
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.optimizer_factory = optimizer_factory
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.scheduler = scheduler
        self.scheduler_factory = scheduler_factory
        self.scheduler_kwargs = dict(scheduler_kwargs or {})
        self.contrastive_loss = contrastive_loss or _nt_xent
        self.reconstruction_loss = reconstruction_loss or F.mse_loss
        self.projection = (
            _component(
                projection,
                projection_kwargs,
                lambda: nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 64)),
            )
            if objective == "contrastive"
            else None
        )
        self.decoder = (
            _component(decoder, decoder_kwargs, lambda: _ConvDec(in_size, in_channels, latent_dim))
            if objective == "autoencoder"
            else None
        )
        self.epoch_losses: list[float] = []
        self._batch_losses: list[torch.Tensor] = []

    def _encode(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.encoder(batch)
        if not torch.is_tensor(features) or features.shape != (len(batch), self.latent_dim):
            raise ValueError(f"The backbone must return (N, {self.latent_dim}) features.")
        return features

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx
        if self.objective == "contrastive":
            augment = self.augment if callable(self.augment) else lambda x: _augment(x, _AUG_PARAMS[self.augment])
            views = torch.cat((augment(batch), augment(batch)), dim=0)
            if views.shape != (2 * len(batch), *batch.shape[1:]):
                raise ValueError("Augmentations must preserve each patch batch's shape.")
            loss = self.contrastive_loss(self.projection(self._encode(views)), self.temperature)
        else:
            reconstruction = self.decoder(self._encode(batch))
            if reconstruction.shape[-2:] != batch.shape[-2:]:
                reconstruction = F.interpolate(
                    reconstruction, size=batch.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = self.reconstruction_loss(reconstruction, batch)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("The training objective must return one finite scalar loss.")
        self._batch_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return loss

    def on_train_epoch_end(self) -> None:
        if self._batch_losses:
            self.epoch_losses.append(float(torch.stack(self._batch_losses).mean().cpu()))
            self._batch_losses.clear()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer_factory = self.optimizer_factory or torch.optim.AdamW
        optimizer = optimizer_factory(
            self.parameters(), **{"lr": self.lr, "weight_decay": self.weight_decay, **self.optimizer_kwargs}
        )
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer, **self.scheduler_kwargs)
        elif self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **{"T_max": max(1, self.max_epochs), **self.scheduler_kwargs}
            )
        else:
            return {"optimizer": optimizer}
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


PatchArray = np.ndarray | Sequence[np.ndarray]


class PatchEncoder:
    """Learn node-local CNN features from image patches.

    Patch tensors stay in CPU memory; Lightning transfers only the active
    batch. One pooled ``StandardScaler`` places every transformed sample in a
    shared feature space.

    Parameters
    ----------
    latent_dim
        Number of learned features per patch.
    objective
        ``"contrastive"`` (default) or the optional ``"autoencoder"`` baseline.
    temperature
        NT-Xent temperature for the contrastive objective.
    augment
        ``"default"`` or ``"strong"`` augmentation preset.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        objective: Literal["contrastive", "autoencoder"] = "contrastive",
        temperature: float = 0.2,
        augment: Literal["default", "strong"] | Callable = "default",
        backbone: nn.Module | Callable | None = None,
        backbone_kwargs: Mapping[str, Any] | None = None,
        projection: nn.Module | Callable | None = None,
        projection_kwargs: Mapping[str, Any] | None = None,
        decoder: nn.Module | Callable | None = None,
        decoder_kwargs: Mapping[str, Any] | None = None,
    ):
        if latent_dim < 1:
            raise ValueError(f"latent_dim={latent_dim!r}: expected a positive integer.")
        if objective not in ("contrastive", "autoencoder"):
            raise ValueError(f"objective={objective!r}: expected 'contrastive' or 'autoencoder'.")
        if temperature <= 0:
            raise ValueError(f"temperature={temperature!r}: expected a positive number.")
        if not callable(augment) and augment not in _AUG_PARAMS:
            raise ValueError(f"augment={augment!r}: expected one of {sorted(_AUG_PARAMS)}.")
        self.latent_dim = int(latent_dim)
        self.objective = objective
        self.temperature = float(temperature)
        self.augment = augment
        self._backbone = backbone
        self._backbone_kwargs = dict(backbone_kwargs or {})
        self._projection = projection
        self._projection_kwargs = dict(projection_kwargs or {})
        self._decoder = decoder
        self._decoder_kwargs = dict(decoder_kwargs or {})
        self._enc: nn.Module | None = None
        self._scaler: StandardScaler | None = None
        self._input_shape: tuple[int, int, int] | None = None
        self._device = "cpu"
        self._history: dict[str, list[float]] = {}
        self._fit_params: dict[str, Any] = {}
        self._random_state: int | None = None
        self._optimizer_updates = 0

    def __repr__(self) -> str:
        status = "fitted" if self.is_trained else "unfitted"
        details = f"latent_dim={self.latent_dim}, objective={self.objective!r}, status={status!r}"
        if self.is_trained:
            details += f", input_shape={self._input_shape!r}, device={self._device!r}"
        return f"PatchEncoder({details})"

    @staticmethod
    def _check_patches(patches: np.ndarray) -> np.ndarray:
        array = np.asarray(patches, dtype=np.float32)
        if array.ndim != 4:
            raise ValueError(
                f"patches must have shape (n, channels, P, P); got ndim={array.ndim}, shape={array.shape}."
            )
        if array.shape[0] == 0:
            raise ValueError("patches must contain at least one observation.")
        if array.shape[1] == 0:
            raise ValueError("patches must contain at least one channel.")
        if array.shape[2] != array.shape[3]:
            raise ValueError(f"patches must be square (P x P); got {array.shape[2]} x {array.shape[3]}.")
        if not np.isfinite(array).all():
            raise ValueError("patches contains NaN or infinite values.")
        return np.ascontiguousarray(array)

    @classmethod
    def _check_collection(cls, patches: PatchArray, *, require_two: bool = False) -> list[np.ndarray]:
        if isinstance(patches, np.ndarray):
            arrays = [cls._check_patches(patches)]
        else:
            arrays = [cls._check_patches(array) for array in patches]
        if not arrays:
            raise ValueError("patches must contain at least one array.")
        expected = arrays[0].shape[1:]
        for i, array in enumerate(arrays[1:], start=1):
            if array.shape[1:] != expected:
                raise ValueError(
                    f"patches[{i}] has shape {array.shape[1:]}; expected shared channel and spatial shape {expected}."
                )
        if require_two and sum(len(array) for array in arrays) < 2:
            raise ValueError("PatchEncoder requires at least two patches for self-supervised training.")
        return arrays

    def fit(
        self,
        patches: PatchArray,
        *,
        max_epochs: int = 200,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        accelerator: str = "auto",
        devices: str | int | list[int] = "auto",
        precision: str | int = "32-true",
        num_workers: int = 0,
        random_state: int = 0,
        optimizer_factory: Callable | None = None,
        optimizer_kwargs: Mapping[str, Any] | None = None,
        scheduler: Literal["cosine"] | None = "cosine",
        scheduler_factory: Callable | None = None,
        scheduler_kwargs: Mapping[str, Any] | None = None,
        contrastive_loss: Callable | None = None,
        reconstruction_loss: Callable | None = None,
        shuffle: bool = True,
        drop_last: bool | None = None,
        exhaust_sampler: bool = False,
        **trainer_kwargs: Any,
    ) -> PatchEncoder:
        """Fit one pooled encoder and embedding scaler.

        ``patches`` may be one 4D array or a sequence of arrays. Sequences are
        trained jointly without concatenation. Extra keywords are forwarded to
        ``Lightning.Trainer``. ``exhaust_sampler=True`` consumes all shuffled
        epoch draws, reproducing a fully exhausted manual DataLoader loop even
        when Lightning stops after a single known batch. The default retains
        Lightning's previous sampling behavior.
        """
        arrays = self._check_collection(patches, require_two=True)
        if max_epochs < 1:
            raise ValueError("max_epochs must be at least 1.")
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2.")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if scheduler not in (None, "cosine"):
            raise ValueError("scheduler must be 'cosine' or None; use scheduler_factory for other schedules.")

        random_state = int(random_state)
        seed_everything(random_state, device=accelerator)
        pl.seed_everything(random_state, workers=True, verbose=False)
        input_channels, input_size, _ = arrays[0].shape[1:]
        self._input_shape = (input_channels, input_size, input_size)

        dataset = ConcatDataset([_ArrayPatchDataset(array) for array in arrays])
        generator = torch.Generator().manual_seed(random_state)
        resolved_batch_size = min(batch_size, len(dataset))
        sampler = _ExhaustingRandomSampler(dataset, generator=generator) if shuffle and exhaust_sampler else None
        loader = DataLoader(
            dataset,
            batch_size=resolved_batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=accelerator in ("auto", "gpu", "cuda") and torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            # BatchNorm cannot train on a one-patch tail batch. Preserve every
            # patch unless the final remainder would contain exactly one.
            drop_last=len(dataset) % resolved_batch_size == 1 if drop_last is None else drop_last,
            generator=generator,
        )
        module = _PatchTrainingModule(
            in_size=input_size,
            in_channels=input_channels,
            latent_dim=self.latent_dim,
            objective=self.objective,
            temperature=self.temperature,
            augment=self.augment,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            backbone=self._backbone,
            backbone_kwargs=self._backbone_kwargs,
            projection=self._projection,
            projection_kwargs=self._projection_kwargs,
            decoder=self._decoder,
            decoder_kwargs=self._decoder_kwargs,
            optimizer_factory=optimizer_factory,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_factory=scheduler_factory,
            scheduler_kwargs=scheduler_kwargs,
            contrastive_loss=contrastive_loss,
            reconstruction_loss=reconstruction_loss,
        )
        lightning_accelerator = "gpu" if accelerator == "cuda" else accelerator
        trainer_options: dict[str, Any] = {
            "max_epochs": max_epochs,
            "accelerator": lightning_accelerator,
            "devices": devices,
            "precision": precision,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": True,
            "deterministic": True,
        }
        trainer_options.update(trainer_kwargs)
        trainer = pl.Trainer(**trainer_options)
        trainer.fit(module, train_dataloaders=loader)

        training_device = trainer.strategy.root_device
        self._enc = module.encoder.eval().to(training_device)
        self._device = str(training_device)
        self._history = {"train_loss": module.epoch_losses}
        self._optimizer_updates = int(trainer.global_step)
        self._random_state = random_state
        self._fit_params = {
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "accelerator": accelerator,
            "resolved_device": self._device,
            "devices": devices,
            "precision": precision,
            "num_workers": num_workers,
            "trainer_kwargs": json.loads(json.dumps(trainer_kwargs, default=str)),
            "n_patches": len(dataset),
            "n_samples": len(arrays),
            "optimizer_updates": self._optimizer_updates,
            "optimizer": _component_name(optimizer_factory or torch.optim.AdamW),
            "optimizer_kwargs": dict(optimizer_kwargs or {}),
            "scheduler": _component_name(scheduler_factory) if scheduler_factory else scheduler,
            "scheduler_kwargs": dict(scheduler_kwargs or {}),
            "shuffle": bool(shuffle),
            "drop_last": loader.drop_last,
            "exhaust_sampler": bool(exhaust_sampler),
        }
        self._scaler = StandardScaler()
        for raw in self._raw_transform(arrays, batch_size=max(batch_size, 4096), num_workers=num_workers):
            self._scaler.partial_fit(raw)
        logger.info(
            "PatchEncoder fitted: objective=%s, n_patches=%d, latent_dim=%d, device=%s",
            self.objective,
            len(dataset),
            self.latent_dim,
            self._device,
        )
        return self

    def _validate_input_shape(self, array: np.ndarray) -> None:
        if self._input_shape is not None and array.shape[1:] != self._input_shape:
            raise ValueError(
                f"patches have channel/spatial shape {array.shape[1:]}; fitted encoder expects {self._input_shape}."
            )

    @torch.no_grad()
    def _raw_transform(
        self,
        arrays: Sequence[np.ndarray],
        *,
        batch_size: int,
        num_workers: int,
    ) -> list[np.ndarray]:
        if self._enc is None:
            raise RuntimeError("PatchEncoder is not fitted. Call .fit(patches) first.")
        self._enc.eval()
        output: list[np.ndarray] = []
        for array in arrays:
            self._validate_input_shape(array)
            loader = DataLoader(
                _ArrayPatchDataset(array),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=self._device.startswith("cuda"),
                persistent_workers=num_workers > 0,
            )
            batches = [self._enc(batch.to(self._device, non_blocking=True)).cpu().numpy() for batch in loader]
            output.append(np.concatenate(batches).astype(np.float32, copy=False))
        return output

    def transform(
        self,
        patches: PatchArray,
        *,
        batch_size: int = 4096,
        num_workers: int = 0,
        standardize: bool = True,
    ) -> np.ndarray | list[np.ndarray]:
        """Embed and standardise one patch array or a sequence of arrays."""
        if self._scaler is None:
            raise RuntimeError("PatchEncoder is not fitted. Call .fit(patches) first.")
        is_single = isinstance(patches, np.ndarray)
        arrays = self._check_collection(patches)
        raw_embeddings = self._raw_transform(arrays, batch_size=batch_size, num_workers=num_workers)
        embeddings = [
            self._scaler.transform(raw).astype(np.float32) if standardize else raw for raw in raw_embeddings
        ]
        return embeddings[0] if is_single else embeddings

    def fit_transform(
        self, patches: PatchArray, *, standardize: bool = True, **fit_kwargs: Any,
    ) -> np.ndarray | list[np.ndarray]:
        """Fit the pooled encoder and return standardised embeddings."""
        self.fit(patches, **fit_kwargs)
        return self.transform(patches, standardize=standardize)

    def transform_adata(
        self,
        adata: AnnData | Sequence[AnnData],
        *,
        patch_key: str = "image_patches",
        key_added: str = "X_cnn",
        batch_size: int = 4096,
        num_workers: int = 0,
        standardize: bool = True,
    ) -> AnnData | list[AnnData]:
        """Write standardised learned features and provenance to AnnData."""
        adatas, is_single = _as_adata_list(adata)
        patch_arrays = _patch_arrays(adatas, patch_key)
        embeddings = self.transform(
            patch_arrays, batch_size=batch_size, num_workers=num_workers, standardize=standardize
        )
        assert isinstance(embeddings, list)
        provenance = self.provenance(patch_key=patch_key, key_added=key_added)
        provenance["standardization"] = "pooled StandardScaler" if standardize else "none"
        for item, embedding in zip(adatas, embeddings, strict=True):
            item.obsm[key_added] = embedding
            item.uns[f"{key_added}_params"] = provenance.copy()
        return adatas[0] if is_single else adatas

    def provenance(self, *, patch_key: str, key_added: str) -> dict[str, Any]:
        """Return serialisable training and representation provenance."""
        return {
            "method": "PatchEncoder",
            "key_added": key_added,
            "patch_key": patch_key,
            "architecture": {
                "encoder": "convolutional" if self._backbone is None else _component_name(self._backbone),
                "latent_dim": self.latent_dim,
                "input_shape": list(self._input_shape) if self._input_shape is not None else None,
            },
            "objective": self.objective,
            "temperature": self.temperature,
            "augmentation": {
                "preset": self.augment,
                "contrast_range": list(_AUG_PARAMS[self.augment][:2]),
                "brightness_range": _AUG_PARAMS[self.augment][2],
                "noise_std": _AUG_PARAMS[self.augment][3],
                "transforms": ["horizontal_flip", "vertical_flip", "transpose", "intensity_jitter", "noise"],
            } if isinstance(self.augment, str) else {"callable": _component_name(self.augment)},
            "standardization": "pooled StandardScaler",
            "random_state": self._random_state,
            "training": self._fit_params.copy(),
            "history": {key: list(values) for key, values in self._history.items()},
            "software": {
                "mantpy": _package_version("mantpy"),
                "torch": str(torch.__version__),
                "pytorch_lightning": str(getattr(pl, "__version__", "unknown")),
                "scikit_learn": _package_version("scikit-learn"),
            },
        }

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save encoder weights, configuration, history, and fitted scaler."""
        if self._enc is None or self._scaler is None or self._input_shape is None:
            raise RuntimeError("PatchEncoder is not fitted. Call .fit(patches) first.")
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Set overwrite=True to replace.")
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._enc.state_dict(), path / "model.pt")
        config = {
            "latent_dim": self.latent_dim,
            "objective": self.objective,
            "temperature": self.temperature,
            "augment": self.augment if isinstance(self.augment, str) else "custom",
            "custom_backbone": self._backbone is not None,
            "backbone_name": _component_name(self._backbone),
            "backbone_kwargs": self._backbone_kwargs,
            "optimizer_updates": self._optimizer_updates,
            "input_shape": list(self._input_shape),
            "device": self._device,
            "history": self._history,
            "fit_params": self._fit_params,
            "random_state": self._random_state,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2, default=str))
        np.savez(
            path / "scaler.npz",
            mean=self._scaler.mean_,
            scale=self._scaler.scale_,
            var=self._scaler.var_,
            n_samples_seen=self._scaler.n_samples_seen_,
        )

    @classmethod
    def load(
        cls, path: str | Path, *, backbone: nn.Module | Callable | None = None,
        backbone_kwargs: Mapping[str, Any] | None = None,
    ) -> PatchEncoder:
        """Load weights onto CPU; a custom architecture must be supplied explicitly."""
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        if config.get("custom_backbone", False) and backbone is None:
            raise ValueError("This checkpoint needs its backbone module or factory supplied explicitly.")
        instance = cls(
            latent_dim=config["latent_dim"],
            objective=config["objective"],
            temperature=config["temperature"],
            augment=config["augment"] if config["augment"] != "custom" else "default",
            backbone=backbone,
            backbone_kwargs=backbone_kwargs if backbone_kwargs is not None else config.get("backbone_kwargs"),
        )
        input_shape = tuple(int(value) for value in config["input_shape"])
        instance._input_shape = input_shape
        instance._enc = _component(
            backbone, instance._backbone_kwargs,
            lambda: _ConvEnc(input_shape[1], input_shape[0], instance.latent_dim),
        )
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        instance._enc.load_state_dict(state)
        instance._enc.eval()
        instance._device = "cpu"
        instance._history = config.get("history", {})
        instance._fit_params = config.get("fit_params", {})
        instance._random_state = config.get("random_state")
        instance._optimizer_updates = config.get("optimizer_updates", 0)

        scaler_state = np.load(path / "scaler.npz")
        scaler = StandardScaler()
        scaler.mean_ = scaler_state["mean"]
        scaler.scale_ = scaler_state["scale"]
        scaler.var_ = scaler_state["var"]
        n_samples_seen = scaler_state["n_samples_seen"]
        scaler.n_samples_seen_ = int(n_samples_seen) if n_samples_seen.ndim == 0 else n_samples_seen
        scaler.n_features_in_ = len(scaler.mean_)
        instance._scaler = scaler
        return instance

    @property
    def is_trained(self) -> bool:
        """Whether model weights and a scaler are fitted."""
        return self._enc is not None and self._scaler is not None

    @property
    def history(self) -> dict[str, list[float]]:
        """Per-epoch training loss."""
        return {key: list(values) for key, values in self._history.items()}

    @property
    def model(self) -> nn.Module:
        """Fitted backbone, without the training projection head."""
        if self._enc is None:
            raise RuntimeError("PatchEncoder is not fitted.")
        return self._enc

    @property
    def optimizer_updates(self) -> int:
        """Actual optimizer steps recorded by Lightning."""
        return self._optimizer_updates


def _as_adata_list(adata: AnnData | Sequence[AnnData]) -> tuple[list[AnnData], bool]:
    if isinstance(adata, AnnData):
        return [adata], True
    adatas = list(adata)
    if not adatas:
        raise ValueError("adata must contain at least one AnnData object.")
    if not all(isinstance(item, AnnData) for item in adatas):
        raise TypeError("adata must be an AnnData object or a sequence of AnnData objects.")
    return adatas, False


def _patch_arrays(adatas: Sequence[AnnData], patch_key: str) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for i, adata in enumerate(adatas):
        if patch_key not in adata.obsm:
            raise KeyError(f"adata[{i}].obsm has no {patch_key!r} patch tensor.")
        array = PatchEncoder._check_patches(adata.obsm[patch_key])
        if len(array) != adata.n_obs:
            raise ValueError(
                f"adata[{i}].obsm[{patch_key!r}] has {len(array)} patches but the AnnData has {adata.n_obs} observations."
            )
        arrays.append(array)
    return arrays


def encode_patches(
    adata: AnnData | Sequence[AnnData],
    *,
    patch_key: str = "image_patches",
    key_added: str = "X_cnn",
    latent_dim: int = 96,
    objective: Literal["contrastive", "autoencoder"] = "contrastive",
    temperature: float = 0.2,
    augment: Literal["default", "strong"] = "default",
    max_epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    accelerator: str = "auto",
    devices: str | int | list[int] = "auto",
    precision: str | int = "32-true",
    num_workers: int = 0,
    random_state: int = 0,
    **trainer_kwargs: Any,
) -> PatchEncoder:
    """Learn and attach standardised patch features in one call.

    One pooled encoder is trained across all supplied objects. Only
    ``obsm[patch_key]`` enters training; ``obs`` is never read. Features are
    written in place to ``obsm[key_added]`` and serialisable provenance to
    ``uns[f"{key_added}_params"]``. The fitted encoder is returned.
    """
    adatas, _ = _as_adata_list(adata)
    arrays = _patch_arrays(adatas, patch_key)
    encoder = PatchEncoder(
        latent_dim=latent_dim,
        objective=objective,
        temperature=temperature,
        augment=augment,
    ).fit(
        arrays,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        num_workers=num_workers,
        random_state=random_state,
        **trainer_kwargs,
    )
    encoder.transform_adata(
        adatas,
        patch_key=patch_key,
        key_added=key_added,
        batch_size=max(batch_size, 4096),
        num_workers=num_workers,
    )
    return encoder


__all__ = ["PatchEncoder", "encode_patches"]
