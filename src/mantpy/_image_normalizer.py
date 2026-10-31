"""Reusable fitted pixel transformations for image cohorts and held-out images."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _array(image):
    return image.to_array() if hasattr(image, "to_array") else np.asarray(image)


def _channel_float32(image, channel):
    if hasattr(image, "to_array"):
        return image[channel].to_array(dtype=np.float32)[0]
    return np.asarray(image[channel], dtype=np.float32)


class ImageNormalizer:
    """Fit a pooled image transformation once and reuse it on held-out images.

    Images are channel-first arrays or Mantpy ImageContainers. The
    ``pooled_zscore`` method uses scikit-learn StandardScaler partial fitting;
    ``pooled_arcsinh_percentile`` applies arcsinh to every channel and clips
    selected channels to pooled percentile bounds. Training images alone
    determine all fitted statistics. Arrays are transformed to float32,
    except ``method='none'``, which returns the input array unchanged.

    Channel order is positional. Supplying ``channel_names`` at construction
    and transform time additionally verifies that order. ``channel_indices``
    selects channels for percentile clipping; by default all are selected.
    """

    def __init__(
        self,
        method="pooled_zscore",
        *,
        cofactor=1.0,
        percentile=99.5,
        channel_indices=None,
        channel_names=None,
        chunk_size=100_000,
    ):
        self.method = method
        self.cofactor = cofactor
        self.percentile = percentile
        self.channel_indices = channel_indices
        self.channel_names = channel_names
        self.chunk_size = chunk_size

    def fit(self, images):
        """Learn pooled statistics from an ordered mapping or sequence of images."""
        if self.method not in {"pooled_zscore", "pooled_arcsinh_percentile", "none"}:
            raise ValueError("Unknown image normalization method.")
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        if self.method == "pooled_arcsinh_percentile":
            if not np.isfinite(self.cofactor) or self.cofactor <= 0:
                raise ValueError("cofactor must be finite and positive.")
            if not np.isfinite(self.percentile) or not 0 < self.percentile <= 100:
                raise ValueError("percentile must lie in (0, 100].")
        samples = list(images.values()) if isinstance(images, Mapping) else list(images)
        if not samples:
            raise ValueError("At least one training image is required.")
        shapes = [tuple(image.shape) for image in samples]
        if any(len(shape) != 3 or min(shape) < 1 for shape in shapes):
            raise ValueError("Images must have nonempty channel-first shape (C, H, W).")
        channels = {shape[0] for shape in shapes}
        if len(channels) != 1:
            raise ValueError("Training images must share channel count and order.")
        self.n_channels_ = channels.pop()
        self.channel_names_ = None if self.channel_names is None else tuple(map(str, self.channel_names))
        if self.channel_names_ is not None and len(self.channel_names_) != self.n_channels_:
            raise ValueError("channel_names must match the image channel count.")
        selected = range(self.n_channels_) if self.channel_indices is None else self.channel_indices
        selected = tuple(selected)
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in selected):
            raise ValueError("channel_indices must contain integer channel positions.")
        self.channel_indices_ = tuple(int(value) for value in selected)
        if len(set(self.channel_indices_)) != len(self.channel_indices_) or any(
            value < 0 or value >= self.n_channels_ for value in self.channel_indices_
        ):
            raise ValueError("channel_indices must be unique valid channel positions.")
        self.scaler_ = None
        self.bounds_ = None
        if self.method == "pooled_zscore":
            from sklearn.preprocessing import StandardScaler

            self.scaler_ = StandardScaler()
            for image in samples:
                array = _array(image)
                pixels = array.reshape(array.shape[0], -1).T
                for start in range(0, len(pixels), self.chunk_size):
                    self.scaler_.partial_fit(pixels[start : start + self.chunk_size])
        elif self.method == "pooled_arcsinh_percentile":
            total = sum(shape[1] * shape[2] for shape in shapes)
            bounds = []
            for channel in self.channel_indices_:
                pooled = np.empty(total, dtype=np.float32)
                offset = 0
                for image in samples:
                    values = _channel_float32(image, channel).reshape(-1)
                    pooled[offset : offset + len(values)] = np.arcsinh(values / self.cofactor)
                    offset += len(values)
                bounds.append(np.percentile(pooled, self.percentile))
            self.bounds_ = np.asarray(bounds, dtype=np.float32)
        self.provenance_ = {"method": self.method}
        if self.bounds_ is not None:
            self.provenance_.update(
                cofactor=float(self.cofactor),
                percentile=float(self.percentile),
                bounds=self.bounds_.copy(),
                bounds_dtype=str(self.bounds_.dtype),
                channel_indices=list(self.channel_indices_),
            )
        elif self.scaler_ is not None:
            self.provenance_.update(mean=self.scaler_.mean_.copy(), scale=self.scaler_.scale_.copy())
        if self.channel_names_ is not None:
            self.provenance_["marker_order"] = list(self.channel_names_)
        return self

    def transform(self, image, *, channel_names=None):
        """Apply fitted training statistics without refitting on this image."""
        if not hasattr(self, "n_channels_"):
            raise ValueError("Call fit() before transform().")
        shape = tuple(image.shape)
        if len(shape) != 3 or shape[0] != self.n_channels_:
            raise ValueError(f"Expected {self.n_channels_} channels in a (C, H, W) image.")
        if channel_names is not None:
            supplied = tuple(map(str, channel_names))
            if len(supplied) != self.n_channels_ or (
                self.channel_names_ is not None and supplied != self.channel_names_
            ):
                raise ValueError("Transform image has a different channel order.")
        if self.method == "none":
            return _array(image)
        if self.scaler_ is not None:
            array = _array(image)
            pixels = array.reshape(array.shape[0], -1).T
            transformed = np.empty(pixels.shape, dtype=np.float32)
            for start in range(0, len(pixels), self.chunk_size):
                stop = min(start + self.chunk_size, len(pixels))
                transformed[start:stop] = self.scaler_.transform(pixels[start:stop]).astype(np.float32)
            return transformed.T.reshape(shape)
        prepared = np.empty(shape, dtype=np.float32)
        bounds = dict(zip(self.channel_indices_, self.bounds_, strict=True))
        for channel in range(self.n_channels_):
            values = np.arcsinh(_channel_float32(image, channel) / self.cofactor)
            bound = bounds.get(channel)
            if bound is not None and bound > 0:
                values = np.clip(values / bound, 0.0, 1.0)
            prepared[channel] = values
        return prepared

    def transform_cohort(self, images):
        """Return transformed images with the original sample keys and order."""
        return {sample: self.transform(image) for sample, image in images.items()}
