"""ImageContainer — multi-layer, spatially-aware image wrapper for IMC/CODEX images.

This class stores **named image layers**
(raw, preprocessed, masks, etc.) in a single object. AnnData integration uses
an H5AD-safe mapping made by :meth:`ImageContainer.to_serializable`; call
:func:`as_image_container` at the consumption boundary to recover the object
interface without placing a live Python object in ``AnnData.uns``.

Supports:

- Multiple named layers (e.g. ``"raw"``, ``"preprocessed"``, ``"mask"``)
- In-memory numpy arrays
- Lazy TIFF loading via tifffile (no extra deps)
- Dask-backed lazy arrays
- Spatial crop by pixel coordinates (returns new container with all layers cropped)
- Tile iteration for out-of-core processing of very large images
- Channel subsetting

Usage::

    ic = mt.im.ImageContainer("big_codex.tif")  # "image" layer auto-created
    ic.add_layer("preprocessed", enhanced_array)  # add named layer
    ic["preprocessed"]  # get layer as (C, H, W) array
    ic.layers  # list all layer names

    # Integration with AnnData:
    adata = mt.io.read_imc(ic, panel=panel, cells=cells)
    ic = mt.im.as_image_container(adata.uns["image_container"])
    mt.pp.preprocess_ecm(adata, method="frangi")  # writes "preprocessed" layer
    mt.pl.ecm_graph_overlay(adata)  # reads from container automatically
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "IMAGE_CONTAINER_SCHEMA",
    "IMAGE_CONTAINER_SCHEMA_VERSION",
    "ImageContainer",
    "as_image_container",
]


IMAGE_CONTAINER_SCHEMA = "mantpy.image_container"
"""Schema identifier used by :meth:`ImageContainer.to_serializable`."""

IMAGE_CONTAINER_SCHEMA_VERSION = 1
"""Current version of Mantpy's serializable image-container contract."""


class ImageContainer:
    """Multi-layer wrapper around spatially-registered multi-channel images.

    Each layer is stored as a ``(C, H, W)`` array sharing the same spatial dimensions.
    The default layer created at construction is named ``"image"`` (matching
    Squidpy convention).

    Parameters
    ----------
    img
        Path to a TIFF/common RGB raster or a pre-loaded 2-D/3-D numpy array.
        With ``lazy=True`` decoding is deferred; TIFFs remain tile-addressable.
    channel_names
        Optional list of channel names (length must match C).
    channel_axis
        Input channel axis. Arrays retain the legacy default of ``0``; TIFFs
        use their axes metadata (falling back to ``0``), and other RGB raster
        paths default to ``-1``. Pass ``-1`` for an in-memory ``(H, W, C)``
        image. Layers are stored in canonical channel-first order regardless.
    lazy
        Load via the Dask/Zarr dependencies included with Mantpy. When
        ``False`` (default) the array is loaded eagerly into RAM.
    scale
        Pixel-to-micron scale factor stored for downstream use.
    layer
        Name for the initial image layer. Default ``"image"``.

    Examples
    --------
    >>> ic = mt.im.ImageContainer("big_codex.tif", lazy=True)
    >>> ic.shape  # (C, H, W)
    >>> ic.layers  # ['image']
    >>> ic.add_layer("preprocessed", enhanced)
    >>> ic["preprocessed"]  # (C, H, W) array
    >>> tile = ic.crop(x=1000, y=2000, w=512, h=512)  # crops all layers
    """

    def __init__(
        self,
        img: str | Path | np.ndarray,
        *,
        channel_names: list[str] | None = None,
        channel_axis: int | None = None,
        lazy: bool = False,
        scale: float = 1.0,
        layer: str = "image",
    ) -> None:
        self._validate_layer_name(layer)
        self._path: Path | None = None
        self._lazy = lazy
        self.scale = scale
        self._layers: dict[str, Any] = {}
        self._default_layer: str = layer
        self.input_channel_axis = channel_axis
        # Layers are canonicalised to (C, H, W), irrespective of the input
        # layout.  Keeping this public makes the storage contract explicit.
        self.channel_axis = 0

        if isinstance(img, np.ndarray):
            arr = img.astype(np.float32) if not lazy else img
            # Arrays retain the historical channel-first default.  RGB users
            # can now opt into their native layout with ``channel_axis=-1``.
            arr = self._as_channel_first(arr, 0 if channel_axis is None else channel_axis)
            self._layers[layer] = arr
        else:
            self._path = Path(img)
            if not self._path.exists():
                raise FileNotFoundError(f"Image file not found: {self._path}")
            is_tiff = self._path.suffix.lower() in {".tif", ".tiff"}
            # TIFF stacks historically defaulted to (C, H, W).  Ordinary
            # RGB rasters are conventionally (H, W, C), so make paths such as
            # PNG/JPEG work without an otherwise surprising keyword.
            if channel_axis is not None:
                resolved_axis = channel_axis
            elif is_tiff:
                resolved_axis = self._tiff_channel_axis(self._path)
            else:
                resolved_axis = -1
            if lazy and is_tiff:
                self._layers[layer] = self._load_dask(self._path, resolved_axis)
            elif lazy:
                self._layers[layer] = self._load_raster_dask(self._path, resolved_axis)
            elif is_tiff:
                self._layers[layer] = self._load_tiff(self._path, resolved_axis)
            else:
                self._layers[layer] = self._load_raster(self._path, resolved_axis)

        if channel_names is not None:
            if len(channel_names) != self._layers[layer].shape[0]:
                raise ValueError(
                    f"channel_names has {len(channel_names)} entries but image has "
                    f"{self._layers[layer].shape[0]} channels."
                )
        self.channel_names = channel_names

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------

    @property
    def layers(self) -> list[str]:
        """List of all layer names."""
        return list(self._layers.keys())

    def add_layer(self, name: str, data: np.ndarray) -> None:
        """Add a named image layer.

        Parameters
        ----------
        name
            Layer name (e.g. ``"preprocessed"``, ``"mask"``).
        data
            ``(C, H, W)`` or ``(H, W)`` array. Must match spatial dimensions
            of existing layers. 2-D arrays are promoted to ``(1, H, W)``.

        Raises
        ------
        ValueError
            If spatial dimensions don't match.
        """
        self._validate_layer_name(name)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        if arr.ndim != 3:
            raise ValueError(f"Layer data must be (C, H, W) or (H, W), got shape {data.shape}.")
        if self._layers:
            ref = next(iter(self._layers.values()))
            if arr.shape[1:] != ref.shape[1:]:
                raise ValueError(
                    f"Spatial dimensions mismatch: new layer has shape {arr.shape[1:]}, "
                    f"existing layers have {ref.shape[1:]}."
                )
        self._layers[name] = arr

    def get_layer(self, name: str, *, compute: bool = True):
        """Get a layer by name as a ``(C, H, W)`` numpy array.

        Triggers Dask compute if the layer is lazy unless ``compute=False``.
        The latter is useful for tile-wise algorithms that must not
        materialise a whole-slide image.
        """
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' not found. Available: {self.layers}")
        arr = self._layers[name]
        if not compute:
            return arr
        if hasattr(arr, "compute"):
            arr = arr.compute()
            self._layers[name] = arr
        return np.asarray(arr, dtype=np.float32)

    def has_layer(self, name: str) -> bool:
        """Check if a layer exists."""
        return name in self._layers

    def drop_layer(self, name: str) -> None:
        """Remove a layer by name."""
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' not found. Available: {self.layers}")
        if name == self._default_layer and len(self._layers) == 1:
            raise ValueError("Cannot drop the only remaining layer.")
        del self._layers[name]

    def to_serializable(self) -> dict[str, Any]:
        """Return an H5AD-safe representation of this container.

        The returned mapping contains only scalars, NumPy arrays, and nested
        mappings supported by AnnData's HDF5/Zarr writers. Image layers are
        materialised in canonical ``(C, H, W)`` order so no live Dask array,
        open TIFF store, or :class:`ImageContainer` is placed in ``AnnData.uns``.

        Returns
        -------
        dict
            Mapping with ``schema``, ``schema_version``, ``default_layer``,
            ``scale``, ``channel_axis``, and ``layers`` fields. The optional
            ``channel_names`` field is a one-dimensional string array.
        """
        payload: dict[str, Any] = {
            "schema": IMAGE_CONTAINER_SCHEMA,
            "schema_version": IMAGE_CONTAINER_SCHEMA_VERSION,
            "default_layer": self._default_layer,
            "scale": float(self.scale),
            "channel_axis": 0,
            "layers": {name: self.get_layer(name) for name in self.layers},
        }
        if self.channel_names is not None:
            payload["channel_names"] = np.asarray(self.channel_names, dtype=str)
        return payload

    @classmethod
    def from_serializable(cls, payload: Mapping[str, Any]) -> ImageContainer:
        """Reconstruct a container view from Mantpy's serializable mapping.

        The reconstructed container shares its layer mapping with a mutable
        payload. Consequently, :meth:`add_layer` and :meth:`drop_layer` keep an
        ``AnnData.uns`` payload updated in memory without placing a live Python
        object there.

        Parameters
        ----------
        payload
            Mapping produced by :meth:`to_serializable`, including one or more
            canonical channel-first arrays under ``payload['layers']``.

        Returns
        -------
        ImageContainer
            Eager in-memory view over the serialized layer arrays.
        """
        if not isinstance(payload, Mapping):
            raise TypeError(
                "Serialized ImageContainer must be a mapping; "
                f"got {type(payload).__name__}."
            )
        if payload.get("schema") != IMAGE_CONTAINER_SCHEMA:
            raise ValueError(
                "Unsupported image-container schema. Expected "
                f"{IMAGE_CONTAINER_SCHEMA!r}, got {payload.get('schema')!r}."
            )
        version = payload.get("schema_version")
        if version != IMAGE_CONTAINER_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported image-container schema version. Expected "
                f"{IMAGE_CONTAINER_SCHEMA_VERSION}, got {version!r}."
            )

        layers = payload.get("layers")
        if not isinstance(layers, Mapping) or not layers:
            raise ValueError("Serialized ImageContainer must contain a non-empty 'layers' mapping.")

        default_layer = payload.get("default_layer")
        if not isinstance(default_layer, str) or default_layer not in layers:
            raise ValueError(
                "Serialized ImageContainer 'default_layer' must name an entry in 'layers'."
            )

        spatial_shape: tuple[int, int] | None = None
        for name, data in layers.items():
            cls._validate_layer_name(name)
            shape = getattr(data, "shape", None)
            if shape is None or len(shape) != 3:
                raise ValueError(
                    f"Serialized layer {name!r} must have shape (C, H, W); got {shape}."
                )
            layer_spatial_shape = (int(shape[1]), int(shape[2]))
            if spatial_shape is None:
                spatial_shape = layer_spatial_shape
            elif layer_spatial_shape != spatial_shape:
                raise ValueError(
                    f"Serialized layer {name!r} has spatial shape {layer_spatial_shape}; "
                    f"expected {spatial_shape}."
                )

        channel_axis = payload.get("channel_axis")
        if channel_axis != 0:
            raise ValueError(
                "Serialized ImageContainer layers must be channel-first "
                f"(channel_axis=0); got {channel_axis!r}."
            )

        channel_names_value = payload.get("channel_names")
        channel_names = None
        if channel_names_value is not None:
            channel_names_array = np.asarray(channel_names_value)
            if channel_names_array.ndim != 1:
                raise ValueError("Serialized ImageContainer 'channel_names' must be one-dimensional.")
            channel_names = [str(name) for name in channel_names_array.tolist()]
            default_channels = int(layers[default_layer].shape[0])
            if len(channel_names) != default_channels:
                raise ValueError(
                    "Serialized ImageContainer has "
                    f"{len(channel_names)} channel names but its default layer has "
                    f"{default_channels} channels."
                )

        ic = cls.__new__(cls)
        ic._path = None
        ic._lazy = False
        ic.scale = float(payload.get("scale", 1.0))
        ic.input_channel_axis = 0
        ic.channel_axis = 0
        ic._default_layer = default_layer
        # Keep the mutable mapping by reference so layer additions/removals
        # remain serializable and immediately visible in AnnData.uns.
        ic._layers = layers if isinstance(layers, MutableMapping) else dict(layers)
        ic.channel_names = channel_names
        return ic

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        """(C, H, W) shape of the default layer."""
        data = self._layers[self._default_layer]
        c, h, w = data.shape
        return (int(c), int(h), int(w))

    @property
    def n_channels(self) -> int:
        """Number of channels ``C`` in the default layer."""
        return self.shape[0]

    @property
    def height(self) -> int:
        """Image height ``H`` (rows) of the default layer."""
        return self.shape[1]

    @property
    def width(self) -> int:
        """Image width ``W`` (columns) of the default layer."""
        return self.shape[2]

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def to_array(self, dtype: type = np.float32, layer: str | None = None) -> np.ndarray:
        """Return a layer as a ``(C, H, W)`` numpy array.

        Parameters
        ----------
        dtype
            Output dtype.
        layer
            Layer name. ``None`` returns the default (``"image"``) layer.
        """
        name = layer if layer is not None else self._default_layer
        return self.get_layer(name).astype(dtype)

    def __getitem__(self, key: str | int | list[int] | slice) -> np.ndarray | ImageContainer:
        """Access layers by name or subset channels by index.

        - String key → returns the layer array ``(C, H, W)``
        - Integer/list/slice → channel subset of default layer, returns new ImageContainer
        """
        if isinstance(key, str):
            return self.get_layer(key)
        # Channel subsetting
        channels = key
        if isinstance(channels, int):
            channels = [channels]
        data = self._layers[self._default_layer][channels]
        names = None
        if self.channel_names is not None:
            if isinstance(channels, slice):
                names = self.channel_names[channels]
            else:
                names = [self.channel_names[i] for i in channels]
        ic = ImageContainer.__new__(ImageContainer)
        ic._path = self._path
        ic._lazy = self._lazy
        ic.scale = self.scale
        ic.input_channel_axis = self.input_channel_axis
        ic.channel_axis = self.channel_axis
        ic._default_layer = self._default_layer
        ic._layers = {self._default_layer: data}
        # Copy other layers with same channel subset where possible
        for lname, ldata in self._layers.items():
            if lname == self._default_layer:
                continue
            if ldata.shape[0] == self._layers[self._default_layer].shape[0]:
                ic._layers[lname] = ldata[channels]
        ic.channel_names = names
        return ic

    # ------------------------------------------------------------------
    # Spatial operations
    # ------------------------------------------------------------------

    def crop(
        self,
        *,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> ImageContainer:
        """Return a spatial crop as a new ImageContainer (all layers cropped).

        Parameters
        ----------
        x, y
            Top-left corner of the crop (pixel coordinates).
        w, h
            Width and height of the crop in pixels.

        Returns
        -------
        ImageContainer
            New container with all layers cropped to the same region.
        """
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + w)
        y1 = min(self.height, y + h)

        ic = ImageContainer.__new__(ImageContainer)
        ic._path = None
        ic._lazy = self._lazy
        ic.scale = self.scale
        ic.input_channel_axis = self.input_channel_axis
        ic.channel_axis = self.channel_axis
        ic._default_layer = self._default_layer
        ic._layers = {name: data[:, y0:y1, x0:x1] for name, data in self._layers.items()}
        ic.channel_names = self.channel_names
        return ic

    def tile_iter(
        self,
        tile_size: int = 1024,
        overlap: int = 0,
    ) -> Generator[tuple[ImageContainer, dict], None, None]:
        """Iterate over non-overlapping (or overlapping) spatial tiles.

        Yields ``(tile_container, info_dict)`` pairs.  ``info_dict`` contains
        ``x0``, ``y0``, ``x1``, ``y1`` — the pixel bounding box of the tile
        in the *original* image coordinate frame.

        Parameters
        ----------
        tile_size
            Side length in pixels of each square tile.
        overlap
            Number of pixels to extend each tile on all sides (for context).
            The reported bounding box in ``info_dict`` is the *inner* tile
            without the overlap padding.

        Yields
        ------
        (ImageContainer, dict)
        """
        step = tile_size
        for y0 in range(0, self.height, step):
            for x0 in range(0, self.width, step):
                x1 = min(self.width, x0 + tile_size)
                y1 = min(self.height, y0 + tile_size)
                xc0 = max(0, x0 - overlap)
                yc0 = max(0, y0 - overlap)
                xc1 = min(self.width, x1 + overlap)
                yc1 = min(self.height, y1 + overlap)
                tile = self.crop(x=xc0, y=yc0, w=xc1 - xc0, h=yc1 - yc0)
                info = {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "xc0": xc0,
                    "yc0": yc0,
                }
                yield tile, info

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def max_projection(self, layer: str | None = None) -> np.ndarray:
        """Return (H, W) max-projection across channels (numpy, float32).

        Parameters
        ----------
        layer
            Layer to project. ``None`` uses the default layer.
        """
        arr = self.to_array(layer=layer)
        proj = arr.max(axis=0).astype(np.float32)
        if proj.max() > 0:
            proj = proj / proj.max()
        return proj

    def show(
        self,
        *,
        layer: str | None = None,
        ax=None,
        cmap: str = "gray",
        save: str | Path | None = None,
        show: bool = True,
    ):
        """Quick max-projection plot of a layer.

        Parameters
        ----------
        layer
            Layer to display. ``None`` uses the default layer.
        ax
            Matplotlib Axes; created if ``None``.
        cmap
            Colormap for display.
        save
            File path to save.
        show
            Call ``plt.show()``.

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        name = layer if layer is not None else self._default_layer
        ax.imshow(self.max_projection(layer=name), cmap=cmap, origin="upper")
        ax.set_title(f"ImageContainer['{name}']  {self.shape}")
        ax.axis("off")

        if save is not None:
            fig.savefig(str(save), bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return ax

    def __repr__(self) -> str:
        src = str(self._path) if self._path else "array"
        lazy_tag = " [lazy]" if self._lazy else ""
        layer_info = ", ".join(self.layers)
        return (
            f"ImageContainer(shape={self.shape}, layers=[{layer_info}], scale={self.scale}, source='{src}'{lazy_tag})"
        )

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_layer_name(name: str) -> None:
        """Reject names that cannot round-trip as HDF5 mapping keys."""
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(
                "ImageContainer layer names must be non-empty strings without '/'; "
                f"got {name!r}."
            )

    @staticmethod
    def _as_channel_first(arr, channel_axis: int):
        """Return an array-like in canonical ``(C, H, W)`` layout."""
        if arr.ndim == 2:
            return arr[np.newaxis]
        if arr.ndim != 3:
            raise ValueError(
                f"img must be (C, H, W), (H, W, C), or a 2-D image; got shape {arr.shape}."
            )
        if not isinstance(channel_axis, int) or not -arr.ndim <= channel_axis < arr.ndim:
            raise ValueError(f"channel_axis={channel_axis!r} is invalid for shape {arr.shape}.")
        axis = channel_axis % arr.ndim
        return arr if axis == 0 else np.moveaxis(arr, axis, 0)

    @classmethod
    def _load_tiff(cls, path: Path, channel_axis: int) -> np.ndarray:
        import tifffile

        arr = tifffile.imread(str(path)).astype(np.float32)
        try:
            return cls._as_channel_first(arr, channel_axis)
        except ValueError as exc:
            raise ValueError(f"TIFF at {path} has unsupported shape {arr.shape}.") from exc

    @staticmethod
    def _tiff_channel_axis(path: Path) -> int:
        """Infer RGB/sample axes while retaining channel-first TIFF defaults."""
        import tifffile

        with tifffile.TiffFile(path) as tif:
            axes = tif.series[0].axes
        if len(axes) == 3:
            for marker in ("C", "S"):
                if marker in axes:
                    return axes.index(marker)
        return 0

    @classmethod
    def _load_dask(cls, path: Path, channel_axis: int):
        try:
            import dask.array as da
            import tifffile
            store = tifffile.imread(str(path), aszarr=True)
            arr = da.from_zarr(store)
        except ImportError as exc:
            raise ImportError(
                "Lazy image loading requires Mantpy's base Dask, Zarr, and tifffile dependencies. "
                "Reinstall Mantpy with: pip install --force-reinstall mantpy"
            ) from exc
        try:
            return cls._as_channel_first(arr, channel_axis).astype(np.float32)
        except ValueError as exc:
            raise ValueError(f"TIFF at {path} has unsupported shape {arr.shape}.") from exc

    @classmethod
    def _load_raster(cls, path: Path, channel_axis: int) -> np.ndarray:
        """Load a common RGB raster (PNG/JPEG/BMP/WebP) via Pillow."""
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        try:
            with Image.open(path) as image:
                arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        except Exception as exc:  # Pillow raises format-specific subclasses
            raise ValueError(f"Could not read image raster at {path}.") from exc
        return cls._as_channel_first(arr, channel_axis)

    @classmethod
    def _load_raster_dask(cls, path: Path, channel_axis: int):
        """Defer decoding of a common RGB raster until its data are requested."""
        try:
            import dask.array as da
            from dask import delayed
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Lazy image loading requires Mantpy's base Dask dependency. "
                "Reinstall Mantpy with: pip install --force-reinstall mantpy"
            ) from exc

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as image:
            width, height = image.size

        @delayed
        def _decode(filename: str) -> np.ndarray:
            with Image.open(filename) as image:
                # Keep the deferred whole-image decode at its native uint8
                # footprint.  Downstream slices are promoted only after the
                # requested preview/tile has been selected.
                return np.asarray(image.convert("RGB"), dtype=np.uint8)

        arr = da.from_delayed(_decode(str(path)), shape=(height, width, 3), dtype=np.uint8)
        return cls._as_channel_first(arr, channel_axis)


def as_image_container(value: ImageContainer | Mapping[str, Any]) -> ImageContainer:
    """Coerce a legacy container or serialized payload to ``ImageContainer``.

    This is the compatibility boundary for code consuming
    ``adata.uns['image_container']``. Readers now store an H5AD-safe mapping at
    that key; objects created by older Mantpy versions may still hold a live
    :class:`ImageContainer` in memory.

    Parameters
    ----------
    value
        A live :class:`ImageContainer` or a mapping returned by
        :meth:`ImageContainer.to_serializable`.

    Returns
    -------
    ImageContainer
        ``value`` unchanged when it is already a container, otherwise a view
        reconstructed from the serialized mapping.
    """
    if isinstance(value, ImageContainer):
        return value
    return ImageContainer.from_serializable(value)
