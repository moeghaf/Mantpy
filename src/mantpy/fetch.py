"""Public data fetching and explicit-path loading utilities.

Provides automated download and caching of:
- Matrisome gene lists (MatrisomAnnotator, matrisome.org)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import requests

from mantpy._version import __version__

if TYPE_CHECKING:
    from anndata import AnnData

    from mantpy.ds import Bunch, ColIVIntestineBunch, LungBunch

_log = logging.getLogger(__name__)

__all__ = [
    "fetch_matrisome",
    "load_balbc_pbs_lung",
    "load_coliv_intestine",
    "load_prostate_cell_segmentation",
    "load_prostate_he_visium",
    "load_schistosoma_ecm_cohort",
]

# Google Sheets exports for Matrisome Project masterlists (UIC, Bhatt lab).
# Source: https://sites.google.com/uic.edu/matrisome/matrisome-annotations
_MATRISOME_URLS: dict[str, list[str]] = {
    "mouse": ["https://docs.google.com/spreadsheets/d/1Te6n2q_cisXeirzBClK-VzA6T-zioOB5/export?format=csv"],
    "human": ["https://docs.google.com/spreadsheets/d/1GwwV3pFvsp7DKBbCgr8kLpf8Eh_xV8ks/export?format=csv"],
    "zebrafish": ["https://docs.google.com/spreadsheets/d/1KtQbMGz2_3Rg2UbOBogvUazUMI3uT2eD/export?format=csv"],
}

# Exact bytes served by the authoritative public exports on 2026-07-24. A
# source update is reviewed and released explicitly instead of silently
# changing a tutorial's gene universe.
_MATRISOME_SHA256: dict[str, str] = {
    "mouse": "b61008101fdaa242020a9a62928519c9e777b8350fbfea6f52da08185bc08ea1",
    "human": "de2f911a7362519e1941cb654a83a1cd0476372841001de014659fefa77b38c6",
    "zebrafish": "09b7e5cab5bd7272542152f629b69819dc32535ff455f5ec5c7049f756d0a77a",
}

MATRISOME_CACHE_FNAME = "{species}_matrisome_masterlist.csv"

_PROSTATE_HE_VISIUM_SAMPLES = {
    "normal": {
        "slug": "Visium_FFPE_Human_Normal_Prostate",
        "image_suffix": ".jpg",
        "diagnosis": "normal prostate",
    },
    "adeno": {
        "slug": "Visium_FFPE_Human_Prostate_Cancer",
        "image_suffix": ".tif",
        "diagnosis": "adenocarcinoma and invasive carcinoma",
    },
    "acinar": {
        "slug": "Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma",
        "image_suffix": ".tif",
        "diagnosis": "acinar cell carcinoma",
    },
}
_PROSTATE_HE_VISIUM_ALIASES = {
    "adenocarcinoma": "adeno",
    "cancer": "adeno",
    "acinar_carcinoma": "acinar",
}
_VISIUM_SPOT_PITCH_UM = 100.0
_VISIUM_SPOT_FOOTPRINT_UM = 65.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _NoAmbientAuth(requests.auth.AuthBase):
    """Disable requests' automatic ``.netrc`` authentication."""

    def __call__(self, request):
        request.headers.pop("Authorization", None)
        return request


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        path = Path(cache_dir).expanduser()
    elif value := os.environ.get("MANTPY_CACHE"):
        path = Path(value).expanduser()
    else:
        path = Path.home() / ".cache" / "mantpy"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_cache(
    url: str,
    dest: Path,
    *,
    min_bytes: int = 1024,
    expected_sha256: str | None = None,
) -> Path:
    """Download *url* to *dest* without forwarding ambient credentials.

    Public dataset downloads must be usable anonymously. This helper
    deliberately does not inspect or forward process credentials.
    """
    if min_bytes < 0:
        raise ValueError(f"min_bytes must be non-negative, got {min_bytes}.")
    dest = Path(dest)
    if (
        dest.exists()
        and dest.stat().st_size >= min_bytes
        and (expected_sha256 is None or _file_sha256(dest) == expected_sha256.lower())
    ):
        _log.info("Using cached %s", dest)
        return dest

    headers = {"User-Agent": f"mantpy/{__version__}"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        _log.info("Downloading %s to %s", url, dest)
        resp = requests.get(
            url,
            headers=headers,
            auth=_NoAmbientAuth(),
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        if tmp.stat().st_size < min_bytes:
            raise RuntimeError(
                f"Downloaded file {dest.name!r} is smaller than the required "
                f"minimum ({tmp.stat().st_size} < {min_bytes} bytes)."
            )
        if expected_sha256 is not None:
            observed = _file_sha256(tmp)
            if observed != expected_sha256.lower():
                raise RuntimeError(
                    f"Downloaded file {dest.name!r} failed SHA-256 verification "
                    f"(expected {expected_sha256.lower()}, observed {observed})."
                )
        tmp.replace(dest)
    except requests.exceptions.RequestException as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download from {url!r}.\nError: {exc}\n"
            f"Download the file manually as {dest.name!r} in the requested cache directory."
        ) from exc
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return dest


def _safe_extract_tar_archive(archive: str | Path, target: str | Path) -> Path:
    """Extract a data-only tar archive beneath *target*.

    Absolute paths, parent traversal, links, devices, and other special
    members are rejected before any member is written. The additional
    validation is kept even though ``tarfile``'s ``data`` filter is also used,
    making the safety boundary explicit on every supported Python version.
    """
    archive = Path(archive)
    target = Path(target)
    root = target.resolve()
    with tarfile.open(archive, mode="r:*") as tar:
        members = tar.getmembers()
        for member in members:
            destination = (root / member.name).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe member {member.name!r} in {archive.name!r}.") from exc
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsafe non-data member {member.name!r} in {archive.name!r}.")
        target.mkdir(parents=True, exist_ok=True)
        tar.extractall(path=target, members=members, filter="data")
    return target


def _extract_10x_spatial_archive(archive: Path, target: Path, *, force: bool = False) -> Path:
    """Extract a 10x ``spatial.tar.gz`` without permitting path traversal."""
    spatial = target / "spatial"
    required = (
        spatial / "tissue_positions_list.csv",
        spatial / "scalefactors_json.json",
    )
    if not force and all(path.exists() for path in required):
        return spatial

    _safe_extract_tar_archive(archive, target)
    if not all(path.exists() for path in required):
        raise RuntimeError(f"10x spatial archive {archive} is missing required files.")
    return spatial


def _visium_image_mpp(spatial_px: np.ndarray, scalefactors: Mapping[str, object]) -> tuple[float, str]:
    """Calibrate full-resolution microns/pixel from the 100 µm spot pitch."""
    coords = np.asarray(spatial_px, dtype=float)
    coords = coords[np.isfinite(coords).all(axis=1)]
    if len(coords) >= 2:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(coords).query(coords, k=2)
        nearest = distances[:, 1]
        nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
        if nearest.size:
            return _VISIUM_SPOT_PITCH_UM / float(np.median(nearest)), "median_100um_spot_pitch"

    spot_diameter = float(scalefactors.get("spot_diameter_fullres", 0.0))
    if spot_diameter > 0:
        return _VISIUM_SPOT_FOOTPRINT_UM / spot_diameter, "65um_spot_footprint_fallback"
    raise ValueError("Could not calibrate image_mpp from Visium positions or scalefactors.")


def load_prostate_he_visium(
    sample: Literal["normal", "adeno", "acinar"] = "adeno",
    *,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
    lazy: bool = True,
) -> Bunch:
    """Load a 10x FFPE prostate H&E/Visium sample.

    The default adenocarcinoma sample delegates to the verified public tutorial
    bundle returned by :func:`mantpy.datasets.prostate_he_visium`. The normal
    and acinar samples retain the historical direct 10x download path. Setting
    ``force_refresh=True`` also selects that direct path for compatibility.

    Parameters
    ----------
    sample
        ``"normal"``, ``"adeno"`` (adenocarcinoma/invasive carcinoma), or
        ``"acinar"``.  The longer aliases ``"adenocarcinoma"`` and
        ``"acinar_carcinoma"`` are also accepted.
    cache_dir
        Cache root. The ``MANTPY_CACHE`` environment variable and then
        ``~/.cache/mantpy`` are used when omitted.
    force_refresh
        Use the historical direct 10x path and re-download its three raw files.
    lazy
        Keep the full-resolution image lazy through Mantpy's Dask/Zarr-backed
        image container. Set to ``False`` only for small tests or machines with
        enough memory.

    Returns
    -------
    mantpy.ds.Bunch
        Attributes ``image``, ``visium``, ``image_mpp``, ``spot_radius_um``,
        ``paths``, and ``provenance``. ``visium.obsm['spatial']`` is in
        full-resolution pixels and ``visium.obsm['spatial_um']`` is in µm.
    """
    key = _PROSTATE_HE_VISIUM_ALIASES.get(str(sample).lower(), str(sample).lower())
    if key not in _PROSTATE_HE_VISIUM_SAMPLES:
        valid = ", ".join(sorted(_PROSTATE_HE_VISIUM_SAMPLES))
        raise ValueError(f"Unknown prostate sample {sample!r}. Choose one of: {valid}.")
    if key == "adeno" and not force_refresh:
        from mantpy.datasets import prostate_he_visium

        return prostate_he_visium(cache_dir=cache_dir, lazy=lazy)

    from mantpy.ds import Bunch
    from mantpy.im import ImageContainer

    spec = _PROSTATE_HE_VISIUM_SAMPLES[key]
    slug = str(spec["slug"])
    base_url = f"https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/{slug}"
    target = _resolve_cache_dir(cache_dir) / "prostate_he_visium" / key
    target.mkdir(parents=True, exist_ok=True)

    filenames = {
        "counts": f"{slug}_filtered_feature_bc_matrix.h5",
        "spatial_archive": f"{slug}_spatial.tar.gz",
        "image": f"{slug}_image{spec['image_suffix']}",
    }
    paths = {name: target / filename for name, filename in filenames.items()}
    if force_refresh:
        for path in paths.values():
            path.unlink(missing_ok=True)
    for name, path in paths.items():
        _download_with_cache(f"{base_url}/{filenames[name]}", path, min_bytes=1)
    spatial_dir = _extract_10x_spatial_archive(paths["spatial_archive"], target, force=force_refresh)
    paths["spatial"] = spatial_dir

    # Squidpy's reader is the current scverse Visium entry point.  It reads the
    # small hires/lowres assets but records (rather than decodes) the source WSI.
    import squidpy as sq

    visium = sq.read.visium(
        target,
        counts_file=paths["counts"].name,
        library_id=slug,
        load_images=True,
        source_image_path=paths["image"],
    )
    duplicate_var_names = int(visium.var_names.duplicated().sum())
    if duplicate_var_names:
        # Only the display/index names change (e.g. TBCE -> TBCE-1); stable
        # 10x identifiers in var['gene_ids'] remain byte-for-byte untouched.
        visium.var_names_make_unique(join="-")
    scalefactors = json.loads((spatial_dir / "scalefactors_json.json").read_text(encoding="utf-8"))
    image_mpp, scale_method = _visium_image_mpp(visium.obsm["spatial"], scalefactors)
    spot_diameter_px = float(scalefactors.get("spot_diameter_fullres", 0.0))
    spot_radius_um = 0.5 * spot_diameter_px * image_mpp if spot_diameter_px > 0 else 0.5 * _VISIUM_SPOT_FOOTPRINT_UM
    visium.obsm["spatial_um"] = np.asarray(visium.obsm["spatial"], dtype=np.float32) * np.float32(image_mpp)

    provenance = {
        "provider": "10x Genomics",
        "dataset": slug,
        "sample": key,
        "diagnosis": spec["diagnosis"],
        "space_ranger_version": "1.3.0",
        "license": "CC BY 4.0",
        "source_url": base_url,
        "scale_method": scale_method,
        "spot_pitch_um": _VISIUM_SPOT_PITCH_UM,
        "var_names_make_unique": {
            "applied": bool(duplicate_var_names),
            "duplicate_count": duplicate_var_names,
            "join": "-",
            "gene_ids_preserved": "var['gene_ids']",
        },
        # This mapping is persisted in ``visium.uns``. Keep it portable and
        # avoid embedding a user's cache root in exported h5ad files.
        "files": {name: path.name for name, path in paths.items()},
    }
    visium.uns.setdefault("mantpy", {})["prostate_he_visium"] = {
        "image_mpp": image_mpp,
        "spot_radius_um": spot_radius_um,
        "provenance": provenance,
    }
    image = ImageContainer(
        paths["image"],
        channel_axis=-1,
        channel_names=["red", "green", "blue"],
        lazy=lazy,
        scale=image_mpp,
    )
    return Bunch(
        image=image,
        visium=visium,
        image_mpp=image_mpp,
        spot_radius_um=spot_radius_um,
        paths=paths,
        provenance=provenance,
    )


def load_prostate_cell_segmentation(
    path: str | Path | None = None,
) -> AnnData:
    """Load the deposited coordinate-only Cellpose result for the prostate tutorial.

    The object contains only nuclear centroids, physical coordinates and the
    recorded segmentation provenance. It contains no ECM patches, graph,
    anatomical labels, transcriptomic values, or derived analysis result. Pass an
    explicit path to load it from another location; with ``path=None`` this
    compatibility API returns ``mt.datasets.prostate_he_visium().cells``.
    """
    if path is None:
        from mantpy.datasets import prostate_he_visium

        return prostate_he_visium().cells

    from mantpy.datasets._loaders import load_prostate_cells

    return load_prostate_cells(path)


def _normalise_matrisome_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        # standard dot-separated (R export) and space-separated (Google Sheets) variants
        "Gene.Symbol": "gene_symbol",
        "Gene Symbol": "gene_symbol",
        "Zebrafish Gene Symbol": "gene_symbol",  # zebrafish sheet
        "Matrisome.Division": "division",
        "Matrisome Division": "division",
        "Matrisome.Category": "category",
        "Matrisome Category": "category",
    }
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    if "gene_symbol" not in rename.values():
        raise ValueError(
            f"Cannot find gene symbol column in matrisome CSV. "
            f"Expected one of {[k for k in col_map if 'Gene' in k]}. "
            f"Found columns: {list(df.columns)}"
        )
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_matrisome(
    species: Literal["mouse", "human", "zebrafish"] = "mouse",
    *,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download the Matrisome Project masterlist for the given species.

    Data source: UIC Matrisome Project Google Sheets
    (https://sites.google.com/uic.edu/matrisome/matrisome-annotations).

    Parameters
    ----------
    species
        ``"mouse"`` (default), ``"human"``, or ``"zebrafish"``.
    cache_dir
        Directory for cached files.  Defaults to ``~/.cache/mantpy/``.

    Returns
    -------
    pd.DataFrame
        Columns: ``gene_symbol`` (str, uppercase), ``category`` (str),
        ``division`` (str).

    Raises
    ------
    ValueError
        If *species* is not a supported value.
    RuntimeError
        If the download fails and no cache exists.  Manual fallback: download
        the masterlist from https://matrisomedb.org/ (select species, click
        Download) and save it to the path shown in the error message.
    """
    valid = frozenset(_MATRISOME_URLS)
    if species not in valid:
        raise ValueError(f"species must be one of {sorted(valid)!r}, got {species!r}.")

    cache = _resolve_cache_dir(cache_dir)
    dest = cache / MATRISOME_CACHE_FNAME.format(species=species)

    for url in _MATRISOME_URLS[species]:
        try:
            _download_with_cache(
                url,
                dest,
                min_bytes=1,
                expected_sha256=_MATRISOME_SHA256[species],
            )
            break
        except RuntimeError:
            if dest.exists() and dest.stat().st_size >= 1 and _file_sha256(dest) == _MATRISOME_SHA256[species]:
                break
            continue
    else:
        if not dest.exists() or _file_sha256(dest) != _MATRISOME_SHA256[species]:
            gsheet_url = _MATRISOME_URLS[species][0].removesuffix("/export?format=csv")
            raise RuntimeError(
                f"Automatic download failed for {species!r} matrisome and no verified cache exists.\n"
                f"Manual options:\n"
                f"  1. Open {gsheet_url}\n"
                f"     File → Download → CSV, then save to: {dest}\n"
                f"  2. Download from https://matrisomedb.org/ (select species + Download)\n"
                f"     and save to: {dest}"
            )

    df = pd.read_csv(dest)
    df = _normalise_matrisome_columns(df)

    keep = [c for c in ("gene_symbol", "category", "division") if c in df.columns]
    df = df[keep].copy()
    df["gene_symbol"] = df["gene_symbol"].astype(str).str.upper()
    df = df[df["gene_symbol"].notna() & (df["gene_symbol"] != "NAN")].reset_index(drop=True)

    _log.info("fetch_matrisome: %d %s matrisome genes loaded.", len(df), species)
    return df


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _bundle_cache_path(root: Path, filename: str) -> Path:
    """Return a flat cache destination for an externally supplied filename."""
    filename = str(filename)
    if (
        not filename
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise RuntimeError(f"Unsafe dataset filename from remote manifest: {filename!r}.")
    return root / filename


def load_coliv_intestine(
    image: str | Path | np.ndarray | None = None,
    *,
    reference_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> ColIVIntestineBunch:
    """Load the raw Collagen-IV channel and held-out intestine annotation.

    This loader is intentionally a raw-data boundary.  It reads one untouched
    two-dimensional ECM channel and the separate reference labels used only to
    evaluate the tutorial.  It does not clip intensities, detect foreground,
    make patches, calculate features, or construct a graph.

    Parameters
    ----------
    image
        Raw Collagen-IV image as a two-dimensional array or a single-plane TIFF.
        Select the ECM plane before calling this loader when the acquisition is
        stored as a multiplexed stack.
    reference_path
        ``.npz`` containing ``grid_y``, ``grid_x``, ``annotation``, and
        ``layer_names``. This path is required; the held-out annotation is
        never discovered from a neighboring repository.

    Returns
    -------
    ColIVIntestineBunch
        Attribute-access bundle with ``image``/``raw_image``, ``reference``,
        ``layer_names``, ``layer_palette``, ``pixel_size_um``, and dataset metadata.

    Notes
    -----
    The intestine example uses one ECM channel.  The reusable
    :func:`mantpy.pp.image_ecm_patches` function accepts either one channel or
    a channel-first stack of multiple ECM markers.
    """
    if image is None:
        if reference_path is not None:
            raise ValueError("reference_path requires an explicit image; omit both to use the public dataset.")
        from mantpy.datasets import coliv_intestine

        return coliv_intestine(cache_dir=cache_dir)

    if reference_path is None:
        raise FileNotFoundError("The held-out reference requires an explicit reference_path=... NPZ file.")
    from mantpy.datasets._loaders import load_coliv_files

    return load_coliv_files(image, reference_path)


def load_balbc_pbs_lung(
    zenodo_dir: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> LungBunch:
    """Load the public BALB/c PBS mouse-lung cohort.

    This compatibility wrapper preserves the historical ``mt.fetch`` entry
    point while delegating all loading and verification to
    :func:`mantpy.datasets.balbc_pbs_lung`.

    Parameters
    ----------
    zenodo_dir
        Optional already-extracted schema-v1 bundle for a verified local
        mirror. Forwarded as ``source_dir`` to the canonical dataset loader.
    cache_dir
        Cache root. The ``MANTPY_CACHE`` environment variable and then
        ``~/.cache/mantpy`` are used when omitted.

    Returns
    -------
    LungBunch
        The same attribute-access dataset bundle returned by
        :func:`mantpy.datasets.balbc_pbs_lung`.
    """
    from mantpy.datasets import balbc_pbs_lung

    return balbc_pbs_lung(source_dir=zenodo_dir, cache_dir=cache_dir)


def load_schistosoma_ecm_cohort(
    zenodo_dir: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    include_raw: bool = False,
    raw_dir: str | Path | None = None,
    verbose: bool = True,
) -> Bunch:
    """Load the public Schistosoma ECM cohort.

    ``zenodo_dir`` and ``raw_dir`` remain accepted as local overrides for
    established scripts. The public bundle already includes raw inputs, so
    ``include_raw`` no longer changes the returned schema.
    """
    from mantpy.datasets import schistosoma_ecm

    source_dir = zenodo_dir if zenodo_dir is not None else raw_dir
    del include_raw
    return schistosoma_ecm(
        cache_dir=cache_dir,
        source_dir=source_dir,
        verbose=verbose,
    )
