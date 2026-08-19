"""Public data fetching and explicit-path loading utilities.

Provides automated download and caching of:
- Matrisome gene lists (MatrisomAnnotator, matrisome.org)
"""

from __future__ import annotations

import hashlib
import logging
import os
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import requests

from mantpy._version import __version__

if TYPE_CHECKING:

    from mantpy.ds import Bunch, ColIVIntestineBunch, LungBunch

_log = logging.getLogger(__name__)

__all__ = [
    "fetch_matrisome",
    "load_balbc_pbs_lung",
    "load_coliv_intestine",
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
