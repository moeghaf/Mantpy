"""Verified download and extraction for public Mantpy datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import pooch
from requests.auth import AuthBase
from tqdm.auto import tqdm

from mantpy._version import __version__

from . import _registry as registry_module

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ALLOWED_DATA_SUFFIXES = (
    ".csv",
    ".h5",
    ".h5ad",
    ".json",
    ".npz",
    ".tar.gz",
    ".ome.tif",
    ".ome.tiff",
    ".tif",
    ".tiff",
)

_DOWNLOAD_CITATION_NOTICES: dict[str, tuple[str, str]] = {
    "coliv_intestine": (
        "Collagen-IV intestine",
        "HuBMAP dataset HBM893.MCGS.487. "
        "https://portal.hubmapconsortium.org/browse/dataset/HBM893.MCGS.487\n"
        "Mantpy tutorial dataset archive: https://doi.org/10.5281/zenodo.21538382",
    ),
    "balbc_pbs_lung": (
        "BALB/c PBS lung",
        "Parkinson, J. E., Bryant, M., Ghafoor, M. et al. "
        "Extracellular matrix phenotyping by imaging mass cytometry defines "
        "distinct cellular matrix environments associated with allergic airway "
        "inflammation. Molecular Systems Biology (2026). "
        "https://doi.org/10.1038/s44320-026-00234-5\n"
        "Mantpy tutorial dataset archive: https://doi.org/10.5281/zenodo.21538382",
    ),
    "schistosoma_ecm": (
        "Schistosoma ECM",
        "Su, Kim, et al. SOX9 plays an essential role in myofibroblast driven "
        "hepatic granuloma integrity and parenchymal repair during "
        "schistosomiasis-induced liver damage. PLOS Pathogens 21(6) (2025). "
        "https://doi.org/10.1371/journal.ppat.1012928\n"
        "Mantpy tutorial dataset archive: https://doi.org/10.5281/zenodo.21538382",
    ),
    "prostate_he_visium": (
        "prostate H&E/Visium",
        "10x Genomics, Human Prostate Cancer: Adenocarcinoma with Invasive "
        "Carcinoma, FFPE (Visium CytAssist Gene Expression, 1.3.0). "
        "https://www.10xgenomics.com/datasets/"
        "human-prostate-cancer-adenocarcinoma-with-invasive-carcinoma-ffpe-1-standard-1-3-0\n"
        "Mantpy tutorial dataset archive: https://doi.org/10.5281/zenodo.21538382",
    ),
}

_DOWNLOAD_LABELS = {
    "coliv_intestine": "Collagen-IV intestine dataset",
    "balbc_pbs_lung": "BALB/c PBS lung dataset",
    "schistosoma_ecm": "Schistosoma ECM dataset",
    "prostate_he_visium": "prostate H&E/Visium dataset",
}


@dataclass(frozen=True)
class PreparedBundle:
    """A verified extracted dataset archive and its manifest."""

    name: str
    root: Path
    manifest: Mapping[str, Any]
    entries: tuple[Mapping[str, Any], ...]

    def entries_for(self, role: str) -> tuple[Mapping[str, Any], ...]:
        """Return manifest entries carrying one role in manifest order."""
        return tuple(entry for entry in self.entries if entry["role"] == role)

    def paths_for(self, role: str) -> dict[str, Path]:
        """Return paths for one role keyed by sample identifier or filename."""
        matches: dict[str, Path] = {}
        for entry in self.entries:
            if entry["role"] != role:
                continue
            key = str(entry.get("sample_id") or entry.get("key") or Path(entry["path"]).stem)
            if key in matches:
                raise RuntimeError(f"Dataset {self.name!r} contains duplicate {role!r} key {key!r}.")
            matches[key] = self.root / str(entry["path"])
        return matches

    def one_path(self, role: str, *, required: bool = True) -> Path | None:
        """Return the unique path carrying ``role``."""
        paths = list(self.paths_for(role).values())
        if not paths and not required:
            return None
        if len(paths) != 1:
            expectation = "exactly one" if required else "at most one"
            raise RuntimeError(f"Dataset {self.name!r} must contain {expectation} {role!r} file; found {len(paths)}.")
        return paths[0]

    def public_paths(self) -> dict[str, Path | dict[str, Path]]:
        """Return a role-indexed mapping of verified local paths."""
        roles = dict.fromkeys(str(entry["role"]) for entry in self.entries)
        result: dict[str, Path | dict[str, Path]] = {}
        for role in roles:
            role_entries = self.entries_for(role)
            keys = [
                str(entry.get("sample_id") or entry.get("key") or Path(entry["path"]).stem) for entry in role_entries
            ]
            if len(keys) != len(set(keys)):
                keys = [str(entry.get("key") or entry["path"]) for entry in role_entries]
            paths = {key: self.root / str(entry["path"]) for key, entry in zip(keys, role_entries, strict=True)}
            result[role] = next(iter(paths.values())) if len(paths) == 1 else paths
        return result

    def quickstart_paths(self) -> dict[str, Path]:
        """Return verified quick-start assets without interpreting their science."""
        result: dict[str, Path] = {}
        for entry in self.entries:
            if not bool(entry.get("quickstart", False)):
                continue
            role = str(entry["role"])
            key = str(entry.get("key") or role)
            sample_id = entry.get("sample_id")
            if sample_id is not None and "key" not in entry:
                key = f"{role}:{sample_id}"
            if key in result:
                raise RuntimeError(f"Dataset {self.name!r} contains duplicate quick-start key {key!r}.")
            result[key] = self.root / str(entry["path"])
        return result


class _NoAmbientAuth(AuthBase):
    """Explicitly disable requests' automatic ``.netrc`` authentication."""

    def __call__(self, request):
        request.headers.pop("Authorization", None)
        return request


class _DownloadProgress:
    """Lazily create one concise tqdm bar when a transfer really starts."""

    def __init__(self, description: str) -> None:
        self.description = description
        self.total = 0
        self._bar: Any | None = None

    def _ensure_bar(self):
        if self._bar is None:
            self._bar = tqdm(
                total=self.total,
                desc=self.description,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                leave=True,
            )
        return self._bar

    def update(self, amount: int) -> None:
        self._ensure_bar().update(amount)

    def reset(self) -> None:
        if self._bar is not None:
            self._bar.reset(total=self.total)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve Mantpy's cache using argument, environment, then user cache."""
    if cache_dir is not None:
        root = Path(cache_dir).expanduser()
    elif value := os.environ.get("MANTPY_CACHE"):
        root = Path(value).expanduser()
    else:
        root = Path.home() / ".cache" / "mantpy"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_download_citation(name: str) -> None:
    notice = _DOWNLOAD_CITATION_NOTICES.get(name)
    if notice is None:
        return
    label, citation = notice
    print(f"Please cite the {label} dataset:\n{citation}")  # noqa: T201


def _file_spec(value: object, *, label: str) -> tuple[str, int, str]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Public dataset registry {label} must be a mapping.")
    filename = value.get("filename")
    size = value.get("size")
    sha256 = value.get("sha256")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise RuntimeError(f"Public dataset registry {label} has an unsafe filename.")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise RuntimeError(
            f"Mantpy's immutable public dataset record is not configured yet ({label} has no positive byte size)."
        )
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256.lower()) is None:
        raise RuntimeError(
            f"Mantpy's immutable public dataset record is not configured yet ({label} has no SHA-256 digest)."
        )
    return filename, size, sha256.lower()


def _validated_registry(name: str) -> tuple[Mapping[str, Any], int, tuple[str, int, str], tuple[str, int, str]]:
    registry = registry_module.DATASET_REGISTRY
    if not isinstance(registry, Mapping) or registry.get("schema_version") != 1:
        raise RuntimeError("Unsupported Mantpy public dataset registry schema.")
    record_id = registry.get("record_id")
    if not isinstance(record_id, int) or isinstance(record_id, bool) or record_id <= 0:
        raise RuntimeError("Mantpy's immutable public dataset record has not been published yet.")
    datasets = registry.get("datasets")
    if not isinstance(datasets, Mapping) or name not in datasets:
        raise KeyError(f"Unknown Mantpy dataset {name!r}.")
    manifest_spec = _file_spec(registry.get("manifest"), label="root manifest")
    archive_spec = _file_spec(datasets[name], label=f"archive {name!r}")
    return registry, record_id, manifest_spec, archive_spec


def _zenodo_url(registry: Mapping[str, Any], record_id: int, filename: str) -> str:
    base_url = str(registry.get("base_url", "https://zenodo.org")).rstrip("/")
    return f"{base_url}/api/records/{record_id}/files/{quote(filename, safe='')}/content"


def _retrieve_file(
    *,
    registry: Mapping[str, Any],
    record_id: int,
    spec: tuple[str, int, str],
    target: Path,
    progressbar: bool,
    progress_label: str | None = None,
) -> Path:
    filename, expected_size, expected_sha256 = spec
    target.mkdir(parents=True, exist_ok=True)
    downloader = pooch.HTTPDownloader(
        progressbar=_DownloadProgress(progress_label or f"Downloading {filename}") if progressbar else False,
        headers={"User-Agent": f"mantpy/{__version__}"},
        auth=_NoAmbientAuth(),
        timeout=120,
    )
    path = Path(
        pooch.retrieve(
            url=_zenodo_url(registry, record_id, filename),
            known_hash=f"sha256:{expected_sha256}",
            fname=filename,
            path=target,
            downloader=downloader,
        )
    )
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise RuntimeError(f"Downloaded {filename!r} has {observed_size} bytes; expected {expected_size}.")
    return path


def _safe_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"Unsafe dataset path {value!r}.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} or ":" in part for part in path.parts):
        raise RuntimeError(f"Unsafe dataset path {value!r}.")
    return path


def _allowed_data_path(path: PurePosixPath) -> bool:
    lower = path.as_posix().lower()
    return lower == "manifest.json" or lower.endswith(_ALLOWED_DATA_SUFFIXES)


def safe_extract_tar(archive: str | Path, target: str | Path) -> Path:
    """Extract a data-only tar archive after validating every member."""
    archive = Path(archive)
    target = Path(target)
    root = target.resolve()
    with tarfile.open(archive, mode="r:*") as tar:
        members = tar.getmembers()
        seen: set[str] = set()
        for member in members:
            relative = _safe_relative_path(member.name)
            folded = relative.as_posix().casefold()
            if folded in seen:
                raise RuntimeError(f"Duplicate member {member.name!r} in {archive.name!r}.")
            seen.add(folded)
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsafe non-data member {member.name!r} in {archive.name!r}.")
            if member.isfile() and not _allowed_data_path(relative):
                raise RuntimeError(f"Unsupported file type for {member.name!r} in {archive.name!r}.")
            destination = (root / Path(*relative.parts)).resolve()
            try:
                destination.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe member {member.name!r} in {archive.name!r}.") from exc
        target.mkdir(parents=True, exist_ok=True)
        tar.extractall(path=target, members=members, filter="data")
    return target


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read verified manifest {path.name!r}.") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Manifest {path.name!r} must contain a JSON object.")
    return value


def _validate_root_manifest(
    path: Path,
    *,
    registry: Mapping[str, Any],
    record_id: int,
) -> Mapping[str, Any]:
    manifest = _read_json(path)
    if manifest.get("schema_version") != 1 or manifest.get("record_id") != record_id:
        raise RuntimeError("Downloaded root manifest does not match the pinned record identity.")
    if manifest.get("license") != registry.get("license"):
        raise RuntimeError("Downloaded root manifest license does not match Mantpy's registry.")
    observed = manifest.get("datasets")
    expected = registry.get("datasets")
    if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
        raise RuntimeError("Downloaded root manifest has no dataset archive mapping.")
    if set(observed) != set(expected):
        raise RuntimeError("Downloaded root manifest dataset set differs from Mantpy's registry.")
    for name, expected_spec in expected.items():
        expected_file = _file_spec(expected_spec, label=f"archive {name!r}")
        observed_file = _file_spec(observed.get(name), label=f"root manifest archive {name!r}")
        if observed_file != expected_file:
            raise RuntimeError(f"Root manifest metadata for {name!r} differs from Mantpy's registry.")
    return manifest


def _validate_bundle_manifest(root: Path, *, name: str) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Dataset archive {name!r} has no manifest.json.")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != name:
        raise RuntimeError(f"Dataset archive {name!r} has the wrong manifest identity.")
    if manifest.get("license") != "CC-BY-4.0":
        raise RuntimeError(f"Dataset archive {name!r} must declare CC-BY-4.0.")
    entries_value = manifest.get("files")
    if not isinstance(entries_value, list) or not entries_value:
        raise RuntimeError(f"Dataset archive {name!r} has no file manifest.")

    entries: list[Mapping[str, Any]] = []
    expected_paths = {"manifest.json"}
    for value in entries_value:
        if not isinstance(value, Mapping):
            raise RuntimeError(f"Dataset archive {name!r} has a malformed file entry.")
        relative = _safe_relative_path(value.get("path"))
        if not _allowed_data_path(relative) or relative.as_posix() == "manifest.json":
            raise RuntimeError(f"Dataset archive {name!r} contains an unsupported data path {relative!s}.")
        role = value.get("role")
        if not isinstance(role, str) or _ROLE_RE.fullmatch(role) is None:
            raise RuntimeError(f"Dataset archive {name!r} has an invalid file role {role!r}.")
        size = value.get("size")
        digest = value.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError(f"Dataset file {relative!s} has an invalid byte size.")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest.lower()) is None:
            raise RuntimeError(f"Dataset file {relative!s} has an invalid SHA-256 digest.")
        relative_str = relative.as_posix()
        if relative_str.casefold() in {path.casefold() for path in expected_paths}:
            raise RuntimeError(f"Dataset archive {name!r} repeats path {relative_str!r}.")
        expected_paths.add(relative_str)
        path = root / Path(*relative.parts)
        if not path.is_file():
            raise RuntimeError(f"Dataset archive {name!r} is missing {relative_str!r}.")
        if path.stat().st_size != size or _sha256(path) != digest.lower():
            raise RuntimeError(f"Dataset file {relative_str!r} failed size/SHA-256 verification.")
        entries.append(dict(value, path=relative_str, sha256=digest.lower()))

    actual_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != ".verified"
    }
    if actual_paths != expected_paths:
        extra = sorted(actual_paths - expected_paths)
        missing = sorted(expected_paths - actual_paths)
        raise RuntimeError(
            f"Dataset archive {name!r} file list differs from its manifest; extra={extra}, missing={missing}."
        )
    provenance = manifest.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise RuntimeError(f"Dataset archive {name!r} provenance must be a mapping.")
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"Dataset archive {name!r} metadata must be a mapping.")
    forbidden = ("\\users\\", "/users/", "/home/", "/mnt/", "file://", "zenodo_token", "authorization")
    portable_text = json.dumps({"metadata": metadata, "provenance": provenance}, sort_keys=True).lower()
    has_drive_path = re.search(r'"[a-z]:\\\\', portable_text) is not None
    if has_drive_path or any(value in portable_text for value in forbidden):
        raise RuntimeError(f"Dataset archive {name!r} contains non-portable manifest metadata.")
    return manifest, tuple(entries)


def _atomic_extract(archive: Path, destination: Path, *, name: str) -> PreparedBundle:
    marker = destination / ".verified"
    if marker.is_file():
        try:
            manifest, entries = _validate_bundle_manifest(destination, name=name)
            return PreparedBundle(name=name, root=destination, manifest=manifest, entries=entries)
        except RuntimeError:
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=destination.parent))
    try:
        safe_extract_tar(archive, temporary)
        manifest, entries = _validate_bundle_manifest(temporary, name=name)
        (temporary / ".verified").write_text(_sha256(archive), encoding="ascii")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PreparedBundle(name=name, root=destination, manifest=manifest, entries=entries)


def prepare_bundle(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    progressbar: bool = True,
) -> PreparedBundle:
    """Return a locally verified bundle, downloading it when necessary."""
    if source_dir is not None:
        root = Path(source_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        manifest, entries = _validate_bundle_manifest(root, name=name)
        return PreparedBundle(name=name, root=root, manifest=manifest, entries=entries)

    registry, record_id, root_spec, archive_spec = _validated_registry(name)
    cache = resolve_cache_dir(cache_dir) / "datasets" / str(record_id)
    manifest_path = _retrieve_file(
        registry=registry,
        record_id=record_id,
        spec=root_spec,
        target=cache,
        progressbar=False,
    )
    _validate_root_manifest(manifest_path, registry=registry, record_id=record_id)
    first_download = not (cache / archive_spec[0]).is_file()
    archive = _retrieve_file(
        registry=registry,
        record_id=record_id,
        spec=archive_spec,
        target=cache,
        progressbar=progressbar,
        progress_label=f"Downloading {_DOWNLOAD_LABELS.get(name, name)}",
    )
    archive_digest = archive_spec[2]
    destination = cache / "extracted" / f"{name}-{archive_digest[:12]}"
    bundle = _atomic_extract(archive, destination, name=name)
    if first_download:
        _print_download_citation(name)
    return bundle
