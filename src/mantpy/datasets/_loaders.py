"""Dataset-specific readers built on the verified archive boundary."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mantpy._palette import ECM_PALETTE_LUNG_PUBLISHED, INTESTINE_LAYERS
from mantpy.ds import Bunch, ColIVIntestineBunch, LungBunch

from ._download import PreparedBundle, prepare_bundle, resolve_cache_dir, safe_extract_tar


def _metadata(bundle: PreparedBundle) -> Mapping[str, Any]:
    value = bundle.manifest.get("metadata", {})
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Dataset {bundle.name!r} metadata must be a mapping.")
    return value


def _provenance(bundle: PreparedBundle) -> dict[str, Any]:
    value = bundle.manifest.get("provenance", {})
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Dataset {bundle.name!r} provenance must be a mapping.")
    return {
        **dict(value),
        "schema_version": 1,
        "dataset": bundle.name,
        "license": str(bundle.manifest["license"]),
    }


def load_coliv_files(image: str | Path | np.ndarray, reference_path: str | Path) -> ColIVIntestineBunch:
    """Load explicit Collagen-IV files without applying preprocessing."""
    import tifffile

    if isinstance(image, str | Path):
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Raw Collagen-IV image not found: {image_path}")
        raw = np.asarray(tifffile.imread(image_path))
        source: str | None = image_path.name
    else:
        raw = np.asarray(image)
        source = None
    raw = np.squeeze(raw)
    if raw.ndim != 2:
        raise ValueError(
            f"image must be one raw 2-D ECM channel; got shape {raw.shape}. "
            "Select the Collagen-IV plane before loading the dataset. "
            "Use mt.pp.image_ecm_patches directly for a channel-first ECM stack."
        )
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"image must contain numeric intensities; got dtype {raw.dtype}.")

    reference_file = Path(reference_path)
    if not reference_file.is_file():
        raise FileNotFoundError(f"Held-out intestine reference not found: {reference_file}")
    with np.load(reference_file, allow_pickle=False) as reference_npz:
        required = {"annotation", "layer_names"}
        missing = required.difference(reference_npz.files)
        if missing:
            raise ValueError(f"Reference {reference_file} is missing keys: {sorted(missing)}")
        y_key, x_key = ("grid_y", "grid_x") if {"grid_y", "grid_x"}.issubset(reference_npz.files) else ("ys", "xs")
        if y_key not in reference_npz.files or x_key not in reference_npz.files:
            raise ValueError(f"Reference {reference_file} must contain grid_y/grid_x (or ys/xs).")
        grid_y = np.asarray(reference_npz[y_key], dtype=int)
        grid_x = np.asarray(reference_npz[x_key], dtype=int)
        labels = np.asarray(reference_npz["annotation"], dtype=int)
        layer_names = tuple(str(value) for value in np.asarray(reference_npz["layer_names"]).tolist())
    if not (len(grid_y) == len(grid_x) == len(labels)):
        raise ValueError("Reference grid coordinates and annotation have different lengths.")

    patch_ids = [f"patch_y{y:04d}_x{x:04d}" for y, x in zip(grid_y, grid_x, strict=True)]
    reference = pd.DataFrame(
        {"grid_y": grid_y, "grid_x": grid_x, "layer": labels},
        index=pd.Index(patch_ids, name="patch_id"),
    )
    layer_palette = dict(zip(layer_names, tuple(INTESTINE_LAYERS.values()), strict=False))
    return ColIVIntestineBunch(
        image=raw,
        raw_image=raw,
        image_path=source,
        reference_path=reference_file.name,
        reference=reference,
        layer_names=layer_names,
        layer_palette=layer_palette,
        channel_name="CollIV",
        pixel_size_um=0.37744,
        roi="893",
        tissue="human small intestine",
        technology="CODEX",
    )


def coliv_intestine(
    *,
    cache_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    progressbar: bool = True,
) -> ColIVIntestineBunch:
    """Download and load the public Collagen-IV intestine dataset.

    Parameters
    ----------
    cache_dir
        Cache root. The ``MANTPY_CACHE`` environment variable and then
        ``~/.cache/mantpy`` are used when omitted.
    source_dir
        Optional already-extracted schema-v1 bundle for a verified local
        mirror. Normal use should leave this unset.
    progressbar
        Show byte progress while downloading uncached record files.

    Returns
    -------
    ColIVIntestineBunch
        Raw image, held-out reference, portable provenance, verified paths,
        and optional quick-start paths.
    """
    bundle = prepare_bundle("coliv_intestine", cache_dir=cache_dir, source_dir=source_dir, progressbar=progressbar)
    image_path = bundle.one_path("raw_image")
    reference_path = bundle.one_path("reference")
    assert image_path is not None and reference_path is not None
    data = load_coliv_files(image_path, reference_path)
    metadata = _metadata(bundle)
    for key in ("pixel_size_um", "roi", "tissue", "technology", "channel_name"):
        if key in metadata:
            data[key] = metadata[key]
    data.update(
        paths=bundle.public_paths(),
        provenance=_provenance(bundle),
        quickstart=bundle.quickstart_paths(),
    )
    return data


def _read_h5ad_collection(bundle: PreparedBundle, role: str) -> dict[str, Any]:
    import anndata as ad

    return {sample: ad.read_h5ad(path) for sample, path in bundle.paths_for(role).items()}


def _boolean_array(values: Any, *, source: Path) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(bool, copy=False)
    if np.issubdtype(array.dtype, np.number):
        if not np.isin(array, [0, 1]).all():
            raise ValueError(f"Artifact overlay {source.name!r} has non-binary is_artifact values.")
        return array.astype(bool)
    normalized = pd.Series(array, dtype="string").str.strip().str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"Artifact overlay {source.name!r} has invalid is_artifact values.")
    return normalized.isin(["true", "1"]).to_numpy(dtype=bool)


def _read_lung_overlays(bundle: PreparedBundle, roi_names: list[str]) -> dict[str, dict[str, Any]]:
    path = bundle.one_path("artifact_overlay")
    assert path is not None
    if path.suffix.lower() != ".csv":
        raise ValueError("Lung artifact overlay must use the public CSV schema.")
    frame = pd.read_csv(path)
    if "sample_id" not in frame:
        raise ValueError("Lung artifact_overlay.csv is missing 'sample_id'.")
    frame["sample_id"] = frame["sample_id"].astype(str)
    overlays = {
        sample: _read_overlay_frame(group.drop(columns="sample_id"), source=path)
        for sample, group in frame.groupby("sample_id", sort=False)
    }

    boxes_path = bundle.one_path("artifact_boxes", required=False)
    if boxes_path is not None:
        import json

        value = json.loads(boxes_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("Lung artifact_boxes.json must contain a sample-to-boxes mapping.")
        for sample, boxes in value.items():
            if str(sample) in overlays:
                overlays[str(sample)]["artifact_boxes"] = boxes
    if set(overlays) != set(roi_names):
        raise ValueError("Lung artifact overlays must contain one definition for every sample.")
    return {sample: overlays[sample] for sample in roi_names}


def _read_overlay_frame(frame: pd.DataFrame, *, source: Path) -> dict[str, Any]:
    required = {"is_artifact", "ecm_cluster_artifact"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Artifact overlay {source.name!r} is missing columns {sorted(missing)}.")
    result: dict[str, Any] = {column: frame[column].to_numpy() for column in frame.columns}
    result["is_artifact"] = _boolean_array(result["is_artifact"], source=source)
    result["ecm_cluster_artifact"] = np.asarray(result["ecm_cluster_artifact"], dtype=int)
    if "ecm_cluster_pristine" in result:
        result["ecm_cluster_pristine"] = np.asarray(result["ecm_cluster_pristine"], dtype=int)
    return result


def _panel_ecm_names(panel: pd.DataFrame) -> list[str]:
    if "name" not in panel:
        raise ValueError("Dataset panel is missing the 'name' column.")
    keep = panel["keep"].eq(1) if "keep" in panel else pd.Series(True, index=panel.index)
    if "ecm" in panel:
        keep &= panel["ecm"].eq(1)
    elif "is_ecm" in panel:
        keep &= panel["is_ecm"].astype(bool)
    return panel.loc[keep, "name"].astype(str).tolist()


def _read_channel_first_tiff(path: Path, *, n_channels: int) -> np.ndarray:
    """Read a TIFF in canonical ``(C, H, W)`` order, memory-mapping when possible."""
    import tifffile

    with tifffile.TiffFile(path) as handle:
        shape = tuple(int(value) for value in handle.series[0].shape)
        axes = str(handle.series[0].axes)
    if len(shape) != 3:
        raise ValueError(f"Lung image {path.name!r} must be three-dimensional; got shape {shape}.")

    channel_axis: int | None = None
    for marker in ("C", "S"):
        if marker in axes and shape[axes.index(marker)] == n_channels:
            channel_axis = axes.index(marker)
            break
    if channel_axis is None:
        matching_axes = [axis for axis, length in enumerate(shape) if length == n_channels]
        if 0 in matching_axes:
            channel_axis = 0
        elif len(matching_axes) == 1:
            channel_axis = matching_axes[0]
    if channel_axis is None:
        raise ValueError(
            f"Lung image {path.name!r} has shape {shape}, which cannot be aligned to "
            f"the {n_channels}-channel retained panel."
        )

    try:
        image = tifffile.memmap(path, mode="r")
    except (OSError, ValueError):
        image = tifffile.imread(path)
    image = np.asarray(image)
    if channel_axis != 0:
        image = np.moveaxis(image, channel_axis, 0)
    if image.shape[0] != n_channels:
        raise ValueError(
            f"Lung image {path.name!r} has {image.shape[0]} channels after axis normalization; expected {n_channels}."
        )
    return image


def balbc_pbs_lung(
    *,
    cache_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    progressbar: bool = True,
) -> LungBunch:
    """Download and load the public six-ROI BALB/c PBS lung dataset.

    Parameters
    ----------
    cache_dir
        Cache root. The ``MANTPY_CACHE`` environment variable and then
        ``~/.cache/mantpy`` are used when omitted.
    source_dir
        Optional already-extracted schema-v1 bundle for a verified local
        mirror. Normal use should leave this unset.
    progressbar
        Show byte progress while downloading uncached record files.

    Returns
    -------
    LungBunch
        Typed-cell AnnData objects, raw image stacks, panel and cohort
        metadata, corruption overlays, portable provenance, and verified
        paths.

    Notes
    -----
    Mantpy prints the dataset citation after the first successful archive
    download. Cached and local-mirror loads remain quiet.
    """
    bundle = prepare_bundle("balbc_pbs_lung", cache_dir=cache_dir, source_dir=source_dir, progressbar=progressbar)
    panel_path = bundle.one_path("panel")
    metadata_path = bundle.one_path("metadata")
    assert panel_path is not None and metadata_path is not None
    panel = pd.read_csv(panel_path)
    panel.columns = [str(column).strip() for column in panel.columns]
    metadata = pd.read_csv(metadata_path)
    metadata.columns = [str(column).strip() for column in metadata.columns]
    cells_by_sample = _read_h5ad_collection(bundle, "cells")
    image_paths = bundle.paths_for("raw_image")
    if not cells_by_sample or set(cells_by_sample) != set(image_paths):
        raise ValueError("Lung cell H5AD files and raw images must have identical sample identifiers.")
    roi_names = metadata["sample_id"].astype(str).tolist() if "sample_id" in metadata else list(cells_by_sample)
    if len(roi_names) != len(set(roi_names)) or set(roi_names) != set(cells_by_sample):
        raise ValueError("Lung metadata must contain each sample identifier exactly once.")
    kept = panel.loc[panel["keep"].eq(1)].reset_index(drop=True) if "keep" in panel else panel.reset_index(drop=True)
    images = {sample: _read_channel_first_tiff(image_paths[sample], n_channels=len(kept)) for sample in roi_names}

    for sample, cells in cells_by_sample.items():
        if "cell_type" not in cells.obs and "celltype" in cells.obs:
            merged = {
                "AT1 (Activated)": "ATI",
                "AT1": "ATI",
                "ATI (Activated)": "ATI",
                "ATII (Activated)": "ATII",
            }
            cells.obs["cell_type"] = (
                cells.obs["celltype"].astype(str).map(lambda value, mapping=merged: mapping.get(value, value))
            )
        if "cell_type" not in cells.obs:
            raise ValueError(f"Lung cell H5AD for {sample!r} has no 'cell_type' annotation.")

    artifact_overlays = _read_lung_overlays(bundle, roi_names)
    ecm_names = _panel_ecm_names(panel)
    settings = _metadata(bundle)
    target_channel = str(settings.get("target_channel", "HABP"))
    matches = kept.index[kept["name"].astype(str) == target_channel].tolist()
    if len(matches) != 1:
        raise ValueError(f"Lung panel must contain target channel {target_channel!r} exactly once.")
    target_channel_idx = int(matches[0])
    k_ecm = int(settings.get("K_ecm", 3))
    showcase_roi = str(settings.get("showcase_roi", roi_names[0]))
    if showcase_roi not in cells_by_sample:
        raise ValueError(f"Lung showcase ROI {showcase_roi!r} is not present in the cohort.")

    ecm_by_sample = _read_h5ad_collection(bundle, "ecm")
    ecm_patches = _read_h5ad_collection(bundle, "ecm_patches")
    optional_csv: dict[str, pd.DataFrame | None] = {}
    for role in ("loo_summary", "loo_curves"):
        path = bundle.one_path(role, required=False)
        optional_csv[role] = pd.read_csv(path) if path is not None else None

    return LungBunch(
        cells_by_sample=cells_by_sample,
        cohort_cells={sample: cells_by_sample[sample] for sample in roi_names},
        adatas_ecm=ecm_by_sample,
        cohort_ecm=ecm_by_sample,
        cohort_ecm_patches=ecm_patches,
        images=images,
        ecm_images=images,
        panel=panel,
        metadata=metadata,
        roi_names=roi_names,
        ecm_names=ecm_names,
        K_ecm=k_ecm,
        showcase_roi=showcase_roi,
        showcase_image=images[showcase_roi],
        target_celltype=str(settings.get("target_celltype", "AEC")),
        target_channel=target_channel,
        target_channel_idx=target_channel_idx,
        ecm_palette={cluster: ECM_PALETTE_LUNG_PUBLISHED[cluster] for cluster in range(k_ecm)},
        artifact_overlays=artifact_overlays,
        adatas_ecm_artifact=None,
        loo_summary=optional_csv["loo_summary"],
        loo_curves=optional_csv["loo_curves"],
        tissue=str(settings.get("tissue", "mouse lung")),
        technology=str(settings.get("technology", "IMC")),
        paths=bundle.public_paths(),
        provenance=_provenance(bundle),
        quickstart=bundle.quickstart_paths(),
    )


def _select_schistosoma_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if "name" not in panel:
        raise ValueError("Schistosoma panel is missing the 'name' column.")
    keep = panel["keep"].eq(1) if "keep" in panel else pd.Series(True, index=panel.index)
    if "ecm" in panel:
        keep &= panel["ecm"].eq(1)
    elif "is_ecm" in panel:
        keep &= panel["is_ecm"].astype(bool)
    selected = panel.loc[keep].reset_index(drop=True)
    if selected.empty:
        raise ValueError("Schistosoma panel marks no retained ECM channels.")
    return selected


def schistosoma_ecm(
    *,
    cache_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    progressbar: bool = True,
    verbose: bool = True,
) -> Bunch:
    """Download and load the public Schistosoma ECM liver cohort.

    Parameters
    ----------
    cache_dir
        Cache root. The ``MANTPY_CACHE`` environment variable and then
        ``~/.cache/mantpy`` are used when omitted.
    source_dir
        Optional already-extracted schema-v1 bundle for a verified local
        mirror. Normal use should leave this unset.
    progressbar
        Show byte progress while downloading uncached record files.
    verbose
        Display the concise cohort summary after loading.

    Returns
    -------
    Bunch
        Raw ECM images and AnnData objects, panel, de-identified sample
        metadata, group ordering, portable provenance, and verified paths.

    Notes
    -----
    Mantpy prints the dataset citation after the first successful archive
    download. Cached and local-mirror loads remain quiet.
    """
    from mantpy.im import as_image_container
    from mantpy.io import read_imc_folder

    bundle = prepare_bundle("schistosoma_ecm", cache_dir=cache_dir, source_dir=source_dir, progressbar=progressbar)
    panel_path = bundle.one_path("panel")
    metadata_path = bundle.one_path("metadata")
    assert panel_path is not None and metadata_path is not None
    panel_all = pd.read_csv(panel_path)
    sample_meta = pd.read_csv(metadata_path)
    if "sample_id" not in sample_meta:
        raise ValueError("Schistosoma metadata is missing 'sample_id'.")
    sample_meta["sample_id"] = sample_meta["sample_id"].astype(str)
    if sample_meta["sample_id"].duplicated().any():
        raise ValueError("Schistosoma metadata contains duplicate sample identifiers.")
    sample_meta = sample_meta.set_index("sample_id", drop=False)
    channel_entries = bundle.entries_for("raw_channel") or bundle.entries_for("raw_image")
    channel_samples = {str(entry.get("sample_id")) for entry in channel_entries}
    if None in {entry.get("sample_id") for entry in channel_entries} or channel_samples != set(sample_meta.index):
        raise ValueError("Every Schistosoma raw channel must carry a metadata sample identifier.")
    roi_names = sample_meta.index.astype(str).tolist()
    images: dict[str, np.ndarray] = {}
    raw_adatas: dict[str, Any] = {}
    panel = _select_schistosoma_panel(panel_all)
    for sample in roi_names:
        entries = [entry for entry in channel_entries if str(entry.get("sample_id")) == sample]
        folders = {(bundle.root / str(entry["path"])).parent for entry in entries}
        if len(folders) != 1:
            raise ValueError(f"Schistosoma channels for {sample!r} must share one directory.")
        condition = str(sample_meta.loc[sample, "condition"]) if "condition" in sample_meta else None
        raw_adatas[sample] = read_imc_folder(
            next(iter(folders)),
            panel,
            pattern="*.ome.tif*",
            sample_id=sample,
            condition=condition,
            normalize="none",
        )
        images[sample] = as_image_container(raw_adatas[sample].uns["image_container"]).get_layer("image")

    settings = _metadata(bundle)
    if "group" not in sample_meta:
        if {"condition", "genotype"}.issubset(sample_meta.columns):
            sample_meta["group"] = sample_meta["condition"].astype(str) + "_" + sample_meta["genotype"].astype(str)
        else:
            raise ValueError("Schistosoma metadata requires 'group' or condition/genotype columns.")
    group_order_value = settings.get("group_order")
    group_order = (
        [str(value) for value in group_order_value]
        if isinstance(group_order_value, list)
        else list(dict.fromkeys(sample_meta["group"].astype(str)))
    )
    if len(group_order) != len(set(group_order)) or set(group_order) != set(sample_meta["group"].astype(str)):
        raise ValueError("Schistosoma group_order must contain every observed group exactly once.")
    for column, order_key in (
        ("condition", "condition_order"),
        ("genotype", "genotype_order"),
        ("group", "group_order"),
    ):
        if column not in sample_meta:
            continue
        order = settings.get(order_key)
        if isinstance(order, list):
            sample_meta[column] = pd.Categorical(sample_meta[column], categories=order, ordered=True)

    best_k = int(settings.get("best_k", 7))
    bundle_result = Bunch(
        raw_adatas=raw_adatas,
        images=images,
        panel=panel,
        ecm_names=panel["name"].astype(str).tolist(),
        sample_meta=sample_meta,
        raw_metadata=sample_meta.copy(),
        group_order=group_order,
        adatas=raw_adatas,
        reference_adatas=raw_adatas,
        best_k=best_k,
        preprocessing=dict(settings.get("preprocessing", {})),
        paths=bundle.public_paths(),
        provenance=_provenance(bundle),
        quickstart=bundle.quickstart_paths(),
    )
    if verbose:
        bundle_result._summarize()
    return bundle_result


def _validate_cells(cells, *, path: Path):
    if cells.n_vars != 0:
        raise ValueError(f"Expected a coordinate-only cell AnnData with zero variables; found {cells.n_vars}.")
    if "spatial" not in cells.obsm or "spatial_um" not in cells.obsm:
        raise ValueError("Deposited cells require obsm['spatial'] and obsm['spatial_um'].")
    if not isinstance(cells.uns.get("segmentation"), Mapping):
        raise ValueError("Deposited cells require uns['segmentation'] provenance.")
    cells.uns["segmentation"]["loaded_from"] = path.name
    return cells


def load_prostate_cells(path: str | Path):
    """Load and validate a coordinate-only prostate cell H5AD."""
    import anndata as ad

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Deposited prostate cell coordinates not found at {source}.")
    return _validate_cells(ad.read_h5ad(source), path=source)


def _extract_spatial_archive(archive: Path, target: Path) -> tuple[Path, Path]:
    """Extract the coordinate-only 10x spatial payload and enforce its exact layout."""
    safe_extract_tar(archive, target)
    spatial = target / "spatial"
    positions_candidates = [
        spatial / "tissue_positions.csv",
        spatial / "tissue_positions_list.csv",
    ]
    present_positions = [path for path in positions_candidates if path.is_file()]
    scalefactors = spatial / "scalefactors_json.json"
    if len(present_positions) != 1 or not scalefactors.is_file():
        raise RuntimeError(
            "Prostate spatial archive must contain scalefactors_json.json and exactly one tissue-positions CSV."
        )
    expected = {present_positions[0].resolve(), scalefactors.resolve()}
    observed = {path.resolve() for path in target.rglob("*") if path.is_file()}
    if observed != expected:
        raise RuntimeError("Prostate spatial archive contains files outside the coordinate-only 10x contract.")
    return present_positions[0], scalefactors


def _read_10x_positions(path: Path) -> pd.DataFrame:
    columns = ["in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]
    if path.name == "tissue_positions_list.csv":
        coordinates = pd.read_csv(path, header=None, index_col=0)
        if coordinates.shape[1] != len(columns):
            raise ValueError(f"10x positions file {path.name!r} must contain six columns including the barcode.")
        coordinates.columns = columns
        return coordinates

    coordinates = pd.read_csv(path, index_col=0)
    coordinates.columns = [str(column).strip() for column in coordinates.columns]
    if not set(columns).issubset(coordinates.columns):
        # Match Squidpy's fallback for Space Ranger files carrying a leading
        # metadata row before the actual header.
        coordinates = pd.read_csv(path, header=1, index_col=0)
        coordinates.columns = [str(column).strip() for column in coordinates.columns]
    missing = set(columns).difference(coordinates.columns)
    if missing:
        raise ValueError(f"10x positions file {path.name!r} is missing columns {sorted(missing)}.")
    return coordinates.loc[:, columns]


def _attach_10x_spatial(
    visium,
    *,
    positions_path: Path,
    scalefactors_path: Path,
    library_id: str,
) -> Mapping[str, Any]:
    import json

    try:
        scalefactors = json.loads(scalefactors_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Prostate 10x scalefactors are not valid JSON.") from exc
    if not isinstance(scalefactors, Mapping):
        raise ValueError("Prostate 10x scalefactors must contain a JSON object.")

    coordinates = _read_10x_positions(positions_path)
    if coordinates.index.has_duplicates:
        raise ValueError("Prostate 10x positions contain duplicate barcodes.")
    coordinates.index = coordinates.index.astype(visium.obs.index.dtype)
    visium.obs = pd.merge(visium.obs, coordinates, how="left", left_index=True, right_index=True)
    spatial_columns = ["pxl_row_in_fullres", "pxl_col_in_fullres"]
    if visium.obs[spatial_columns].isna().any(axis=None):
        raise ValueError("Prostate 10x counts contain barcodes without spatial coordinates.")
    visium.obsm["spatial"] = visium.obs[spatial_columns].to_numpy()
    visium.obs.drop(columns=spatial_columns, inplace=True)
    spatial_uns = visium.uns.setdefault("spatial", {}).setdefault(library_id, {})
    spatial_uns.setdefault("metadata", {})
    spatial_uns["images"] = {}
    spatial_uns["scalefactors"] = dict(scalefactors)
    return scalefactors


def prostate_he_visium(
    *,
    cache_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    progressbar: bool = True,
    lazy: bool = True,
) -> Bunch:
    """Download and load the public prostate H&E/Visium tutorial dataset.

    Parameters
    ----------
    cache_dir
        Cache root. The ``MANTPY_CACHE`` environment variable and then
        ``~/.cache/mantpy`` are used when omitted.
    source_dir
        Optional already-extracted schema-v1 bundle for a verified local
        mirror. Normal use should leave this unset.
    progressbar
        Show byte progress while downloading uncached record files.
    lazy
        Keep the full-resolution H&E image lazy when the image extra is
        installed.

    Returns
    -------
    Bunch
        H&E image, Visium AnnData, coordinate-only cells, pathology path,
        pinned Matrisome table, spatial calibration, portable provenance,
        and verified paths.
    """
    import os
    import shutil
    import tempfile

    import squidpy as sq

    from mantpy.fetch import _visium_image_mpp, fetch_matrisome
    from mantpy.im import ImageContainer

    bundle = prepare_bundle("prostate_he_visium", cache_dir=cache_dir, source_dir=source_dir, progressbar=progressbar)
    image_path = bundle.one_path("raw_image")
    counts_path = bundle.one_path("counts")
    spatial_archive = bundle.one_path("spatial_archive")
    cells_path = bundle.one_path("cells")
    pathology_path = bundle.one_path("pathology")
    assert (
        image_path is not None
        and counts_path is not None
        and spatial_archive is not None
        and cells_path is not None
        and pathology_path is not None
    )
    temporary_root = resolve_cache_dir(cache_dir) / "datasets" / ".temporary"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prostate-visium-", dir=temporary_root) as temporary_value:
        temporary = Path(temporary_value)
        positions_path, scalefactors_path = _extract_spatial_archive(spatial_archive, temporary)
        local_counts = temporary / counts_path.name
        try:
            os.link(counts_path, local_counts)
        except OSError:
            shutil.copyfile(counts_path, local_counts)
        library_id = str(_metadata(bundle).get("library_id", "prostate_adeno"))
        visium = sq.read.visium(
            temporary,
            counts_file=local_counts.name,
            library_id=library_id,
            load_images=False,
        )
        scalefactors = _attach_10x_spatial(
            visium,
            positions_path=positions_path,
            scalefactors_path=scalefactors_path,
            library_id=library_id,
        )
    duplicate_var_names = int(visium.var_names.duplicated().sum())
    if duplicate_var_names:
        visium.var_names_make_unique(join="-")
    cells = load_prostate_cells(cells_path)
    settings = _metadata(bundle)
    derived_mpp, scale_method = _visium_image_mpp(visium.obsm["spatial"], scalefactors)
    image_mpp = float(settings.get("image_mpp", derived_mpp))
    spot_diameter_px = float(scalefactors.get("spot_diameter_fullres", 0.0))
    derived_radius = 0.5 * spot_diameter_px * image_mpp if spot_diameter_px > 0 else 32.5
    spot_radius_um = float(settings.get("spot_radius_um", derived_radius))
    if not np.isfinite(image_mpp) or image_mpp <= 0:
        raise ValueError("Prostate bundle requires a positive image_mpp value.")
    if not np.isfinite(spot_radius_um) or spot_radius_um <= 0:
        raise ValueError("Prostate bundle requires a positive spot_radius_um value.")
    if "spatial" not in visium.obsm:
        raise ValueError("Prostate Visium H5AD requires obsm['spatial'].")
    if "spatial_um" not in visium.obsm:
        visium.obsm["spatial_um"] = np.asarray(visium.obsm["spatial"], dtype=np.float32) * np.float32(image_mpp)
    portable_provenance = _provenance(bundle)
    visium.uns.setdefault("mantpy", {})["prostate_he_visium"] = {
        "image_mpp": image_mpp,
        "spot_radius_um": spot_radius_um,
        "scale_method": scale_method,
        "provenance": portable_provenance,
        "var_names_make_unique": {
            "applied": bool(duplicate_var_names),
            "duplicate_count": duplicate_var_names,
            "join": "-",
        },
    }
    image = ImageContainer(
        image_path,
        channel_axis=-1,
        channel_names=["red", "green", "blue"],
        lazy=lazy,
        scale=image_mpp,
    )
    # The Matrisome table is fetched live from a Google Sheets export against a
    # pinned SHA-256. It is the only input to this loader that does not come
    # from the checksummed Zenodo record, so a single upstream edit to that
    # sheet would otherwise break the whole dataset — including for callers who
    # only want the H&E, Visium and cell data. Degrade instead; the strict
    # behaviour stays in :func:`mantpy.fetch.fetch_matrisome` itself.
    try:
        matrisome = fetch_matrisome("human", cache_dir=resolve_cache_dir(cache_dir) / "resources")
    except Exception as exc:  # noqa: BLE001 - an upstream outage must not sink the loader
        warnings.warn(
            "Could not fetch the human Matrisome table, so `matrisome` is None; "
            "the H&E, Visium and cell data are unaffected. Call "
            "`mantpy.fetch.fetch_matrisome` directly to see the underlying error. "
            f"({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        matrisome = None
    return Bunch(
        image=image,
        visium=visium,
        cells=cells,
        pathology=pathology_path,
        matrisome=matrisome,
        image_mpp=image_mpp,
        spot_radius_um=spot_radius_um,
        paths=bundle.public_paths(),
        provenance=portable_provenance,
        quickstart=bundle.quickstart_paths(),
    )
