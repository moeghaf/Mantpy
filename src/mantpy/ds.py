"""Multi-ROI dataset container for Mantpy."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import networkx as nx
from anndata import AnnData

from mantpy._constants import ECM_GRAPH_KEY

if TYPE_CHECKING:
    import pandas as pd
    import spatialdata

__all__ = [
    "Bunch",
    "ColIVIntestineBunch",
    "LungBunch",
    "MantpyDataset",
]


class MantpyDataset:
    """Cohort container for multiple ROIs / samples.

    Each ROI is a separate :class:`anndata.AnnData` object produced by
    :func:`~mantpy.io.read_imc`.  ``MantpyDataset`` provides a thin wrapper
    that lets you iterate, apply Mantpy functions, and export the cohort to
    SpatialData or a list of PyTorch Geometric graphs.

    Parameters
    ----------
    adatas
        Mapping of ``{sample_id: AnnData}``.  The keys become the region
        identifiers used throughout the dataset.

    Examples
    --------
    Build a dataset from two ROIs and run the ECM pipeline on all of them::

        import mantpy as mt
        import numpy as np

        ds = mt.MantpyDataset({"spleen": adata_spleen, "thymus": adata_thymus})

        # Apply any Mantpy function to every ROI
        ds.apply(
            lambda a: mt.gr.build_ecm_graph(
                a,
                edge_method="delaunay",
                edge_features=["log_distance", "norm_angle"],
            )
        )

        # Get PyG dataset for DataLoader
        pyg_list = ds.pyg_dataset()
    """

    def __init__(self, adatas: dict[str, AnnData]) -> None:
        self._adatas: dict[str, AnnData] = dict(adatas)

    # ------------------------------------------------------------------ access

    def __getitem__(self, sample_id: str) -> AnnData:
        return self._adatas[sample_id]

    def __len__(self) -> int:
        return len(self._adatas)

    def __iter__(self):
        return iter(self._adatas)

    @property
    def sample_ids(self) -> list[str]:
        """Ordered list of sample identifiers."""
        return list(self._adatas.keys())

    @property
    def adatas(self) -> dict[str, AnnData]:
        """Dict view of ``{sample_id: AnnData}``."""
        return dict(self._adatas)

    # ----------------------------------------------------------------- mutate

    def apply(self, fn: Callable[[AnnData], None]) -> None:
        """Apply a callable to every ROI in-place.

        Parameters
        ----------
        fn
            Any callable that accepts an :class:`~anndata.AnnData` and modifies
            it in place.  Typical use: pass a Mantpy ``pp``/``gr``/``tl``
            function (with parameters pre-bound via ``lambda`` or
            ``functools.partial``).

        Example
        -------
        ::

            ds.apply(
                lambda a: mt.pp.extract_ecm_patches(
                    a,
                    img,
                    features=["mean", "coherence"],
                    mask=mask,
                )
            )
        """
        for adata in self._adatas.values():
            fn(adata)

    # ------------------------------------------------------------------ export

    def graphs(self, key: str = ECM_GRAPH_KEY) -> dict[str, nx.Graph]:
        """Return ``{sample_id: NetworkX graph}`` for all ROIs that have ``key``.

        Parameters
        ----------
        key
            Graph key in ``adata.uns``.  Defaults to ``"ecm_graph"``.
        """
        return {sid: a.uns[key] for sid, a in self._adatas.items() if key in a.uns}

    def pyg_dataset(
        self,
        graph_key: str = ECM_GRAPH_KEY,
        include_edge_features: bool = True,
    ) -> list:
        """Return a list of PyTorch Geometric ``Data`` objects — one per ROI.

        Calls :func:`~mantpy.gr.to_pyg` on every ROI that has ``graph_key``
        in ``adata.uns``.  The resulting list can be passed directly to a PyG
        ``DataLoader``.

        Parameters
        ----------
        graph_key
            Graph key to export.
        include_edge_features
            Forwarded to :func:`~mantpy.gr.to_pyg`.
        """
        import mantpy.gr as gr

        out = []
        for sid, adata in self._adatas.items():
            if graph_key not in adata.uns:
                continue
            data = gr.to_pyg(adata, graph_key=graph_key, include_edge_features=include_edge_features)
            data.sample_id = sid
            out.append(data)
        return out

    def to_spatialdata(self) -> spatialdata.SpatialData:
        """Bundle all ROIs into a :class:`spatialdata.SpatialData` object.

        Requires the ``[spatial]`` optional dependency::

            pip install mantpy[spatial]
        """
        from mantpy.io import to_spatialdata

        return to_spatialdata(self._adatas)

    def __repr__(self) -> str:
        ids = ", ".join(self.sample_ids[:5])
        suffix = ", ..." if len(self) > 5 else ""
        return f"MantpyDataset({len(self)} ROIs: {ids}{suffix})"


#: Keys listed inline by ``Bunch.__repr__`` before it elides the rest.
#: Bundles range from 4 keys (schistosoma) to 18 (lung); without a cap the
#: wide ones repr as an unreadable wall.
_REPR_MAX_KEYS = 6


class Bunch(dict):
    """sklearn-style attribute-access container for cohort-loader returns.

    Subclasses ``dict`` so both styles work simultaneously:

    >>> bundle = Bunch(adatas={"roi1": ...}, best_k=7)
    >>> bundle.adatas is bundle["adatas"]
    True

    Used by :func:`mantpy.fetch.load_*` cohort loaders to return a
    bundle of ``AnnData`` objects + sample metadata + per-figure
    knobs.  The dict-style access keeps any external code that
    expects the legacy dict return shape working; the attribute-style
    access keeps notebooks one-line and discoverable.

    The class also implements ``_summarize`` so loaders can opt into
    auto-printing a summary + rendering ``sample_meta.head()`` in
    Jupyter on load — the equivalent of typing ``data`` on a bare
    line, but compatible with ``data = ...`` assignment.

    Bundles do not share one key vocabulary: the schistosoma cohort
    calls its ROI dict ``adatas``, the lung cohort splits cells and
    matrix into ``cohort_cells``/``cohort_ecm``.  The three renderers
    below therefore read the schema off ``_ROI_KEYS``/``_K_KEYS``/
    ``_META_KEYS`` rather than hard-coding key names; a subclass
    describes its own bundle by overriding them (see :class:`LungBunch`).
    Each is a tuple of candidate names, tried in order, first match wins.
    """

    #: Class-level on purpose: ``__init__`` aliases ``__dict__`` to the
    #: dict itself, so an *instance* attribute would show up as bundle data.
    _ROI_KEYS: tuple[str, ...] = ("adatas",)
    _K_KEYS: tuple[str, ...] = ("best_k",)
    _META_KEYS: tuple[str, ...] = ("sample_meta",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mirror items into __dict__ so attribute access works.
        self.__dict__ = self

    def _first_key(self, names: tuple[str, ...]) -> str | None:
        """First declared key present in the bundle, else ``None``.

        Goes through ``dict.__contains__`` rather than ``self.get`` so a
        bundle carrying a data key called ``get`` cannot shadow lookup.
        """
        for name in names:
            if dict.__contains__(self, name):
                return name
        return None

    def _spec_value(self, names: tuple[str, ...]):
        key = self._first_key(names)
        return dict.get(self, key) if key is not None else None

    def __repr__(self) -> str:
        cls = type(self)
        roi_key = self._first_key(cls._ROI_KEYS)
        k_key = self._first_key(cls._K_KEYS)
        n = len(dict.get(self, roi_key)) if roi_key is not None else None
        best_k = dict.get(self, k_key) if k_key is not None else None
        head_parts: list[str] = []
        if n is not None:
            head_parts.append(f"{n} ROIs")
        if best_k is not None:
            head_parts.append(f"K={best_k}")
        # Legacy names stay excluded so a default-schema bundle reprs exactly
        # as it did before the schema spec existed.
        named = {roi_key, k_key, "adatas", "best_k"} - {None}
        extra_keys = sorted(k for k in self if k not in named)
        shown = extra_keys[:_REPR_MAX_KEYS]
        hidden = len(extra_keys) - len(shown)
        tail = ""
        if shown:
            tail = f" + {', '.join(shown)}"
            if hidden:
                tail += f", +{hidden} more"
        head = "; ".join(head_parts) if head_parts else "empty"
        return f"<Bunch {head}{tail}>"

    def _repr_html_(self) -> str:
        cls = type(self)
        roi = self._spec_value(cls._ROI_KEYS)
        n = len(roi) if roi is not None else 0
        best_k = self._spec_value(cls._K_KEYS)
        sample_meta = self._spec_value(cls._META_KEYS)
        parts = [f"<p><b>{n} ROIs"]
        if best_k is not None:
            parts.append(f";  K = {best_k} ECM clusters")
        parts.append("</b></p>")
        if sample_meta is not None and hasattr(sample_meta, "_repr_html_"):
            parts.append(sample_meta.head()._repr_html_())
        return "".join(parts)

    def _summarize(self) -> None:
        """Print summary line + display ``sample_meta.head()`` in Jupyter.

        Called by cohort loaders when ``verbose=True`` (the default).
        Falls back to plain ``print`` when IPython is unavailable.
        """
        cls = type(self)
        roi = self._spec_value(cls._ROI_KEYS)
        n_rois = len(roi) if roi is not None else 0
        best_k = self._spec_value(cls._K_KEYS)
        msg = f"Loaded {n_rois} ROIs"
        if best_k is not None:
            msg += f";  K = {best_k} ECM clusters"
        print(msg)  # noqa: T201  user-facing notebook summary; stdout is the contract
        sample_meta = self._spec_value(cls._META_KEYS)
        if sample_meta is None:
            return
        try:
            from IPython.display import display

            display(sample_meta.head())
        except ImportError:
            # Stdout fallback is the documented + tested behaviour when IPython is absent.
            print(sample_meta.head().to_string())  # noqa: T201


class ColIVIntestineBunch(Bunch):
    """Raw Collagen-IV image and held-out reference for the intestine tutorial.

    The loader deliberately exposes the unprocessed image separately from the
    anatomical reference.  No normalised image, patch features, graph, or model
    output is carried by this object.
    """

    _ROI_KEYS = ()
    _K_KEYS = ()
    _META_KEYS = ()

    def __repr__(self) -> str:
        image = dict.get(self, "image")
        reference = dict.get(self, "reference")
        shape = getattr(image, "shape", None)
        dtype = getattr(image, "dtype", None)
        n_reference = len(reference) if reference is not None else 0
        channel = dict.get(self, "channel_name", "ECM")
        roi = dict.get(self, "roi", "unknown")
        return (
            f"ColIVIntestineBunch · ROI {roi} · raw {channel} {shape} {dtype}\n"
            f"    held-out reference  {n_reference:,} patches"
        )

    def _repr_html_(self) -> str:
        return f"<pre>{repr(self)}</pre>"


class LungBunch(Bunch):
    """Bundle returned by :func:`mantpy.fetch.load_balbc_pbs_lung`.

    Splits the cohort across ``cohort_cells`` and ``cohort_ecm`` rather
    than a single ``adatas``; both are keyed by the same ``roi_names``,
    so either gives the ROI count.

    ``_META_KEYS`` is empty deliberately.  The bundle's ``metadata`` is
    the 36-row manifest for the whole experiment (three slides, RangeIndex,
    ``sample_id`` as a column), not the six-ROI cohort table — so letting
    ``_summarize`` render ``metadata.head()`` would greet the reader with
    five C57BL6J slide-1 rows in a BALB/c loader.  Better to show no table
    than the wrong one.

    The repr follows :class:`anndata.AnnData`'s house style — a header plus
    indented detail — and every figure in it is *measured from the bundle*.
    Nothing about tissue or technology is asserted, because the bundle has
    no field carrying either; a hard-coded ``"mouse lung / IMC"`` here would
    be a lie the moment a second cohort reused the class.
    """

    _ROI_KEYS = ("cohort_cells", "cohort_ecm")
    _K_KEYS = ("K_ecm",)
    _META_KEYS = ()

    def _cohort_meta(self) -> pd.DataFrame | None:
        """``metadata`` narrowed to the cohort's own ROIs (see class docstring)."""
        md, rois = dict.get(self, "metadata"), dict.get(self, "roi_names")
        if md is None or rois is None or "sample_id" not in getattr(md, "columns", []):
            return None
        return md[md["sample_id"].isin(list(rois))]

    def _facts(self) -> dict[str, str]:
        """Measure the bundle. Every value here is read, never assumed."""
        f: dict[str, str] = {}
        cells = dict.get(self, "cohort_cells") or {}
        ecm = dict.get(self, "cohort_ecm") or {}

        # Strain/condition are recorded per ROI on the ECM objects. Only report
        # them when every ROI agrees — a mixed cohort should say so, not pick one.
        for field, key in (("strain", "strain"), ("condition", "condition")):
            vals = {a.uns.get(key) for a in ecm.values() if a.uns.get(key) is not None}
            if len(vals) == 1:
                f[field] = str(vals.pop())

        md = self._cohort_meta()
        if md is not None and len(md):
            if "Mouse" in md:
                f["n_mice"] = str(md["Mouse"].nunique())
            if "Slide" in md and md["Slide"].nunique() == 1:
                f["slide"] = str(md["Slide"].iloc[0])

        if cells:
            f["n_cells"] = f"{sum(a.n_obs for a in cells.values()):,}"
            # `celltype` is the source column; `cell_type` is a loader-added
            # collapse of the "(Activated)" suffixes, so the two disagree on
            # the count. Name the column rather than print a bare number.
            for col in ("celltype", "cell_type"):
                seen = {str(v) for a in cells.values() if col in a.obs for v in a.obs[col].unique()}
                if seen:
                    f["n_cell_types"], f["celltype_col"] = str(len(seen)), col
                    break

        patches = [a.uns["ecm_patches"] for a in ecm.values() if "ecm_patches" in a.uns]
        if patches:
            n = sum(len(p) for p in patches)
            f["n_patches"] = f"{n:,}"
            bg = sum(int((p["ecm_cluster"] < 0).sum()) for p in patches if "ecm_cluster" in p)
            if n:
                f["pct_background"] = f"{100 * bg / n:.0f}"
        if ecm:
            n_var = {a.n_vars for a in ecm.values()}
            if len(n_var) == 1:
                f["n_markers"] = str(n_var.pop())
        names = dict.get(self, "ecm_names")
        if names is not None:
            f["n_ecm_markers"] = str(len(names))

        art = dict.get(self, "adatas_ecm_artifact")
        if art:
            flagged = sum(
                int(a.uns["ecm_patches"]["is_artifact"].sum())
                for a in art.values()
                if "ecm_patches" in a.uns and "is_artifact" in a.uns["ecm_patches"]
            )
            f["n_artefacts"] = f"{flagged:,}"
        return f

    def _summary_lines(self) -> list[str]:
        try:
            f = self._facts()
        except Exception:  # noqa: BLE001  # a repr must never be the thing that breaks a session
            return []
        n_rois = len(dict.get(self, "cohort_cells") or ())
        header = " · ".join(
            str(x)
            for x in (dict.get(self, "tissue"), dict.get(self, "technology"), f.get("strain"), f.get("condition"))
            if x
        )
        cohort = f"{n_rois} ROI" + ("s" if n_rois != 1 else "")
        if f.get("n_mice"):
            n_mice = f["n_mice"]
            cohort += f" from {n_mice} mouse" if n_mice == "1" else f" from {n_mice} mice"
        if f.get("slide"):
            cohort += f" (slide {f['slide']})"
        lines = [header or cohort]
        if header:
            lines.append(f"cohort    {cohort}")

        if f.get("n_cells"):
            bit = f"cells     {f['n_cells']}"
            if f.get("n_cell_types"):
                bit += f"  ·  {f['n_cell_types']} types (obs['{f['celltype_col']}'])"
            lines.append(bit)
        if f.get("n_patches"):
            bit = f"ECM       {f['n_patches']} patches"
            if f.get("n_ecm_markers") and f.get("n_markers"):
                bit += f"  ·  {f['n_ecm_markers']} of {f['n_markers']} markers"
            k = self._spec_value(type(self)._K_KEYS)
            if k is not None:
                bit += f"  ·  K = {k}"
            if f.get("pct_background"):
                bit += f"  ·  {f['pct_background']}% background"
            lines.append(bit)
        if f.get("n_artefacts"):
            lines.append(f"artefact  {f['n_artefacts']} patches flagged (adatas_ecm_artifact)")
        return lines

    def __repr__(self) -> str:
        lines = self._summary_lines()
        if not lines:
            return super().__repr__()
        return "\n    ".join([f"LungBunch · {lines[0]}", *lines[1:]])

    def _repr_html_(self) -> str:
        lines = self._summary_lines()
        if not lines:
            return super()._repr_html_()
        rows = "".join(f"<div style='margin-left:1.5em'>{ln}</div>" for ln in lines[1:])
        return f"<p><b>{lines[0]}</b></p><pre style='margin:0'>{rows}</pre>"

    def _summarize(self) -> None:
        print(repr(self))  # noqa: T201  user-facing notebook summary
