"""Analysis tools for Mantpy."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from mantpy._constants import (
    CELL_ECM_GRAPH_KEY,
    ECM_GRAPH_KEY,
    ECM_PATCHES_KEY,
    INTERACTION_TEST_KEY,
    NEIGHBOURHOOD_CLUSTERS_KEY,
    NODE_TYPE_CELL,
    NODE_TYPE_ECM,
    SPATIAL_KEY,
)
from mantpy._core._permutation import permutation_test as _permutation_test
from mantpy._utils import log_params as _log_params

_log = logging.getLogger(__name__)

__all__ = [
    "AblationROCResult",
    "CentralVoidResult",
    "CellECMContactResult",
    "ClusterCompositionResult",
    "CooccurrencePermResult",
    "ECMNeighbourAgreementSummary",
    "HeldOutDenoiseResult",
    "LesionSummary",
    "PristineFlagSummary",
    "ReconstructionSummary",
    "GraphSmoothingResult",
    "SpatialTransferResult",
    "ablation_roc_curves",
    "cell_ecm_enrichment",
    "cell_ecm_enrichment_matrix",
    "cell_ecm_contact",
    "cell_ecm_topology_sensitivity",
    "cluster_coherence",
    "cluster_cooccurrence",
    "cluster_cooccurrence_permutation_test",
    "compute_cluster_composition",
    "cross_compartment_ablation",
    "denoise_ecm_clusters",
    "denoise_held_out_roi",
    "ecm_neighbour_label_agreement",
    "ecm_to_anndata",
    "graph_modularity",
    "grouped_metric_summary",
    "interaction_test",
    "lesion_central_void",
    "lesion_size_by_sample",
    "lesion_topology_stats",
    "lesion_topology_stats_df",
    "loo_denoise_evaluation",
    "loo_pristine_flag_rate",
    "loo_reconstruction_evaluation",
    "neighbourhood_clustering",
    "pick_representative_samples",
    "summarize_pristine_flag_rate",
    "summarize_reconstruction_evaluation",
    "score_partition",
    "select_n_domains",
    "smooth_graph_signal",
    "summarize_largest_lesion",
    "top_enriched_cluster",
    "transfer_spatial_features",
]


def interaction_test(
    adata: AnnData,
    *,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    n_iter: int = 1000,
    p: float = 0.05,
    n_jobs: int = 1,
    seed: int | None = None,
    key_added: str = INTERACTION_TEST_KEY,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """Permutation test for significant cell-ECM interactions.

    For each (cell_type, ecm_cluster) pair in the graph, builds a null
    distribution by shuffling ECM cluster labels, then classifies each pair as
    enriched (+1), avoided (-1), or non-significant (0).

    Parameters
    ----------
    adata
        AnnData with the unified graph in ``adata.uns[graph_key]``.
    graph_key
        Key of the graph in ``adata.uns``.
    n_iter
        Number of permutation iterations.
    p
        Significance threshold (empirical p-value).
    n_jobs
        Parallel jobs.  ``-1`` uses all available CPUs.
    seed
        Seed for the label permutations. ``None`` (the default) draws fresh
        entropy from the operating system, so the null is not reproducible;
        pass an integer to fix it. Results are independent of ``n_jobs``.
    key_added
        Key under which to store the result in ``adata.uns``.
    inplace
        Modify ``adata`` in place, or return a modified copy.

    Returns
    -------
    Unlike most ``mt.tl`` functions, ``interaction_test`` always returns the
    significance DataFrame as a convenience (even when ``inplace=True``), so
    the result is immediately available for plotting without an extra
    ``adata.uns`` lookup. The same DataFrame is also written to
    ``adata.uns[key_added]`` (rows = cell types, columns = ECM clusters,
    values = ``+1`` enriched / ``-1`` avoided / ``0`` non-significant). When
    ``inplace=False`` the write target is the returned copy of ``adata``,
    not the original.
    """
    if graph_key not in adata.uns:
        raise ValueError(f"Key '{graph_key}' not found in adata.uns. Run `mt.gr.build_cell_ecm_graph(adata)` first.")

    if not inplace:
        adata = adata.copy()

    G = adata.uns[graph_key]
    sigval = _permutation_test(G, n_iter=n_iter, p=p, n_jobs=n_jobs, seed=seed)
    adata.uns[key_added] = sigval

    _log_params(
        adata,
        "tl",
        {
            "interaction_test": {
                "graph_key": graph_key,
                "n_iter": n_iter,
                "p": p,
            }
        },
    )

    return sigval


def neighbourhood_clustering(
    adata: AnnData,
    *,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    n_clusters: int = 4,
    include_ecm: bool = True,
    key_added: str = NEIGHBOURHOOD_CLUSTERS_KEY,
    inplace: bool = True,
) -> AnnData | None:
    """Cluster cells by their interaction neighbourhood (IDIN).

    For each cell, counts how many neighbours of each cell type (and optionally
    each ECM cluster) it has, forming a neighbourhood feature vector.  K-means
    on these vectors produces IDIN (Interaction-Defined Immune Niche) clusters.

    Parameters
    ----------
    adata
        AnnData with ``adata.uns[graph_key]``.
    graph_key
        Key of the unified graph in ``adata.uns``.
    n_clusters
        Number of K-means clusters.
    include_ecm
        Include ECM cluster interaction counts in the feature vector.
    key_added
        ``adata.obs`` column for cluster labels.
    inplace
        Modify ``adata`` in place (returns ``None``), or return a modified copy.

    Writes
    ------
    ``adata.obs[key_added]``
        Integer cluster labels per cell.
    ``adata.uns['neighbourhood_clusters_centroids']``
        DataFrame of K-means centroids.
    """
    if graph_key not in adata.uns:
        raise ValueError(f"Key '{graph_key}' not found in adata.uns. Run `mt.gr.build_cell_ecm_graph(adata)` first.")

    if not inplace:
        adata = adata.copy()

    G = adata.uns[graph_key]

    cell_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("node_type") == NODE_TYPE_CELL]
    if not cell_nodes:
        raise ValueError(
            "No cell nodes found in the graph. "
            "For an ECM-only graph, neighbourhood_clustering is not applicable — "
            "all nodes are ECM patches."
        )

    cell_types = sorted({d["cell_type"] for _, d in cell_nodes})
    ecm_clusters = sorted({str(d["ecm_cluster"]) for _, d in G.nodes(data=True) if d.get("node_type") == NODE_TYPE_ECM})

    feature_cols = [f"ct_{ct}" for ct in cell_types]
    if include_ecm:
        feature_cols += [f"ecm_{ec}" for ec in ecm_clusters]

    node_to_idx = {n: i for i, (n, _) in enumerate(cell_nodes)}
    feat_matrix = np.zeros((len(cell_nodes), len(feature_cols)), dtype=np.float32)

    ct_col = {ct: i for i, ct in enumerate(cell_types)}
    ec_col = {ec: len(cell_types) + i for i, ec in enumerate(ecm_clusters)} if include_ecm else {}

    for node, _ in cell_nodes:
        idx = node_to_idx[node]
        for nbr in G.neighbors(node):
            nbr_data = G.nodes[nbr]
            ntype = nbr_data.get("node_type")
            if ntype == NODE_TYPE_CELL:
                ct = nbr_data.get("cell_type")
                if ct in ct_col:
                    feat_matrix[idx, ct_col[ct]] += 1
            elif ntype == NODE_TYPE_ECM and include_ecm:
                ec = str(nbr_data.get("ecm_cluster", ""))
                if ec in ec_col:
                    feat_matrix[idx, ec_col[ec]] += 1

    from sklearn.cluster import KMeans

    n_clusters = min(n_clusters, len(cell_nodes))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(feat_matrix)

    centroids_df = pd.DataFrame(km.cluster_centers_, columns=feature_cols)
    adata.obs[key_added] = labels.astype(str)
    adata.uns[NEIGHBOURHOOD_CLUSTERS_KEY + "_centroids"] = centroids_df

    _log_params(
        adata,
        "tl",
        {
            "neighbourhood_clustering": {
                "graph_key": graph_key,
                "n_clusters": n_clusters,
                "include_ecm": include_ecm,
            }
        },
    )

    if not inplace:
        return adata
    return None


class ClusterCompositionResult(NamedTuple):
    """Result of :func:`compute_cluster_composition`.

    Attributes
    ----------
    per_sample
        ``(n_samples, K)`` DataFrame.  Rows are ``sample_id`` (in
        ``sample_meta`` order); columns are cluster IDs; values are the
        per-sample fraction of patches in each cluster.  When
        ``drop_background=True`` was passed, each row sums to 1.0.
    per_group_mean
        ``(n_groups, K)`` DataFrame.  Rows are group names (in
        ``group_order``); columns are the same cluster IDs as
        ``per_sample``; values are the within-group mean fraction.
    """

    per_sample: pd.DataFrame
    per_group_mean: pd.DataFrame

    @property
    def cluster_cols(self) -> list:
        """Cluster IDs (the shared column index of both DataFrames)."""
        return self.per_group_mean.columns.tolist()


def compute_cluster_composition(
    adatas: dict[str, AnnData],
    *,
    sample_meta: pd.DataFrame,
    group_col: str = "group",
    cluster_col: str = "ecm_cluster",
    patches_key: str = "ecm_patches",
    drop_background: bool = True,
    group_order: Sequence[str] | None = None,
) -> ClusterCompositionResult:
    """Per-sample and per-group ECM-cluster composition across a cohort.

    For each AnnData in ``adatas``, computes the fraction of patches
    assigned to each cluster (read from
    ``adata.uns[patches_key][cluster_col]``).  Returns:

    - ``per_sample``: wide table, rows = ``sample_id``, columns =
      cluster IDs, values = per-sample fraction.  Optionally drops
      negative cluster IDs (background) and renormalises so each row
      sums to 1.0.
    - ``per_group_mean``: rows = group names (from
      ``sample_meta[group_col]``), columns = the same cluster IDs,
      values = within-group mean fraction across samples.

    Centralises the long-form, pivot, renormalisation, and group-mean
    steps used to compare cluster composition across a cohort.

    Parameters
    ----------
    adatas
        Mapping of ``{sample_id: AnnData}``.
    sample_meta
        Per-sample metadata DataFrame indexed by ``sample_id`` with a
        ``group_col`` column.  Typically the ``sample_meta`` field of a
        :class:`~mantpy.ds.Bunch` returned by a cohort loader.
    group_col
        Column in ``sample_meta`` that partitions the cohort.  Defaults
        to ``"group"``.
    cluster_col
        Column in ``adata.uns[patches_key]`` carrying cluster IDs.
    patches_key
        ``adata.uns`` key for the per-patch DataFrame.
    drop_background
        When ``True`` (default), drops cluster IDs ``< 0`` from the
        ``per_sample`` columns and renormalises so each row sums to
        ``1.0``.  Pass ``False`` to keep raw fractions including the
        background column.
    group_order
        Optional explicit ordering for the ``per_group_mean`` index.
        ``None`` uses the first-appearance order of
        ``sample_meta[group_col]``.

    Returns
    -------
    ClusterCompositionResult
        NamedTuple with ``per_sample``, ``per_group_mean``, and a
        ``cluster_cols`` property listing the shared cluster IDs.

    Examples
    --------
    >>> import mantpy as mt  # doctest: +SKIP
    >>> comp = mt.tl.compute_cluster_composition(
    ...     adatas,
    ...     sample_meta=sample_meta,
    ... )  # doctest: +SKIP
    >>> comp.per_group_mean.shape  # (n_groups, K)  # doctest: +SKIP
    """
    rows: list[dict] = []
    for name in sample_meta.index:
        s = adatas[name].uns[patches_key][cluster_col].value_counts(normalize=True).sort_index()
        g = sample_meta.loc[name, group_col]
        for cl, frac in s.items():
            rows.append({"sample": name, "group": g, "cluster": int(cl), "frac": float(frac)})
    comp = pd.DataFrame(rows)
    wide = comp.pivot(index="sample", columns="cluster", values="frac").fillna(0.0).reindex(sample_meta.index)

    if drop_background:
        valid_cols = sorted(c for c in wide.columns if c >= 0)
        denom = wide[valid_cols].sum(axis=1).replace(0, np.nan)
        per_sample = wide[valid_cols].div(denom, axis=0).fillna(0.0)
    else:
        per_sample = wide

    if group_order is None:
        group_order = list(dict.fromkeys(sample_meta[group_col]))

    per_group_mean = pd.DataFrame(index=list(group_order), columns=per_sample.columns, dtype=float)
    for g in group_order:
        rois_g = sample_meta[sample_meta[group_col] == g].index.tolist()
        per_group_mean.loc[g] = per_sample.loc[rois_g].mean(axis=0).values

    return ClusterCompositionResult(per_sample=per_sample, per_group_mean=per_group_mean)


class LesionSummary(NamedTuple):
    """Per-ROI summary of the largest reconnected lesion.

    Used as the input bundle for shared colour-scale computation across
    multiple ROIs.

    Attributes
    ----------
    n_nodes
        Node count of the lesion (size).
    max_degree
        Maximum node degree across the lesion subgraph.
    max_k_core
        Maximum ``k`` for which the lesion's k-core is non-empty.
    hole_area
        Alpha-shape interior-void area (``central_hole_area``).
    """

    n_nodes: int
    max_degree: int
    max_k_core: int
    hole_area: float


def summarize_largest_lesion(
    adata: AnnData,
    *,
    target_cluster: int | str,
    radius: float,
) -> LesionSummary | None:
    """Summary statistics of the largest reconnected lesion in one ROI.

    Pulls the radius-reconnected largest component of ``target_cluster``
    via :func:`mantpy.gr.largest_component_radius_reconnect`, then
    computes per-lesion node count, max degree, max k-core, and
    alpha-shape interior-void area (via
    :func:`mantpy.tl.lesion_central_void`).

    Returns ``None`` when ``target_cluster`` has fewer than 2 nodes in
    the ROI. Composite plotting code can use the result to compute shared
    colour limits across multiple metric overlays.

    Parameters
    ----------
    adata
        ROI AnnData with an ECM graph at ``adata.uns['ecm_graph']``.
    target_cluster, radius
        Forwarded to :func:`mantpy.gr.largest_component_radius_reconnect`.

    Returns
    -------
    LesionSummary | None
        NamedTuple with ``n_nodes``, ``max_degree``, ``max_k_core``,
        ``hole_area``; ``None`` when no valid lesion is found.
    """
    import networkx as nx

    from mantpy.gr import largest_component_radius_reconnect

    rec = largest_component_radius_reconnect(adata, target_cluster=target_cluster, radius=radius)
    if rec is None or rec["n_nodes"] < 2:
        return None
    big_coords = rec["coords"]
    sub = rec["subgraph"].copy()
    sub.remove_edges_from(nx.selfloop_edges(sub))
    cores = nx.core_number(sub) if sub.number_of_nodes() else {}
    return LesionSummary(
        n_nodes=int(rec["n_nodes"]),
        max_degree=int(max((sub.degree(v) for v in sub.nodes()), default=0)),
        max_k_core=int(max(cores.values()) if cores else 0),
        hole_area=float(lesion_central_void(big_coords, radius=radius).area),
    )


def lesion_size_by_sample(
    lesions_by_sample: dict[str, list[dict]],
    *,
    only_kept: bool = True,
) -> dict[str, int]:
    """Total lesion node count per sample (cohort-level size score).

    For each sample in ``lesions_by_sample``, sums ``n_nodes`` across
    the per-lesion records produced by
    :func:`mantpy.gr.extract_components_radius_reconnect`.

    Parameters
    ----------
    lesions_by_sample
        Mapping ``{sample_id: list[record]}`` where each record has
        ``n_nodes`` and ``kept`` keys (the output shape of
        :func:`mantpy.gr.extract_components_radius_reconnect`).
    only_kept
        When ``True`` (default), only lesions with ``record['kept']``
        contribute (drops min-size-filtered fragments).

    Returns
    -------
    dict[str, int]
        ``{sample_id: total_n_nodes}`` — typically passed as ``scores=``
        to :func:`pick_representative_samples`.
    """
    return {
        name: int(sum(r["n_nodes"] for r in records if (not only_kept or r["kept"])))
        for name, records in lesions_by_sample.items()
    }


def pick_representative_samples(
    sample_meta: pd.DataFrame,
    scores: dict[str, float],
    *,
    group_col: str = "group",
    selector: str = "median",
    min_score: float | None = None,
    selector_per_group: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Pick one representative sample per group, based on per-sample scores.

    For each group in ``sample_meta[group_col]``:

    1. Collect the candidate sample IDs (rows where ``group_col == g``).
    2. Apply ``min_score`` filter (if set).  If the filter empties the
       candidate list, fall back to the unfiltered list.
    3. Apply the selector (``"median"``, ``"max"``, ``"min"``, or
       ``"first"``).  Per-group overrides via ``selector_per_group``
       (handy for "all Naive groups → first ROI, all Infected groups →
       median").
    4. Per-sample manual overrides via ``overrides``.

    Parameters
    ----------
    sample_meta
        Per-sample metadata indexed by ``sample_id``.
    scores
        ``{sample_id: numeric}`` cohort-level score (e.g. total lesion
        size from :func:`lesion_size_by_sample`).
    group_col
        Column in ``sample_meta`` defining the partition.
    selector
        Default selection rule per group: ``"median"`` picks the
        middle-of-sorted-by-score sample; ``"max"`` / ``"min"`` pick
        the extreme; ``"first"`` returns the first candidate (ignores
        score).
    min_score
        If set, samples with ``scores[name] < min_score`` are filtered
        out unless that would empty the group.
    selector_per_group
        Optional per-group override of ``selector``.  Keys are group
        names, values are one of the selector strings.
    overrides
        Optional final ``{group: sample_id}`` overrides applied after
        selection.  Use to pin a specific ROI for visual reasons.

    Returns
    -------
    dict[str, str]
        ``{group: sample_id}`` — one representative per group, in
        ``sample_meta[group_col]`` first-appearance order.
    """
    valid_selectors = {"median", "max", "min", "first"}
    if selector not in valid_selectors:
        raise ValueError(f"Unknown selector {selector!r}.  Choose from {sorted(valid_selectors)}.")
    selector_per_group = selector_per_group or {}
    overrides = overrides or {}
    for s in selector_per_group.values():
        if s not in valid_selectors:
            raise ValueError(f"selector_per_group: unknown selector {s!r}.  Choose from {sorted(valid_selectors)}.")

    picks: dict[str, str] = {}
    for g in dict.fromkeys(sample_meta[group_col]):
        candidates = sample_meta[sample_meta[group_col] == g].index.tolist()
        if not candidates:
            continue
        sel = selector_per_group.get(g, selector)
        if sel == "first":
            picks[g] = candidates[0]
            continue
        scored = [(n, float(scores.get(n, 0.0))) for n in candidates]
        if min_score is not None:
            filtered = [t for t in scored if t[1] >= min_score]
            if filtered:
                scored = filtered
        scored.sort(key=lambda t: t[1])
        if sel == "median":
            picks[g] = scored[len(scored) // 2][0]
        elif sel == "max":
            picks[g] = scored[-1][0]
        elif sel == "min":
            picks[g] = scored[0][0]

    picks.update(overrides)
    return picks


def ecm_to_anndata(
    adata: AnnData,
    *,
    patches_key: str = ECM_PATCHES_KEY,
    feature_prefix: str = "feat_",
    sample_id: str | None = None,
) -> AnnData:
    """Materialize ECM patches as a standalone AnnData (rows = patches).

    Reads ``adata.uns[patches_key]`` (the DataFrame written by
    :func:`mantpy.pp.extract_ecm_patches`) and returns a separate AnnData
    where each row is one ECM patch. This unlocks the full ``squidpy``
    toolchain on ECM patches without modifying the source ``adata``::

        >>> ecm_adata = mt.tl.ecm_to_anndata(adata)
        >>> import squidpy as sq
        >>> sq.gr.spatial_neighbors(ecm_adata, n_neighs=6, coord_type="generic")
        >>> sq.gr.nhood_enrichment(ecm_adata, cluster_key="ecm_cluster")

    Parameters
    ----------
    adata
        Source AnnData containing the ECM patch DataFrame in
        ``adata.uns[patches_key]``.
    patches_key
        Key under which patches were stored. Defaults to
        :data:`mantpy._constants.ECM_PATCHES_KEY` (``"ecm_patches"``).
    feature_prefix
        Prefix used for feature columns in the patch DataFrame. Columns
        starting with this prefix become ``X`` columns; the rest go to
        ``obs``.
    sample_id
        Optional provenance tag stored in ``uns["mantpy_provenance"]``.
        Defaults to ``adata.uns.get("sample_id")``.

    Returns
    -------
    AnnData
        ``X`` = ``(n_patches, n_features)`` float32 matrix of patch features.
        ``obs`` = passthrough of non-feature columns (``ecm_cluster``,
        ``x``, ``y``, ...). ``ecm_cluster`` is cast to categorical.
        ``obsm["spatial"]`` = ``(n_patches, 2)`` patch centroids.
        ``var_names`` = descriptive feature names from
        ``adata.uns["ecm_feature_names"]`` when available
        (e.g. ``"mean__Col1a1"``); otherwise the feature column names with
        ``feature_prefix`` stripped.
    """
    if patches_key not in adata.uns:
        raise ValueError(
            f"Key '{patches_key}' not found in adata.uns. Run `mt.pp.extract_ecm_patches(adata, img)` first."
        )

    patch_df = adata.uns[patches_key]
    feat_cols = [c for c in patch_df.columns if c.startswith(feature_prefix)]
    if not feat_cols:
        raise ValueError(
            f"No feature columns (prefix={feature_prefix!r}) found in "
            f"adata.uns[{patches_key!r}]. Columns: {list(patch_df.columns)}"
        )

    X = patch_df[feat_cols].to_numpy(dtype=np.float32, copy=True)

    obs_cols = [c for c in patch_df.columns if c not in feat_cols]
    obs = patch_df[obs_cols].copy().reset_index(drop=True)
    obs.index = obs.index.astype(str)
    if "ecm_cluster" in obs.columns:
        obs["ecm_cluster"] = obs["ecm_cluster"].astype("category")

    var_names: list[str]
    feat_meta_raw = adata.uns.get("ecm_feature_names")
    feat_meta: list[dict] | None = None
    if isinstance(feat_meta_raw, str):
        try:
            import json as _json

            feat_meta = _json.loads(feat_meta_raw)
        except (ValueError, TypeError):
            # Malformed JSON in adata.uns; treat feature metadata as absent.
            feat_meta = None
    elif isinstance(feat_meta_raw, list):
        feat_meta = feat_meta_raw

    if feat_meta is not None and len(feat_meta) == len(feat_cols):
        var_names = [f"{m.get('extractor', 'feat')}__{m.get('protein', i)}" for i, m in enumerate(feat_meta)]
    else:
        var_names = [c[len(feature_prefix) :] if c.startswith(feature_prefix) else c for c in feat_cols]

    var = pd.DataFrame(index=pd.Index(var_names, dtype=str))
    if feat_meta is not None and len(feat_meta) == len(feat_cols):
        var["extractor"] = [m.get("extractor") for m in feat_meta]
        var["protein"] = [m.get("protein") for m in feat_meta]

    coords = patch_df[["x", "y"]].to_numpy(dtype=np.float32)

    ecm_adata = AnnData(X=X, obs=obs, var=var)
    ecm_adata.obsm["spatial"] = coords
    ecm_adata.uns["mantpy_provenance"] = {
        "source_sample": sample_id if sample_id is not None else adata.uns.get("sample_id"),
        "patches_key": patches_key,
        "n_patches": int(len(patch_df)),
    }

    return ecm_adata


# ---------------------------------------------------------------------------
# Lesion / spatial-component topology statistics
# ---------------------------------------------------------------------------


def lesion_topology_stats(
    subgraph,
    *,
    coords: np.ndarray | None = None,
    metrics: list[str] | None = None,
    central_hole_radius: float | None = None,
) -> dict:
    """Compute graph topology + (optional) geometry statistics on one component.

    Designed to be called with the per-component output of
    :func:`mantpy.gr.extract_components_radius_reconnect`::

        for rec in lesions:
            stats = mt.tl.lesion_topology_stats(
                rec["subgraph"],
                coords=rec["coords"],
            )

    Parameters
    ----------
    subgraph
        NetworkX graph for one connected component (lesion / domain / island).
        Edge weights, when present, are interpreted as Euclidean distances
        (used for ``mean_edge_len`` / ``max_edge_len``).
    coords
        Optional ``(N, 2)`` array of (x, y) positions; required for the
        ``"central_hole"`` metric group.  Ignored otherwise.
    metrics
        Names of metric groups to compute.  ``None`` returns *every* metric
        *except* ``"central_hole"`` (which is opt-in because it needs both
        ``coords`` and ``central_hole_radius`` and pulls in the ``gudhi``
        optional dep).  Choose any subset of:

        - ``"basic"``        — ``n_nodes``, ``n_edges``, ``density``
        - ``"degree"``       — ``mean_degree``, ``max_degree``, ``std_degree``
        - ``"edge_length"``  — ``mean_edge_len``, ``median_edge_len``,
          ``max_edge_len`` (requires edge ``weight``)
        - ``"clustering"``   — ``avg_clustering``, ``transitivity``
        - ``"k_core"``       — ``max_k_core`` (deepest core shell)
        - ``"centrality"``   — ``mean_betweenness``, ``mean_closeness``
        - ``"shortest_path"`` — ``diameter``, ``avg_shortest_path``,
          ``closeness_centralization`` (Freeman centralisation index)
        - ``"connectivity"`` — ``algebraic_connectivity`` (Fiedler value)
        - ``"central_hole"`` — ``central_hole_area``: largest bounded
          un-filled region of the alpha shape (see
          :func:`lesion_central_void`).  Requires ``coords`` and
          ``central_hole_radius``.
    central_hole_radius
        Spatial reconnect radius (px) forwarded to
        :func:`lesion_central_void` as its ``radius`` argument.  Required
        when ``"central_hole"`` is in ``metrics``; ignored otherwise.

    Returns
    -------
    dict
        Metric name -> value.  Metrics that the component is too small to
        support (e.g. shortest-path metrics on a 2-node lesion) return
        ``float('nan')``.

    Notes
    -----
    The k-core depth is the most informative single descriptor of internal
    organisation: high k-core = tightly interconnected interior (e.g. the
    coherent ring-with-core of a mature granuloma); low k-core = sparse
    fragments.
    """
    import networkx as nx

    all_groups = (
        "basic",
        "degree",
        "edge_length",
        "clustering",
        "k_core",
        "centrality",
        "shortest_path",
        "connectivity",
        "central_hole",
    )
    # ``central_hole`` is opt-in: requires the optional ``gudhi`` dependency
    # plus ``central_hole_radius``, so it must be requested explicitly.
    default_groups = tuple(g for g in all_groups if g != "central_hole")
    if metrics is None:
        metrics = list(default_groups)
    unknown = [m for m in metrics if m not in all_groups]
    if unknown:
        raise ValueError(f"Unknown metric group(s) {unknown}.  Choose from {list(all_groups)}.")
    if "central_hole" in metrics:
        if coords is None:
            raise ValueError(
                "metrics=['central_hole', ...] requires the `coords` argument "
                "(the (N, 2) point positions of the lesion)."
            )
        if central_hole_radius is None:
            raise ValueError(
                "metrics=['central_hole', ...] requires `central_hole_radius` "
                "(the alpha-shape filtration radius in pixels; usually the same "
                "value passed to `mt.gr.extract_components_radius_reconnect`)."
            )

    G = subgraph
    n = G.number_of_nodes()
    m = G.number_of_edges()
    out: dict = {}

    if "basic" in metrics:
        out["n_nodes"] = int(n)
        out["n_edges"] = int(m)
        out["density"] = float(nx.density(G)) if n > 1 else 0.0
    if n == 0:
        for grp in metrics:
            if grp == "basic":
                continue
            for k in _LESION_METRIC_NAMES.get(grp, ()):
                out.setdefault(k, float("nan"))
        return out

    if "degree" in metrics:
        degrees = np.asarray([d for _, d in G.degree()], dtype=float)
        out["mean_degree"] = float(degrees.mean())
        out["max_degree"] = float(degrees.max()) if degrees.size else 0.0
        out["std_degree"] = float(degrees.std())
    if "edge_length" in metrics:
        if m > 0:
            try:
                edge_lens = np.asarray(
                    [d.get("weight", np.nan) for _, _, d in G.edges(data=True)],
                    dtype=float,
                )
                out["mean_edge_len"] = float(np.nanmean(edge_lens))
                out["median_edge_len"] = float(np.nanmedian(edge_lens))
                out["max_edge_len"] = float(np.nanmax(edge_lens))
            except ValueError:
                for k in _LESION_METRIC_NAMES["edge_length"]:
                    out[k] = float("nan")
        else:
            for k in _LESION_METRIC_NAMES["edge_length"]:
                out[k] = float("nan")
    if "clustering" in metrics:
        if n > 2:
            out["avg_clustering"] = float(nx.average_clustering(G))
            out["transitivity"] = float(nx.transitivity(G))
        else:
            out["avg_clustering"] = float("nan")
            out["transitivity"] = float("nan")
    if "k_core" in metrics:
        try:
            Gc = G.copy()
            Gc.remove_edges_from(nx.selfloop_edges(Gc))
            cores = nx.core_number(Gc)
            out["max_k_core"] = float(max(cores.values())) if cores else 0.0
        except nx.NetworkXException:
            out["max_k_core"] = float("nan")
    if "centrality" in metrics:
        if n > 3:
            try:
                bc = nx.betweenness_centrality(G, normalized=True, seed=0)
                out["mean_betweenness"] = float(np.mean(list(bc.values())))
            except nx.NetworkXException:
                out["mean_betweenness"] = float("nan")
            try:
                cl_dict = nx.closeness_centrality(G)
                out["mean_closeness"] = float(np.mean(list(cl_dict.values())))
            except nx.NetworkXException:
                out["mean_closeness"] = float("nan")
        else:
            out["mean_betweenness"] = float("nan")
            out["mean_closeness"] = float("nan")
    if "shortest_path" in metrics:
        if n > 3 and nx.is_connected(G):
            try:
                cl_dict = nx.closeness_centrality(G)
                cl_vals = np.asarray(list(cl_dict.values()), dtype=float)
                out["diameter"] = float(nx.diameter(G))
                out["avg_shortest_path"] = float(nx.average_shortest_path_length(G, weight=None))
                # Freeman closeness centralisation, bounded [0, 1]:
                # 0 = uniform, 1 = single hub dominates.
                Cstar = float(cl_vals.max())
                num = float(np.sum(Cstar - cl_vals))
                denom = ((n - 1) * (n - 2)) / (2 * n - 3) if (2 * n - 3) > 0 else 0.0
                out["closeness_centralization"] = num / denom if denom > 0 else float("nan")
            except Exception:  # noqa: BLE001 — best-effort: any path/centralisation failure → NaN metrics
                for k in _LESION_METRIC_NAMES["shortest_path"]:
                    out[k] = float("nan")
        else:
            for k in _LESION_METRIC_NAMES["shortest_path"]:
                out[k] = float("nan")
    if "connectivity" in metrics:
        if n > 3 and nx.is_connected(G) and n <= 5000:
            try:
                out["algebraic_connectivity"] = float(
                    nx.algebraic_connectivity(
                        G,
                        weight=None,
                        method="tracemin_lu",
                        seed=0,
                    )
                )
            except (nx.NetworkXException, RuntimeError):
                out["algebraic_connectivity"] = float("nan")
        else:
            out["algebraic_connectivity"] = float("nan")
    if "central_hole" in metrics:
        try:
            out["central_hole_area"] = float(lesion_central_void(coords, radius=central_hole_radius).area)
        except ImportError:
            raise
        except Exception:  # noqa: BLE001  degenerate geometry → NaN; missing gudhi re-raised above
            out["central_hole_area"] = float("nan")
    return out


_LESION_METRIC_NAMES: dict[str, tuple[str, ...]] = {
    "basic": ("n_nodes", "n_edges", "density"),
    "degree": ("mean_degree", "max_degree", "std_degree"),
    "edge_length": ("mean_edge_len", "median_edge_len", "max_edge_len"),
    "clustering": ("avg_clustering", "transitivity"),
    "k_core": ("max_k_core",),
    "centrality": ("mean_betweenness", "mean_closeness"),
    "shortest_path": ("diameter", "avg_shortest_path", "closeness_centralization"),
    "connectivity": ("algebraic_connectivity",),
    "central_hole": ("central_hole_area",),
}


def lesion_topology_stats_df(
    records_by_sample: dict[str, list[dict]],
    *,
    sample_meta: pd.DataFrame | None = None,
    metrics: list[str] | None = None,
    only_kept: bool = True,
    central_hole_radius: float | None = None,
) -> pd.DataFrame:
    """Run :func:`lesion_topology_stats` across many samples → tidy DataFrame.

    Parameters
    ----------
    records_by_sample
        Mapping ``sample_id -> list of component records`` (the dicts returned
        by :func:`mantpy.gr.extract_components_radius_reconnect`).
    sample_meta
        Optional DataFrame indexed by ``sample_id`` whose columns are merged
        as extra columns on every lesion row (typically condition + genotype).
    metrics
        Forwarded to :func:`lesion_topology_stats`.  Add ``"central_hole"``
        to fold alpha-shape central-void area into the table in one call,
        avoiding the post-hoc per-lesion :func:`lesion_central_void` loop.
    only_kept
        If ``True``, drop any component with ``kept=False`` (the small-fragment
        filter applied by ``extract_components_radius_reconnect``).
    central_hole_radius
        Forwarded to :func:`lesion_topology_stats` as ``central_hole_radius``;
        required when ``"central_hole"`` is in ``metrics``.

    Returns
    -------
    pandas.DataFrame
        One row per lesion.  Columns: ``sample_id``, ``lesion_id``, every
        column from ``sample_meta``, then every metric requested.

    Examples
    --------
    Compute topology + central-hole area in one call::

        lesion_df = mt.tl.lesion_topology_stats_df(
            lesions_by_sample,
            sample_meta=sample_meta,
            metrics=["basic", "degree", "k_core", "central_hole"],
            central_hole_radius=RADIUS_PX,
            only_kept=True,
        )
    """
    rows: list[dict] = []
    for sample_id, recs in records_by_sample.items():
        for k, rec in enumerate(recs):
            if only_kept and not rec.get("kept", True):
                continue
            stats = lesion_topology_stats(
                rec["subgraph"],
                coords=rec.get("coords"),
                metrics=metrics,
                central_hole_radius=central_hole_radius,
            )
            row = {"sample_id": sample_id, "lesion_id": k, **stats}
            rows.append(row)
    df = pd.DataFrame(rows)
    if sample_meta is not None and len(df):
        # Drop metadata columns that already exist here. `sample_meta` is indexed
        # by sample_id and commonly repeats it as a column too, so a plain merge
        # on right_index collides with our own `sample_id` and pandas emits
        # `sample_id_x`/`sample_id_y` alongside it — three columns for one value.
        overlapping = [name for name in sample_meta.columns if name in df.columns]
        df = df.merge(
            sample_meta.drop(columns=overlapping),
            left_on="sample_id",
            right_index=True,
            how="left",
        )
    return df


# ---------------------------------------------------------------------------
# Cluster-cluster spatial co-occurrence
# ---------------------------------------------------------------------------


def cluster_cooccurrence(
    adatas: dict | list,
    *,
    n_clusters: int,
    cluster_attr: str = "ecm_cluster",
    graph_key: str = ECM_GRAPH_KEY,
    ignore_negative_labels: bool = True,
    eps: float = 1e-9,
) -> np.ndarray:
    """Aggregate ``log2(observed / expected)`` cluster-cluster co-occurrence.

    For every edge in every supplied ROI's spatial graph, increment the
    ``inter[c_u, c_v]`` matrix (symmetric).  The expected count under a
    null model where edges connect clusters in proportion to their node
    abundance is ``inter.sum() * p ⊗ p`` where ``p`` is the marginal
    cluster frequency.  The returned matrix is then
    ``log2((observed + eps) / expected)``.

    Subtract two of these matrices to get a Δ-enrichment heatmap (e.g.
    Infected_KO − Infected_WT to see which cluster pairs are KO-enriched).

    Parameters
    ----------
    adatas
        Either a dict ``{sample_id: AnnData}`` or a list of AnnData objects.
        Only the spatial graph at ``adata.uns[graph_key]`` and the
        ``cluster_attr`` node attribute are used.
    n_clusters
        Total number of cluster labels (matrix will be ``n_clusters ×
        n_clusters``).
    cluster_attr
        Node attribute carrying the cluster label.
    graph_key
        Key under which the NetworkX graph lives in ``adata.uns``.
    ignore_negative_labels
        If ``True``, edges touching nodes with cluster label ``< 0``
        (typically ``-1`` for background patches) are skipped.
    eps
        Small additive constant to avoid ``log2(0)`` in empty cells.

    Returns
    -------
    np.ndarray
        Symmetric ``n_clusters × n_clusters`` log₂-enrichment matrix.
        Non-finite entries are zeroed.
    """
    iterable = adatas.values() if isinstance(adatas, dict) else list(adatas)
    inter = np.zeros((n_clusters, n_clusters), dtype=float)
    node_count = np.zeros(n_clusters, dtype=float)
    for adata in iterable:
        if graph_key not in adata.uns:
            continue
        G = adata.uns[graph_key]
        labels = {v: int(d.get(cluster_attr, -1)) for v, d in G.nodes(data=True)}
        for _v, lbl in labels.items():
            if 0 <= lbl < n_clusters:
                node_count[lbl] += 1
        for u, v in G.edges():
            cu, cv = labels[u], labels[v]
            if ignore_negative_labels and (cu < 0 or cv < 0):
                continue
            if not (0 <= cu < n_clusters and 0 <= cv < n_clusters):
                continue
            inter[cu, cv] += 1
            if cu != cv:
                inter[cv, cu] += 1
    p = node_count / max(node_count.sum(), 1.0)
    expected = inter.sum() * np.outer(p, p)
    expected = np.where(expected > 0, expected, np.nan)
    log_e = np.log2((inter + eps) / expected)
    return np.where(np.isfinite(log_e), log_e, 0.0)


class CooccurrencePermResult(NamedTuple):
    """Result of :func:`cluster_cooccurrence_permutation_test`.

    NamedTuple — iterates / unpacks exactly like the legacy
    ``(delta_obs, p_uncorr, p_fdr)`` 3-tuple, so pre-existing call sites
    that do ``delta, p_u, p_f = ...`` keep working unchanged.  New code
    should prefer attribute access (``result.p_fdr``, ``result.delta_obs``)
    for readability.

    Attributes
    ----------
    delta_obs
        ``(K, K)`` array — observed Δ log₂(obs/exp).
    p_uncorr
        ``(K, K)`` array — per-cell two-sided permutation p-values.
    p_fdr
        ``(K, K)`` array — Benjamini-Hochberg-adjusted p-values across
        the unique upper-triangle-including-diagonal cells, mirrored to
        the lower triangle so the matrix stays symmetric.
    """

    delta_obs: np.ndarray
    p_uncorr: np.ndarray
    p_fdr: np.ndarray


def cluster_cooccurrence_permutation_test(
    adatas: dict | list,
    *,
    group_a_rois: list,
    group_b_rois: list,
    n_clusters: int,
    n_perm: int = 5000,
    seed: int = 0,
    cluster_attr: str = "ecm_cluster",
    graph_key: str = ECM_GRAPH_KEY,
    eps: float = 1e-9,
) -> CooccurrencePermResult:
    """Two-sided permutation test on Δ log₂(obs/exp) cluster co-occurrence.

    For two groups of ROIs (e.g. KO and WT), tests the null hypothesis
    that the group labels are exchangeable across ROIs.  For each cell of
    the K×K log-enrichment matrix, the p-value is
    ``(#{|Δ_perm| ≥ |Δ_obs|} + 1) / (n_perm + 1)``.

    Parameters
    ----------
    adatas
        Either a dict ``{sample_id: AnnData}`` or a list of AnnData with
        ``.uns["sample_id"]`` set.  Must contain every ROI in
        ``group_a_rois + group_b_rois``.
    group_a_rois, group_b_rois
        Lists of ROI/sample IDs for the two groups being compared.
        ``delta = log_a − log_b``.
    n_clusters
        Total number of cluster labels (K).
    n_perm
        Number of permutations under the null.  ``5000`` is a good
        compromise between p-value precision and runtime; ``≥1000`` is
        the practical minimum.
    seed
        Random seed for the permutation RNG (reproducibility).
    cluster_attr
        Node attribute carrying the cluster label.
    graph_key
        Key under which the spatial graph lives in ``adata.uns``.
    eps
        Additive constant inside the log to avoid ``log2(0)``.

    Returns
    -------
    :class:`CooccurrencePermResult`
        NamedTuple with fields ``delta_obs``, ``p_uncorr``, ``p_fdr`` —
        in that order, so legacy positional unpacking
        ``delta_obs, p_uncorr, p_fdr = ...`` keeps working unchanged.

    Examples
    --------
    Test which cluster-cluster co-occurrences differ between KO and WT::

        result = mt.tl.cluster_cooccurrence_permutation_test(
            adatas,
            group_a_rois=infected_ko_rois,
            group_b_rois=infected_wt_rois,
            n_clusters=7,
            n_perm=5000,
        )
        # Mask non-significant cells for plotting
        delta_significant = np.where(result.p_fdr < 0.05, result.delta_obs, np.nan)

    Legacy tuple unpacking still works::

        delta, p_uncorr, p_fdr = mt.tl.cluster_cooccurrence_permutation_test(...)
    """
    from mantpy._utils import bh_fdr_correction

    iterable = (
        adatas
        if isinstance(adatas, dict)
        else {getattr(a, "uns", {}).get("sample_id", str(i)): a for i, a in enumerate(adatas)}
    )

    # Per-ROI interaction matrices for fast permutation aggregation.
    per_roi = {}
    for name in list(group_a_rois) + list(group_b_rois):
        adata = iterable.get(name)
        if adata is None or graph_key not in adata.uns:
            continue
        G = adata.uns[graph_key]
        labels = {v: int(d.get(cluster_attr, -1)) for v, d in G.nodes(data=True)}
        inter = np.zeros((n_clusters, n_clusters), dtype=float)
        node_count = np.zeros(n_clusters, dtype=float)
        for _v, lbl in labels.items():
            if 0 <= lbl < n_clusters:
                node_count[lbl] += 1
        for u, v in G.edges():
            cu, cv = labels[u], labels[v]
            if cu < 0 or cv < 0:
                continue
            if not (0 <= cu < n_clusters and 0 <= cv < n_clusters):
                continue
            inter[cu, cv] += 1
            if cu != cv:
                inter[cv, cu] += 1
        per_roi[name] = (inter, node_count)

    def _agg(roi_names):
        inter = np.zeros((n_clusters, n_clusters))
        node_count = np.zeros(n_clusters)
        for name in roi_names:
            if name in per_roi:
                i, nc = per_roi[name]
                inter += i
                node_count += nc
        p = node_count / max(node_count.sum(), 1.0)
        expected = inter.sum() * np.outer(p, p)
        expected = np.where(expected > 0, expected, np.nan)
        log_e = np.log2((inter + eps) / expected)
        return np.where(np.isfinite(log_e), log_e, 0.0)

    log_a_obs = _agg(group_a_rois)
    log_b_obs = _agg(group_b_rois)
    delta_obs = log_a_obs - log_b_obs
    abs_obs = np.abs(delta_obs)

    rng = np.random.default_rng(seed)
    all_rois = list(group_a_rois) + list(group_b_rois)
    n_a = len(group_a_rois)
    ge_count = np.zeros_like(delta_obs, dtype=int)
    for _ in range(n_perm):
        shuffled = list(all_rois)
        rng.shuffle(shuffled)
        delta_perm = _agg(shuffled[:n_a]) - _agg(shuffled[n_a:])
        ge_count += (np.abs(delta_perm) >= abs_obs).astype(int)
    p_uncorr = (ge_count + 1) / (n_perm + 1)

    iu = np.triu_indices(n_clusters)
    fdr_flat = bh_fdr_correction(p_uncorr[iu])
    p_fdr = np.ones_like(p_uncorr)
    for (i, j), pa in zip(zip(*iu, strict=False), fdr_flat, strict=False):
        p_fdr[i, j] = pa
        p_fdr[j, i] = pa
    return CooccurrencePermResult(delta_obs, p_uncorr, p_fdr)


class CentralVoidResult(NamedTuple):
    """Result of :func:`lesion_central_void`.

    NamedTuple — iterates / unpacks exactly like the legacy
    ``(area, component, kept_tris, kept_edges, d_tri)`` 5-tuple, so
    pre-existing call sites that do ``area, *_ = ...`` or full 5-tuple
    unpacking keep working unchanged.  New code should prefer attribute
    access (``result.area``, ``result.kept_tris``, ...) for readability.

    Attributes
    ----------
    area
        Total area (px²) of the largest bounded un-filled component.
    component
        List of Delaunay simplex indices that make up the largest void.
    kept_tris
        Set of (sorted) vertex-index tuples for triangles in the alpha
        complex (i.e. the "filled" interior).
    kept_edges
        Set of (sorted) vertex-index tuples for edges in the alpha
        complex.
    d_tri
        ``scipy.spatial.Delaunay`` object of ``coords`` (or ``None`` if
        fewer than 4 points).
    """

    area: float
    component: list
    kept_tris: set
    kept_edges: set
    d_tri: Any  # scipy.spatial.Delaunay | None


def lesion_central_void(
    coords,
    *,
    radius: float | None = None,
    alpha_sq: float | None = None,
) -> CentralVoidResult:
    """Largest bounded *un-filled* region inside a lesion's alpha shape.

    Builds the 2-D alpha complex of ``coords`` at filtration value
    ``alpha_sq`` (defaults to ``(radius / 2)²``), then finds the bounded
    connected components of *un-filled* Delaunay simplices.  The largest
    such component is the lesion's "central void" — the geometric hole
    you see in a doughnut-shaped lesion.

    Parameters
    ----------
    coords
        ``(N, 2)`` ndarray of 2-D point coordinates (e.g. the radius-
        reconnected ECM-5 patch positions of one lesion).
    radius
        Spatial reconnect radius (px).  ``alpha_sq`` defaults to
        ``(radius / 2) ** 2`` if not given explicitly.
    alpha_sq
        Squared circumradius cutoff for the alpha complex.  Triangles
        with circumradius² ≤ ``alpha_sq`` are "filled"; the rest are
        un-filled and contribute to the void.

    Returns
    -------
    :class:`CentralVoidResult`
        NamedTuple with fields ``area``, ``component``, ``kept_tris``,
        ``kept_edges``, ``d_tri`` — in that order, so legacy positional
        unpacking ``area, comp, kept_tris, kept_edges, d_tri = ...`` and
        partial unpacking ``area, *_ = ...`` keep working unchanged.

    Notes
    -----
    Requires the ``gudhi`` package for the alpha complex and quantifies
    central-void topology in lesion masks.

    Examples
    --------
    Compute the central-void area for every kept lesion::

        lesions = mt.gr.extract_components_radius_reconnect(
            adata,
            target_cluster=5,
            radius=40,
            min_nodes=5,
        )
        for rec in lesions:
            if rec["kept"]:
                result = mt.tl.lesion_central_void(rec["coords"], radius=40)
                print(result.area)

    Legacy tuple-unpacking still works::

        area, comp, kept_tris, kept_edges, d_tri = mt.tl.lesion_central_void(
            rec["coords"],
            radius=40,
        )
    """
    import gudhi
    import networkx as nx
    from scipy.spatial import Delaunay

    coords = np.asarray(coords, dtype=float)
    if alpha_sq is None:
        if radius is None:
            raise ValueError("Provide either `radius` or `alpha_sq`.")
        alpha_sq = (radius / 2.0) ** 2
    if len(coords) < 4:
        return CentralVoidResult(0.0, [], set(), set(), None)

    ac = gudhi.AlphaComplex(points=coords.tolist())
    st = ac.create_simplex_tree()
    kept_tris, kept_edges = set(), set()
    for simplex, filt in st.get_filtration():
        if filt > alpha_sq:
            continue
        if len(simplex) == 3:
            kept_tris.add(tuple(sorted(simplex)))
        elif len(simplex) == 2:
            kept_edges.add(tuple(sorted(simplex)))

    d_tri = Delaunay(coords)
    open_idx = [si for si, t in enumerate(d_tri.simplices) if tuple(sorted(t.tolist())) not in kept_tris]
    if not open_idx:
        return CentralVoidResult(0.0, [], kept_tris, kept_edges, d_tri)

    open_set = set(open_idx)
    G_open = nx.Graph()
    G_open.add_nodes_from(open_idx)
    for si in open_idx:
        for ni in d_tri.neighbors[si]:
            if ni >= 0 and ni in open_set:
                G_open.add_edge(si, ni)
    boundary = {si for si in range(len(d_tri.simplices)) if -1 in d_tri.neighbors[si]}

    best_area, best_comp = 0.0, []
    for comp in nx.connected_components(G_open):
        if set(comp) & boundary:
            continue
        area = 0.0
        for si in comp:
            p = coords[d_tri.simplices[si]]
            v1 = p[1] - p[0]
            v2 = p[2] - p[0]
            area += 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        if area > best_area:
            best_area, best_comp = area, sorted(comp)
    return CentralVoidResult(float(best_area), best_comp, kept_tris, kept_edges, d_tri)


# ---------------------------------------------------------------------------
# Cell -> ECM cluster enrichment with effect size + BH-FDR.
# Complements `interaction_test` (which collapses to ternary +1/-1/0) by
# exposing log2(observed/expected) and per-cluster p-values for ONE focal
# cell type.  Pools edges across one or many AnnDatas so cohort-level tests
# are a single call.
# ---------------------------------------------------------------------------
def cell_ecm_enrichment(
    adata_or_adatas: AnnData | dict[str, AnnData] | list[AnnData],
    cell_type: str,
    *,
    K_ecm: int | None = None,
    n_perm: int = 5000,
    alpha_fdr: float = 0.05,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    cell_attr: str = "cell_type",
    ecm_attr: str = "ecm_cluster",
    random_state: int = 0,
    key_added: str = "cell_ecm_enrichment",
    inplace: bool = True,
) -> pd.DataFrame:
    """Calculate log2(observed/expected) ECM-cluster enrichment around a cell type.

    For each ECM cluster k, counts cell-ECM edges in the unified graph whose
    cell endpoint has ``cell_attr == cell_type`` and whose ECM endpoint has
    ``ecm_attr == k``.  Builds the null by permuting the cell-type labels
    **within each ROI** on the fixed graph (``n_perm`` times) and recounting,
    so the test preserves the spatial graph structure and each ROI's edge
    cluster frequencies — unlike an i.i.d. label draw, which ignores both and
    inflates significance.  Expected counts per ROI are ``edge cluster
    frequency × number of target edges``, pooled across ROIs.  Reports
    observed, expected, log2 enrichment, empirical two-sided p-value (with a
    ``+1`` correction, so the floor is ``1/(n_perm+1)``), and BH-FDR-corrected
    q-value per cluster. Set ``n_perm=0`` to skip inferential testing and return
    descriptive effect sizes with missing p/q values. Per-ROI effect sizes are written to
    ``adata.uns[key_added + '_per_roi']``.

    Parameters
    ----------
    adata_or_adatas
        One AnnData, a list, or a ``{label: AnnData}`` mapping.  All inputs
        must have the unified graph at ``adata.uns[graph_key]``.  When
        multiple AnnDatas are supplied, edge counts are pooled across them
        (a cohort-level test).
    cell_type
        Value of ``cell_attr`` identifying the focal cell type.
    K_ecm
        Number of ECM signal clusters (0..K_ecm-1).  Inferred from the data
        if ``None``.  Background ECM patches (cluster < 0) are excluded.
    n_perm
        Number of label shuffles for the null. Set to zero for descriptive
        effect sizes without p-values or FDR calls.
    alpha_fdr
        Significance level for the FDR call (returned as ``"significant"``
        boolean column).
    graph_key
        ``adata.uns`` key for the unified cell-ECM graph.
    cell_attr, ecm_attr
        Node-attribute names on cell / ECM nodes respectively.
    random_state
        Seed for the permutation RNG.
    key_added
        ``adata.uns`` key under which to write the result.  In the multi-
        AnnData case, the result is written to every input AnnData.
    inplace
        Write to ``adata.uns[key_added]`` (returns the DataFrame as well).

    Returns
    -------
    DataFrame with columns
        ``cluster``, ``n_obs``, ``n_exp``, ``log2_enr``, ``p_value``,
        ``p_fdr``, ``significant``.
    """
    from mantpy._utils import bh_fdr_correction

    if n_perm < 0:
        raise ValueError("n_perm must be non-negative.")

    if isinstance(adata_or_adatas, AnnData):
        adatas = {"_": adata_or_adatas}
    elif isinstance(adata_or_adatas, dict):
        adatas = adata_or_adatas
    else:
        adatas = {f"_{i}": a for i, a in enumerate(adata_or_adatas)}

    eps = 1e-9
    # Per-ROI structure: the cell-type of every cell node, the signal
    # cell-ECM edges as (cell index, ECM cluster), and (below) the ROI's edge
    # cluster frequencies.  Permuting cell-type labels WITHIN each ROI on the
    # fixed graph preserves spatial structure and ROI nesting, unlike an
    # i.i.d. cluster-label draw.
    per_roi: list[dict[str, Any]] = []
    for label, a in adatas.items():
        if graph_key not in a.uns:
            raise ValueError(
                f"Key '{graph_key}' not found in adata.uns. Run `mt.gr.build_cell_ecm_graph(adata)` first."
            )
        G = a.uns[graph_key]
        # node-attr lookup once per graph
        attrs = dict(G.nodes(data=True))
        cell_nodes = [n for n, d in attrs.items() if d.get("node_type") == NODE_TYPE_CELL]
        cell_pos = {n: i for i, n in enumerate(cell_nodes)}
        ct = np.array([str(attrs[n].get(cell_attr, "")) for n in cell_nodes], dtype=object)
        e_cell: list[int] = []
        e_clu: list[int] = []
        for u, v in G.edges():
            au, av = attrs.get(u, {}), attrs.get(v, {})
            t_u = au.get("node_type")
            t_v = av.get("node_type")
            if t_u == NODE_TYPE_CELL and t_v == NODE_TYPE_ECM:
                cell_node, ecm_n = u, av
            elif t_v == NODE_TYPE_CELL and t_u == NODE_TYPE_ECM:
                cell_node, ecm_n = v, au
            else:
                continue
            cl = ecm_n.get(ecm_attr, -1)
            try:
                cl = int(cl)
            except (TypeError, ValueError):
                continue
            if cl < 0:
                continue
            e_cell.append(cell_pos[cell_node])
            e_clu.append(cl)
        per_roi.append(
            {
                "label": label,
                "ct": ct,
                "e_cell": np.asarray(e_cell, dtype=np.int64),
                "e_clu": np.asarray(e_clu, dtype=np.int64),
            }
        )

    all_clu = (
        np.concatenate([r["e_clu"] for r in per_roi])
        if any(r["e_clu"].size for r in per_roi)
        else np.empty(0, dtype=np.int64)
    )
    if all_clu.size == 0:
        raise ValueError("No cell-ECM edges with signal clusters found.")
    if K_ecm is None:
        K_ecm = int(all_clu.max()) + 1
    cluster_levels = np.arange(K_ecm, dtype=np.int32)

    # Fixed per-ROI edge cluster frequencies = expected composition under a
    # random reassignment of which edges are "target" (matches the null mean).
    for r in per_roi:
        ef = np.bincount(r["e_clu"], minlength=K_ecm)[:K_ecm].astype(np.float64)
        r["edge_freq"] = ef / ef.sum() if ef.sum() > 0 else ef

    target = str(cell_type)

    def _enr_counts(ct_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Pooled (observed, expected) target-edge cluster counts."""
        obs = np.zeros(K_ecm, dtype=np.float64)
        exp = np.zeros(K_ecm, dtype=np.float64)
        for r, ct_r in zip(per_roi, ct_list, strict=False):
            if r["e_cell"].size == 0:
                continue
            is_t = ct_r[r["e_cell"]] == target
            if not is_t.any():
                continue
            nb = np.bincount(r["e_clu"][is_t], minlength=K_ecm)[:K_ecm].astype(np.float64)
            obs += nb
            exp += r["edge_freq"] * nb.sum()
        return obs, exp

    obs_counts, exp_counts = _enr_counts([r["ct"] for r in per_roi])
    if obs_counts.sum() == 0:
        raise ValueError(f"No cell-ECM edges incident to cell_type='{cell_type}' found.")
    log2_enr = np.log2((obs_counts + eps) / (exp_counts + eps))

    if n_perm > 0:
        # Null: permute cell-type labels within each ROI on the fixed graph.
        rng = np.random.default_rng(random_state)
        perm_log2 = np.zeros((n_perm, K_ecm), dtype=np.float64)
        for i in range(n_perm):
            ct_perm = [rng.permutation(r["ct"]) for r in per_roi]
            ob, ex = _enr_counts(ct_perm)
            perm_log2[i] = np.log2((ob + eps) / (ex + eps))

        # Two-sided empirical p with +1 correction (floor = 1/(n_perm+1)).
        p_uncorr = ((np.abs(perm_log2) >= np.abs(log2_enr)[None, :]).sum(axis=0) + 1) / (n_perm + 1)
        p_fdr = bh_fdr_correction(p_uncorr)
    else:
        p_uncorr = np.full(K_ecm, np.nan, dtype=np.float64)
        p_fdr = np.full(K_ecm, np.nan, dtype=np.float64)

    df = pd.DataFrame(
        {
            "cluster": cluster_levels,
            "n_obs": obs_counts.astype(int),
            "n_exp": exp_counts,
            "log2_enr": log2_enr,
            "p_value": p_uncorr,
            "p_fdr": p_fdr,
            "significant": p_fdr < alpha_fdr,
        }
    )

    # Per-ROI effect sizes (same statistic computed within each ROI).
    per_roi_rows: list[dict[str, Any]] = []
    for r in per_roi:
        if r["e_cell"].size == 0:
            continue
        is_t = r["ct"][r["e_cell"]] == target
        nb = np.bincount(r["e_clu"][is_t], minlength=K_ecm)[:K_ecm].astype(np.float64)
        ex = r["edge_freq"] * nb.sum()
        l2 = np.log2((nb + eps) / (ex + eps))
        for k in range(K_ecm):
            per_roi_rows.append(
                {
                    "roi": r["label"],
                    "cluster": int(k),
                    "n_obs": int(nb[k]),
                    "n_exp": float(ex[k]),
                    "log2_enr": float(l2[k]),
                }
            )
    per_roi_df = pd.DataFrame(per_roi_rows)

    if inplace:
        for a in adatas.values():
            a.uns[key_added] = df
            a.uns[f"{key_added}_per_roi"] = per_roi_df
            _log_params(
                a,
                "tl",
                {
                    "cell_ecm_enrichment": {
                        "cell_type": cell_type,
                        "K_ecm": int(K_ecm),
                        "n_perm": int(n_perm),
                        "alpha_fdr": float(alpha_fdr),
                        "null": "within_roi_celltype_permutation" if n_perm else "not_run_descriptive",
                    }
                },
            )
    return df


def cell_ecm_topology_sensitivity(
    cells_by_sample: dict[str, AnnData],
    *,
    cell_type: str,
    K_ecm: int,
    topologies: Sequence[str] = ("knn", "delaunay"),
    knn_k: int = 5,
    knn_Dmax: float | None = None,
    n_perm: int = 2000,
    random_state: int | None = 0,
    **enrichment_kwargs: Any,
) -> pd.DataFrame:
    """Compare cell--ECM enrichment across cross-edge topologies.

    For each requested topology, the function copies the cohort, rebuilds only
    the cell--ECM cross edges, and reruns :func:`cell_ecm_enrichment`. Cell and
    ECM within-compartment graphs are therefore held fixed. The returned wide
    frame has one row per ECM cluster and one column per topology.
    """
    if not cells_by_sample:
        raise ValueError("cells_by_sample is empty; expected at least one sample.")
    allowed = {"knn", "delaunay"}
    unknown = [method for method in topologies if method not in allowed]
    if unknown:
        raise ValueError(f"Unsupported topologies {unknown}; expected only {sorted(allowed)}.")

    results: dict[str, pd.Series] = {}
    from mantpy.gr import build_cell_ecm_graph

    for method in topologies:
        cohort = {sample: adata.copy() for sample, adata in cells_by_sample.items()}
        for adata in cohort.values():
            build_cell_ecm_graph(
                adata,
                edge_method=method,
                k=knn_k,
                Dmax_CE=knn_Dmax if method == "knn" else None,
            )
        enrichment = cell_ecm_enrichment(
            cohort,
            cell_type=cell_type,
            K_ecm=K_ecm,
            n_perm=n_perm,
            random_state=random_state,
            inplace=False,
            **enrichment_kwargs,
        )
        results[method] = enrichment.set_index("cluster")["log2_enr"]
    return pd.DataFrame(results)


def cell_ecm_enrichment_matrix(
    cohort_cells: AnnData | dict[str, AnnData] | list[AnnData],
    *,
    K_ecm: int,
    cell_types: Sequence[str] | None = None,
    n_perm: int = 5000,
    alpha_fdr: float = 0.05,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    cell_attr: str = "cell_type",
    ecm_attr: str = "ecm_cluster",
    random_state: int = 0,
) -> pd.DataFrame:
    """Cohort-wide cell-type x ECM-cluster enrichment matrix (one tidy frame).

    Runs :func:`cell_ecm_enrichment` for *every* cell type in turn (same
    within-ROI permutation null, same pooling across ROIs) and stacks the
    results into a single long DataFrame. Crucially, BH-FDR is applied **once
    across the whole matrix** of (cell_type x cluster) p-values — not per
    cell type — so the multiple-testing correction reflects the full set of
    tests in the returned matrix. With ``n_perm=0``, no inferential test or
    multiple-testing correction is run.

    Differs from :func:`cell_ecm_enrichment` (one focal cell type, per-test
    FDR) by sweeping all cell types and correcting jointly; differs from
    :func:`interaction_test` (which collapses to a ternary ±1/0 call) by
    exposing the continuous ``log2(observed/expected)`` effect size.

    Parameters
    ----------
    cohort_cells
        One AnnData, a list, or a ``{roi_name: AnnData}`` mapping, each with
        the unified cell-ECM graph at ``adata.uns[graph_key]``. Edges are
        pooled across all inputs (a cohort-level test).
    K_ecm
        Number of ECM signal clusters (``0..K_ecm-1``). Background patches
        (cluster ``< 0``) are excluded.
    cell_types
        Cell types to test (rows of the matrix). ``None`` (default) tests
        every cell type observed on the cell nodes, in sorted order. A cell
        type with no incident cell-ECM edges is skipped with a warning rather
        than raising.
    n_perm
        Number of within-ROI label shuffles per cell type (the null). Set to
        zero for a descriptive effect-size matrix with missing p/q values.
    alpha_fdr
        Significance threshold for the matrix-wide BH-FDR call.
    graph_key, cell_attr, ecm_attr
        Forwarded to :func:`cell_ecm_enrichment`.
    random_state
        Seed for the permutation RNG (shared across cell types).

    Returns
    -------
    DataFrame
        Tidy long frame, one row per ``(cell_type, cluster)``, with columns
        ``cell_type``, ``cluster``, ``n_obs``, ``n_exp``, ``log2_enr``,
        ``p_value``, ``p_fdr``, ``significant``. ``p_fdr`` is the
        matrix-wide BH-corrected q-value; ``significant`` is ``p_fdr <
        alpha_fdr``. Pivot on ``cell_type`` x ``cluster`` for a heatmap (see
        :func:`mantpy.pl.cell_ecm_enrichment_heatmap`).

    Examples
    --------
    >>> import mantpy as mt  # doctest: +SKIP
    >>> mat = mt.tl.cell_ecm_enrichment_matrix(cohort, K_ecm=3)  # doctest: +SKIP
    >>> mat[mat.cell_type == "AEC"]  # doctest: +SKIP
    """
    from mantpy._utils import bh_fdr_correction

    if isinstance(cohort_cells, AnnData):
        adatas = {"_": cohort_cells}
    elif isinstance(cohort_cells, dict):
        adatas = cohort_cells
    else:
        adatas = {f"_{i}": a for i, a in enumerate(cohort_cells)}

    # Resolve the cell-type row order from the graph cell nodes.
    if cell_types is None:
        observed: set[str] = set()
        for a in adatas.values():
            if graph_key not in a.uns:
                raise ValueError(
                    f"Key '{graph_key}' not found in adata.uns. Run `mt.gr.build_cell_ecm_graph(adata)` first."
                )
            G = a.uns[graph_key]
            for _n, d in G.nodes(data=True):
                if d.get("node_type") == NODE_TYPE_CELL:
                    observed.add(str(d.get(cell_attr, "")))
        observed.discard("")
        cell_types = sorted(observed)
    else:
        cell_types = [str(c) for c in cell_types]

    rows: list[pd.DataFrame] = []
    for ct in cell_types:
        try:
            enr = cell_ecm_enrichment(
                cohort_cells,
                cell_type=ct,
                K_ecm=K_ecm,
                n_perm=n_perm,
                alpha_fdr=alpha_fdr,
                graph_key=graph_key,
                cell_attr=cell_attr,
                ecm_attr=ecm_attr,
                random_state=random_state,
                inplace=False,
            )
        except ValueError:
            # No cell-ECM edges incident to this cell type — skip the row.
            continue
        sub = enr[["cluster", "n_obs", "n_exp", "log2_enr", "p_value"]].copy()
        sub.insert(0, "cell_type", ct)
        rows.append(sub)

    if not rows:
        raise ValueError("No cell type produced any cell-ECM edges; cannot build a matrix.")
    matrix = pd.concat(rows, ignore_index=True)
    if n_perm > 0:
        # Matrix-wide BH-FDR across every (cell_type, cluster) p-value.
        matrix["p_fdr"] = bh_fdr_correction(matrix["p_value"].to_numpy())
        matrix["significant"] = matrix["p_fdr"] < alpha_fdr
    else:
        matrix["p_fdr"] = np.nan
        matrix["significant"] = False
    return matrix


def top_enriched_cluster(
    enr_df: pd.DataFrame,
    *,
    direction: str = "positive",
    require_significant: bool = True,
    cluster_col: str = "cluster",
    log2_col: str = "log2_enr",
    sig_col: str = "significant",
) -> int:
    """Return the most-enriched ECM cluster from a ``cell_ecm_enrichment`` result.

    Convenience selector for the typical downstream step after a
    cell-ECM-cluster enrichment test: drill into the single most-enriched
    (or most-depleted) cluster.

    Parameters
    ----------
    enr_df
        DataFrame returned by :func:`cell_ecm_enrichment` (or any frame with
        the columns named in ``cluster_col``, ``log2_col``, ``sig_col``).
    direction
        ``"positive"`` (default) returns the cluster with the largest
        positive log2 enrichment; ``"negative"`` returns the most depleted.
    require_significant
        If ``True`` (default), restrict the selection to rows whose
        ``sig_col`` value is truthy.
    cluster_col, log2_col, sig_col
        Column names in ``enr_df``.

    Returns
    -------
    int
        The cluster integer label.

    Raises
    ------
    ValueError
        If no row satisfies the direction + significance filter, or if
        ``direction`` is not ``"positive"`` / ``"negative"``.

    Examples
    --------
    >>> import mantpy as mt  # doctest: +SKIP
    >>> enr = mt.tl.cell_ecm_enrichment(cohort, cell_type="AEC", K_ecm=3)
    >>> top_k = mt.tl.top_enriched_cluster(enr, direction="positive")
    """
    if direction not in ("positive", "negative"):
        raise ValueError(f"direction={direction!r}: expected 'positive' or 'negative'. Pass one of these.")
    for col in (cluster_col, log2_col):
        if col not in enr_df.columns:
            raise KeyError(f"column {col!r} not found in enr_df. Available columns: {list(enr_df.columns)}.")

    df = enr_df
    if require_significant:
        if sig_col not in enr_df.columns:
            raise KeyError(
                f"column {sig_col!r} not found in enr_df (required when require_significant=True). "
                f"Available columns: {list(enr_df.columns)}."
            )
        df = df[df[sig_col].astype(bool)]

    if direction == "positive":
        df = df[df[log2_col] > 0]
        if df.empty:
            raise ValueError(
                "no rows with positive log2 enrichment "
                + ("(after significance filter)" if require_significant else "")
                + "; cannot pick a top-enriched cluster."
            )
        idx = df[log2_col].idxmax()
    else:
        df = df[df[log2_col] < 0]
        if df.empty:
            raise ValueError(
                "no rows with negative log2 enrichment "
                + ("(after significance filter)" if require_significant else "")
                + "; cannot pick a top-depleted cluster."
            )
        idx = df[log2_col].idxmin()

    return int(df.loc[idx, cluster_col])


def ecm_neighbour_label_agreement(
    cells_by_sample: dict[str, AnnData],
    ecm_by_sample: dict[str, AnnData],
    *,
    subset_key: str = "is_artifact",
    cluster_key: str = "ecm_cluster",
    graph_key: str = ECM_GRAPH_KEY,
    signal_only: bool = True,
) -> ECMNeighbourAgreementSummary:
    """Summarize local label agreement for a selected set of ECM patches.

    For every patch selected by ``subset_key``, the function calculates the
    fraction of its ECM-graph neighbours with the same ``cluster_key`` label.
    With ``signal_only=True``, both selected patches and their candidate
    neighbours must have non-negative labels, so removed background patches do
    not dilute the result. Selected patches without an eligible neighbour are
    counted but cannot be scored. The reported cohort mean gives every
    scoreable selected patch equal weight; the ROI range is calculated from
    per-sample means.

    Parameters
    ----------
    cells_by_sample
        Cell AnnData mapping carrying an ECM graph in ``uns[graph_key]``.
    ecm_by_sample
        Row-aligned ECM carriers containing ``uns['ecm_patches']``.
    subset_key
        Boolean patch-table column selecting patches to score.
    cluster_key
        Integer patch-label column whose local agreement is measured.
    graph_key
        Key of the ECM-only NetworkX graph on each cell AnnData.
    signal_only
        Exclude negative-label background patches from both the selected set
        and each patch's neighbour denominator.
    """
    if not cells_by_sample or set(cells_by_sample) != set(ecm_by_sample):
        raise ValueError("Cell and ECM cohorts must be non-empty and have identical sample keys.")

    rows: list[dict[str, Any]] = []
    pooled: list[float] = []
    n_selected = 0
    for sample, cells in cells_by_sample.items():
        graph = cells.uns.get(graph_key)
        if graph is None:
            raise KeyError(f"Sample {sample!r} has no ECM graph in uns[{graph_key!r}].")
        patches = ecm_by_sample[sample].uns.get(ECM_PATCHES_KEY)
        if patches is None:
            raise KeyError(f"Sample {sample!r} has no patch table in uns[{ECM_PATCHES_KEY!r}].")
        missing = [key for key in (subset_key, cluster_key) if key not in patches]
        if missing:
            raise KeyError(f"Sample {sample!r} patch table is missing columns {missing}.")

        labels = patches[cluster_key].astype(int).to_numpy()
        selected = patches[subset_key].astype(bool).to_numpy(copy=True)
        if signal_only:
            selected &= labels >= 0
        sample_selected = int(selected.sum())
        n_selected += sample_selected

        agreements: list[float] = []
        for index in np.flatnonzero(selected):
            node = f"ecm_{index}"
            if node not in graph:
                raise ValueError(
                    f"Sample {sample!r} graph is not row-aligned with its ECM patch table; "
                    f"missing node {node!r}."
                )
            neighbour_indices = np.asarray(
                [int(str(neighbour).rsplit("_", 1)[1]) for neighbour in graph.neighbors(node)],
                dtype=int,
            )
            if signal_only:
                neighbour_indices = neighbour_indices[labels[neighbour_indices] >= 0]
            if neighbour_indices.size:
                agreements.append(float((labels[neighbour_indices] == labels[index]).mean()))

        if not agreements:
            raise ValueError(f"Sample {sample!r} has no selected patches with eligible ECM neighbours.")
        pooled.extend(agreements)
        rows.append(
            {
                "sample": sample,
                "n_selected": sample_selected,
                "n_scored": len(agreements),
                "mean_agreement": float(np.mean(agreements)),
            }
        )

    per_sample = pd.DataFrame(rows)
    return ECMNeighbourAgreementSummary(
        mean=float(np.mean(pooled)),
        minimum=float(per_sample["mean_agreement"].min()),
        maximum=float(per_sample["mean_agreement"].max()),
        n_samples=len(per_sample),
        n_patches=n_selected,
        n_scored=len(pooled),
        per_sample=per_sample,
    )


class HeldOutDenoiseResult(NamedTuple):
    """Outputs from :func:`denoise_held_out_roi`."""

    cells: AnnData
    ecm: AnnData
    model: Any

    @property
    def n_relabelled(self) -> int:
        """Number of patches whose label changed."""
        patches = self.ecm.uns[ECM_PATCHES_KEY]
        actual = patches["ecm_cluster"].astype(int).to_numpy()
        denoised = patches["denoised_cluster"].astype(int).to_numpy()
        return int((actual != denoised).sum())

    def __repr__(self) -> str:
        patches = self.ecm.uns[ECM_PATCHES_KEY]
        labels = patches["ecm_cluster"].astype(int).to_numpy()
        return "\n".join(
            [
                "Held-out ECM reconstruction",
                f"  patches       {len(patches):,}",
                f"  signal        {int((labels >= 0).sum()):,}",
                f"  relabelled    {self.n_relabelled:,}",
                f"  model          {type(self.model).__name__}",
            ]
        )


class AblationROCResult(NamedTuple):
    """One model's curves and fold summary from an ablation run."""

    curves: pd.DataFrame
    summary: pd.DataFrame


class PristineFlagSummary(NamedTuple):
    """Cohort summary of held-out false-positive flag rates."""

    mean: float
    minimum: float
    maximum: float
    n_samples: int

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Pristine-patch false positives",
                f"  samples   {self.n_samples}",
                f"  mean      {self.mean:.1%}",
                f"  range     {self.minimum:.1%}--{self.maximum:.1%}",
            ]
        )


class ECMNeighbourAgreementSummary(NamedTuple):
    """Agreement between selected ECM labels and their signal neighbours."""

    mean: float
    minimum: float
    maximum: float
    n_samples: int
    n_patches: int
    n_scored: int
    per_sample: pd.DataFrame

    def __repr__(self) -> str:
        return "\n".join(
            [
                "ECM-neighbour label agreement",
                f"  samples          {self.n_samples}",
                f"  selected patches {self.n_patches:,}",
                f"  with neighbours  {self.n_scored:,}",
                f"  mean agreement   {self.mean:.1%}",
                f"  ROI range        {self.minimum:.1%}--{self.maximum:.1%}",
            ]
        )


class ReconstructionSummary(NamedTuple):
    """Cohort means from a leave-one-sample-out reconstruction evaluation."""

    n_samples: int
    before_accuracy: float
    after_accuracy: float
    artifact_recovered: float
    correction_precision: float
    pristine_relabelled: float
    pristine_relabelled_minimum: float
    pristine_relabelled_maximum: float

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Leave-one-sample-out ECM reconstruction",
                f"  samples                    {self.n_samples}",
                f"  accuracy                   {self.before_accuracy:.1%} -> {self.after_accuracy:.1%}",
                f"  injected errors recovered  {self.artifact_recovered:.1%}",
                f"  corrections correct        {self.correction_precision:.1%}",
                f"  pristine patches relabelled {self.pristine_relabelled:.1%} "
                f"({self.pristine_relabelled_minimum:.1%}--{self.pristine_relabelled_maximum:.1%})",
            ]
        )


def denoise_ecm_clusters(
    cell_adata: AnnData,
    ecm_adata: AnnData,
    *,
    model: Any,
    anomaly_threshold: float = 0.30,
    spatial_radius: float = 14.0,
    spatial_min_neighbours: int = 2,
    anomaly_key: str = "anomaly_score",
    denoised_key: str = "denoised_cluster",
    cluster_key: str = "ecm_cluster",
    copy: bool = False,
) -> AnnData | None:
    """Score ECM patches with a fitted scorer and clean them spatially.

    Applies the spatially coherent cleaning rule used for the AEC--ECM niche analysis.
    The scorer returns a per-patch anomaly score and candidate cluster label.
    A patch is relabelled only when:

    * its current label is a signal cluster (not background),
    * the model's argmax is also a signal cluster (background excluded),
    * the anomaly score crosses ``anomaly_threshold``, **and**
    * the candidate has at least ``spatial_min_neighbours`` other
      candidates within ``spatial_radius`` px (suppresses isolated
      false-positive relabels).

    All other patches keep their original label. Anomaly score, denoised
    cluster, and the model's per-class probabilities are written to
    ``ecm_adata.uns['ecm_patches']`` (always) and mirrored to
    ``ecm_adata.obs`` when ``n_obs == n_patches``.

    Parameters
    ----------
    cell_adata
        Cell-side AnnData carrying the unified
        ``adata.uns['cell_ecm_graph']`` from
        :func:`mantpy.gr.build_cell_ecm_graph`.
    ecm_adata
        ECM-side AnnData. Must carry
        ``ecm_adata.uns['ecm_patches']`` with the ``cluster_key`` column.
    model
        Fitted model exposing ``score(cell_adata, ecm_adata)``. Its output
        must contain ``anomaly_score``, ``denoised_cluster``, ``p_actual``,
        and ``p_pred`` columns.
    anomaly_threshold
        Minimum ``anomaly_score`` (0–1, after min-max normalisation by
        the scorer) required to consider a relabel.
    spatial_radius
        Distance in pixels within which to count other candidate
        patches. Defaults to ``14.0`` (matches the BALB/c PBS lung
        cohort's ECM-ECM kNN distance ceiling).
    spatial_min_neighbours
        Minimum number of *other* candidates that must lie within
        ``spatial_radius``. Pass ``0`` to disable the spatial-coherence
        filter (each candidate stands alone).
    anomaly_key, denoised_key
        Column names for the anomaly score (float, 0–1) and the denoised
        cluster label (int).
    cluster_key
        Name of the existing cluster column in
        ``ecm_adata.uns['ecm_patches']``.
    copy
        Return a copy of ``ecm_adata`` instead of mutating in place
        (scverse convention).

    Returns
    -------
    AnnData or None
        The mutated (or copied) ``ecm_adata``, or ``None`` when
        ``copy=False``.

    Examples
    --------
    >>> import mantpy as mt  # doctest: +SKIP
    >>> model = mt.nn.NeighbourCompositionBaseline(context="cell")  # doctest: +SKIP
    >>> model.fit([(c0, e0), (c1, e1)])  # doctest: +SKIP
    >>> mt.tl.denoise_ecm_clusters(cells_target, ecm_target, model=model)  # doctest: +SKIP
    """
    if not (0.0 <= anomaly_threshold <= 1.0):
        raise ValueError(f"anomaly_threshold must lie in [0, 1] (got {anomaly_threshold}).")
    if spatial_radius < 0:
        raise ValueError(f"spatial_radius must be >= 0 (got {spatial_radius}).")
    if spatial_min_neighbours < 0:
        raise ValueError(f"spatial_min_neighbours must be >= 0 (got {spatial_min_neighbours}).")
    if not hasattr(model, "score"):
        raise TypeError(f"model must be a fitted scorer exposing .score(); got {type(model).__name__}.")

    target = ecm_adata.copy() if copy else ecm_adata

    # Read the existing cluster vector from uns['ecm_patches']; this is
    # canonical for mantpy's ECM patch pipeline.  Two valid layouts:
    #   (a) AnnData has n_obs = n_patches (the `mt.tl.ecm_to_anndata`
    #       output) — write the new columns to .obs as well.
    #   (b) AnnData has n_obs = 0 and the patch DataFrame lives in
    #       .uns['ecm_patches'] (the bundled BALB/c PBS lung layout) —
    #       only the patch DataFrame is touched.
    patches = target.uns.get("ecm_patches")
    if patches is None or cluster_key not in patches.columns:
        raise KeyError(f"ecm_adata.uns['ecm_patches'] missing or lacks the '{cluster_key}' column.")
    if not {"x", "y"}.issubset(patches.columns) and spatial_min_neighbours > 0:
        raise KeyError(
            "ecm_adata.uns['ecm_patches'] must contain 'x' and 'y' "
            "columns for the spatial-coherence filter "
            "(or pass spatial_min_neighbours=0 to disable it)."
        )

    scores = model.score(cell_adata, target)
    expected = {"anomaly_score", "denoised_cluster", "p_actual", "p_pred"}
    missing = expected - set(scores.columns)
    if missing:
        raise RuntimeError(f"model.score() returned DataFrame missing columns: {sorted(missing)}")

    n_patches = len(patches)
    if len(scores) != n_patches:
        raise RuntimeError(
            f"model.score() returned {len(scores)} rows but the patch "
            f"DataFrame in ecm_adata.uns['ecm_patches'] has {n_patches} rows."
        )

    anom = scores["anomaly_score"].to_numpy()
    pred = scores["denoised_cluster"].to_numpy()
    actual = patches[cluster_key].astype(int).to_numpy()

    # Cleaning rule:
    #   1. Current label must be a signal cluster (not background).
    #   2. Prediction must differ from the current label.
    #   3. Prediction must also be a signal cluster (not background).
    #   4. Anomaly score must cross `anomaly_threshold`.
    K = None
    if hasattr(model, "K_ecm") and model.K_ecm is not None:
        K = int(model.K_ecm)
    actual_is_signal = (actual >= 0) if K is None else ((actual >= 0) & (actual < K))
    pred_is_signal = (pred >= 0) if K is None else ((pred >= 0) & (pred < K))
    candidate = actual_is_signal & pred_is_signal & (pred != actual) & (anom >= anomaly_threshold)

    # 5. Spatial-coherence filter (cKDTree): each candidate must have at
    #    least `spatial_min_neighbours` OTHER candidates within
    #    `spatial_radius` px. Disabled when spatial_min_neighbours == 0.
    relabel = candidate.copy()
    if spatial_min_neighbours > 0 and candidate.any():
        from scipy.spatial import cKDTree

        cand_xy = patches.loc[candidate, ["x", "y"]].to_numpy(dtype=float)
        tree = cKDTree(cand_xy)
        counts = tree.query_ball_point(cand_xy, spatial_radius, return_length=True)
        # Subtract 1 because each candidate matches itself at d=0.
        keep = (np.asarray(counts) - 1) >= spatial_min_neighbours
        relabel = np.zeros_like(candidate)
        relabel[np.where(candidate)[0][keep]] = True

    denoised = actual.copy()
    denoised[relabel] = pred[relabel]

    # Always write to the canonical patch DataFrame in uns.
    patches[anomaly_key] = anom.astype(np.float32)
    patches[denoised_key] = denoised.astype(int)
    target.uns["ecm_patches"] = patches  # re-bind defensively

    # Also write to .obs when the AnnData is one-obs-per-patch (the
    # `mt.tl.ecm_to_anndata` layout).  Skip on the n_obs=0 layout.
    if target.n_obs == n_patches:
        target.obs[anomaly_key] = anom.astype(np.float32)
        target.obs[denoised_key] = pd.Categorical(
            denoised.astype(int),
            categories=sorted(set(np.unique(denoised).tolist()) | set(np.unique(actual).tolist())),
        )

    if copy:
        return target
    return None


def denoise_held_out_roi(
    cells_by_sample: dict[str, AnnData],
    ecm_by_sample: dict[str, AnnData],
    *,
    held_out: str,
    target_pool: dict[str, AnnData] | None = None,
    model_factory: Any = None,
    model_kwargs: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    graph_kwargs: dict[str, Any] | None = None,
    denoise_kwargs: dict[str, Any] | None = None,
) -> HeldOutDenoiseResult:
    """Train on the other samples and denoise one held-out ECM sample.

    This is the compact, reusable form of the tutorial's held-out fitting
    scaffold. Training always uses the pristine ``ecm_by_sample`` objects.
    The held-out target comes from ``target_pool`` when supplied (for example,
    an artefact-injected validation cohort), otherwise from ``ecm_by_sample``.
    A copied cell object is rebuilt against the target patches before scoring,
    so the input mappings are never modified.

    Returns the rebuilt target cells, denoised ECM object, and fitted model in
    a :class:`HeldOutDenoiseResult`.
    """
    if held_out not in cells_by_sample or held_out not in ecm_by_sample:
        raise KeyError(f"Held-out sample {held_out!r} must occur in both input mappings.")
    if target_pool is not None and held_out not in target_pool:
        raise KeyError(f"Held-out sample {held_out!r} is missing from target_pool.")

    train_names = [name for name in cells_by_sample if name != held_out]
    if not train_names:
        raise ValueError("Held-out denoising needs at least one training sample.")
    missing_ecm = [name for name in train_names if name not in ecm_by_sample]
    if missing_ecm:
        raise KeyError(f"Training samples missing from ecm_by_sample: {missing_ecm}.")

    if model_factory is None:
        from mantpy.nn import NeighbourCompositionBaseline

        model_factory = NeighbourCompositionBaseline
        model_kwargs = {"context": "cell", **dict(model_kwargs or {})}

    model = model_factory(**dict(model_kwargs or {}))
    model.fit(
        [(cells_by_sample[name], ecm_by_sample[name]) for name in train_names],
        **dict(fit_kwargs or {}),
    )

    target_cells = cells_by_sample[held_out].copy()
    source = ecm_by_sample if target_pool is None else target_pool
    target_ecm = source[held_out].copy()
    target_cells.uns[ECM_PATCHES_KEY] = target_ecm.uns[ECM_PATCHES_KEY].copy()

    from mantpy.gr import ensure_cell_ecm_graph

    graph_recipe = dict(graph_kwargs or {})
    graph_recipe["rebuild"] = True
    ensure_cell_ecm_graph(target_cells, **graph_recipe)
    denoise_ecm_clusters(target_cells, target_ecm, model=model, **dict(denoise_kwargs or {}))
    return HeldOutDenoiseResult(cells=target_cells, ecm=target_ecm, model=model)


def loo_reconstruction_evaluation(
    cells_by_sample: dict[str, AnnData],
    ecm_by_sample: dict[str, AnnData],
    *,
    artefact_pool: dict[str, AnnData],
    roi_names: Sequence[str] | None = None,
    model_factory: Any = None,
    model_kwargs: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    graph_kwargs: dict[str, Any] | None = None,
    denoise_kwargs: dict[str, Any] | None = None,
    truth_col: str = "is_artifact",
    cluster_key: str = "ecm_cluster",
    denoised_key: str = "denoised_cluster",
) -> pd.DataFrame:
    """Evaluate completed ECM-label reconstruction in every held-out sample.

    For each sample, trains the requested model on the remaining pristine
    samples, applies the spatially gated denoising rule to that sample's
    artefact-injected patch labels, and compares both the injected and final
    labels with the pristine reference labels. Unlike
    :func:`loo_denoise_evaluation`, which measures anomaly ranking, this helper
    evaluates the *labels actually changed by the cleaner*.

    Parameters
    ----------
    cells_by_sample, ecm_by_sample
        Pristine cell and ECM mappings with matching sample keys.
    artefact_pool
        Artefact-injected ECM mapping. Patch order and count must match the
        pristine ECM reference for every sample and ``truth_col`` must identify
        injected patches.
    roi_names
        Explicit held-out order. Defaults to sorted cell-cohort keys.
    model_factory, model_kwargs, fit_kwargs, graph_kwargs, denoise_kwargs
        Forwarded to :func:`denoise_held_out_roi`. The default model is the
        cell-context :class:`mantpy.nn.NeighbourCompositionBaseline`; the
        default cleaning thresholds are those of
        :func:`denoise_ecm_clusters`.
    truth_col, cluster_key, denoised_key
        Patch-table columns for injected truth, observed labels and completed
        reconstruction labels.

    Returns
    -------
    pandas.DataFrame
        One row per held-out sample with patch counts, injected fraction,
        accuracy before and after reconstruction, artefact recovery, correction
        precision and the pristine-patch relabelling rate.
    """
    rois = list(roi_names) if roi_names is not None else sorted(cells_by_sample)
    if len(rois) < 2:
        raise ValueError("Leave-one-out reconstruction needs at least two samples.")
    for roi in rois:
        if roi not in cells_by_sample or roi not in ecm_by_sample or roi not in artefact_pool:
            raise KeyError(f"Sample {roi!r} must occur in cells, ECM, and artefact mappings.")

    rows: list[dict[str, Any]] = []
    for roi in rois:
        result = denoise_held_out_roi(
            cells_by_sample,
            ecm_by_sample,
            held_out=roi,
            target_pool=artefact_pool,
            model_factory=model_factory,
            model_kwargs=model_kwargs,
            fit_kwargs=fit_kwargs,
            graph_kwargs=graph_kwargs,
            denoise_kwargs=denoise_kwargs,
        )
        reference_patches = ecm_by_sample[roi].uns.get(ECM_PATCHES_KEY)
        reconstructed_patches = result.ecm.uns.get(ECM_PATCHES_KEY)
        if reference_patches is None or reconstructed_patches is None:
            raise KeyError(f"Sample {roi!r} is missing uns[{ECM_PATCHES_KEY!r}].")
        for column in (cluster_key, denoised_key, truth_col):
            source = reference_patches if column == cluster_key else reconstructed_patches
            if column not in source:
                raise KeyError(f"Sample {roi!r} patch table is missing column {column!r}.")
        if len(reference_patches) != len(reconstructed_patches):
            raise ValueError(
                f"Sample {roi!r} has {len(reference_patches)} pristine patches but "
                f"{len(reconstructed_patches)} artefact patches; patch order/count must match."
            )

        reference = reference_patches[cluster_key].astype(int).to_numpy()
        observed = reconstructed_patches[cluster_key].astype(int).to_numpy()
        reconstructed = reconstructed_patches[denoised_key].astype(int).to_numpy()
        is_artifact = reconstructed_patches[truth_col].astype(bool).to_numpy()
        is_signal = reference >= 0
        artifact_signal = is_artifact & is_signal
        pristine_signal = (~is_artifact) & is_signal
        changed = (reconstructed != observed) & is_signal

        n_signal = int(is_signal.sum())
        n_artifact = int(artifact_signal.sum())
        rows.append(
            {
                "roi": roi,
                "n_signal": n_signal,
                "n_artifact": n_artifact,
                "n_relabelled": int(changed.sum()),
                "corrupted_fraction": float(n_artifact / n_signal) if n_signal else float("nan"),
                "before_accuracy": float((observed[is_signal] == reference[is_signal]).mean())
                if n_signal
                else float("nan"),
                "after_accuracy": float((reconstructed[is_signal] == reference[is_signal]).mean())
                if n_signal
                else float("nan"),
                "artifact_recovered": float(
                    (reconstructed[artifact_signal] == reference[artifact_signal]).mean()
                )
                if n_artifact
                else float("nan"),
                "correction_precision": float((reconstructed[changed] == reference[changed]).mean())
                if changed.any()
                else float("nan"),
                "pristine_relabel_rate": float(
                    (reconstructed[pristine_signal] != reference[pristine_signal]).mean()
                )
                if pristine_signal.any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_reconstruction_evaluation(results: pd.DataFrame) -> ReconstructionSummary:
    """Return compact cohort means from reconstruction-evaluation rows.

    Each held-out sample contributes equally, matching the fold-level summary
    used by :func:`loo_reconstruction_evaluation` and the tutorial figures.
    The pristine-patch relabelling range is retained so a cohort mean cannot
    hide an unusually high held-out sample.
    """
    required = {
        "before_accuracy",
        "after_accuracy",
        "artifact_recovered",
        "correction_precision",
        "pristine_relabel_rate",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise KeyError(f"Reconstruction results are missing columns {missing}.")
    if results.empty:
        raise ValueError("Reconstruction results are empty.")
    metrics = results[list(required)].apply(pd.to_numeric, errors="coerce")
    if metrics.isna().all().any():
        invalid = list(metrics.columns[metrics.isna().all()])
        raise ValueError(f"Reconstruction metrics contain no finite values for {invalid}.")

    pristine = metrics["pristine_relabel_rate"]
    return ReconstructionSummary(
        n_samples=len(results),
        before_accuracy=float(metrics["before_accuracy"].mean()),
        after_accuracy=float(metrics["after_accuracy"].mean()),
        artifact_recovered=float(metrics["artifact_recovered"].mean()),
        correction_precision=float(metrics["correction_precision"].mean()),
        pristine_relabelled=float(pristine.mean()),
        pristine_relabelled_minimum=float(pristine.min()),
        pristine_relabelled_maximum=float(pristine.max()),
    )


def _score_one_fold(
    *,
    held_out: str,
    fold_grp: Any,
    model: Any,
    cells_by_sample: dict[str, AnnData],
    ecm_by_sample: dict[str, AnnData],
    artefact_pool: dict[str, AnnData] | None,
    graph_kwargs: dict[str, Any],
    denoise_kwargs: dict[str, Any] | None,
    truth_col: str,
    signal_only: bool,
    curve_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    scored: dict[str, AnnData],
    verbose: bool,
) -> None:
    """Score a single held-out ROI with an already-trained ``model``.

    Shared body of :func:`loo_denoise_evaluation` so leave-one-ROI-out and
    leave-one-group-out reuse identical graph-rebuild / scoring / curve
    bookkeeping. ``fold_grp`` is the held-out group label (``None`` in the
    classic per-ROI mode); when given, it is recorded in a ``group``
    column on the summary row. Appends to ``curve_rows`` / ``summary_rows``
    and writes the scored ECM AnnData into ``scored`` in place.
    """
    cell_target = cells_by_sample[held_out].copy()
    if artefact_pool is not None:
        ecm_target = artefact_pool[held_out].copy()
    else:
        ecm_target = ecm_by_sample[held_out].copy()
    cell_target.uns[ECM_PATCHES_KEY] = ecm_target.uns[ECM_PATCHES_KEY]

    from mantpy.gr import ensure_cell_ecm_graph

    graph_recipe = dict(graph_kwargs)
    graph_recipe["rebuild"] = True
    ensure_cell_ecm_graph(cell_target, **graph_recipe)

    if denoise_kwargs is not None:
        denoise_ecm_clusters(cell_target, ecm_target, model=model, **denoise_kwargs)

    if artefact_pool is not None:
        metrics = model.score_anomaly(
            cell_target,
            ecm_target,
            truth_col=truth_col,
            signal_only=signal_only,
        )
        roc_a = metrics["roc_auc"]
        pr_a = metrics["pr_auc"]
        for x_, y_ in zip(metrics["fpr"], metrics["tpr"], strict=False):
            curve_rows.append({"fold": held_out, "kind": "roc", "x": float(x_), "y": float(y_)})
        for r_, p_ in zip(metrics["recall"], metrics["precision"], strict=False):
            curve_rows.append({"fold": held_out, "kind": "pr", "x": float(r_), "y": float(p_)})
        # Make sure anomaly_score is on the held-out patches even
        # when the spatial-coherence rule wasn't run.
        patches = ecm_target.uns[ECM_PATCHES_KEY]
        if "anomaly_score" not in patches.columns:
            patches["anomaly_score"] = metrics["scores"]["anomaly_score"].to_numpy()
    else:
        roc_a = float("nan")
        pr_a = float("nan")

    row: dict[str, Any] = {"fold": held_out, "roc_auc": roc_a, "pr_auc": pr_a}
    if fold_grp is not None:
        row["group"] = fold_grp
    summary_rows.append(row)
    scored[held_out] = ecm_target

    if verbose:
        dev = getattr(model, "device", "?")
        grp_txt = f"  [group {fold_grp}]" if fold_grp is not None else ""
        _log.info(
            "  fold %s%s   ROC-AUC = %.3f   PR-AUC = %.3f   (device: %s)",
            held_out,
            grp_txt,
            roc_a,
            pr_a,
            dev,
        )


def loo_denoise_evaluation(
    cells_by_sample: dict[str, AnnData],
    ecm_by_sample: dict[str, AnnData],
    *,
    artefact_pool: dict[str, AnnData] | None = None,
    roi_names: Sequence[str] | None = None,
    model_factory: Any = None,
    model_kwargs: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    graph_kwargs: dict[str, Any] | None = None,
    denoise_kwargs: dict[str, Any] | None = None,
    truth_col: str = "is_artifact",
    cluster_key: str = "ecm_cluster",
    celltype_key: str = "cell_type",
    signal_only: bool = True,
    groupby: str | None = None,
    group_map: dict[str, Any] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, AnnData]]:
    """Leave-one-ROI-out (or leave-one-group-out) evaluation of a denoiser.

    For each ROI in turn: train a fresh denoiser on the other ROIs'
    pristine ``(cell, ecm)`` pairs, swap the held-out ROI's ECM patches
    for the artefact-injected version (when ``artefact_pool`` is given),
    rebuild the held-out ROI's three graphs, score with the trained
    model, and collect per-fold ROC + PR curves + AUCs.

    When ``group_map`` (or ``groupby``) is supplied, the cross-validation
    becomes **leave-one-group-out** instead: each held-out fold is an
    entire group of ROIs (e.g. all ROIs from one mouse), the model trains
    only on ROIs *not* in that group — so no same-group ROI ever leaks
    into training — and each held-out ROI is still scored individually
    (one ``summary`` row per ROI, tagged with its group in the ``group``
    column). This answers "does detection generalise across animals?"
    without same-mouse leakage. The default (``group_map=None``) is the
    classic leave-one-ROI-out and is unchanged.

    Collapses a multi-step leave-one-out scaffold into a single call. Works
    on any cohort that exposes the same ``cells_by_sample`` /
    ``cohort_ecm`` / (optional) artefact-overlay layout as
    :func:`mantpy.fetch.load_balbc_pbs_lung`.

    Parameters
    ----------
    cells_by_sample
        ``{roi_name: cell_adata}`` map. Each ``cell_adata`` must carry
        ``obsm['spatial']``, ``obs[celltype_key]``, and (typically) an
        already-attached ``uns['ecm_patches']``.
    ecm_by_sample
        ``{roi_name: ecm_adata}`` map of pristine ECM AnnDatas, one per
        ROI. Each must carry ``uns['ecm_patches'][cluster_key]``.
    artefact_pool
        Optional ``{roi_name: ecm_adata}`` map of artefact-injected
        ECMs. When provided, the held-out ROI's ``ecm_patches`` are
        swapped for the artefact version before scoring; the
        ``truth_col`` column on that DataFrame is the supervision
        signal for ROC/PR. When ``None``, the LOO loop still runs but
        the AUC columns are NaN (useful for a pure-denoising sanity
        check).
    roi_names
        Optional explicit fold ordering. Defaults to
        ``sorted(cells_by_sample)``.
    model_factory
        Callable that returns a fresh, untrained model exposing
        ``.fit(pristine_pairs, ...)`` and ``.score_anomaly(...)``.
        Defaults to :class:`mantpy.nn.NeighbourCompositionBaseline` with
        ``context='cell'``.
    model_kwargs
        Kwargs forwarded to ``model_factory()`` on every fold (e.g.
        ``{'context': 'joint', 'max_iter': 2000}``).
    fit_kwargs
        Kwargs forwarded to ``model.fit()`` on every fold. The
        ``cluster_key`` and ``celltype_key`` arguments are filled in
        automatically.
    graph_kwargs
        Kwargs forwarded to :func:`mantpy.gr.ensure_cell_ecm_graph` on
        the held-out ROI. Defaults to
        ``{'cell_k': 5, 'ecm_k': 5, 'cell_ecm_k': 5}`` (matches
        :func:`mantpy.gr.build_cell_ecm_graph`'s own defaults).
    denoise_kwargs
        Kwargs forwarded to :func:`mantpy.tl.denoise_ecm_clusters`. When
        ``None`` (default) the denoiser is run but the denoised labels
        are not written via the spatial-coherence rule — only the
        anomaly scores feed ROC/PR.
    truth_col
        Boolean column on the artefact ROI's ``ecm_patches`` that flags
        the positive class for ROC/PR (default ``'is_artifact'``).
    cluster_key, celltype_key
        Standard mantpy patch-cluster / cell-type column names; passed
        to ``fit()`` and used by the signal-mask filter.
    signal_only
        Restrict ROC/PR to signal patches (``ecm_cluster >= 0``).
    groupby
        Optional ``obs`` column name on each cell AnnData holding the
        group label of the ROI (e.g. ``'Mouse'``). The per-ROI group is
        read as the most common value of ``cell_adata.obs[groupby]``.
        Ignored when ``group_map`` is given (which takes precedence).
        When both are ``None``, every ROI is its own group (classic
        leave-one-ROI-out).
    group_map
        Optional explicit ``{roi_name: group}`` mapping. Takes precedence
        over ``groupby``. ROIs sharing a group are held out together, and
        none of them is ever in the training set for that fold.
    verbose
        Print one line per fold with AUCs.

    Returns
    -------
    curves
        Long-format ``DataFrame`` with columns
        ``[fold, kind, x, y]`` ready for
        :func:`mantpy.pl.classifier_roc`. ``kind`` is one of ``'roc'``
        (``x=fpr, y=tpr``) or ``'pr'`` (``x=recall, y=precision``).
    summary
        Per-ROI ``DataFrame`` with columns ``[fold, roc_auc, pr_auc]``,
        plus a ``group`` column when ``groupby`` / ``group_map`` is used.
    scored
        ``{roi_name: ecm_adata}`` of the held-out ECM AnnDatas after
        scoring. The artefact version when ``artefact_pool`` is given,
        otherwise the pristine version. ``uns['ecm_patches']`` carries
        the per-patch ``anomaly_score`` (and the denoised column if
        ``denoise_kwargs`` was given).

    Examples
    --------
    >>> import mantpy as mt  # doctest: +SKIP
    >>> data = mt.fetch.load_balbc_pbs_lung()  # doctest: +SKIP
    >>> curves, summary, scored = mt.tl.loo_denoise_evaluation(  # doctest: +SKIP
    ...     cells_by_sample=data["cells_by_sample"],
    ...     ecm_by_sample=data["cohort_ecm"],
    ...     artefact_pool=data["adatas_ecm_artifact"],
    ...     graph_kwargs={
    ...         "cell_edge_method": "delaunay",
    ...         "ecm_edge_method": "grid",  # 8-connected lattice
    ...         "cell_ecm_edge_method": "delaunay",
    ...     },
    ... )
    """
    if model_factory is None:
        from mantpy.nn import NeighbourCompositionBaseline

        model_factory = NeighbourCompositionBaseline
        model_kwargs = {"context": "cell", **dict(model_kwargs or {})}
    else:
        model_kwargs = dict(model_kwargs or {})
    fit_kwargs = dict(fit_kwargs or {})
    graph_kwargs = dict(graph_kwargs or {})
    fit_kwargs.setdefault("cluster_key", cluster_key)
    fit_kwargs.setdefault("celltype_key", celltype_key)

    rois = list(roi_names) if roi_names is not None else sorted(cells_by_sample)
    if not rois:
        raise ValueError("cells_by_sample is empty; nothing to evaluate.")
    for r in rois:
        if r not in cells_by_sample:
            raise KeyError(f"ROI '{r}' missing from cells_by_sample.")
        if r not in ecm_by_sample:
            raise KeyError(f"ROI '{r}' missing from ecm_by_sample.")
    if artefact_pool is not None:
        for r in rois:
            if r not in artefact_pool:
                raise KeyError(
                    f"ROI '{r}' missing from artefact_pool. Pass artefact_pool=None to skip artefact swap-in."
                )

    # ------------------------------------------------------------------
    # Resolve the per-ROI group label. The default (no groupby/group_map)
    # gives each ROI its own group, so the loop below is identical to the
    # classic leave-one-ROI-out. With a group map, ROIs sharing a group
    # are held out together (leave-one-group-out), preventing same-group
    # leakage into the training fold.
    # ------------------------------------------------------------------
    grouped = (groupby is not None) or (group_map is not None)
    if group_map is not None:
        roi_group = {r: group_map[r] for r in rois if r in group_map}
        missing_grp = [r for r in rois if r not in group_map]
        if missing_grp:
            raise KeyError(
                f"ROIs {missing_grp} missing from group_map. Provide a group for every ROI, or pass group_map=None."
            )
    elif groupby is not None:
        roi_group = {}
        for r in rois:
            obs = cells_by_sample[r].obs
            if groupby not in obs.columns:
                raise KeyError(
                    f"groupby column '{groupby}' not found in "
                    f"cells_by_sample['{r}'].obs. Available columns: "
                    f"{list(obs.columns)}."
                )
            roi_group[r] = obs[groupby].mode().iloc[0]
    else:
        roi_group = {r: r for r in rois}

    # Held-out folds = unique groups, in first-seen order; train on every
    # ROI whose group differs from the held-out group.
    fold_groups: list[Any] = []
    for r in rois:
        if roi_group[r] not in fold_groups:
            fold_groups.append(roi_group[r])

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    scored: dict[str, AnnData] = {}

    model = None
    for fold_grp in fold_groups:
        held_rois = [r for r in rois if roi_group[r] == fold_grp]
        train_rois = [r for r in rois if roi_group[r] != fold_grp]
        if not train_rois:
            raise ValueError(
                f"Held-out group {fold_grp!r} leaves no training ROIs; leave-one-group-out needs >= 2 groups."
            )
        pristine_pairs = [(cells_by_sample[r], ecm_by_sample[r]) for r in train_rois]
        model = model_factory(**model_kwargs)
        model.fit(pristine_pairs, **fit_kwargs)

        for held_out in held_rois:
            _score_one_fold(
                held_out=held_out,
                fold_grp=fold_grp if grouped else None,
                model=model,
                cells_by_sample=cells_by_sample,
                ecm_by_sample=ecm_by_sample,
                artefact_pool=artefact_pool,
                graph_kwargs=graph_kwargs,
                denoise_kwargs=denoise_kwargs,
                truth_col=truth_col,
                signal_only=signal_only,
                curve_rows=curve_rows,
                summary_rows=summary_rows,
                scored=scored,
                verbose=verbose,
            )

    curves = pd.DataFrame(curve_rows)
    summary = pd.DataFrame(summary_rows)
    if not grouped and "group" in summary.columns:
        summary = summary.drop(columns="group")
    if verbose and artefact_pool is not None and len(summary):
        _log.info(
            "\n%s LOO   mean ROC-AUC = %.3f ± %.3f   mean PR-AUC = %.3f ± %.3f",
            type(model).__name__,
            summary.roc_auc.mean(),
            summary.roc_auc.std(),
            summary.pr_auc.mean(),
            summary.pr_auc.std(),
        )

    return curves, summary, scored


def loo_pristine_flag_rate(
    cells_by_sample: dict[str, AnnData],
    ecm_by_sample: dict[str, AnnData],
    *,
    artefact_pool: dict[str, AnnData],
    roi_names: Sequence[str] | None = None,
    model_factory: Any = None,
    model_kwargs: dict[str, Any] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    graph_kwargs: dict[str, Any] | None = None,
    truth_col: str = "is_artifact",
    cluster_key: str = "ecm_cluster",
    celltype_key: str = "cell_type",
    target_recall: float = 0.90,
    signal_only: bool = True,
) -> pd.DataFrame:
    """Estimate over-cleaning on pristine data with held-out models.

    Each fold trains on the other pristine samples. The artefact-injected
    held-out sample sets the anomaly threshold required to recover
    ``target_recall`` of known artefacts; that same threshold is then applied
    to its pristine counterpart. The output contains one pristine flagged
    fraction per sample, making the negative-control calculation a single
    reproducible call rather than tutorial bookkeeping.
    """
    if not 0 < target_recall <= 1:
        raise ValueError(f"target_recall must lie in (0, 1] (got {target_recall}).")
    rois = list(roi_names) if roi_names is not None else sorted(cells_by_sample)
    if len(rois) < 2:
        raise ValueError("Leave-one-out evaluation needs at least two samples.")
    for name in rois:
        if name not in cells_by_sample or name not in ecm_by_sample or name not in artefact_pool:
            raise KeyError(f"Sample {name!r} must occur in cells, ECM, and artefact mappings.")

    if model_factory is None:
        from mantpy.nn import NeighbourCompositionBaseline

        model_factory = NeighbourCompositionBaseline
        model_kwargs = {"context": "cell", **dict(model_kwargs or {})}

    from mantpy.gr import ensure_cell_ecm_graph

    rows: list[dict[str, Any]] = []
    for held_out in rois:
        model = model_factory(**dict(model_kwargs or {}))
        training_kwargs = dict(fit_kwargs or {})
        training_kwargs.setdefault("cluster_key", cluster_key)
        training_kwargs.setdefault("celltype_key", celltype_key)
        model.fit(
            [(cells_by_sample[name], ecm_by_sample[name]) for name in rois if name != held_out],
            **training_kwargs,
        )

        corrupt_cells = cells_by_sample[held_out].copy()
        corrupt_ecm = artefact_pool[held_out].copy()
        corrupt_cells.uns[ECM_PATCHES_KEY] = corrupt_ecm.uns[ECM_PATCHES_KEY].copy()
        graph_recipe = dict(graph_kwargs or {})
        graph_recipe["rebuild"] = True
        ensure_cell_ecm_graph(corrupt_cells, **graph_recipe)
        corrupt_scores = model.score_anomaly(
            corrupt_cells,
            corrupt_ecm,
            truth_col=truth_col,
            signal_only=signal_only,
        )
        y_true = np.asarray(corrupt_scores["y_true"], dtype=bool)
        y_score = np.asarray(corrupt_scores["y_score"], dtype=float)
        positives = y_score[y_true]
        threshold = float(np.quantile(positives, 1.0 - target_recall)) if positives.size else float("inf")

        pristine_cells = cells_by_sample[held_out].copy()
        pristine_ecm = ecm_by_sample[held_out].copy()
        pristine_patches = pristine_ecm.uns[ECM_PATCHES_KEY].copy()
        pristine_patches[truth_col] = False
        pristine_ecm.uns[ECM_PATCHES_KEY] = pristine_patches
        pristine_cells.uns[ECM_PATCHES_KEY] = pristine_patches.copy()
        ensure_cell_ecm_graph(pristine_cells, **graph_recipe)
        pristine_scores = model.score_anomaly(
            pristine_cells,
            pristine_ecm,
            truth_col=truth_col,
            signal_only=signal_only,
        )
        scores = np.asarray(pristine_scores["y_score"], dtype=float)
        rows.append(
            {
                "roi": held_out,
                "threshold": threshold,
                "pristine_flagged": float((scores >= threshold).mean()),
                "n_pristine": int(scores.size),
                "n_positive_artifacts": int(positives.size),
            }
        )
    return pd.DataFrame(rows)


def summarize_pristine_flag_rate(
    results: pd.DataFrame,
    *,
    rate_col: str = "pristine_flagged",
) -> PristineFlagSummary:
    """Return the mean and range from :func:`loo_pristine_flag_rate`."""
    if rate_col not in results:
        raise KeyError(f"Column {rate_col!r} is missing from results.")
    values = results[rate_col].dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"Column {rate_col!r} contains no finite rates.")
    return PristineFlagSummary(
        mean=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        n_samples=int(values.size),
    )


def cross_compartment_ablation(
    cells_by_sample: dict[str, AnnData],
    ecm_by_sample: dict[str, AnnData],
    *,
    artefact_pool: dict[str, AnnData],
    roi_names: Sequence[str] | None = None,
    contexts: Sequence[str] = ("cell", "ecm", "joint"),
    include_prior: bool = True,
    graph_kwargs: dict[str, Any] | None = None,
    groupby: str | None = None,
    group_map: dict[str, Any] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Cross-compartment artefact-detection ablation in a single call.

    Runs the *same* multinomial-logistic-regression model
    (:class:`mantpy.nn.NeighbourCompositionBaseline`) through the identical
    leave-one-ROI-out (or leave-one-group-out) harness, varying ONLY the
    neighbour features it sees, plus an optional internal label-frequency
    reference. Because every model shares the same folds, any AUC gap is
    attributable to the *information content* of the compartment:

    - ``'cell'``  — neighbouring cell-type composition (cell→ECM edges).
      The cross-compartment signal that matrix-only analysis misses.
    - ``'ecm'``   — neighbouring ECM-cluster composition (ECM-ECM edges).
      The logistic-regression analogue of naive spatial smoothing.
    - ``'joint'`` — both concatenated. Kept for the caveat that corrupted
      ECM-neighbour features can mislead the joint model below cell-only.
    - ``'prior'`` — global cluster frequency only (the floor).

    Returns a tidy frame ready for a grouped bar or ROC plot. It differs
    from :func:`loo_denoise_evaluation` (which evaluates *one*
    model and also returns ROC/PR curves + scored AnnDatas) by sweeping the
    compartment axis and returning only the per-fold AUC table.

    Parameters
    ----------
    cells_by_sample, ecm_by_sample
        ``{roi_name: adata}`` maps, as for :func:`loo_denoise_evaluation`.
    artefact_pool
        ``{roi_name: ecm_adata}`` artefact-injected ECMs supplying the
        ``is_artifact`` supervision. Required (the ablation has no signal
        without it).
    roi_names
        Optional explicit fold ordering. Defaults to ``sorted(cells_by_sample)``.
    contexts
        Which :class:`NeighbourCompositionBaseline` contexts to run. Any
        subset / ordering of ``('cell', 'ecm', 'joint')``.
    include_prior
        Also run the internal label-frequency reference (model name ``'prior'``).
    graph_kwargs
        Forwarded to :func:`mantpy.gr.ensure_cell_ecm_graph` on each
        held-out ROI (e.g. the grid-ECM recipe).
    groupby, group_map
        Passed straight through to :func:`loo_denoise_evaluation` to switch
        to leave-one-group-out (e.g. ``groupby='Mouse'`` for
        leave-one-mouse-out). See that function for the semantics.
    verbose
        Forwarded to :func:`loo_denoise_evaluation`.

    Returns
    -------
    DataFrame
        Tidy long frame with columns ``[model, fold, roc_auc, pr_auc]``
        (plus ``group`` when ``groupby`` / ``group_map`` is used). ``model``
        is one of ``'cell'``, ``'ecm'``, ``'joint'``, ``'prior'``.

    Examples
    --------
    >>> import mantpy as mt  # doctest: +SKIP
    >>> data = mt.fetch.load_balbc_pbs_lung()  # doctest: +SKIP
    >>> abl = mt.tl.cross_compartment_ablation(  # doctest: +SKIP
    ...     data["cells_by_sample"],
    ...     data["cohort_ecm"],
    ...     artefact_pool=data["adatas_ecm_artifact"],
    ...     graph_kwargs=dict(
    ...         cell_k=5,
    ...         cell_Dmax=15.0,
    ...         ecm_edge_method="grid",
    ...         ecm_grid_connectivity=8,
    ...         cell_ecm_k=5,
    ...         cell_ecm_Dmax=15.0,
    ...     ),
    ... )
    >>> abl[abl.model == "cell"].roc_auc.mean()  # doctest: +SKIP
    0.96
    >>> # Leave-one-mouse-out in one call:
    >>> lomo = mt.tl.cross_compartment_ablation(  # doctest: +SKIP
    ...     data["cells_by_sample"],
    ...     data["cohort_ecm"],
    ...     artefact_pool=data["adatas_ecm_artifact"],
    ...     groupby="Mouse",
    ... )
    """
    from mantpy.nn import NeighbourCompositionBaseline
    from mantpy.nn._baselines import PriorFrequencyBaseline

    if artefact_pool is None:
        raise ValueError(
            "cross_compartment_ablation needs an artefact_pool (the "
            "is_artifact supervision); there is no signal without it."
        )
    bad = [c for c in contexts if c not in ("cell", "ecm", "joint")]
    if bad:
        raise ValueError(f"contexts {bad} invalid; choose from 'cell', 'ecm', 'joint'.")

    jobs: list[tuple[str, Any, dict[str, Any]]] = [(c, NeighbourCompositionBaseline, {"context": c}) for c in contexts]
    if include_prior:
        jobs.append(("prior", PriorFrequencyBaseline, {}))

    frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for model_name, factory, model_kwargs in jobs:
        if verbose:
            _log.info("=== %s ===", model_name)
        curves, summary, _scored = loo_denoise_evaluation(
            cells_by_sample=cells_by_sample,
            ecm_by_sample=ecm_by_sample,
            artefact_pool=artefact_pool,
            roi_names=roi_names,
            model_factory=factory,
            model_kwargs=model_kwargs,
            fit_kwargs={},
            graph_kwargs=graph_kwargs,
            denoise_kwargs=None,
            groupby=groupby,
            group_map=group_map,
            verbose=verbose,
        )
        summary = summary.copy()
        summary.insert(0, "model", model_name)
        frames.append(summary)
        if not curves.empty:
            curves = curves.copy()
            curves.insert(0, "model", model_name)
            curve_frames.append(curves)

    out = pd.concat(frames, ignore_index=True)
    # Stable column order: model, fold, [group], roc_auc, pr_auc.
    front = ["model", "fold"]
    if "group" in out.columns:
        front.append("group")
    out = out[front + [c for c in out.columns if c not in front]]
    out = out.reset_index(drop=True)
    curve_table = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    out.attrs["roc_curves_records"] = curve_table.to_dict(orient="records")
    return out


def ablation_roc_curves(
    results: pd.DataFrame,
    *,
    model: str = "cell",
) -> AblationROCResult:
    """Extract one model's ROC curves without rerunning an ablation.

    :func:`cross_compartment_ablation` retains the fold-level curves in the
    returned table's metadata. This helper exposes the plotting pair expected
    by :func:`mantpy.pl.classifier_roc`.
    """
    curve_records = results.attrs.get("roc_curves_records")
    if not isinstance(curve_records, list) or not curve_records:
        raise ValueError("results do not contain curves from cross_compartment_ablation.")
    curves = pd.DataFrame.from_records(curve_records)
    if "model" not in results or "model" not in curves:
        raise KeyError("Ablation results and stored curves must contain a 'model' column.")
    if model not in set(results["model"].astype(str)):
        raise ValueError(f"Model {model!r} is not present in the ablation results.")
    model_curves = curves.loc[curves["model"].astype(str).eq(model)].drop(columns="model")
    model_summary = results.loc[results["model"].astype(str).eq(model)].drop(columns="model")
    return AblationROCResult(
        curves=model_curves.reset_index(drop=True),
        summary=model_summary.reset_index(drop=True),
    )


def grouped_metric_summary(
    results: pd.DataFrame,
    *,
    groupby: str | Sequence[str],
    metrics: str | Sequence[str] = "roc_auc",
) -> pd.DataFrame:
    """Summarize tidy repeated-measure metrics by one or more groups.

    Returns one row per group and metric with ``mean``, sample ``std``, and
    non-missing ``n``. This keeps notebook reporting consistent with Mantpy's
    plotting helpers while remaining useful for any model-by-fold result.
    """
    groups = [groupby] if isinstance(groupby, str) else list(groupby)
    metric_columns = [metrics] if isinstance(metrics, str) else list(metrics)
    missing = [column for column in (*groups, *metric_columns) if column not in results]
    if missing:
        raise KeyError(f"Results are missing columns {missing}.")
    long = results.melt(
        id_vars=groups,
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    summary = (
        long.groupby([*groups, "metric"], dropna=False)["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    return summary


def score_partition(
    labels: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    """Score a clustering against a reference labelling by majority-vote matching.

    Maps each predicted cluster to the reference class it overlaps most
    (majority vote), then reports macro-F1, per-class F1, and the adjusted Rand
    index. The mapping makes the score invariant to cluster-id permutation and
    to **over-clustering** — several predicted clusters may map to the same
    reference class, so deliberately splitting a class into sub-clusters is not
    penalised. Macro-F1 weights every reference class equally, so a thin or rare
    class counts as much as a dominant one (unlike ARI, which is pair-dominated
    and rewards merging rare classes).

    Parameters
    ----------
    labels
        Predicted integer cluster labels, shape ``(n_items,)``.
    reference
        Reference labels (e.g. a hand annotation), shape ``(n_items,)``. May be
        any hashable label type; classes are taken in sorted order.

    Returns
    -------
    dict
        ``macro_f1`` (float), ``per_class_f1`` (list of float, one per reference
        class in sorted order), ``ari`` (float), and ``classes`` (the reference
        class labels, sorted, matching ``per_class_f1``).

    Examples
    --------
    >>> import numpy as np
    >>> truth = np.array([0, 0, 1, 1, 2, 2])
    >>> pred = np.array([3, 3, 5, 5, 9, 9])  # permuted + arbitrary ids
    >>> mt.tl.score_partition(pred, truth)["macro_f1"]  # doctest: +SKIP
    1.0

    """
    from sklearn.metrics import adjusted_rand_score, f1_score

    pred = np.asarray(labels).ravel()
    ref = np.asarray(reference).ravel()
    if pred.shape[0] != ref.shape[0]:
        raise ValueError(f"labels has length {pred.shape[0]} but reference has length {ref.shape[0]}.")
    classes, truth_idx = np.unique(ref, return_inverse=True)
    k = len(classes)
    mapped = np.empty(pred.shape[0], dtype=np.int64)
    for c in np.unique(pred):
        m = pred == c
        mapped[m] = np.bincount(truth_idx[m], minlength=k).argmax()
    per_class = f1_score(truth_idx, mapped, labels=list(range(k)), average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(per_class)),
        "per_class_f1": [float(x) for x in per_class],
        "ari": float(adjusted_rand_score(truth_idx, pred)),
        "classes": classes.tolist(),
    }


def _knn_adjacency(coords: np.ndarray, k: int) -> sp.csr_matrix:
    """Symmetric binary k-nearest-neighbour adjacency over points (CSR)."""
    from sklearn.neighbors import NearestNeighbors

    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"coords must be 2-D (n_points, n_dims); got shape {pts.shape}.")
    n = pts.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k={k!r}: expected 1 <= k < n_points ({n}).")
    idx = NearestNeighbors(n_neighbors=k + 1).fit(pts).kneighbors(pts, return_distance=False)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1 : k + 1].ravel()
    w = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    return w.maximum(w.T)


def graph_modularity(
    labels: np.ndarray,
    coords: np.ndarray | None = None,
    *,
    adjacency: Any | None = None,
    k: int = 6,
) -> float:
    """Newman modularity of a partition on a spatial graph, clipped to ``[0, 1]``.

    Modularity is the fraction of within-cluster edges minus its expectation under a
    degree-preserving null. On a spatial graph it is high when clusters are spatially
    compact and low when they are interleaved, so it is a label-free quality signal for
    a spatial-domain partition. Because of the modularity *resolution limit*
    (Fortunato & Barthélemy, 2007) it tends to favour **over-segmentation** — pair it with
    :func:`cluster_coherence` (which favours under-segmentation) for a balanced selector
    (see :func:`select_n_domains`).

    Parameters
    ----------
    labels
        Integer cluster labels, shape ``(n_nodes,)``.
    coords
        Node coordinates, shape ``(n_nodes, n_dims)``. Used to build a symmetric ``k``-NN
        graph when ``adjacency`` is not given.
    adjacency
        Precomputed symmetric adjacency (``scipy.sparse`` matrix or dense array), shape
        ``(n_nodes, n_nodes)``. Takes precedence over ``coords``.
    k
        Number of neighbours when building the graph from ``coords``.

    Returns
    -------
    float
        Modularity in ``[0, 1]`` (negative values are clipped to 0).

    See Also
    --------
    cluster_coherence : the complementary under-segmentation signal.
    select_n_domains : balances the two to pick the number of domains.
    """
    lab = np.asarray(labels).ravel()
    if adjacency is None:
        if coords is None:
            raise ValueError("provide either coords (to build a k-NN graph) or a precomputed adjacency.")
        adjacency = _knn_adjacency(coords, k)
    a_csr = sp.csr_matrix(adjacency)
    up = sp.triu(a_csr, 1).tocoo()
    m = int(up.nnz)
    if m == 0:
        return 0.0
    deg = np.asarray(a_csr.sum(1)).ravel()
    nk = int(lab.max()) + 1
    within = float((lab[up.row] == lab[up.col]).sum()) / m
    frac = np.bincount(lab, weights=deg, minlength=nk) / (2.0 * m)
    return float(np.clip(within - np.sum(frac * frac), 0.0, 1.0))


def cluster_coherence(labels: np.ndarray, adjacency: Any) -> dict[str, float]:
    """Spatial coherence of a partition: per-cluster largest-connected-component fraction.

    For each cluster, the fraction of its members that lie in the **largest connected
    component** of the subgraph ``adjacency`` induces on them. A cluster split across
    disconnected regions of the tissue lowers its fraction. The minimum across clusters is
    flat at 1.0 while every cluster is contiguous and collapses once a cluster fragments,
    so it favours **under-segmentation** — the complement of :func:`graph_modularity`.

    Parameters
    ----------
    labels
        Integer cluster labels, shape ``(n_nodes,)``.
    adjacency
        Spatial adjacency over the same nodes (``scipy.sparse`` matrix or dense array),
        e.g. a foreground lattice or a k-NN graph.

    Returns
    -------
    dict
        ``mean`` and ``min`` of the per-cluster largest-component fraction.

    See Also
    --------
    graph_modularity : the complementary over-segmentation signal.
    select_n_domains : balances the two to pick the number of domains.
    """
    from scipy.sparse.csgraph import connected_components

    lab = np.asarray(labels).ravel()
    a_csr = sp.csr_matrix(adjacency)
    fr = []
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        if idx.size == 0:
            continue
        _, comp = connected_components(a_csr[idx][:, idx], directed=False)
        fr.append(np.bincount(comp).max() / idx.size)
    fr = np.asarray(fr, dtype=float)
    return {"mean": float(fr.mean()), "min": float(fr.min())}


def select_n_domains(
    embedding: np.ndarray,
    coords: np.ndarray,
    *,
    adjacency: Any | None = None,
    k_range: Sequence[int] = range(2, 9),
    k_modularity: int = 6,
    random_state: int = 0,
) -> dict[str, Any]:
    """Choose the number of spatial domains label-free by balancing modularity and coherence.

    Over-segmentation inflates :func:`graph_modularity` (the resolution limit keeps
    rewarding finer splits) while under-segmentation inflates the minimum
    :func:`cluster_coherence` (one contiguous blob is maximally coherent). Standardising
    each across the candidate set and summing them, ``z(Q) + z(coh_min)``, cancels the two
    opposing biases, so the score peaks near the true number of domains. For each ``K`` in
    ``k_range`` the embedding is KMeans-clustered, modularity is scored on a ``k``-NN graph
    over ``coords`` and coherence-min on ``adjacency`` (defaulting to that same graph), and
    the ``K`` maximising the standardised sum is returned.

    Parameters
    ----------
    embedding
        Node feature/embedding matrix to cluster, shape ``(n_nodes, n_features)``.
    coords
        Node coordinates, shape ``(n_nodes, n_dims)`` — the spatial graph for modularity.
    adjacency
        Optional spatial adjacency for the coherence term (e.g. a foreground lattice).
        Defaults to the ``k``-NN graph built from ``coords``.
    k_range
        Candidate cluster counts to score.
    k_modularity
        Neighbours for the modularity k-NN graph over ``coords``.
    random_state
        Seed for KMeans (``n_init=10``).

    Returns
    -------
    dict
        ``labels`` (the clustering at the selected K), ``n_domains`` (the selected K), and
        ``table`` (a :class:`pandas.DataFrame` with columns ``n_domains``, ``modularity``,
        ``coherence_min``, ``score`` for every candidate K).

    See Also
    --------
    graph_modularity, cluster_coherence : the two terms being balanced.
    """
    from sklearn.cluster import KMeans

    ks = list(k_range)
    if not ks:
        raise ValueError("k_range is empty; provide at least one candidate cluster count.")
    coord_adj = _knn_adjacency(coords, k_modularity)
    coh_adj = coord_adj if adjacency is None else sp.csr_matrix(adjacency)
    labs, rows = {}, []
    for kk in ks:
        lab = KMeans(kk, n_init=10, random_state=random_state).fit_predict(embedding)
        labs[kk] = lab
        rows.append(
            {
                "n_domains": kk,
                "modularity": graph_modularity(lab, adjacency=coord_adj),
                "coherence_min": cluster_coherence(lab, coh_adj)["min"],
            }
        )
    table = pd.DataFrame(rows)

    def _z(v: np.ndarray) -> np.ndarray:
        return (v - v.mean()) / (v.std() + 1e-9)

    table["score"] = _z(table["modularity"].to_numpy()) + _z(table["coherence_min"].to_numpy())
    best = int(table.loc[table["score"].idxmax(), "n_domains"])
    return {"labels": labs[best], "n_domains": best, "table": table}


# ---------------------------------------------------------------------------
# Sparse joint-graph and cross-modal spatial analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellECMContactResult:
    """Summary returned by :func:`cell_ecm_contact`."""

    adata: AnnData
    degree_key: str
    contact_key: str
    n_cells: int
    n_contacted: int
    threshold: int

    @property
    def fraction_contacted(self) -> float:
        """Fraction of cell nodes with cross-degree above ``threshold``."""
        return self.n_contacted / self.n_cells if self.n_cells else float("nan")

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Cell--ECM contact",
                f"  cells        {self.n_cells:,}",
                f"  contacted    {self.n_contacted:,} ({self.fraction_contacted:.1%})",
                f"  stored       obs[{self.degree_key!r}], obs[{self.contact_key!r}]",
            ]
        )


@dataclass(frozen=True)
class GraphSmoothingResult:
    """Summary returned by :func:`smooth_graph_signal`."""

    adata: AnnData
    key_added: str
    storage: Literal["obs", "obsm"]
    feature_names: tuple[str, ...]
    graph_key: str
    n_iter: int
    alpha: float
    n_finite: tuple[int, ...]

    def __repr__(self) -> str:
        location = f"{self.storage}[{self.key_added!r}]"
        finite = self.n_finite[0] if len(self.n_finite) == 1 else min(self.n_finite)
        suffix = "" if len(self.n_finite) == 1 else " minimum per feature"
        return "\n".join(
            [
                "Graph-smoothed signal",
                f"  graph        obsp[{self.graph_key!r}]",
                f"  signals      {len(self.feature_names):,}",
                f"  finite       {finite:,}{suffix}",
                f"  iterations   {self.n_iter} (alpha={self.alpha:g})",
                f"  stored       {location}",
            ]
        )


@dataclass(frozen=True)
class SpatialTransferResult:
    """Summary returned by :func:`transfer_spatial_features`."""

    target: AnnData
    key_added: str
    storage: Literal["obs", "obsm"]
    feature_names: tuple[str, ...]
    method: Literal["radius", "nearest"]
    n_targets: int
    n_covered: int
    coverage_key: str
    count_key: str

    @property
    def fraction_covered(self) -> float:
        """Fraction of target observations receiving at least one finite value."""
        return self.n_covered / self.n_targets if self.n_targets else float("nan")

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Spatial feature transfer",
                f"  method       {self.method}",
                f"  features     {len(self.feature_names):,}",
                f"  covered      {self.n_covered:,} / {self.n_targets:,} ({self.fraction_covered:.1%})",
                f"  stored       {self.storage}[{self.key_added!r}]",
            ]
        )


def _square_connectivity(adata: AnnData, graph_key: str) -> sp.csr_matrix:
    if graph_key not in adata.obsp:
        raise KeyError(f"adata.obsp[{graph_key!r}] is missing; build or select the graph first.")
    adjacency = sp.csr_matrix(adata.obsp[graph_key], dtype=float)
    expected = (adata.n_obs, adata.n_obs)
    if adjacency.shape != expected:
        raise ValueError(f"adata.obsp[{graph_key!r}] has shape {adjacency.shape}; expected {expected}.")
    if adjacency.data.size and not np.all(np.isfinite(adjacency.data)):
        raise ValueError(f"adata.obsp[{graph_key!r}] contains non-finite edge weights.")
    if adjacency.data.size and np.any(adjacency.data < 0):
        raise ValueError(f"adata.obsp[{graph_key!r}] contains negative edge weights.")
    adjacency.eliminate_zeros()
    return adjacency


def _node_type_mask(
    adata: AnnData,
    node_type: str | Sequence[str] | None,
    *,
    node_type_key: str,
) -> np.ndarray:
    if node_type is None:
        return np.ones(adata.n_obs, dtype=bool)
    if node_type_key not in adata.obs:
        raise KeyError(f"adata.obs[{node_type_key!r}] is missing; node-type restriction cannot be applied.")
    labels = (node_type,) if isinstance(node_type, str) else tuple(node_type)
    if not labels:
        raise ValueError("node_type must contain at least one label.")
    return np.asarray(adata.obs[node_type_key].isin(labels), dtype=bool)


def cell_ecm_contact(
    adata: AnnData,
    *,
    graph_key: str = "cell_ecm_connectivities",
    node_type_key: str = "node_type",
    cell_label: str = NODE_TYPE_CELL,
    ecm_label: str = NODE_TYPE_ECM,
    degree_key: str = "ecm_degree",
    contact_key: str = "ecm_contact",
    threshold: int = 0,
    inplace: bool = True,
) -> CellECMContactResult:
    """Derive cell-to-ECM cross-degree and binary contact from a sparse joint graph.

    The function consumes the typed, observation-native joint representation
    produced by :func:`mantpy.gr.compose_cell_ecm_graph`. Only edges from cell
    rows to ECM columns are counted, so the result is unchanged whether the
    stored cross-compartment matrix contains one or both orientations. Values
    on ECM observations are stored as ``NaN`` because cell contact is not
    defined for those nodes.

    Parameters
    ----------
    adata
        Joint AnnData with node roles in ``obs[node_type_key]`` and a square
        sparse cross-compartment adjacency in ``obsp[graph_key]``.
    graph_key
        Cross-compartment connectivity matrix in ``adata.obsp``.
    node_type_key
        Observation column containing the cell and ECM role labels.
    cell_label, ecm_label
        Values identifying cell and ECM nodes.
    degree_key, contact_key
        Output columns in ``adata.obs``. Contact is one when cross-degree is
        strictly greater than ``threshold`` and zero otherwise.
    threshold
        Minimum cross-degree excluded from binary contact. The default of zero
        means that any cell--ECM edge defines contact.
    inplace
        Modify ``adata`` or an independent copy. The modified object is always
        available as ``result.adata``.
    """
    if not isinstance(threshold, int | np.integer) or threshold < 0:
        raise ValueError("threshold must be a non-negative integer.")
    if node_type_key not in adata.obs:
        raise KeyError(f"adata.obs[{node_type_key!r}] is missing; expected typed cell and ECM nodes.")
    target = adata if inplace else adata.copy()
    adjacency = _square_connectivity(target, graph_key)
    roles = np.asarray(target.obs[node_type_key].astype(str))
    cell_mask = roles == str(cell_label)
    ecm_mask = roles == str(ecm_label)
    if not np.any(cell_mask):
        raise ValueError(f"adata.obs[{node_type_key!r}] contains no {cell_label!r} nodes.")
    if not np.any(ecm_mask):
        raise ValueError(f"adata.obs[{node_type_key!r}] contains no {ecm_label!r} nodes.")

    # Count distinct non-zero cell-to-ECM entries, not edge weights.
    cross = adjacency[cell_mask][:, ecm_mask]
    degree = np.asarray(cross.getnnz(axis=1), dtype=np.int64)
    contact = degree > int(threshold)
    degree_all = np.full(target.n_obs, np.nan, dtype=np.float32)
    contact_all = np.full(target.n_obs, np.nan, dtype=np.float32)
    degree_all[cell_mask] = degree
    contact_all[cell_mask] = contact.astype(np.float32)
    target.obs[degree_key] = degree_all
    target.obs[contact_key] = contact_all

    provenance = {
        "graph_key": graph_key,
        "node_type_key": node_type_key,
        "cell_label": str(cell_label),
        "ecm_label": str(ecm_label),
        "degree_key": degree_key,
        "contact_key": contact_key,
        "threshold": int(threshold),
        "n_cells": int(cell_mask.sum()),
        "n_contacted": int(contact.sum()),
    }
    target.uns[f"{contact_key}_params"] = provenance
    _log_params(target, "tl", {"cell_ecm_contact": provenance})
    return CellECMContactResult(
        adata=target,
        degree_key=degree_key,
        contact_key=contact_key,
        n_cells=int(cell_mask.sum()),
        n_contacted=int(contact.sum()),
        threshold=int(threshold),
    )


def _resolve_adata_values(
    adata: AnnData,
    values: str | Sequence[str] | np.ndarray,
    *,
    source: Literal["auto", "obs", "var"],
) -> tuple[np.ndarray, tuple[str, ...], str]:
    if isinstance(values, np.ndarray):
        matrix = np.asarray(values)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2 or matrix.shape[0] != adata.n_obs:
            raise ValueError(
                f"values array must have shape (n_obs,) or (n_obs, n_features); got {matrix.shape}."
            )
        names = tuple(f"input_{index}" for index in range(matrix.shape[1]))
        return np.asarray(matrix, dtype=float), names, "array"

    names = (values,) if isinstance(values, str) else tuple(values)
    if not names:
        raise ValueError("values must name at least one obs column or var.")
    if len(set(names)) != len(names):
        raise ValueError("values contains duplicate names.")
    invalid = [name for name in names if not isinstance(name, str) or not name]
    if invalid:
        raise TypeError("all values names must be non-empty strings.")

    in_obs = [name in adata.obs.columns for name in names]
    in_var = [name in adata.var_names for name in names]
    if source == "auto":
        ambiguous = [name for name, obs, var in zip(names, in_obs, in_var, strict=True) if obs and var]
        if ambiguous:
            raise ValueError(
                f"Names occur in both obs and var: {ambiguous}. Pass source='obs' or source='var' explicitly."
            )
        resolved = []
        for name, obs, var in zip(names, in_obs, in_var, strict=True):
            if obs:
                resolved.append("obs")
            elif var:
                resolved.append("var")
            else:
                raise KeyError(f"{name!r} is absent from both adata.obs and adata.var_names.")
        source_label = "mixed" if len(set(resolved)) > 1 else resolved[0]
    elif source == "obs":
        missing = [name for name, present in zip(names, in_obs, strict=True) if not present]
        if missing:
            raise KeyError(f"Observation columns are missing: {missing}.")
        resolved = ["obs"] * len(names)
        source_label = "obs"
    elif source == "var":
        missing = [name for name, present in zip(names, in_var, strict=True) if not present]
        if missing:
            raise KeyError(f"Variables are missing: {missing}.")
        resolved = ["var"] * len(names)
        source_label = "var"
    else:
        raise ValueError("source must be 'auto', 'obs', or 'var'.")

    columns: list[np.ndarray] = []
    for name, location in zip(names, resolved, strict=True):
        if location == "obs":
            try:
                column = np.asarray(adata.obs[name], dtype=float)
            except (TypeError, ValueError) as error:
                raise TypeError(f"adata.obs[{name!r}] must be numeric.") from error
        else:
            if not adata.var_names.is_unique:
                raise ValueError("adata.var_names must be unique when resolving values from X.")
            index = int(adata.var_names.get_loc(name))
            column_data = adata.X[:, index]
            column = column_data.toarray().ravel() if sp.issparse(column_data) else np.asarray(column_data).ravel()
            try:
                column = column.astype(float, copy=False)
            except (TypeError, ValueError) as error:
                raise TypeError(f"adata[:, {name!r}].X must be numeric.") from error
        columns.append(column)
    return np.column_stack(columns), names, source_label


def smooth_graph_signal(
    adata: AnnData,
    values: str | Sequence[str] | np.ndarray,
    *,
    graph_key: str = "joint_connectivities",
    source: Literal["auto", "obs", "var"] = "auto",
    key_added: str | None = None,
    alpha: float = 0.75,
    n_iter: int = 12,
    node_type: str | Sequence[str] | None = None,
    node_type_key: str = "node_type",
    inplace: bool = True,
) -> GraphSmoothingResult:
    """Smooth one or many signals over a sparse graph without crossing missing support.

    At each iteration Mantpy computes the edge-weighted mean of finite
    neighbours and updates ``x <- (1-alpha) * x + alpha * neighbour_mean``.
    The finite support is fixed from the input: missing observations remain
    ``NaN`` and do not contribute to any neighbour average. This makes the
    operation safe for spatially transferred measurements whose capture
    coverage does not span every graph node.

    ``values`` can name numeric columns in ``obs``, variables in ``var_names``,
    a mixture of the two when names are unambiguous, or provide an aligned
    array. A scalar output is stored in ``obs``; multiple signals are stored in
    ``obsm`` together with their ordered names in ``uns[f'{key_added}_params']``.
    Set ``node_type='cell'`` to keep both propagation and output restricted to
    cell nodes even when a joint adjacency is supplied.
    """
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and between 0 and 1.")
    if not isinstance(n_iter, int | np.integer) or n_iter < 0:
        raise ValueError("n_iter must be a non-negative integer.")
    target = adata if inplace else adata.copy()
    adjacency = _square_connectivity(target, graph_key)
    matrix, feature_names, source_label = _resolve_adata_values(target, values, source=source)
    selected = _node_type_mask(target, node_type, node_type_key=node_type_key)

    initial_finite = np.isfinite(matrix) & selected[:, None]
    smoothed = np.where(initial_finite, matrix, np.nan).astype(float, copy=False)
    # Remove all edges incident to an excluded node once, then normalise per
    # feature at every iteration because missing support can differ by feature.
    adjacency = adjacency.multiply(selected[:, None]).multiply(selected[None, :]).tocsr()
    for _ in range(int(n_iter)):
        finite = initial_finite
        neighbour_sum = adjacency @ np.where(finite, smoothed, 0.0)
        neighbour_weight = adjacency @ finite.astype(float)
        neighbour_mean = np.divide(
            neighbour_sum,
            neighbour_weight,
            out=np.zeros_like(neighbour_sum, dtype=float),
            where=neighbour_weight > 0,
        )
        update = initial_finite & (neighbour_weight > 0)
        smoothed[update] = (1.0 - float(alpha)) * smoothed[update] + float(alpha) * neighbour_mean[update]
        smoothed[~initial_finite] = np.nan

    output_key = key_added or (f"{feature_names[0]}_smoothed" if len(feature_names) == 1 else "X_smoothed")
    if not isinstance(output_key, str) or not output_key:
        raise ValueError("key_added must be a non-empty string.")
    if len(feature_names) == 1:
        target.obs[output_key] = smoothed[:, 0].astype(np.float32)
        storage: Literal["obs", "obsm"] = "obs"
    else:
        target.obsm[output_key] = smoothed.astype(np.float32)
        storage = "obsm"

    provenance = {
        "graph_key": graph_key,
        "input_source": source_label,
        "feature_names": list(feature_names),
        "key_added": output_key,
        "storage": storage,
        "alpha": float(alpha),
        "n_iter": int(n_iter),
        "node_type_key": node_type_key,
        "node_types": None
        if node_type is None
        else [node_type]
        if isinstance(node_type, str)
        else list(node_type),
        "finite_counts": [int(value) for value in initial_finite.sum(axis=0)],
        "missing_policy": "fixed support; missing values remain NaN",
    }
    target.uns[f"{output_key}_params"] = provenance
    _log_params(target, "tl", {"smooth_graph_signal": provenance})
    return GraphSmoothingResult(
        adata=target,
        key_added=output_key,
        storage=storage,
        feature_names=feature_names,
        graph_key=graph_key,
        n_iter=int(n_iter),
        alpha=float(alpha),
        n_finite=tuple(int(value) for value in initial_finite.sum(axis=0)),
    )


def _spatial_coordinates(adata: AnnData, key: str) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"adata.obsm[{key!r}] is missing.")
    coords = np.asarray(adata.obsm[key], dtype=float)
    if coords.ndim != 2 or coords.shape[0] != adata.n_obs or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm[{key!r}] must have shape (n_obs, n_dimensions>=2); got {coords.shape}.")
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"adata.obsm[{key!r}] contains non-finite coordinates.")
    return coords


def _spatial_source_mask(adata: AnnData, source_mask: str | np.ndarray | Sequence[bool] | None) -> np.ndarray:
    if source_mask is None:
        return np.ones(adata.n_obs, dtype=bool)
    if isinstance(source_mask, str):
        if source_mask not in adata.obs:
            raise KeyError(f"adata.obs[{source_mask!r}] is missing.")
        raw = np.asarray(adata.obs[source_mask])
    else:
        raw = np.asarray(source_mask)
    if raw.ndim != 1 or raw.shape[0] != adata.n_obs:
        raise ValueError(f"source_mask must have shape ({adata.n_obs},); got {raw.shape}.")
    if pd.isna(raw).any():
        raise ValueError("source_mask contains missing values.")
    return raw.astype(bool, copy=False)


def transfer_spatial_features(
    source_adata: AnnData,
    target_adata: AnnData,
    keys: str | Sequence[str],
    *,
    method: Literal["radius", "nearest"] = "radius",
    radius: float | None = None,
    max_distance: float | None = None,
    aggregation: Literal["mean", "sum", "max"] = "mean",
    source: Literal["auto", "obs", "var"] = "auto",
    source_spatial_key: str = SPATIAL_KEY,
    target_spatial_key: str = SPATIAL_KEY,
    source_mask: str | np.ndarray | Sequence[bool] | None = None,
    key_added: str | None = None,
    inplace: bool = True,
) -> SpatialTransferResult:
    """Transfer observation or gene features between aligned spatial objects.

    ``method='radius'`` aggregates all selected source observations within
    ``radius`` of each target. ``method='nearest'`` uses one nearest source and
    can reject it with ``max_distance``. Unsupported targets remain ``NaN``;
    Mantpy records a coverage flag, contributing-source count, ordered feature
    names, coordinate keys, mask, distance rule and aggregation alongside the
    output.

    Parameters
    ----------
    source_adata, target_adata
        Spatial AnnData objects in the same physical coordinate system.
    keys
        Numeric ``source_adata.obs`` columns and/or ``var_names`` to transfer.
    method
        Radius aggregation or nearest-neighbour transfer.
    radius
        Positive search radius, required for ``method='radius'``.
    max_distance
        Optional positive distance cap for nearest-neighbour transfer.
    aggregation
        Feature-wise finite-value aggregation for radius transfer.
    source
        Resolve all names from ``obs`` or ``var`` explicitly, or use ``auto``
        to resolve each unambiguous name independently.
    source_spatial_key, target_spatial_key
        Coordinate arrays in the respective ``obsm`` mappings.
    source_mask
        Optional boolean array or source ``obs`` column selecting eligible
        observations.
    key_added
        Scalar output ``obs`` column or multi-feature ``obsm`` key.
    inplace
        Modify ``target_adata`` or an independent copy. The modified target is
        always available as ``result.target``.
    """
    if method not in {"radius", "nearest"}:
        raise ValueError("method must be 'radius' or 'nearest'.")
    if aggregation not in {"mean", "sum", "max"}:
        raise ValueError("aggregation must be 'mean', 'sum', or 'max'.")
    if method == "radius":
        if radius is None or not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be a positive finite value for method='radius'.")
        if max_distance is not None:
            raise ValueError("max_distance applies only to method='nearest'; use radius for a radius transfer.")
    elif radius is not None:
        raise ValueError("radius applies only to method='radius'.")
    elif aggregation != "mean":
        raise ValueError("aggregation applies only to method='radius'; nearest transfer selects one value.")
    if max_distance is not None and (not np.isfinite(max_distance) or max_distance <= 0):
        raise ValueError("max_distance must be a positive finite value.")

    target = target_adata if inplace else target_adata.copy()
    source_values, feature_names, source_label = _resolve_adata_values(source_adata, keys, source=source)
    source_coords = _spatial_coordinates(source_adata, source_spatial_key)
    target_coords = _spatial_coordinates(target, target_spatial_key)
    if source_coords.shape[1] != target_coords.shape[1]:
        raise ValueError(
            "Source and target coordinates must have the same dimensionality; "
            f"got {source_coords.shape[1]} and {target_coords.shape[1]}."
        )
    eligible = _spatial_source_mask(source_adata, source_mask)
    if not np.any(eligible):
        raise ValueError("source_mask selects no source observations.")
    source_coords = source_coords[eligible]
    source_values = source_values[eligible]

    from scipy.spatial import cKDTree

    tree = cKDTree(source_coords)
    output = np.full((target.n_obs, len(feature_names)), np.nan, dtype=float)
    n_sources = np.zeros(target.n_obs, dtype=np.int64)
    if method == "nearest":
        distance, index = tree.query(target_coords, k=1)
        supported = np.ones(target.n_obs, dtype=bool)
        if max_distance is not None:
            supported &= distance <= float(max_distance)
        output[supported] = source_values[index[supported]]
        n_sources[supported] = 1
    else:
        neighbours = tree.query_ball_point(target_coords, r=float(radius))
        lengths = np.fromiter((len(indices) for indices in neighbours), dtype=np.int64, count=target.n_obs)
        n_sources[:] = lengths
        nonempty = np.flatnonzero(lengths)
        if nonempty.size:
            row_index = np.repeat(nonempty, lengths[nonempty])
            col_index = np.concatenate([np.asarray(neighbours[index], dtype=np.int64) for index in nonempty])
            weights = sp.csr_matrix(
                (np.ones(row_index.size, dtype=float), (row_index, col_index)),
                shape=(target.n_obs, source_coords.shape[0]),
            )
            finite = np.isfinite(source_values)
            if aggregation in {"mean", "sum"}:
                totals = weights @ np.where(finite, source_values, 0.0)
                counts = weights @ finite.astype(float)
                if aggregation == "mean":
                    output = np.divide(
                        totals,
                        counts,
                        out=np.full_like(totals, np.nan, dtype=float),
                        where=counts > 0,
                    )
                else:
                    output = np.where(counts > 0, totals, np.nan)
            else:
                for target_index in nonempty:
                    block = source_values[np.asarray(neighbours[target_index], dtype=np.int64)]
                    finite_any = np.any(np.isfinite(block), axis=0)
                    if np.any(finite_any):
                        with np.errstate(all="ignore"):
                            maxima = np.nanmax(block[:, finite_any], axis=0)
                        output[target_index, finite_any] = maxima

    output_key = key_added or (feature_names[0] if len(feature_names) == 1 else "X_transferred")
    if not isinstance(output_key, str) or not output_key:
        raise ValueError("key_added must be a non-empty string.")
    if len(feature_names) == 1:
        target.obs[output_key] = output[:, 0].astype(np.float32)
        storage: Literal["obs", "obsm"] = "obs"
    else:
        target.obsm[output_key] = output.astype(np.float32)
        storage = "obsm"
    coverage = np.any(np.isfinite(output), axis=1)
    coverage_key = f"{output_key}_covered"
    count_key = f"{output_key}_n_sources"
    target.obs[coverage_key] = coverage
    target.obs[count_key] = n_sources

    mask_label = source_mask if isinstance(source_mask, str) else "array" if source_mask is not None else None
    provenance = {
        "method": method,
        "radius": None if radius is None else float(radius),
        "max_distance": None if max_distance is None else float(max_distance),
        "aggregation": aggregation,
        "input_source": source_label,
        "feature_names": list(feature_names),
        "source_spatial_key": source_spatial_key,
        "target_spatial_key": target_spatial_key,
        "source_mask": mask_label,
        "n_source_obs": int(source_adata.n_obs),
        "n_eligible_source_obs": int(eligible.sum()),
        "n_target_obs": int(target.n_obs),
        "n_covered": int(coverage.sum()),
        "fraction_covered": float(coverage.mean()) if target.n_obs else float("nan"),
        "coverage_key": coverage_key,
        "count_key": count_key,
        "key_added": output_key,
        "storage": storage,
        "unsupported_policy": "NaN",
    }
    target.uns[f"{output_key}_params"] = provenance
    _log_params(target, "tl", {"transfer_spatial_features": provenance})
    return SpatialTransferResult(
        target=target,
        key_added=output_key,
        storage=storage,
        feature_names=feature_names,
        method=method,
        n_targets=int(target.n_obs),
        n_covered=int(coverage.sum()),
        coverage_key=coverage_key,
        count_key=count_key,
    )
