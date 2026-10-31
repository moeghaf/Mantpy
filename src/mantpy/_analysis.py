"""Small, array-based analysis contracts shared by the public tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _categories(values, categories=None) -> pd.Index:
    if categories is None:
        values = pd.Series(list(values), dtype=object).dropna()
        categories = list(pd.unique(values))
        if all(isinstance(value, (int, float, np.number)) for value in categories):
            categories = sorted(categories)
    index = pd.Index(list(categories), name="cluster", tupleize_cols=False)
    if not index.is_unique or index.hasnans:
        raise ValueError("categories must contain unique, non-missing labels.")
    return index


def _ids(values, size: int, name: str) -> pd.Index:
    index = pd.RangeIndex(size) if values is None else pd.Index(values)
    if len(index) != size or not index.is_unique or index.hasnans:
        raise ValueError(f"{name} must contain {size} unique, non-missing IDs.")
    return index


def _aligned(values, ids: pd.Index, name: str):
    if isinstance(values, (pd.Series, pd.DataFrame)):
        if not values.index.is_unique or not ids.isin(values.index).all():
            raise ValueError(f"{name} must contain every requested ID exactly once.")
        return values.reindex(ids)
    array = np.asarray(values)
    if len(array) != len(ids):
        raise ValueError(f"{name} has {len(array)} rows; expected {len(ids)}.")
    return array


def _nonnegative_matrix(matrix, name: str) -> sp.csr_matrix:
    matrix = sp.csr_matrix(matrix, dtype=float, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.data.size and (
        not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
    ):
        raise ValueError(f"{name} must have finite, non-negative weights.")
    return matrix


def _metadata(samples, sample_meta, unit_col=None) -> pd.DataFrame:
    if sample_meta is None:
        sample_meta = pd.DataFrame(index=pd.Index(samples, name="sample_id"))
    if not sample_meta.index.is_unique:
        raise ValueError("sample_meta must have unique sample IDs.")
    if not sample_meta.index.isin(samples).all():
        raise KeyError("sample_meta contains samples absent from adatas.")
    if unit_col is not None:
        if unit_col not in sample_meta:
            raise KeyError(f"sample_meta[{unit_col!r}] is missing.")
        if sample_meta[unit_col].isna().any():
            raise ValueError("Biological unit labels must not be missing.")
    return sample_meta


@dataclass(frozen=True)
class ClusterCompositionDetails:
    """Composition tables; group means weight biological units equally."""

    per_sample: pd.DataFrame
    per_group_mean: pd.DataFrame
    counts: pd.DataFrame
    per_unit: pd.DataFrame
    per_unit_counts: pd.DataFrame
    unit_meta: pd.DataFrame
    unit_aggregation: str

    @property
    def cluster_cols(self) -> list:
        return self.per_sample.columns.tolist()


def composition_details(
    adatas, *, sample_meta, group_col, cluster_col, patches_key,
    drop_background, group_order, unit_col, unit_aggregation,
    categories, background_labels, label_source,
) -> ClusterCompositionDetails:
    meta = _metadata(list(adatas), sample_meta, unit_col)
    if group_col not in meta:
        raise KeyError(f"sample_meta[{group_col!r}] is missing.")
    if unit_aggregation not in {"pool_counts", "mean_roi_fraction"}:
        raise ValueError("unit_aggregation must be 'pool_counts' or 'mean_roi_fraction'.")
    if label_source not in {"auto", "obs", "patches"}:
        raise ValueError("label_source must be 'auto', 'obs' or 'patches'.")
    labels = {}
    for sample in meta.index:
        adata = adatas[sample]
        patches = adata.uns.get(patches_key)
        if label_source == "patches" or (
            label_source == "auto" and patches is not None and cluster_col in patches
        ):
            labels[sample] = pd.Series(patches[cluster_col]).dropna()
        elif cluster_col in adata.obs:
            labels[sample] = adata.obs[cluster_col].dropna()
        else:
            raise KeyError(f"Sample {sample!r} has no {cluster_col!r} labels.")
    cats = _categories(
        [label for series in labels.values() for label in pd.unique(series)], categories,
    )
    if drop_background:
        if background_labels is None:
            keep = [
                not (isinstance(c, (int, float, np.number)) and c < 0)
                for c in cats
            ]
        else:
            keep = ~cats.isin(background_labels)
        cats = cats[keep]
    counts = pd.DataFrame(0, index=meta.index, columns=cats, dtype=np.int64)
    for sample, series in labels.items():
        counts.loc[sample] = series.value_counts().reindex(cats, fill_value=0).values
    fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    units = pd.Series(meta.index, index=meta.index) if unit_col is None else meta[unit_col]
    unit_order = list(pd.unique(units))
    unit_counts = counts.groupby(units, sort=False).sum().reindex(unit_order)
    unit_counts.index.name = unit_col or meta.index.name
    if unit_aggregation == "pool_counts":
        per_unit = unit_counts.div(
            unit_counts.sum(axis=1).replace(0, np.nan), axis=0,
        ).fillna(0.0)
    else:
        per_unit = fractions.groupby(units, sort=False).mean().reindex(unit_order)
    unit_meta = pd.DataFrame(index=per_unit.index, columns=[group_col])
    for unit in unit_order:
        groups = pd.unique(meta.loc[units == unit, group_col])
        if len(groups) != 1:
            raise ValueError(f"Unit {unit!r} belongs to more than one group.")
        unit_meta.loc[unit, group_col] = groups[0]
    if group_order is None:
        group_order = list(pd.unique(meta[group_col]))
    group_mean = pd.DataFrame(index=list(group_order), columns=cats, dtype=float)
    for group in group_order:
        group_mean.loc[group] = per_unit.loc[unit_meta[group_col] == group].mean(axis=0)
    return ClusterCompositionDetails(
        fractions, group_mean, counts, per_unit, unit_counts,
        unit_meta, unit_aggregation,
    )


@dataclass(frozen=True)
class CooccurrenceDetails:
    """Ordered endpoint counts and their abundance-product reference."""

    enrichment: np.ndarray
    observed: np.ndarray
    expected: np.ndarray
    node_counts: np.ndarray
    categories: tuple
    per_sample: dict[str, Any]
    per_unit: dict[str, Any]
    unit_aggregation: str


def cooccurrence_details(
    adatas, *, categories, graph_key, cluster_attr, sample_meta,
    unit_col, unit_aggregation, missing, eps, count_fn, score_fn,
) -> CooccurrenceDetails:
    if missing not in {"zero", "nan"}:
        raise ValueError("missing must be 'zero' or 'nan'.")
    if unit_aggregation not in {"pool_counts", "mean_roi_enrichment"}:
        raise ValueError("unit_aggregation must be 'pool_counts' or 'mean_roi_enrichment'.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    cohort = adatas if isinstance(adatas, Mapping) else dict(enumerate(adatas))
    meta = _metadata(list(cohort), sample_meta, unit_col)
    cats = tuple(categories)
    shape = (len(cats), len(cats))

    def result(observed, nodes, per_sample=None, per_unit=None, enrichment=None):
        p = nodes / max(nodes.sum(), 1.0)
        expected = observed.sum() * np.outer(p, p)
        score = score_fn(observed, nodes, eps=eps, missing=missing)
        return CooccurrenceDetails(
            score if enrichment is None else enrichment,
            observed, expected, nodes, cats,
            per_sample or {}, per_unit or {}, unit_aggregation,
        )

    per_sample = {}
    for sample in meta.index:
        adata = cohort[sample]
        if graph_key in adata.uns:
            graph = adata.uns[graph_key]
        elif graph_key in adata.obsp:
            import networkx as nx

            adjacency = _nonnegative_matrix(adata.obsp[graph_key], "adjacency")
            if adjacency.shape != (adata.n_obs, adata.n_obs):
                raise ValueError("Observation graph must be square and aligned to obs.")
            if (adjacency != adjacency.T).nnz:
                raise ValueError("Cluster co-occurrence requires an undirected, symmetric graph.")
            if cluster_attr not in adata.obs:
                raise KeyError(f"adata.obs[{cluster_attr!r}] is missing.")
            graph = nx.from_scipy_sparse_array(adjacency)
            nx.set_node_attributes(
                graph, dict(enumerate(adata.obs[cluster_attr].tolist())), cluster_attr,
            )
        else:
            graph = None
        if graph is None:
            observed, nodes = np.zeros(shape), np.zeros(len(cats))
        else:
            observed, nodes = count_fn(
                graph, n_clusters=len(cats),
                cluster_attr=cluster_attr, categories=cats,
            )
        per_sample[sample] = result(observed, nodes)
    units = pd.Series(meta.index, index=meta.index) if unit_col is None else meta[unit_col]
    per_unit = {}
    for unit in pd.unique(units):
        samples = meta.index[units == unit]
        observed = sum((per_sample[s].observed for s in samples), np.zeros(shape))
        nodes = sum((per_sample[s].node_counts for s in samples), np.zeros(len(cats)))
        mean = None
        if unit_aggregation == "mean_roi_enrichment":
            scores = np.stack([per_sample[s].enrichment for s in samples])
            finite = np.isfinite(scores)
            mean = np.divide(
                np.where(finite, scores, 0).sum(axis=0), finite.sum(axis=0),
                out=np.full(shape, np.nan), where=finite.sum(axis=0) > 0,
            )
        per_unit[unit] = result(observed, nodes, enrichment=mean)
    observed = sum((r.observed for r in per_sample.values()), np.zeros(shape))
    nodes = sum((r.node_counts for r in per_sample.values()), np.zeros(len(cats)))
    mean = None
    if (unit_col is not None or unit_aggregation == "mean_roi_enrichment") and per_unit:
        scores = np.stack([r.enrichment for r in per_unit.values()])
        finite = np.isfinite(scores)
        mean = np.divide(
            np.where(finite, scores, 0).sum(axis=0), finite.sum(axis=0),
            out=np.full(shape, np.nan), where=finite.sum(axis=0) > 0,
        )
    return result(observed, nodes, per_sample, per_unit, mean)


def neighbourhood_features(
    adjacency, labels=None, *, values=None, categories=None,
    target_ids=None, source_ids=None, normalization="proportions",
    hops=1, via=None, exclude_self=True, prefix=None,
) -> pd.DataFrame:
    """Aggregate source labels or features along explicit weighted graph paths.

    Rows of ``adjacency`` are targets; columns are sources (or intermediate
    nodes for two hops). Labels and continuous ``values`` are aligned to
    ``source_ids``. ``via`` maps intermediates to final sources. With two
    hops and no ``via``, a square graph is traversed twice. Paths ending at
    the target's own stable ID are excluded, including directed return
    paths; other repeated paths retain their multiplicity.

    ``counts``, ``proportions`` and ``log1p_counts`` describe categorical
    composition. ``mean`` divides aggregated continuous values by the
    total remaining path weight. Isolated targets have zero features.
    DataFrame inputs are realigned by IDs; array inputs are positional.
    """
    adjacency = _nonnegative_matrix(adjacency, "adjacency")
    if not isinstance(hops, (int, np.integer)) or hops not in {1, 2}:
        raise ValueError("hops must be 1 or 2.")
    if hops == 1 and via is not None:
        raise ValueError("via is only meaningful for hops=2.")
    if hops == 2:
        via = adjacency if via is None else _nonnegative_matrix(via, "via")
        if adjacency.shape[1] != via.shape[0]:
            raise ValueError("adjacency columns must match via rows.")
        adjacency = (adjacency @ via).tocsr()
    positional_square = adjacency.shape[0] == adjacency.shape[1]
    explicit_ids = target_ids is not None or source_ids is not None
    targets = _ids(target_ids, adjacency.shape[0], "target_ids")
    sources = _ids(source_ids, adjacency.shape[1], "source_ids")
    if exclude_self and (explicit_ids or positional_square):
        matching = sources.get_indexer(targets)
        rows = np.flatnonzero(matching >= 0)
        if len(rows):
            adjacency = adjacency.tolil()
            adjacency[rows, matching[rows]] = 0
            adjacency = adjacency.tocsr()
            adjacency.eliminate_zeros()
    aliases = {
        "count": "counts", "fraction": "proportions",
        "log1p_count": "log1p_counts", "logcount": "log1p_counts",
    }
    normalization = aliases.get(normalization, normalization)
    if values is not None:
        if labels is not None:
            raise ValueError("Supply labels or values, not both.")
        aligned = _aligned(values, sources, "values")
        names = list(aligned.columns) if isinstance(aligned, pd.DataFrame) else None
        matrix = np.asarray(aligned, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("values must be a finite source-by-feature matrix.")
        names = names or list(range(matrix.shape[1]))
        if normalization not in {"counts", "mean"}:
            raise ValueError("Continuous values support normalization='counts' or 'mean'.")
        totals = np.asarray(adjacency @ matrix)
        denominator = np.asarray(adjacency.sum(axis=1)).ravel()
    else:
        if labels is None:
            raise ValueError("Supply labels or continuous values.")
        aligned = np.asarray(_aligned(labels, sources, "labels"))
        if aligned.ndim != 1:
            raise ValueError("labels must be a one-dimensional source vector.")
        cats = _categories(aligned, categories)
        codes = cats.get_indexer(aligned)
        valid = codes >= 0
        onehot = sp.csr_matrix(
            (np.ones(valid.sum()), (np.flatnonzero(valid), codes[valid])),
            shape=(len(sources), len(cats)),
        )
        totals = (adjacency @ onehot).toarray()
        names = cats.tolist()
        denominator = totals.sum(axis=1)
        if normalization not in {"counts", "proportions", "log1p_counts"}:
            raise ValueError("Categorical normalization must be counts, proportions or log1p_counts.")
    if normalization in {"proportions", "mean"}:
        totals = np.divide(
            totals, denominator[:, None], out=np.zeros_like(totals),
            where=denominator[:, None] > 0,
        )
    elif normalization == "log1p_counts":
        totals = np.log1p(totals)
    columns = names if prefix is None else [f"{prefix}{name}" for name in names]
    result = pd.DataFrame(totals, index=targets, columns=columns)
    result.attrs.update(hops=hops, normalization=normalization, exclude_self=exclude_self)
    return result


@dataclass(frozen=True)
class ContactMarkerContrastResult:
    contrast: pd.Series
    contacted_mean: pd.Series
    uncontacted_mean: pd.Series
    contact_mask: pd.Series
    weights: pd.Series
    n_contacted: int
    n_uncontacted: int


@dataclass(frozen=True)
class CellECMEnrichmentResult:
    """Pooled and per-ROI contact enrichment from the same counting engine."""

    pooled: pd.DataFrame
    per_roi: pd.DataFrame

    @property
    def per_sample(self) -> pd.DataFrame:
        return self.per_roi


def contact_marker_contrast(
    connectivity, marker_features, *, cell_labels=None, cell_type=None,
    patch_labels=None, cluster=None, patch_ids=None, weighting="unique_patch",
    accumulation_dtype="float64", cell_ids=None,
) -> ContactMarkerContrastResult:
    """Compare supplied marker values in contacted versus uncontacted patches.

    ``connectivity`` is cell-by-patch. ``unique_patch`` counts each patch
    once; ``edge`` weights contacted patches by their number of selected
    cell edges. Stored edge magnitudes are ignored in both modes.
    Uncontacted patches always count once. An empty arm produces NaN,
    and no marker normalization or biological-unit pooling is performed.
    ``accumulation_dtype="input"`` preserves the original dtype and uses
    NumPy means for array inputs or pandas means for DataFrame inputs.
    Indexed ``cell_labels`` align to ``cell_ids``; arrays remain positional.
    """
    adjacency = _nonnegative_matrix(connectivity, "connectivity")
    cells = _ids(cell_ids, adjacency.shape[0], "cell_ids")
    if patch_ids is None and isinstance(marker_features, pd.DataFrame):
        patch_ids = marker_features.index
    ids = _ids(patch_ids, adjacency.shape[1], "patch_ids")
    aligned = _aligned(marker_features, ids, "marker_features")
    if accumulation_dtype not in {"float64", "input"}:
        raise ValueError("accumulation_dtype must be 'float64' or 'input'.")
    numpy_input = not isinstance(aligned, pd.DataFrame)
    if isinstance(aligned, pd.DataFrame):
        features = aligned.astype(float) if accumulation_dtype == "float64" else aligned
    else:
        array = np.asarray(aligned, dtype=float if accumulation_dtype == "float64" else None)
        if array.ndim != 2:
            raise ValueError("marker_features must be patch-by-marker.")
        features = pd.DataFrame(array, index=ids)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("marker_features must be finite.")
    if weighting not in {"unique_patch", "edge"}:
        raise ValueError("weighting must be 'unique_patch' or 'edge'.")
    if cell_labels is not None:
        cell_labels = np.asarray(_aligned(cell_labels, cells, "cell_labels"))
        if cell_labels.ndim != 1:
            raise ValueError("cell_labels must be a one-dimensional cell vector.")
    if cell_type is not None:
        if cell_labels is None:
            raise ValueError("cell_labels must align with connectivity rows.")
        selected = np.asarray(cell_labels) == cell_type
        adjacency = adjacency[selected]
    degree = np.asarray(adjacency.getnnz(axis=0)).ravel()
    contact = degree > 0
    eligible = np.ones(len(ids), dtype=bool)
    if cluster is not None:
        if patch_labels is None:
            raise ValueError("patch_labels is required when restricting cluster.")
        eligible = np.asarray(_aligned(patch_labels, ids, "patch_labels")) == cluster
    yes = eligible & contact
    no = eligible & ~contact
    weights = degree.astype(float) if weighting == "edge" else np.ones(len(ids))
    weights[~yes] = 0
    if accumulation_dtype == "input" and numpy_input and weighting == "unique_patch":
        array = features.to_numpy()
        yes_mean = pd.Series(
            array[yes].mean(axis=0) if yes.any() else np.full(array.shape[1], np.nan),
            index=features.columns,
        )
        no_mean = pd.Series(
            array[no].mean(axis=0) if no.any() else np.full(array.shape[1], np.nan),
            index=features.columns,
        )
    else:
        if weighting == "unique_patch":
            yes_mean = features.loc[yes].mean(axis=0)
        else:
            yes_mean = (
                features.loc[yes].mul(weights[yes], axis=0).sum(axis=0) / weights[yes].sum()
                if yes.any() else pd.Series(np.nan, index=features.columns)
            )
        no_mean = features.loc[no].mean(axis=0)
    return ContactMarkerContrastResult(
        yes_mean - no_mean, yes_mean, no_mean,
        pd.Series(contact, index=ids, name="contact"),
        pd.Series(weights, index=ids, name="contact_weight"),
        int(yes.sum()), int(no.sum()),
    )


@dataclass(frozen=True)
class ComponentSummaryResult:
    components: pd.DataFrame
    membership: pd.DataFrame
    per_sample: pd.DataFrame
    per_unit: pd.DataFrame
    records: dict


def component_summary(
    adatas, *, target_cluster, mode="existing", radius=None,
    cluster_attr="ecm_cluster", graph_key="ecm_graph", pos_attrs=("x", "y"),
    min_nodes=1, sample_meta=None, unit_col=None,
    unit_aggregation="pool_components", metrics=None,
) -> ComponentSummaryResult:
    """Extract components and summarize them without changing their graph rule.

    ``existing`` inherits input graph edges; ``radius`` explicitly rebuilds
    target-cluster edges and requires ``radius``. Tables include rejected
    components and original node IDs. Summaries use kept components only.
    Empty samples/units have count zero and undefined medians (NaN).
    ``pool_components`` computes medians over a unit's components;
    ``mean_roi_summary`` averages ROI medians and sums component counts.
    """
    from mantpy import gr
    from mantpy.tl import lesion_topology_stats_df

    if mode not in {"existing", "radius"}:
        raise ValueError("mode must be 'existing' or 'radius'.")
    if mode == "radius" and radius is None:
        raise ValueError("radius is required for mode='radius'.")
    if mode == "radius" and (not np.isfinite(radius) or radius < 0):
        raise ValueError("radius must be finite and non-negative.")
    if mode == "existing" and radius is not None:
        raise ValueError("radius is only meaningful for mode='radius'.")
    if not isinstance(min_nodes, (int, np.integer)) or min_nodes < 1:
        raise ValueError("min_nodes must be a positive integer.")
    if unit_aggregation not in {"pool_components", "mean_roi_summary"}:
        raise ValueError("unit_aggregation must be 'pool_components' or 'mean_roi_summary'.")
    cohort = adatas if isinstance(adatas, Mapping) else {"sample": adatas}
    meta = _metadata(list(cohort), sample_meta, unit_col)
    records = {}
    membership = []
    for sample in meta.index:
        kwargs = {
            "target_cluster": target_cluster, "cluster_attr": cluster_attr,
            "graph_key": graph_key, "pos_attrs": pos_attrs, "min_nodes": min_nodes,
        }
        records[sample] = (
            gr.extract_components_radius_reconnect(cohort[sample], radius=radius, **kwargs)
            if mode == "radius" else gr.extract_components_existing(cohort[sample], **kwargs)
        )
        for component_id, record in enumerate(records[sample]):
            degrees = (
                [record["subgraph"].degree(node) for node in record["node_ids"]]
                if mode == "existing"
                else [degree for _, degree in record["subgraph"].degree()]
            )
            for node_id, degree in zip(record["node_ids"], degrees, strict=True):
                membership.append({
                    "sample_id": sample, "component_id": component_id, "node_id": node_id,
                    "kept": record.get("kept", True), "degree": degree,
                })
    table = lesion_topology_stats_df(
        records, metrics=["basic", "degree"] if metrics is None else metrics,
        only_kept=False,
    ).rename(columns={"lesion_id": "component_id"})
    if table.empty:
        table = pd.DataFrame(columns=["sample_id", "component_id", "n_nodes", "n_edges"])
    table["kept"] = [
        records[row.sample_id][int(row.component_id)].get("kept", True)
        for row in table.itertuples()
    ]
    table["mean_degree"] = [
        2 * records[row.sample_id][int(row.component_id)]["n_edges"]
        / max(records[row.sample_id][int(row.component_id)]["n_nodes"], 1)
        for row in table.itertuples()
    ]
    # n_nodes is available even when callers request a different metric family.
    table["n_nodes"] = [
        records[row.sample_id][int(row.component_id)]["n_nodes"]
        for row in table.itertuples()
    ]
    membership = pd.DataFrame(
        membership, columns=["sample_id", "component_id", "node_id", "kept", "degree"],
    )
    kept = table.loc[table.kept.astype(bool)]

    def summary(rows):
        return {
            "n_components": len(rows),
            "median_nodes": rows.n_nodes.median(),
            "median_mean_degree": rows.mean_degree.median(),
        }

    per_sample = pd.DataFrame.from_dict(
        {sample: summary(kept.loc[kept.sample_id == sample]) for sample in meta.index},
        orient="index",
    ).reindex(
        index=meta.index,
        columns=["n_components", "median_nodes", "median_mean_degree"],
    )
    units = pd.Series(meta.index, index=meta.index) if unit_col is None else meta[unit_col]
    unit_rows = {}
    for unit in pd.unique(units):
        samples = meta.index[units == unit]
        if unit_aggregation == "pool_components":
            unit_rows[unit] = summary(kept.loc[kept.sample_id.isin(samples)])
        else:
            sub = per_sample.loc[samples]
            unit_rows[unit] = {
                "n_components": int(sub.n_components.sum()),
                "median_nodes": sub.median_nodes.mean(),
                "median_mean_degree": sub.median_mean_degree.mean(),
            }
    per_unit = pd.DataFrame.from_dict(unit_rows, orient="index").reindex(
        columns=["n_components", "median_nodes", "median_mean_degree"],
    )
    per_unit.index.name = unit_col or meta.index.name
    return ComponentSummaryResult(table, membership, per_sample, per_unit, records)


def _spatial_candidate_gate(candidate, coordinates, radius, min_neighbours):
    """Shared synchronous gate: count other candidates, including coincident IDs."""
    candidate = np.asarray(candidate, dtype=bool)
    if min_neighbours <= 0 or not candidate.any():
        return candidate.copy()
    from scipy.spatial import cKDTree

    xy = np.asarray(coordinates, dtype=float)
    rows = np.flatnonzero(candidate)
    tree = cKDTree(xy[rows])
    counts = tree.query_ball_point(xy[rows], radius, return_length=True)
    keep = np.asarray(counts) - 1 >= min_neighbours
    result = np.zeros_like(candidate)
    result[rows[keep]] = True
    return result


@dataclass(frozen=True)
class LabelReconstructionResult:
    labels: np.ndarray
    anomaly_scores: np.ndarray
    relabelled: np.ndarray
    predicted_labels: np.ndarray
    confidence: np.ndarray
    probabilities: np.ndarray


def reconstruct_labels(
    observed, probabilities, *, threshold, categories=None, signal=None,
    score="minmax_neglogp", min_confidence=0., coordinates=None,
    spatial_radius=None, min_neighbours=0, majority_smoothing=False,
    temperature=1., eps=1e-12, ids=None,
) -> LabelReconstructionResult:
    """Reconstruct labels from aligned class probabilities in one synchronous pass.

    ``raw_neglogp`` is -log P(observed label). ``minmax_neglogp`` rescales
    those values within the supplied signal rows, returning zero for a
    constant score. ``one_minus_p`` and ``margin`` are also supported.
    Labels absent from ``categories`` are background and remain unchanged.
    Temperature is applied to probabilities before scoring; fitting it
    remains the caller's training-only step. Spatial gating counts other
    candidate patches. Majority smoothing changes the predicted label,
    while confidence remains the original maximum class probability.
    A Series ``observed`` defines row IDs unless ``ids`` is supplied.
    Indexed probability, signal and coordinate inputs align to these IDs.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative.")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must lie in [0, 1].")
    if not isinstance(min_neighbours, (int, np.integer)) or min_neighbours < 0:
        raise ValueError("min_neighbours must be a non-negative integer.")
    if not np.isfinite(temperature) or temperature <= 0 or not 0 < eps < 1:
        raise ValueError("temperature must be positive and eps must lie in (0, 1).")
    if spatial_radius is not None and (
        not np.isfinite(spatial_radius) or spatial_radius < 0
    ):
        raise ValueError("spatial_radius must be finite and non-negative.")
    if ids is None:
        if isinstance(observed, pd.Series):
            ids = observed.index
        elif isinstance(probabilities, pd.DataFrame):
            ids = probabilities.index
    observation_ids = _ids(ids, len(observed), "ids")
    observed_array = np.asarray(_aligned(observed, observation_ids, "observed"))
    if observed_array.ndim != 1:
        raise ValueError("observed must be one-dimensional.")
    if isinstance(probabilities, pd.DataFrame):
        probabilities = _aligned(probabilities, observation_ids, "probabilities")
        if categories is None:
            categories = probabilities.columns
        else:
            if not pd.Index(categories).isin(probabilities.columns).all():
                raise ValueError("Probability columns must contain all categories.")
            probabilities = probabilities.loc[:, list(categories)]
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim != 2 or proba.shape[0] != len(observed_array) or proba.shape[1] < 1:
        raise ValueError("probabilities must be an observed-row-by-class matrix.")
    if not np.isfinite(proba).all() or np.any(proba < 0) or np.any(proba > 1):
        raise ValueError("probabilities must be finite values in [0, 1].")
    if len(proba) and not np.allclose(proba.sum(axis=1), 1., rtol=1e-6, atol=1e-8):
        raise ValueError("Each probability row must sum to one.")
    cats = _categories(range(proba.shape[1]) if categories is None else categories)
    # Explicit probability-column order must not be numerically sorted.
    if categories is not None:
        cats = _categories([], categories)
    if len(cats) != proba.shape[1]:
        raise ValueError("categories must match probability columns.")
    codes = cats.get_indexer(observed_array)
    is_signal = codes >= 0
    if signal is not None:
        mask = np.asarray(_aligned(signal, observation_ids, "signal"), dtype=bool)
        if mask.shape != is_signal.shape:
            raise ValueError("signal must align with observed.")
        is_signal &= mask
    if temperature != 1 and len(proba):
        logp = np.log(np.maximum(proba, eps)) / temperature
        logp -= logp.max(axis=1, keepdims=True)
        proba = np.exp(logp)
        proba /= proba.sum(axis=1, keepdims=True)
    else:
        proba = proba.copy()
    rows = np.flatnonzero(is_signal)
    p_observed = np.ones(len(proba))
    p_observed[rows] = proba[rows, codes[rows]]
    neglog = -np.log(np.maximum(p_observed, eps))
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    if score == "raw_neglogp":
        anomaly = neglog
    elif score == "minmax_neglogp":
        anomaly = np.zeros(len(proba))
        if len(rows):
            lo, hi = neglog[rows].min(), neglog[rows].max()
            if hi > lo:
                anomaly[rows] = (neglog[rows] - lo) / (hi - lo)
    elif score == "one_minus_p":
        anomaly = 1 - p_observed
    elif score == "margin":
        anomaly = confidence - p_observed
    else:
        raise ValueError("Unknown score; use raw_neglogp, minmax_neglogp, one_minus_p or margin.")
    anomaly = np.where(is_signal, anomaly, 0.)
    spatial_active = spatial_radius is not None and spatial_radius > 0
    if spatial_active and (majority_smoothing or min_neighbours > 0):
        if coordinates is None:
            raise ValueError("coordinates must be a finite (n_rows, 2) array.")
        xy = np.asarray(_aligned(coordinates, observation_ids, "coordinates"), dtype=float)
        if xy.shape != (len(proba), 2) or not np.isfinite(xy).all():
            raise ValueError("coordinates must be a finite (n_rows, 2) array.")
        if majority_smoothing and len(rows):
            from scipy.spatial import cKDTree
            neighborhoods = cKDTree(xy[rows]).query_ball_point(xy[rows], spatial_radius)
            smoothed = predicted.copy()
            for row, neighbors in zip(rows, neighborhoods, strict=True):
                votes = np.bincount(predicted[rows[neighbors]], minlength=len(cats))
                smoothed[row] = votes.argmax()
            predicted = smoothed
    elif majority_smoothing:
        raise ValueError("majority_smoothing requires a positive spatial_radius.")
    candidates = (
        is_signal & (predicted != codes) & (anomaly >= threshold)
        & (confidence >= min_confidence)
    )
    if spatial_active and min_neighbours > 0:
        candidates = _spatial_candidate_gate(candidates, xy, spatial_radius, min_neighbours)
    elif min_neighbours > 0:
        raise ValueError("min_neighbours requires a positive spatial_radius.")
    labels = observed_array.copy()
    predicted_labels = cats.to_numpy()[predicted]
    # A fixed-width string array must not truncate a longer replacement label.
    if labels.dtype.kind in {"U", "S"}:
        labels = labels.astype(object)
    labels[candidates] = predicted_labels[candidates]
    return LabelReconstructionResult(
        labels, anomaly, candidates, predicted_labels, confidence, proba,
    )
