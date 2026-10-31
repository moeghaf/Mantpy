# Descriptive analysis APIs

These functions replace reusable notebook calculations. Dataset selection,
cross-validation, parameter selection, permutations and interpretation stay
visible in the analysis notebook.

## Composition with explicit biological units

```python
composition = mt.tl.compute_cluster_composition(
    adatas,
    sample_meta=sample_meta,
    unit_col="mouse",
    unit_aggregation="pool_counts",
    categories=cluster_order,
    return_details=True,
)
composition.per_unit
composition.per_group_mean
```

`pool_counts` adds ROI patch counts within each mouse before calculating
fractions. Group means then give each mouse equal weight.
`mean_roi_fraction` gives each ROI equal weight within its mouse instead.
The detailed result also supplies `counts`, `per_unit_counts` and `unit_meta`.

Without `unit_col`, samples remain the statistical units. The default return
is the original two-item `ClusterCompositionResult`, so tuple unpacking
continues to work. `label_source="auto"` reads the legacy
`uns["ecm_patches"]` table when present, otherwise labels in `obs`.
Select `"obs"` or `"patches"` explicitly when both representations exist.
Numeric background labels below zero are excluded by default; other
background labels can be supplied with `background_labels`.
Empty foreground samples have zero fractions, consistent with the original
default. Explicit `categories` retain absent states and their column order.

## Co-occurrence counts and unit summaries

```python
cooccurrence = mt.tl.cluster_cooccurrence(
    adatas,
    categories=cluster_order,
    sample_meta=sample_meta,
    unit_col="mouse",
    missing="nan",
    return_details=True,
)
cooccurrence.per_unit[mouse].enrichment
cooccurrence.observed
cooccurrence.expected
```

Each non-self undirected edge contributes both endpoint orientations;
self-loops contribute once. Edge magnitudes are ignored. Isolated labelled
nodes enter the abundance denominator. Expected counts are
`observed.sum() * outer(node_frequency, node_frequency)`.

The default pools all ROI counts and returns the original NumPy enrichment
matrix, with undefined cells zeroed. With `unit_col`, counts are pooled
within each biological unit, followed by an equal-unit mean of enrichment.
`unit_aggregation="mean_roi_enrichment"` averages ROI enrichment instead.
`missing="nan"` exposes undefined cells and averages only defined values.
The detailed top-level `observed`, `expected` and `node_counts` describe
pooled tissue; they are **not** the counts underlying an equal-unit mean
matrix. Individual results are available in `per_sample` and `per_unit`.

Both legacy NetworkX graphs in `uns[graph_key]` and symmetric observation
graphs in `obsp[graph_key]` are supported. The latter use labels from `obs`.
Directed or asymmetric graphs require an explicit upstream decision.

## Weighted neighborhood features

```python
composition_features = mt.tl.neighbourhood_features(
    adjacency,
    source_labels,
    categories=category_order,
    target_ids=target_ids,
    source_ids=source_ids,
    normalization="proportions",
)
two_hop = mt.tl.neighbourhood_features(
    adjacency, source_labels,
    categories=category_order,
    target_ids=target_ids,
    source_ids=source_ids,
    hops=2, via=second_adjacency,
    normalization="counts",
)
marker_means = mt.tl.neighbourhood_features(
    adjacency, values=source_marker_table,
    target_ids=target_ids, source_ids=source_ids,
    normalization="mean",
)
```

Rows of the first matrix identify targets. Its columns identify sources for
one hop or intermediate nodes for two hops. `via` maps intermediate nodes
to the final sources. Without `via`, two hops traverse a square graph twice.
Returning paths to the target's own stable ID are excluded by default,
including directed return paths. Other path multiplicities are retained.
For two hops through a bipartite incidence matrix, pass its transpose
explicitly as `via`.

Categorical normalization accepts `counts`, `proportions` or `log1p_counts`;
`count`, `fraction` and `log1p_count` remain convenient aliases.
Missing categories contribute no counts. Isolated targets have zero
features. Continuous values support weighted sums (`counts`) or weighted
means (`mean`) and receive no hidden standardization.

Series and DataFrames align by explicit stable IDs; arrays are positional.
Use distinct cell and patch ID namespaces for cross-compartment matrices,
or pass `exclude_self=False`. Omitting IDs supplies integer positional IDs;
self-return exclusion then applies only to a square final path matrix.
Rectangular cross-compartment matrices preserve every edge by default.

## Marker contrasts for cell–matrix contacts

```python
contrast = mt.tl.contact_marker_contrast(
    cell_patch_incidence, marker_table,
    cell_labels=cell_types, cell_type=selected_type,
    patch_labels=patch_clusters, cluster=selected_cluster,
    patch_ids=patch_ids,
    weighting="unique_patch",
)
contrast.contrast
```

The result contains marker means for contacted and uncontacted patches,
their difference, the contact mask, contact weights and the two sample
counts. `unique_patch` gives each patch equal weight.
`edge` weights contacted patches by the number of selected cell edges;
stored edge magnitudes are ignored. Uncontacted patches always count once.
An empty comparison arm produces NaN. Supplied marker values are used
directly. Average per-ROI or per-mouse contrasts explicitly downstream.
The default accumulates in float64. `accumulation_dtype="input"` preserves
the supplied dtype and uses NumPy means for arrays or pandas means for
DataFrames, allowing an existing float32 calculation to retain its rounding.
Supply `cell_ids` to align a cell-label Series to connectivity rows.
Array cell labels remain positional.

The existing focal-type and all-type enrichment functions also expose
their pooled and per-ROI tables without writing to the inputs:

```python
enrichment = mt.tl.cell_ecm_enrichment_matrix(
    cohort, K_ecm=n_clusters, n_perm=0,
    eps=0, return_details=True,
)
enrichment.pooled
enrichment.per_roi
```

`eps=0` gives exact ratios, negative infinity for zero observed count
with positive expectation, and NaN for undefined zero/zero cells.
The default retains the existing `1e-9` pseudocount and DataFrame result.
Per-ROI edge frequencies remain the expected-count reference; they are
not replaced by frequencies pooled across the cohort.

## Components, topology and original node membership

```python
regions = mt.tl.component_summary(
    adatas, target_cluster=selected_cluster,
    mode="radius", radius=radius, min_nodes=min_nodes,
    sample_meta=sample_meta, unit_col="mouse",
)
regions.components
regions.membership
regions.per_unit
```

`mode="existing"` inherits graph edges. `mode="radius"` rebuilds edges using
the existing radius-reconnection engine. Those are different graph rules
and require an explicit choice. The existing component and topology
engines supply the calculation.

`components` contains every component and its `kept` flag; `membership`
maps original node IDs to component IDs and within-component degrees.
Summaries use kept components only. Empty units have component count zero
and NaN medians. `pool_components` calculates medians over all components
within a unit; `mean_roi_summary` averages ROI medians while summing their
component counts. The original extraction records remain available in
`records`. More topology metric families can be requested with `metrics`.

## Label reconstruction from probabilities

```python
reconstruction = mt.tl.reconstruct_labels(
    observed_labels, predicted_probabilities,
    categories=class_order,
    threshold=threshold,
    score="minmax_neglogp",
    min_confidence=min_confidence,
    coordinates=patch_coordinates,
    spatial_radius=spatial_radius,
    min_neighbours=min_neighbours,
    temperature=temperature,
)
reconstruction.labels
reconstruction.anomaly_scores
reconstruction.relabelled
```

One call performs one synchronous pass. `raw_neglogp` is
`-log(P(observed_label))`; `minmax_neglogp` rescales those scores over the
supplied signal rows in that ROI, returning zero for constant scores.
`one_minus_p` and `margin` are also available. Categories define probability
column order; labels outside them are background and remain unchanged.
An explicit `signal` mask can further restrict the eligible rows.

Temperature changes probabilities before scoring. Its training-only fit,
the probability model and repeated passes remain the caller's choices.
Spatial gating counts **other candidate patches**, not all nearby patches.
Majority smoothing is optional and retains the historical confidence
definition: the maximum original class probability.
The result includes predicted labels, confidence and calibrated
probabilities as well as the final labels and relabel mask.
An observed-label Series defines stable row IDs. Probability DataFrames,
signal Series and coordinate DataFrames are realigned to them.
With array observed labels, supply `ids` for indexed auxiliary inputs;
a probability DataFrame otherwise defines the row order.

The legacy `denoise_ecm_clusters` API keeps its existing scorer and
defaults. It shares the same spatial-candidate gate with this array API.
