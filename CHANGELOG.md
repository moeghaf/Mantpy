# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [1.3.0] - 2026-11-10

### Added

Helpers for scoring and controlling a single-image patch analysis. Both are new
functions; no existing output changes.

- `mt.tl.score_patch_labels` scores patch labels, or a pixel label image,
  against a pixel annotation of the same image. Every annotated pixel inside an
  analysed patch counts once. It returns the pixel contingency table, its
  adjusted Rand index, a one-to-one matching of clusters to annotated classes,
  the matched class of every patch (ready for `mt.pl.patch_map`), the recovery
  of each class and the annotated class of every patch.
- `mt.gr.patch_graph_control` derives a negative-control graph from a stored
  patch graph: `"shuffled_edges"` permutes the edge destinations with a seed,
  `"no_edges"` removes every edge. The source graph is left in place.
- `mt.pl.patch_map` takes `legend=True` (or a mapping of `Axes.legend` options) to
  add one entry per drawn category. The default, no legend, is unchanged.

Group contrasts and plot options for a cohort analysis. The defaults are unchanged.

- `mt.tl.cluster_cooccurrence` takes `group_col`, the column of `sample_meta` with
  the experimental group of each sample. The returned details then hold `unit_meta`
  and a `contrast(case=..., control=...)` method: the mean enrichment of the case
  units minus that of the control units, each unit counted once, as a table of
  cluster pairs. It parallels `contrast` on the composition details.
- `mt.pl.plot_mean_composition` takes `title` (`None` for no title) and
  `group_labels` (write the group names, or labels of your own, under the bars). Rows
  of `group_mean` are now matched to `group_order` by label when the index holds
  those labels.
- `mt.pl.plot_delta_masked` takes `show_values=True` to write the value in every cell
  of the unmasked heat map, accepts a table as well as an array, and leaves undefined
  (NaN) cells white instead of failing to set the colour range.
- `mt.pl.ecm_centroid_heatmap` takes `colorbar_label` (default `"z"`).
- `mt.pp.renumber_ecm_clusters` gives the ECM clusters of a cohort new numbers from a
  `{current: new}` mapping, for example to keep the numbers of an earlier run when two
  clusters of similar size swap. Every patch stays in its cluster. The patch tables and
  the stored clustering record are updated.

Per-sample marker contrasts, repeated label reconstruction and per-sample points on two bar
plots. The defaults of existing functions are unchanged.

- `mt.tl.cell_ecm_marker_contrast` compares, in every sample of a cohort, the ECM marker
  profile of the patches that touch a cell type with that of the other patches, optionally
  inside one ECM cluster. It reads the stored cell-ECM graph and the patch table, calls
  `mt.tl.contact_marker_contrast` once per sample and returns the per-sample contrasts, their
  mean with every sample counted once, the patch counts and the contacted patches
  (`CellECMMarkerContrastResult`). A sample in which one of the two groups is empty has no
  contrast and is left out of the mean.
- `mt.pl.marker_contrast_bars` draws that result: one horizontal bar per marker for the mean
  and one point per sample, with `sample_groups` to give the samples of each group (for
  example each mouse) their own marker shape.
- `mt.pl.cell_ecm_enrichment_bars` takes `show_samples=True` to draw one point per sample
  on each bar, with `sample_groups`, `sample_size` and `legend`. A sample with no observed
  contact is drawn as an open grey marker below the bars, not at its ratio. `annotate="none"`
  draws the bars without text (for example when no permutation test was run), and the details
  object of `mt.tl.cell_ecm_enrichment(..., return_details=True)` is accepted in place of
  the AnnData.
- `mt.tl.fit_temperature` chooses, from a grid, the temperature that gives the observed labels
  the lowest mean negative log-probability under a classifier's class probabilities. The
  result is the `temperature` argument of `mt.tl.label_anomaly_scores` and
  `mt.tl.reconstruct_labels`.
- `mt.tl.reconstruct_labels_iteratively` repeats `mt.tl.reconstruct_labels` for up to
  `max_passes` passes. It takes a function that returns the class probabilities for a set of
  labels, so context that depends on the labels is recomputed after every pass, and it stops
  when a pass changes no label. It returns the final labels, the rows that changed and the
  result of every pass (`IterativeLabelReconstructionResult`).
- `mt.pl.ecm_centroid_heatmap` takes `values="mean_intensity"` to show the cluster means
  themselves in place of their z-scores across the clusters, for patch features that are
  already on a common scale.

### Fixed

- `GraphMAE.fit` on a graph without edges (for example the `"no_edges"` control) no longer
  emits NumPy `Mean of empty slice` warnings or stores `NaN` edge statistics. The embeddings
  are unchanged.
- `mt.pl.ecm_centroid_heatmap` no longer raises `TypeError` when `K_ecm` is left out;
  the number of clusters is taken from the patch table, as documented.

## [1.2.0] - 2026-11-09

### Added

Helpers for the patch-encoder, composition and classifier steps of a cohort
analysis. Every addition is opt-in: existing defaults, outputs, stored keys and
provenance logs are unchanged when the new names and options are not used.

- `mt.pp.ecm_patch_tiles` returns the pixel block of every ECM patch as one
  float32 array of shape `(patches, ECM channels, patch_size, patch_size)`, in
  cohort and patch-table row order, ready for `PatchEncoder`. With
  `subset="signal"` it returns only the patches with a non-negative label. It
  reads the patch tables and the images and writes nothing.
- `mt.pp.cluster_ecm_patches` accepts `embedding=`, a patch-by-feature matrix
  (for example the output of `PatchEncoder.transform`) that is clustered in
  place of the patch-table features, and `order_by=`, where `"size"` numbers
  the clusters from the largest to the smallest. Both default to `None`, which
  keeps the previous labels, results and provenance exactly. The provenance
  keys `feature_source` and `label_mapping`, and `label_order` for K-means, are
  recorded only when one of the two options is used.
- `mt.tl.ecm_patch_table` stacks the patch tables of a cohort into one long
  table with `sample_id` and `node_id` columns, optionally joined to sample
  metadata. `node_id` is the name of the patch in the ECM graph, so the table
  joins to component membership and to node-level classifier results. It writes
  nothing.
- `ClusterCompositionDetails.contrast`, on the result of
  `mt.tl.compute_cluster_composition(..., return_details=True)`, compares the
  mean per-unit cluster fractions of case groups against control groups and
  returns a table accepted by `mt.tl.top_enriched_cluster`. The fields of the
  result and the default two-item return are unchanged.
- `NodeClassifier.composition_baseline` returns the held-out predictions of a
  logistic regression that sees only per-graph cluster fractions (or supplied
  features), evaluated on the folds the classifier was fitted with. It stores
  nothing on the classifier, and the existing methods are unchanged.
- `NodeClassifier.attribution_table` returns the integrated-gradient node
  importance of many graphs as one long table with sample, group, fold, node,
  coordinate and cluster columns. `integrated_gradients` and its defaults are
  unchanged.
- `mt.tl.cluster_membership_correlation` reports, for every sample and cluster,
  the Pearson correlation between a node score and membership of that cluster.
  It is a new function and changes no existing output.

Helpers for cell-ECM context models and label-reconstruction benchmarks. All
are new functions; no existing output changes.

- `mt.gr.graph_layers` returns the cell-cell, ECM-ECM and cell-ECM layers of a
  built cell-ECM graph as sparse matrices aligned to the cells and to the patch
  table, with matching matrices of edge lengths.
- `mt.gr.distance_weights` turns edge lengths into edge weights (uniform,
  inverse, inverse square, or Gaussian with a fixed or median width).
- `mt.tl.context_features` describes each ECM patch by the cell types and ECM
  clusters of its graph neighbours (cell, ECM or joint context; one or two hops).
- `mt.tl.corrupt_labels` replaces a seeded fraction of labels in square blocks
  and as isolated units, for label-reconstruction benchmarks.
- `mt.tl.label_anomaly_scores` scores how unexpected each observed label is,
  with an optional adjustment for the training label frequencies.
- `mt.tl.score_label_reconstruction` reports detection and correction metrics
  against the true labels, overall and per class.

## [1.1.1] - 2026-11-09

### Added

- Shape options for the built-in patch encoder in `PatchEncoder` and
  `encode_patches`: `conv_blocks` (number of convolution blocks), `conv_width`
  (channels of the first block), `pool_size` and `pool_type` (average or max
  pooling before the readout) and `conv_activation` (ReLU or GELU). The defaults
  keep the previous architecture, so existing results are unchanged. The options
  are saved with the encoder and recorded in its provenance.

### Changed

- With `pool_size` set, `PatchEncoder.fit` uses `deterministic="warn"` by default,
  because adaptive pooling has no deterministic CUDA backward pass. Pass
  `deterministic=` to override. Fits without pooling are unaffected.
- Encoders saved with non-default shape options need Mantpy 1.1.1 or newer to load.

## [1.1.0] - 2026-10-31

### Added

- Configurable patch encoders with custom PyTorch backbones, heads, augmentations,
  optimizers, losses and raw embedding export.
- GraphMAE training history, callbacks and explicit latent scaling.
- Reusable classifier fitting, biological-unit splits, ROI-balanced loss and
  integrated-gradient results with seed-level attributions.
- Biological-unit composition and detailed co-occurrence summaries,
  neighbourhood features, contact-marker contrasts, component membership and
  probability-based label reconstruction.
- Fitted image normalization for training and held-out images, tiles-only patch
  extraction, supplied sparse graph adapters and ID-aligned patch maps.
- Explicit natural-log distance features and optional directed float64 edge
  standardization for reproducing recorded graph representations.

### Fixed

- Patch maps support read-only masks returned by Pandas 3.1.

## [1.0.2] - 2026-10-28

Release accompanying the Nature Communications submission. The source is
identical to the `1.0.2rc3` pre-release wheel used for the paper's execution
checks (SHA-256 `9ad8eed649a269aa2f92dacc58e9747fab4d04bf4d46c35e4275a3b10ff2589c`)
except for version metadata and this changelog entry; `mantpy._palette` gains
the shared representation, neutral and graph-control palettes used by the
paper's figure code.

### Added

- The bioRxiv preprint DOI (`10.1101/2025.06.04.657781`) in `CITATION.cff`,
  the README citation section and the package metadata (`Preprint` URL).
- Opt-in graph rules for regular ECM patch lattices and for reproducing graphs
  built with SciPy. Existing defaults, edges, weights and provenance logs are
  unchanged when the new options are not used.
  - `mt.gr.build_cell_ecm_graph(edge_method="radius", r=...)` joins each cell
    to every ECM patch centre within `r`; `edge_method="containing_ring"` joins
    each cell to the patch that contains it and that patch's eight lattice
    neighbours (`patch_size` is inferred from the patch spacing if not given).
  - `delaunay_max_length` caps Delaunay edge length in `mt.gr.build_cell_graph`,
    `mt.gr.build_ecm_graph(s)` and `mt.gr.build_cell_ecm_graph`; the `Dmax`
    caps stay kNN-only.
  - `delaunay_jitter` and `delaunay_seed` triangulate a seeded, Gaussian-perturbed
    copy of the points so co-circular points (patch lattices) are triangulated
    reproducibly; lengths and caps use the original coordinates.
  - `knn_backend="scipy"` builds kNN layers with `scipy.spatial.cKDTree`, which
    can keep different neighbours than the default scikit-learn ball tree when
    candidates are tied at the k-th distance.
  - `mt.gr.ensure_cell_ecm_graph` and `mt.gr.build_cell_ecm_graphs` forward these
    options (`cell_r`, `ecm_r`, `cell_ecm_r`, `cell_ecm_patch_size`,
    `*_delaunay_max_length`, `knn_backend`, `delaunay_jitter`, `delaunay_seed`).
    `build_cell_ecm_graphs(cell_graph="mantpy", cell_edge_method="radius",
    cell_radius=r)` now builds a radius cell graph instead of raising, and
    `ensure_cell_ecm_graph` can now build radius cell and ECM layers.

### Fixed

- Legacy `mt.gr.to_hetero_pyg` now exports both directions of undirected
  within-type edges and separate cell-to-ECM and ECM-to-cell relations, with
  one correctly oriented feature row per edge. Previously feature tensors
  had twice as many rows as edge indices and reverse directions were missing.
  Partially missing feature vectors now raise a clear error. New graph builders
  record directional endpoints; older within-type graphs without this metadata
  warn when orientation is ambiguous and should be rebuilt.
- `mt.io.to_spatialdata` now creates indexed patch-centroid points and links
  the patch table to those instances, rather than incorrectly annotating image
  elements. Images and points use separate ROI coordinate systems. The original
  patch IDs remain available as `instance_id`, and the ROI as `sample_id`;
  `region` now names the corresponding `"{sample_id}_ecm_patches"` points.
- `mt.tl.ecm_to_anndata` preserves source patch indices as string observation
  names. Legacy graph builders and PyG exports retain explicit source-ID
  mappings. Observation-native `to_pyg` now attaches requested graph labels
  and the documented graph metadata before returning, matching the legacy path.
- `background_quantile=0` in `mt.pp.extract_ecm_patches` now retains the
  complete patch grid, including tied minimum-intensity patches. Earlier
  versions removed every patch at the minimum even when no background
  filtering was requested. Positive quantiles retain the existing strict
  threshold convention.
- `mt.nn.NodeClassifier` now trains against its explicit `labels` argument,
  as documented, including when an input graph has no `y` attribute or its
  `y` differs from those labels. Earlier versions silently trained on
  `graph.y`; results can change when the two disagree.
- The intestine loader's citation notice now identifies processed HuBMAP
  image HBM926.SHNZ.594 and its raw dataset HBM734.XBSR.357.
- `mt.datasets.schistosoma_ecm` exposes an archived cluster count as
  `historical_best_k`, rather than `best_k`, and no longer displays it as a
  recommended cluster count. Code reading `data.best_k` must use the
  explicitly historical field instead; the loader does not select a K for
  a new analysis.
- `CITATION.cff` now lists all ten preprint authors.
- `mt.tl.cluster_cooccurrence` and `mt.tl.cluster_cooccurrence_permutation_test`
  now count every edge between distinct nodes in both orientations.
  Previously a same-cluster edge was counted once and a cross-cluster edge
  twice before normalising by the abundance-product reference, which
  under-weighted same-cluster co-occurrence by a factor of two relative to
  cross-cluster pairs, shifted every cell, and changed permutation p-values
  and FDR values. Results produced with 1.0.1 or earlier (including
  0.2.0rc1) differ. Directed graphs now raise `ValueError`; edge weights
  remain ignored and a self-loop still counts once.
- `mt.gr.build_ecm_graph(s)` and `mt.gr.build_cell_graph` with `edge_method="knn"`, and the
  fewer-than-three-point fallback of `edge_method="delaunay"`, no longer create self-loops
  when points share coordinates. `knn_edges` now excludes each query point by identity
  instead of skipping the first returned neighbour, so coincident distinct points remain
  neighbours at distance 0. Selections for coordinates without duplicates are unchanged;
  graphs built from duplicate coordinates can differ from 1.0.1 and earlier.

- `mt.gr.build_patch_graph`, the kNN adjacency used by `mt.tl.graph_modularity`
  and `mt.tl.select_n_domains`, and perimeter-ring graph features from
  `mt.pp.extract_structure_ecm` now exclude each point by identity when coordinates
  coincide. Distinct coincident nodes remain neighbours. Patch/domain k validation,
  undirected union, edge features and perimeter neighbour budgets are preserved;
  graphs and structure features computed from duplicate coordinates can change.

### Changed

- Squidpy is constrained to the compatible 1.8 series. The tutorial-data
  DOI is documented separately from the software citation and is not
  presented as a software archive identifier.

## [1.0.0] - 2026-08-19

Preprint release.

### Removed

- The prostate H&E/Visium tutorial dataset loader and the H&E-specific
  preprocessing, patch-extraction, plotting and segmentation paths
  (`mt.datasets.prostate_he_visium`, `mt.pp.preprocess_he`,
  `mt.pp.he_ecm_patches`, `mt.pp.he_ecm_patch_summary`, `mt.pl.he_overview`,
  the `mt.fetch.load_prostate_*` compatibility wrappers and the
  `stain="hematoxylin"` path of `mt.pp.segment_cells_tiled`). Mantpy now
  focuses on the spatial-proteomics workflows described in the preprint. The
  archived Zenodo data record is unchanged.

### Changed

- Building a combined graph with only one graph layer present now emits a
  `UserWarning` instead of an INFO-level log record, so the fallback is
  audible in notebooks.

### Added

- `CITATION.cff` citation metadata for the software and the preprint.

## [0.2.0]

### Added

- AnnData-native ECM image patches, spatial graph construction, cell–ECM
  integration, clustering, permutation analysis, and topology tools.
- Optional patch encoders, GraphMAE representations, node classification, and
  neighbour-composition baselines.
- Reusable plotting, palette, and explicit publication-style helpers for Mantpy
  analysis outputs.
- Verified one-line loaders for the public tutorial datasets, with safe
  caching, integrity checks, progress reporting, and source citations.
- Scanpy, Squidpy, PyTorch Geometric, SpatialData, and H5AD interoperability.
