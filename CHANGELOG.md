# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

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
