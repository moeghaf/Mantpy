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

### Fixed

- `CITATION.cff` now lists all ten preprint authors.

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
