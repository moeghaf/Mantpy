<p align="center">
  <img src="https://raw.githubusercontent.com/moeghaf/Mantpy/main/docs/_static/mantpy_logo.png" alt="mantpy" width="320"/>
</p>

# Mantpy: extracellular-matrix analysis for spatial omics

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/moeghaf/Mantpy/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/mantpy

Mantpy is a scverse-based framework for graph analysis of the extracellular
matrix (ECM) in spatial omics. It represents cells and ECM patches as distinct,
linked node types, so matrix structure can be analysed on its own or together
with cellular context. Mantpy works with `AnnData` and interoperates with
Scanpy, Squidpy, and other single-cell and spatial tools.

<p align="center">
  <img src="https://raw.githubusercontent.com/moeghaf/Mantpy/main/docs/_static/mantpy_overview.png" alt="Mantpy overview: input, graph construction and applications" width="820"/>
</p>

## Installation

Mantpy requires Python 3.11 or newer.

```bash
pip install mantpy
```

Optional extras add heavier dependencies only when needed:

```bash
pip install "mantpy[gnn]"      # graph learning and explainability
pip install "mantpy[patch]"    # learned image-patch features
pip install "mantpy[spatial]"  # SpatialData integration
pip install "mantpy[segment]"  # Cellpose segmentation
```

## Quick start

This runs as written — `toy_ecm_roi` synthesises a small region of interest in
memory, so there is nothing to download:

```python
import mantpy as mt

# A small synthetic ROI: image stack, channel panel, cell table.
roi = mt.datasets.toy_ecm_roi()

# Read multiplexed imaging and a cell table into AnnData.
adata = mt.io.read_imc(
    roi.image, panel=roi.panel, cells=roi.cells,
    sample_id="toy", condition="ctrl",
)

# Normalise channels and segment the ECM into patch nodes.
mt.pp.normalize(adata)
mt.pp.extract_ecm_patches(
    adata,
    roi.image,
    ecm_channel="ColIV",
    ecm_K="auto",
    features=["mean"],
)

# Build cell, ECM, and joint cell-ECM graph layers.
mt.gr.build_graph(adata, mode="cell")
mt.gr.build_graph(adata, mode="ecm")
mt.gr.build_graph(adata, mode="cell_ecm")

# Quantify spatial organisation.
mt.tl.cell_ecm_enrichment(adata, cell_type="B")
mt.tl.neighbourhood_clustering(adata, n_clusters=4)

# Visualise results.
mt.pl.cell_ecm_graph(adata)
mt.pl.neighbourhood_clusters(adata)
```

Point `mt.io.read_imc` at your own files to run the same pipeline on real data:

```python
adata = mt.io.read_imc("image.tiff", panel="panel.csv", cells="cells.csv")
```

The toy ROI is synthetic — its clusters and statistics describe the generator,
not biology. Use it to learn the API, then move to real data.

## Tutorials

Step-by-step guides, each executed when the documentation is built:

- [Loading data](https://mantpy.readthedocs.io/en/latest/tutorials/loading-data.html)
- [Building ECM graphs](https://mantpy.readthedocs.io/en/latest/tutorials/ecm-graphs.html)
- [Cell–ECM graphs](https://mantpy.readthedocs.io/en/latest/tutorials/cell-ecm-graphs.html)
- [Exporting to AnnData](https://mantpy.readthedocs.io/en/latest/tutorials/export-anndata.html)
- [Exporting to PyTorch Geometric](https://mantpy.readthedocs.io/en/latest/tutorials/export-pytorch-geometric.html)

The full manuscript analyses run on real cohorts in the
[reproducibility repository][tutorials], each opening in Colab.

## Public datasets

One-line, checksummed loaders for every tutorial dataset, cached outside the
package and reused offline:

```python
intestine = mt.datasets.coliv_intestine()   #  93 MB
lung = mt.datasets.balbc_pbs_lung()         #  71 MB
liver = mt.datasets.schistosoma_ecm()       #  38 MB
```

Every loader returns the same `Bunch` shape as `toy_ecm_roi`, so moving an
example from synthetic to real data is a one-line change. The bundles are frozen
under CC BY 4.0 in an immutable
[Zenodo record](https://doi.org/10.5281/zenodo.21538382).

## Documentation

[Documentation][] · [API reference][api documentation] · [Changelog][changelog]

## Contact

For questions and help requests, use the [scverse discourse][]. To report a
bug, use the [issue tracker][].

[scverse discourse]: https://discourse.scverse.org/
[tutorials]: https://github.com/moeghaf/mantpy_reproducibility
[issue tracker]: https://github.com/moeghaf/Mantpy/issues
[tests]: https://github.com/moeghaf/Mantpy/actions/workflows/test.yaml
[documentation]: https://mantpy.readthedocs.io
[changelog]: https://mantpy.readthedocs.io/en/latest/changelog.html
[api documentation]: https://mantpy.readthedocs.io/en/latest/api.html
