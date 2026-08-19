---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Loading data

Mantpy analyses start from an {class}`~anndata.AnnData` holding cells, their
coordinates and their marker intensities, plus the raw image the ECM channels
are measured from.

This page uses {func}`~mantpy.datasets.toy_ecm_roi`, which synthesises a small
region of interest in memory. It needs no download and runs in under a second,
so every example here executes when these docs are built.

:::{note}
The toy ROI is synthetic. Its clusters and statistics describe the generator,
not biology. Once the API makes sense, swap the first line for one of the real
loaders — the rest of the code is unchanged.
:::

## A region of interest

```{code-cell} python
import mantpy as mt

roi = mt.datasets.toy_ecm_roi()
roi.image.shape, roi.image.dtype
```

Three pieces come back, matching what the real loaders return:

```{code-cell} python
roi.panel
```

The `ecm` column is what splits the panel in two. Channels flagged `1` describe
the matrix and are used to build ECM patches; the rest are cell markers.

```{code-cell} python
roi.cells.head()
```

## Building an AnnData

{func}`~mantpy.io.read_imc` combines the three into a single object.
`centroid-0` is read as the y coordinate and `centroid-1` as x.

```{code-cell} python
adata = mt.io.read_imc(
    roi.image,
    panel=roi.panel,
    cells=roi.cells,
    sample_id="toy",
    condition="ctrl",
)
adata
```

Cell positions land in `obsm["spatial"]`, which is the layout squidpy and
scanpy expect, so their spatial tooling works on Mantpy objects directly.

```{code-cell} python
adata.obs.head()
```

```{code-cell} python
adata.obsm["spatial"][:5]
```

## What the image looks like

```{code-cell} python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True)
for ax, idx in zip(axes, [0, 2, 4]):
    ax.imshow(roi.image[idx], cmap="magma")
    ax.set_title(roi.panel["name"][idx])
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].scatter(
    adata.obsm["spatial"][:, 0],
    adata.obsm["spatial"][:, 1],
    s=6, c="white", edgecolors="black", linewidths=0.3,
)
axes[0].set_title("ColIV + cells")
plt.show()
```

The ECM channel carries a gradient crossed by a band; the cell channels are
blobs on the cell coordinates. That structure is what the next page clusters.

## Using a real dataset instead

Every loader in {mod}`mantpy.datasets` returns the same shape, so moving to real
data is a one-line change:

```python
data = mt.datasets.coliv_intestine()   # downloads ~93 MB on first call, then caches
```

Downloads are checksummed against an immutable Zenodo record and cached outside
the package. Set `MANTPY_CACHE` to control where.

## Next

- [Building ECM graphs](ecm-graphs.md) — patches, clustering and the ECM graph
- [Cell–ECM graphs](cell-ecm-graphs.md) — joining cells to the matrix
- [Exporting to AnnData](export-anndata.md)
- [Exporting to PyTorch Geometric](export-pytorch-geometric.md)
