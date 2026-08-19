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

# Exporting to AnnData

Mantpy objects are already `AnnData`, so most of the scverse ecosystem works on
them unchanged. The one thing that needs a deliberate step is the **matrix**:
ECM patches are not cells, so they cannot live on the observation axis of an
object whose observations are cells.

{func}`~mantpy.tl.ecm_to_anndata` promotes them into an `AnnData` of their own,
where patches become observations and patch features become variables.

```{code-cell} python
import mantpy as mt

roi = mt.datasets.toy_ecm_roi()
adata = mt.io.read_imc(
    roi.image, panel=roi.panel, cells=roi.cells,
    sample_id="toy", condition="ctrl",
)
mt.pp.extract_ecm_patches(adata, roi.image, patch_size=8, ecm_K=3, features=["mean"])
mt.gr.build_ecm_graph(adata, k=5)
```

## Patches as observations

```{code-cell} python
ecm = mt.tl.ecm_to_anndata(adata)
ecm
```

129 patches × 2 features. The patch centroid is in `obsm["spatial"]` and the
cluster label in `obs`, so scanpy and squidpy treat this exactly like any other
spatial object:

```{code-cell} python
ecm.obs.head()
```

```{code-cell} python
ecm.var
```

`var` records where each feature came from — which extractor produced it and
which channel it summarises. That matters once you have several extractors, as
the column names alone stop being self-describing.

## Standard tooling now applies

Because it is an ordinary `AnnData`, the usual workflow works with no Mantpy
involved:

```{code-cell} python
import scanpy as sc

sc.pp.neighbors(ecm, n_neighbors=8, use_rep="X")
sc.tl.umap(ecm)
sc.pl.umap(ecm, color="ecm_cluster", size=60, show=True)
```

## Writing to disk

Here is the one sharp edge worth knowing about.

```{code-cell} python
mt.gr.build_cell_graph(adata, k=5)
mt.gr.build_cell_ecm_graph(adata, k=5)

try:
    adata.write_h5ad("cells.h5ad")
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
```

The ECM and joint graphs are stored as NetworkX objects in `uns`, and the h5ad
format has no way to serialise them. This is not a Mantpy bug — it is the
`AnnData` writer refusing to guess.

Two ways round it. Export the ECM side, which is graph-free by construction:

```{code-cell} python
ecm.write_h5ad("ecm.h5ad")
print("ecm.h5ad written")
```

Or drop the unserialisable keys from the cell object first. The graphs are
cheap to rebuild from the patches and coordinates, so this loses nothing
permanent:

```{code-cell} python
cells = adata.copy()
for key in ("ecm_graph", "cell_ecm_graph", "cell_graph_nx",
            "image_container", "img", "ecm_image"):
    cells.uns.pop(key, None)

cells.write_h5ad("cells.h5ad")
print("cells.h5ad written")
```

The cell–cell graph survives this, because it lives in `obsp` as a sparse
matrix rather than in `uns` as NetworkX:

```{code-cell} python
import anndata as ad

reloaded = ad.read_h5ad("cells.h5ad")
print("obsp kept:", sorted(reloaded.obsp))
print("uns kept :", sorted(reloaded.uns))
```

```{code-cell} python
:tags: [remove-cell]

import pathlib
for f in ("cells.h5ad", "ecm.h5ad"):
    pathlib.Path(f).unlink(missing_ok=True)
```

## Next

- [Exporting to PyTorch Geometric](export-pytorch-geometric.md)
