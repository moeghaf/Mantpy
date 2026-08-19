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

# Building ECM graphs

Cells are discrete objects, so a cell graph is easy to define: one node per
cell. The matrix is not. It is a continuous texture with no natural units, so
before it can become a graph it has to be divided into something countable.

Mantpy does that with **patches**: the ECM channels are tiled into small
squares, each square is summarised into a feature vector, and those vectors are
clustered into ECM types. The patches become the nodes of the ECM graph.

```{code-cell} python
import mantpy as mt

roi = mt.datasets.toy_ecm_roi()
adata = mt.io.read_imc(
    roi.image, panel=roi.panel, cells=roi.cells,
    sample_id="toy", condition="ctrl",
)
```

## From image to patches

{func}`~mantpy.pp.extract_ecm_patches` tiles the ECM channels, summarises each
tile, and clusters the result. It writes into `adata` and returns nothing —
the same convention scanpy uses.

```{code-cell} python
mt.pp.extract_ecm_patches(
    adata,
    roi.image,
    patch_size=8,      # tile edge in pixels
    ecm_K=3,           # number of ECM clusters
    features=["mean"], # per-channel summary within each tile
)

mt.pp.ecm_patch_summary(adata)
```

`patch_size` is the parameter that matters most. It sets the spatial resolution
of everything downstream: too large and distinct matrix regions get averaged
together, too small and each patch is dominated by noise. It is a real
modelling choice, not a performance knob.

The patches themselves are a table:

```{code-cell} python
adata.uns["ecm_patches"].head()
```

`x` and `y` are the patch centroid; `ecm_cluster` is the assigned ECM type;
`feat_0`/`feat_1` are the per-channel summaries — here the mean of ColIV and FN,
because those are the two channels flagged `ecm=1` in the panel.

:::{note}
`ecm_K=3` is asserted, not discovered. On real data use
{func}`~mantpy.pp.select_ecm_cluster_count` to choose it from the data, or
{func}`~mantpy.pp.select_ecm_leiden_resolution` for the Leiden route.
:::

## Patches to a graph

{func}`~mantpy.gr.build_ecm_graph` connects neighbouring patches. `k=5` joins
each patch to its five nearest neighbours.

```{code-cell} python
mt.gr.build_ecm_graph(adata, k=5)

ecm_graph = adata.uns["ecm_graph"]
ecm_graph
```

```{code-cell} python
mt.pl.ecm_graph(adata)
```

The clusters form contiguous regions rather than scattered speckle, which is
what you want to see — it means the patch features are capturing real spatial
structure rather than noise. On your own data this plot is the first check
worth making after choosing `patch_size` and `ecm_K`.

## Choosing how patches connect

`edge_method` changes the neighbourhood definition:

```{code-cell} python
for method in ("knn", "delaunay", "grid"):
    mt.gr.build_ecm_graph(adata, k=5, edge_method=method)
    g = adata.uns["ecm_graph"]
    print(f"{method:10s} {g.number_of_edges():4d} edges")
```

`knn` gives every patch the same degree. `delaunay` follows the spatial
tessellation and adapts to local density. `grid` connects patches to their
immediate tile neighbours, which is the most literal reading of "adjacent
matrix". There is no universally right answer — pick the one whose notion of
adjacency matches the question being asked.

```{code-cell} python
mt.gr.build_ecm_graph(adata, k=5)  # back to the default for the next page
```

## Next

- [Cell–ECM graphs](cell-ecm-graphs.md) — joining cells to the matrix
- [Exporting to AnnData](export-anndata.md)
