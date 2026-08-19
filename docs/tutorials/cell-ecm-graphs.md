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

# Cell–ECM graphs

The point of Mantpy is the join: a single graph containing both cells and
matrix, so questions like *which cell types sit against which kind of ECM* have
a concrete answer rather than an eyeballed one.

Getting there takes three builders on the same `AnnData` — cells, matrix, then
the bridge between them.

```{code-cell} python
import mantpy as mt

roi = mt.datasets.toy_ecm_roi()
adata = mt.io.read_imc(
    roi.image, panel=roi.panel, cells=roi.cells,
    sample_id="toy", condition="ctrl",
)
mt.pp.extract_ecm_patches(adata, roi.image, patch_size=8, ecm_K=3, features=["mean"])
```

## The three graphs

```{code-cell} python
mt.gr.build_cell_graph(adata, k=5)       # cell  <-> cell
mt.gr.build_ecm_graph(adata, k=5)        # patch <-> patch
mt.gr.build_cell_ecm_graph(adata, k=5)   # cell  <-> patch, plus both of the above

mt.gr.joint_graph_summary(adata)
```

Order matters: {func}`~mantpy.gr.build_cell_ecm_graph` merges the two existing
graphs and adds the cross edges, so both must exist first. The summary is worth
reading closely — it separates the three edge classes, and a joint graph with
very few cell–ECM edges usually means `Dmax_CE` is too tight for the tissue.

Everything lands on the same object:

```{code-cell} python
print("obsp:", sorted(adata.obsp))
print("uns graphs:", [k for k in sorted(adata.uns) if "graph" in k])
```

Cell–cell edges go to `obsp` because they are cell × cell and belong with the
observations. The ECM and joint graphs go to `uns` as NetworkX objects, because
their nodes are not cells and so do not fit the AnnData axes.

## Seeing it

```{code-cell} python
mt.pl.graph_triptych(adata)
```

Left is the cell graph, middle the ECM graph, right the join. The third panel
is the object every downstream analysis actually runs on.

```{code-cell} python
mt.pl.cell_ecm_graph(adata)
```

## Which cells touch which matrix

With the joint graph built, contact becomes a measurement rather than an
impression. {func}`~mantpy.tl.cell_ecm_enrichment` compares observed cell–ECM
adjacency against a permutation null:

```{code-cell} python
result = mt.tl.cell_ecm_enrichment(adata, cell_type="B", n_perm=200)
result
```

:::{warning}
`n_perm=200` is set low here so this page builds quickly. It is far too few for
a real result — the default is 5000, and permutation p-values cannot resolve
below `1/n_perm`. Do not copy this number into an analysis.
:::

Because the toy ROI is synthetic, whatever comes back describes the generator.
On real data this is the step that turns "these cells look like they sit near
collagen" into a number with a null behind it.

## Two builders, two shapes

There is a second route: {func}`~mantpy.gr.compose_cell_ecm_graph` takes two
separate `AnnData` objects and returns a new joint one with sparse matrices in
`obsp`, rather than mutating a single object and storing NetworkX in `uns`.

Use `build_cell_ecm_graph` when cells and patches already share one object and
you want the NetworkX graph for plotting or topology. Use
`compose_cell_ecm_graph` when they are separate — for instance cells from
segmentation and patches from a different modality — or when you want a purely
sparse representation. They are not interchangeable; pick one per analysis.

## Next

- [Exporting to AnnData](export-anndata.md)
- [Exporting to PyTorch Geometric](export-pytorch-geometric.md)
