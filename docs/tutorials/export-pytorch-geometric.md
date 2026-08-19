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

# Exporting to PyTorch Geometric

A cell–ECM graph is a graph, so it can be fed to a graph neural network
directly. {func}`~mantpy.gr.to_pyg` converts a Mantpy graph into a
{class}`torch_geometric.data.Data` object.

:::{note}
This page needs the `gnn` extra:

```bash
pip install "mantpy[gnn]"
```
:::

```{code-cell} python
import mantpy as mt

roi = mt.datasets.toy_ecm_roi()
adata = mt.io.read_imc(
    roi.image, panel=roi.panel, cells=roi.cells,
    sample_id="toy", condition="ctrl",
)
mt.pp.extract_ecm_patches(adata, roi.image, patch_size=8, ecm_K=3, features=["mean"])
mt.gr.build_cell_graph(adata, k=5)
mt.gr.build_ecm_graph(adata, k=5)
mt.gr.build_cell_ecm_graph(adata, k=5)
```

## A homogeneous graph

```{code-cell} python
data = mt.gr.to_pyg(adata, graph_key="cell_ecm_graph")
data
```

Cells and patches are both nodes, with node features built from whatever the
graph carries — position, ECM features, and any topology attributes present.

```{code-cell} python
print("node features :", tuple(data.x.shape))
print("edges         :", tuple(data.edge_index.shape))
print("positions     :", tuple(data.pos.shape) if data.pos is not None else None)
```

Either graph can be exported on its own by naming it:

```{code-cell} python
mt.gr.to_pyg(adata, graph_key="ecm_graph")
```

## Keeping cells and patches distinct

Flattening cells and ECM patches into one node type discards the thing that
makes the graph interesting. {func}`~mantpy.gr.to_hetero_pyg` keeps them
separate as a {class}`torch_geometric.data.HeteroData`:

```{code-cell} python
hetero = mt.gr.to_hetero_pyg(adata)
hetero
```

Node types and edge types are addressable, so a model can learn different
transforms for cell–cell, ECM–ECM and cell–ECM relations:

```{code-cell} python
print("node types:", hetero.node_types)
print("edge types:", hetero.edge_types)
```

For most cell–ECM questions this is the better starting point. Reach for the
homogeneous export when a model or baseline specifically requires one node type.

## Graph labels

`data.y` is only set when you ask for it:

```{code-cell} python
print("y set by default:", hasattr(data, "y") and data.y is not None)
```

Pass `label_key` to attach a per-graph label from `obs`. The column must hold a
single value, since it labels the graph rather than the nodes:

```{code-cell} python
adata.obs["condition_code"] = 0
labelled = mt.gr.to_pyg(adata, graph_key="cell_ecm_graph", label_key="condition_code")
int(labelled.y)
```

:::{warning}
`obs["y"]` is *not* used as a label. `read_imc` writes the y-centroid there, so
treating it as a class would label every graph with a truncated coordinate.
Name your label column explicitly.
:::

## Many ROIs at once

Real training sets have many regions. {class}`~mantpy.ds.MantpyDataset` wraps a
collection and hands back a list of `Data` objects ready for a PyG loader:

```{code-cell} python
import mantpy as mt

dataset = mt.MantpyDataset({"roi_a": adata, "roi_b": adata.copy()})
graphs = dataset.pyg_dataset()
len(graphs), graphs[0]
```

```{code-cell} python
from torch_geometric.loader import DataLoader

loader = DataLoader(graphs, batch_size=2, shuffle=False)
batch = next(iter(loader))
batch
```

From here it is ordinary PyTorch Geometric — nothing Mantpy-specific remains.
{mod}`mantpy.nn` also ships {class}`~mantpy.nn.GraphMAE` for self-supervised
node representations if you would rather not start from scratch.

## Next

- [Advanced: the full analyses](advanced.md) — the paper workflows on real
  cohorts
- The [API reference](../api.md) for everything not covered here
