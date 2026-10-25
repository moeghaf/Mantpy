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

Undirected edges produce both message-passing directions. Within-type relations
contain reciprocal indices; cross edges appear in separate `cell-ecm` and
`ecm-cell` relations. If a legacy graph carries directional edge features,
each index column has its corresponding feature row. A relation may have no
features, but incomplete feature rows raise an error. Set
`include_edge_features=False` when only connectivity is needed.

New graph builders record the `feat_source` and `feat_target` endpoints of each
`feat_fwd` vector. This keeps directional features aligned even when NetworkX
nodes are reordered. Older cross edges use the documented cell-to-ECM forward
convention. For older within-type edges without endpoint metadata, the exporter
assumes the NetworkX iteration order and warns when forward and reverse values
differ. Rebuild these graphs before relying on their directional features.

For most cell–ECM questions this is the better starting point. Reach for the
homogeneous export when a model or baseline specifically requires one node type.

## Mapping predictions back to their source

`node_ids` records the NetworkX keys, and `source_ids` records the source
observation or patch index saved by the graph builder. Both are strings in
tensor-row order. When an externally constructed node has no `source_id`,
its graph key is used instead. Heterogeneous exports keep the mappings on each
node store:

```{code-cell} python
list(zip(hetero["ecm"].node_ids[:3], hetero["ecm"].source_ids[:3]))
```

Observation-native exports use observation names as `node_ids`, retaining
`source_obs_name` as `source_ids` when available. Their node features come from
the selected `X`/`obsm` matrix. Sparse joint graphs currently export connectivity,
node features and positions without edge-feature tensors.

The exports retain `graph_key` and `sample_id` when present; heterogeneous
exports also record `graph_format`. Homogeneous exports support the explicit
numeric graph label below and the legacy `islet_graph_feat` vector as
`graph_feat`. Arbitrary graph attributes and `AnnData.uns` entries are not
automatically serialized into PyG.

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
