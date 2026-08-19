# The cell–ECM graph

Spatial proteomics technologies such as imaging mass cytometry (IMC) and CODEX
profile cells and the extracellular matrix (ECM) together in intact tissue, yet
most analysis tools are cell-centric and treat the ECM as unstructured
background. Mantpy represents a tissue image as a single **heterogeneous spatial
graph** with two node types — **cells** and **ECM patches** — so the matrix can
be analysed jointly with, or independently of, the cells.

## Nodes

- **Cell nodes** come from single-cell segmentation. They are featurised by
  marker intensity and clustered into cell-type phenotypes.
- **ECM nodes** are non-overlapping, non-background **patches** that tile the
  image. They are featurised by ECM-marker intensity and clustered into **ECM
  phenotypes** (analogous to cell types). The representation works from a single
  ECM marker upward, scaling to richly multiplexed cell–ECM panels.

## Edges and subgraphs

Edges connect any pair of node types using **k-nearest-neighbour**, **radius**,
or **Delaunay-triangulation** rules — configurable per relation type. The three
relation types (cell–cell, ECM–ECM, cell–ECM) decompose into three subgraphs
that you can analyse separately or together:

| `mode` | Graph | Captures |
|--------|-------|----------|
| `"cell"` | cell–cell | cellular organisation |
| `"ecm"` | ECM–ECM | matrix structure |
| `"cell_ecm"` | bipartite cell–ECM (merged with the cell and ECM layers) | the cell–matrix interface |

All three are built through one entry point:

```python
import mantpy as mt

mt.gr.build_graph(adata, mode="cell")
mt.gr.build_graph(adata, mode="ecm")
mt.gr.build_graph(adata, mode="cell_ecm")
```

Any graph exports directly to a PyTorch Geometric `Data`/`HeteroData` object via
{func}`mantpy.gr.to_pyg` for use with custom message-passing models.

## What you can do with it

The cell–ECM graph supports three complementary classes of analysis:

1. **Joint cell–ECM analysis** — test cell–matrix associations
   ({func}`mantpy.tl.cell_ecm_enrichment`) and reconstruct matrix labels from
   the surrounding cellular context.
2. **Cell-independent ECM remodelling** — quantify how matrix states are
   arranged relative to one another and how that changes in disease.
3. **Label-free tissue domains** — recover histological domains from a single
   matrix channel, without supervision
   ({func}`mantpy.tl.select_n_domains`).

Worked examples are maintained in the
[Mantpy reproducibility tutorials](https://github.com/moeghaf/mantpy_reproducibility).

## Interoperability

Mantpy is built on the [scverse](https://scverse.org) ecosystem. It uses
`AnnData` (and, optionally, `SpatialData`) as its data backbone, so it
interoperates with Scanpy, Squidpy and the wider single-cell and spatial-omics
tooling. Mantpy is an ECM-focused complement to squidpy.
