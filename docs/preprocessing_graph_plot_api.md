# Image, graph and patch-map adapters

These interfaces preserve the image and graph choices already selected in an
analysis. They allow notebooks to use package APIs without silently rebuilding
their graph topology or changing patch identities.

## Fitted image normalization

```python
normalizer = mt.pp.ImageNormalizer(
    "pooled_arcsinh_percentile",
    cofactor=cofactor,
    percentile=percentile,
    channel_indices=ecm_indices,
    channel_names=marker_names,
).fit(training_images)
held_out_image = normalizer.transform(
    test_image, channel_names=marker_names,
)
```

Images are channel-first arrays or Mantpy `ImageContainer` objects. Fit uses
only the supplied training images; transform reuses their statistics.
`pooled_zscore` retains the incremental scikit-learn scaler calculation.
`pooled_arcsinh_percentile` pools transformed float32 pixels one channel at a
time, then clips the selected channels to their shared percentile bounds.
Other channels receive arcsinh without percentile scaling.
`none` returns the untransformed array.

Channel positions must be integer indices. Supplying marker names verifies
their order at transform time. Fitted statistics are available in
`provenance_`, and the fitted object can be serialized with pickle.

The existing `extract_ecm_patches_cohort` delegates its normalization to this
class. Its patch extraction options, ECM-marker mask and normalization
provenance remain the same. An independent comparison against the previous
cohort implementation found identical prepared arrays and provenance for
all three modes on a two-image fixture.

For a patch encoder that only needs tiles, skip image descriptors:

```python
patches = mt.pp.image_ecm_patches(
    image,
    patch_size=patch_size,
    feature_families=(),
    pixel_thresholds=pixel_thresholds,
    patch_thresholds=patch_thresholds,
)
patches.obsm["image_patches"]
```

This retains foreground selection, grid positions, original patch IDs and
tiles. The observation feature matrix has zero columns and the descriptor
backend does not run.

## Adopt existing graph connections

```python
joint = mt.gr.joint_graph_from_connectivities(
    cells, patches,
    cell_connectivities=cell_connections,
    ecm_connectivities=patch_connections,
    cell_ecm_connectivities=cell_patch_connections,
    cell_ecm_distances=cell_patch_distances,
)
```

Cell and patch matrices are square; cross connections have shape
`n_cells × n_patches`. Cross connections are mirrored, while direction and
weights within each compartment remain unchanged.
Distance values are optional. Their presence does not create edges, and
zero distance is valid for connected coincident nodes.

The result is an observation-native AnnData with typed sparse graph layers.
Cells precede patches, with namespaced observation IDs and original input IDs
in `obs["source_obs_name"]`. Feature spaces are not padded or combined.
The result can be saved to H5AD. Coordinates retain their supplied units;
when physical coordinates are absent, tensor export uses the original
spatial coordinates without inventing a conversion.

```python
graph = mt.gr.to_networkx(joint)
homogeneous = mt.gr.to_pyg(joint)
heterogeneous = mt.gr.to_hetero_pyg(joint)
```

`to_networkx` retains observation order, attributes and weights. Asymmetric
connections export as a directed graph by default.
`legacy_ids=True` provides compartment-local `cell_i` and `ecm_i` names
while retaining source IDs as attributes.
Tensor exports require the existing optional PyTorch Geometric dependency;
they perform no fitting.

`extract_components_existing` obtains cluster-induced components using
those supplied edges. Directed graphs use weak components. Radius
reconnection remains an explicit alternative through
`extract_components_radius_reconnect`.

## Render patches by their original grid positions

```python
ax = mt.pl.patch_map(
    patches, labels,
    palette=cluster_colors,
    extent=image_extent,
)
```

Labels can be an observation-column name, a positional vector or an indexed
Series. Series values align to original patch IDs. Explicit grid columns
take precedence over centroids; recorded full grid shape retains empty
boundary rows and columns. Missing categorical values and the background
label leave empty pixels.

Continuous values use a colormap, with optional `vmin`, `vmax` and colorbar.
An explicit palette selects categorical rendering unless overridden.
`extent` places the map in image or physical coordinates.
Cropping preserves that extent for either upper or lower image origin.
