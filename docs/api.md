# API

Public functions are documented through their module-qualified names, such as
`mantpy.pp.extract_ecm_patches` and `mantpy.gr.build_graph`.

## Input and output (`mantpy.io`)

Read multiplexed images and channel metadata into `AnnData`.

```{eval-rst}
.. currentmodule:: mantpy.io

.. autosummary::
    :toctree: generated

    InputSummary
    PanelSummary
    cell_features_from_mask
    input_summary
    panel_summary
    read_imc
    read_imc_folder
    read_codex
    read_ecm_image
    sample_group_map
    to_spatialdata
```

## Image containers (`mantpy.im`)

`mantpy.im.ImageContainer` provides named image layers and optional lazy
Dask/Zarr-backed access. AnnData stores the container payload as an H5AD-safe
mapping. Restore the object interface at the point of use:

```python
import mantpy as mt

image = mt.im.as_image_container(adata.uns["image_container"])
```

`mantpy.im.as_image_container` accepts both a serialized mapping read from H5AD
and an in-memory `mantpy.im.ImageContainer`.

The public serialization identifiers are `IMAGE_CONTAINER_SCHEMA` and
`IMAGE_CONTAINER_SCHEMA_VERSION`.

```{eval-rst}
.. currentmodule:: mantpy.im

.. autosummary::
    :toctree: generated

    ImageContainer
    as_image_container
```

## Preprocessing (`mantpy.pp`)

Image normalisation, ECM patch extraction, segmentation, and patch
phenotyping. `mantpy.pp.extract_ecm_patches` creates observation-native ECM
patches, while `mantpy.pp.normalize` provides channel-level transformations.

```{eval-rst}
.. currentmodule:: mantpy.pp

.. autosummary::
    :toctree: generated

    BackgroundRemovalSummary
    ClusterCountSelection
    CellSegmentationSummary
    ECMClusteringResult
    ECMLabelOverlaySummary
    ECMLeidenResolutionSelection
    ECMPatchSummary
    HEPreprocessingResult
    HEECMPatchSummary
    PatchComparison
    annotate_structure
    attach_ecm_patches
    apply_ecm_label_overlay
    cell_segmentation_summary
    cluster_ecm_patches
    compare_ecm_patches
    ecm_patches_from_images
    ecm_patch_summary
    ecm_label_overlay_summary
    extract_ecm_patches
    extract_ecm_patches_cohort
    extract_structure_ecm
    he_ecm_patches
    he_ecm_patch_summary
    image_ecm_patches
    normalize
    preprocess_ecm
    preprocess_he
    remove_background_patches
    segment_cells
    segment_cells_tiled
    select_ecm_cluster_count
    select_ecm_leiden_resolution
    split_structures
```

## Graphs (`mantpy.gr`)

Build cell, ECM, and joint cell-ECM graphs and export them to PyTorch
Geometric. `mantpy.gr.build_patch_graph` returns a dict-compatible
`PatchGraphResult` and stores its topology on the patch `AnnData`.
`EDGE_FEATURE_REGISTRY` describes the supported edge-feature names.

```{eval-rst}
.. currentmodule:: mantpy.gr

.. autosummary::
    :toctree: generated

    ECMGraphBuildResult
    GraphBuildResult
    JointGraphSummary
    PatchGraphResult
    build_cell_ecm_graph
    build_cell_ecm_graphs
    build_cell_graph
    build_ecm_graph
    build_ecm_graphs
    build_graph
    build_patch_graph
    compose_cell_ecm_graph
    ensure_cell_ecm_graph
    exp_neg_dist
    extract_components_radius_reconnect
    joint_graph_summary
    largest_component_radius_reconnect
    to_hetero_pyg
    to_pyg
```

## Analysis tools (`mantpy.tl`)

Statistical tests, spatial clustering, embeddings, and predictions.

```{eval-rst}
.. currentmodule:: mantpy.tl

.. autosummary::
    :toctree: generated

    AblationROCResult
    CentralVoidResult
    CellECMContactResult
    ClusterCompositionResult
    CooccurrencePermResult
    ECMNeighbourAgreementSummary
    GraphSmoothingResult
    HeldOutDenoiseResult
    LesionSummary
    PristineFlagSummary
    ReconstructionSummary
    SpatialTransferResult
    ablation_roc_curves
    cell_ecm_enrichment
    cell_ecm_enrichment_matrix
    cell_ecm_contact
    cell_ecm_topology_sensitivity
    cluster_coherence
    cluster_cooccurrence
    cluster_cooccurrence_permutation_test
    compute_cluster_composition
    cross_compartment_ablation
    denoise_ecm_clusters
    denoise_held_out_roi
    ecm_neighbour_label_agreement
    ecm_to_anndata
    graph_modularity
    grouped_metric_summary
    interaction_test
    lesion_central_void
    lesion_size_by_sample
    lesion_topology_stats
    lesion_topology_stats_df
    loo_denoise_evaluation
    loo_pristine_flag_rate
    loo_reconstruction_evaluation
    neighbourhood_clustering
    pick_representative_samples
    score_partition
    select_n_domains
    smooth_graph_signal
    summarize_largest_lesion
    summarize_pristine_flag_rate
    summarize_reconstruction_evaluation
    top_enriched_cluster
    transfer_spatial_features
```

## Plotting (`mantpy.pl`)

Visualise graphs, interactions, clusters, images, and learned representations.

```{eval-rst}
.. currentmodule:: mantpy.pl

.. autosummary::
    :toctree: generated

    categorical_palette
    cell_ecm_enrichment_bars
    cell_ecm_enrichment_heatmap
    cell_ecm_enrichment_per_roi
    cell_ecm_graph
    cell_graph
    channel_overlay_on_neighbours
    classifier_roc
    cross_compartment_ablation_bars
    ecm_centroid_table
    ecm_centroid_heatmap
    ecm_cluster_comparison
    ecm_resolution_selection
    ecm_graph
    ecm_graph_overlay
    ecm_image
    graph_triptych
    he_overview
    image_panel
    interaction_heatmap
    neighbourhood_clusters
    niche_bubble
    niche_bubble_table
    node_value_overlay
    patch_domain_map
    plot_cluster_map
    plot_delta_masked
    plot_lesion_metric_view
    plot_marker_otsu_composite
    plot_mean_composition
    show_image
```

## Plot style (`mantpy.style`)

Mantpy leaves Matplotlib settings unchanged on import. Apply the generic,
export-friendly preset explicitly when desired:

```python
import mantpy as mt

mt.style.apply_publication_style()
```

`mantpy.style.apply_publication_style(**overrides)` returns the exact
Matplotlib settings it applied; keyword arguments override individual values.

```{eval-rst}
.. currentmodule:: mantpy.style

.. autosummary::
    :toctree: generated

    apply_publication_style
```

## Colour palettes (`mantpy.palette`)

Semantic colours, graph-edge colours, categorical palettes, and the
factor-aware palette helper are available through `mt.palette` or the directly
importable `mantpy.palette` module. The supported data attributes are:

`CELL`, `ECM`, `NEUTRAL_DARK`, `NEUTRAL_MID`, `NEUTRAL_LIGHT`,
`NEUTRAL_LIGHTER`, `NEUTRAL_LIGHTEST`, `CONTROL`, `DISEASE`, `WT`, `KO`,
`INTESTINE_LAYERS`, `ECM_CLUSTERS`, `OKABE_ITO`, `TOL_BRIGHT_7_CUSTOM`,
`TRUBETSKOY_12`, `OKABE_ITO_4DISTINCT`, `CLUSTER_BG_COLOR`,
`CONDITION_HUE_PALETTE`, `GENOTYPE_LUMINANCE_PALETTE`,
`SEQUENTIAL_INTENSITY`, `DIVERGING`, `ROC_FOLD`, `ROC_MEAN`,
`ECM_PALETTE_LUNG_PUBLISHED`, `EDGE_CELL_CELL`, `EDGE_ECM_ECM`,
`EDGE_CELL_ECM`, `CELL_PALETTE_LUNG_EXT`, and `PLACEHOLDER_GREY`.

```{eval-rst}
.. currentmodule:: mantpy.palette

.. autosummary::
    :toctree: generated

    factor_palette
```

## Dataset containers (`mantpy.ds`)

Reusable containers for related `AnnData` objects, graphs, and metadata.

```{eval-rst}
.. currentmodule:: mantpy.ds

.. autosummary::
    :toctree: generated

    Bunch
    ColIVIntestineBunch
    LungBunch
    MantpyDataset
```

## Public data access (`mantpy.fetch`)

Compatibility loaders and public upstream resources. New code should use the
one-line `mantpy.datasets` loaders below; the `load_*` functions retain explicit
local-file workflows used by earlier scripts.

```{eval-rst}
.. currentmodule:: mantpy.fetch

.. autosummary::
    :toctree: generated

    fetch_matrisome
    load_balbc_pbs_lung
    load_coliv_intestine
    load_schistosoma_ecm_cohort
```

## Tutorial datasets (`mantpy.datasets`)

Download the complete verified inputs for each public tutorial. Archives are
stored outside the wheel, checked against the immutable record and inner file
manifests, and cached under `~/.cache/mantpy` by default.

```python
import mantpy as mt

data = mt.datasets.coliv_intestine()
data = mt.datasets.balbc_pbs_lung()
data = mt.datasets.schistosoma_ecm()
```

Each loader accepts `cache_dir=...`; `MANTPY_CACHE` is used when that argument
is omitted. Every result includes `paths`, sanitized `provenance`, and a
`quickstart` mapping of verified optional intermediates. The Collagen-IV
quick-start keys are `analysis`, `external_labels`, and `external_metadata`
when those assets are present.

`toy_ecm_roi` is the exception: it synthesises a small ROI in memory, so it
needs no download, no cache and no network. Use it to learn the API — its
cluster labels and statistics describe the generator, not biology.

```{eval-rst}
.. currentmodule:: mantpy.datasets

.. autosummary::
    :toctree: generated

    balbc_pbs_lung
    coliv_intestine
    schistosoma_ecm
    toy_ecm_roi
```

## Neural networks (`mantpy.nn`)

Optional classifiers and self-supervised models.
`mantpy.nn.GraphMAE` learns node-level representations, and
`mantpy.nn.PatchEncoder` learns image-patch representations. Install the
relevant optional extra before using these APIs.

```{eval-rst}
.. currentmodule:: mantpy.nn

.. autosummary::
    :toctree: generated

    ECMClusterGraphBundle
    GraphMAE
    NeighbourCompositionBaseline
    NodeClassifier
    PatchEncoder
    build_ohe_cluster_graphs
    encode_graphmae
    encode_patches
```
