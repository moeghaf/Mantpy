"""Stable and transitional APIs required by the final tutorials."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import mantpy as mt


def _api_section(api_text: str, module_name: str) -> str:
    """Return one namespace's API section, excluding later namespaces."""
    heading = re.search(rf"^## .*\(`mantpy\.{re.escape(module_name)}`\)\s*$", api_text, flags=re.MULTILINE)
    assert heading is not None, f"docs/api.md has no section for mantpy.{module_name}"
    following_heading = re.search(r"^## ", api_text[heading.end() :], flags=re.MULTILINE)
    end = heading.end() + following_heading.start() if following_heading is not None else len(api_text)
    return api_text[heading.end() : end]


def _autosummary_entries(section: str, module_name: str) -> set[str]:
    """Collect entries from autosummary directives for the expected module."""
    assert f".. currentmodule:: mantpy.{module_name}" in section
    entries: set[str] = set()
    lines = section.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != ".. autosummary::":
            continue
        directive_indent = len(line) - len(line.lstrip())
        for candidate in lines[index + 1 :]:
            stripped = candidate.strip()
            if not stripped:
                continue
            indent = len(candidate) - len(candidate.lstrip())
            if indent <= directive_indent:
                break
            if not stripped.startswith(":") and re.fullmatch(r"[A-Za-z_]\w*", stripped):
                entries.add(stripped)
    return entries


def _inline_api_names(section: str, module_name: str) -> set[str]:
    """Collect explicitly formatted API names from one namespace section."""
    pattern = rf"`(?:mantpy\.{re.escape(module_name)}\.)?([A-Za-z_]\w*)`"
    return set(re.findall(pattern, section))


def test_palette_and_style_are_importable_public_modules() -> None:
    """Public convenience namespaces must also work as normal module imports."""
    assert importlib.import_module("mantpy.palette") is mt.palette
    assert importlib.import_module("mantpy.style") is mt.style


@pytest.mark.parametrize(
    "module_name", ["io", "im", "pp", "gr", "tl", "pl", "ds", "datasets", "fetch", "palette", "style", "nn"]
)
def test_every_export_is_documented(module_name: str) -> None:
    """Every export must be documented inside its namespace's own API section."""
    api_text = (Path(__file__).parents[1] / "docs" / "api.md").read_text(encoding="utf-8")
    module = importlib.import_module(f"mantpy.{module_name}")
    section = _api_section(api_text, module_name)
    autosummary = _autosummary_entries(section, module_name)
    inline_names = _inline_api_names(section, module_name)
    missing = [
        name
        for name in module.__all__
        if name not in autosummary and (callable(getattr(module, name)) or name not in inline_names)
    ]
    assert missing == []


REQUIRED_API = {
    "datasets": {
        "balbc_pbs_lung",
        "coliv_intestine",
        "schistosoma_ecm",
    },
    "fetch": {
        "fetch_matrisome",
        "load_balbc_pbs_lung",
        "load_coliv_intestine",
        "load_schistosoma_ecm_cohort",
    },
    "io": {
        "input_summary",
        "panel_summary",
        "read_codex",
        "read_ecm_image",
        "read_imc",
        "read_imc_folder",
        "sample_group_map",
        "to_spatialdata",
    },
    "im": {"ImageContainer", "as_image_container"},
    "pp": {
        "apply_ecm_label_overlay",
        "attach_ecm_patches",
        "cell_segmentation_summary",
        "cluster_ecm_patches",
        "compare_ecm_patches",
        "ecm_label_overlay_summary",
        "ecm_patch_summary",
        "ecm_patches_from_images",
        "image_ecm_patches",
        "normalize",
        "preprocess_ecm",
        "remove_background_patches",
        "select_ecm_cluster_count",
        "select_ecm_leiden_resolution",
    },
    "gr": {
        "build_cell_ecm_graphs",
        "build_ecm_graphs",
        "build_graph",
        "build_patch_graph",
        "compose_cell_ecm_graph",
        "extract_components_radius_reconnect",
        "joint_graph_summary",
        "to_pyg",
        "to_hetero_pyg",
    },
    "tl": {
        "ablation_roc_curves",
        "cell_ecm_contact",
        "cell_ecm_enrichment",
        "cell_ecm_enrichment_matrix",
        "cell_ecm_topology_sensitivity",
        "cluster_cooccurrence",
        "cluster_cooccurrence_permutation_test",
        "compute_cluster_composition",
        "cross_compartment_ablation",
        "denoise_ecm_clusters",
        "denoise_held_out_roi",
        "ecm_neighbour_label_agreement",
        "grouped_metric_summary",
        "lesion_size_by_sample",
        "lesion_topology_stats_df",
        "loo_denoise_evaluation",
        "loo_pristine_flag_rate",
        "loo_reconstruction_evaluation",
        "pick_representative_samples",
        "score_partition",
        "smooth_graph_signal",
        "summarize_largest_lesion",
        "summarize_pristine_flag_rate",
        "summarize_reconstruction_evaluation",
        "top_enriched_cluster",
        "transfer_spatial_features",
    },
    "nn": {
        "ECMClusterGraphBundle",
        "GraphMAE",
        "NeighbourCompositionBaseline",
        "NodeClassifier",
        "PatchEncoder",
        "build_ohe_cluster_graphs",
        "encode_graphmae",
        "encode_patches",
    },
    "pl": {
        "categorical_palette",
        "cell_ecm_enrichment_bars",
        "cell_ecm_enrichment_heatmap",
        "cell_ecm_enrichment_per_roi",
        "channel_overlay_on_neighbours",
        "classifier_roc",
        "cross_compartment_ablation_bars",
        "ecm_centroid_heatmap",
        "ecm_cluster_comparison",
        "graph_triptych",
        "niche_bubble",
        "node_value_overlay",
        "patch_domain_map",
        "show_image",
    },
    "ds": {"Bunch", "MantpyDataset"},
    "palette": {"factor_palette"},
    "style": {"apply_publication_style"},
}


def test_lazy_root_namespaces_are_listed_by_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as namespace_patch:
        namespace_patch.delitem(mt.__dict__, "nn", raising=False)
        namespace_patch.delitem(mt.__dict__, "pl", raising=False)

        assert {"nn", "pl"} <= set(dir(mt))


@pytest.mark.parametrize(
    ("namespace", "names"),
    [
        ("palette", {"factor_palette"}),
        ("gr", {"EDGE_FEATURE_REGISTRY", "exp_neg_dist"}),
    ],
)
def test_public_helpers_are_declared_in_all(namespace: str, names: set[str]) -> None:
    assert names <= set(getattr(mt, namespace).__all__)


@pytest.mark.parametrize(("namespace", "names"), REQUIRED_API.items())
def test_final_tutorial_api_is_exported(namespace: str, names: set[str]) -> None:
    module = getattr(mt, namespace)
    missing = sorted(name for name in names if not hasattr(module, name))
    assert not missing, f"mt.{namespace} is missing tutorial API: {missing}"


@pytest.mark.parametrize(
    ("namespace", "name"),
    [
        ("pp", "enhance_ecm"),
        ("tl", "select_exemplars"),
        ("tl", "score_clusters"),
        ("gr", "build_islet_ecm_graph"),
        ("io", "figures_dir"),
        ("fetch", "load_cell_ecm_graphs"),
        ("fetch", "load_fig6_data"),
        ("fetch", "load_cross_tissue_ecm_baseline"),
        ("fetch", "fetch_matrisome_db_tissue"),
        ("fetch", "merge_matrisome_with_tissue"),
        ("fetch", "load_spatial_overlay_data"),
        ("pp", "hotpixel_filter"),
        ("pp", "filter_ecm_patches"),
        ("gr", "cell_neighbor_enrichment"),
        ("gr", "cell_centrality"),
        ("gr", "cell_co_occurrence"),
        ("gr", "ecm_neighbor_enrichment"),
        ("gr", "ecm_graph_from_image"),
        ("gr", "graphs_from_adatas"),
        ("gr", "build_sub_ecm_graph"),
        ("gr", "drop_boundary_patches"),
        ("gr", "prune_long_edges"),
        ("gr", "prune_sparse_nodes"),
        ("gr", "keep_largest_component"),
        ("gr", "add_topology_features"),
        ("gr", "compute_khop_reachability"),
        ("pl", "FigureBundle"),
        ("pl", "cell_ecm_robustness_figure"),
        ("pl", "ecm_reconstruction_figure"),
        ("pl", "joint_cell_ecm_figure"),
        ("pl", "schistosoma_combined_figure"),
        ("pl", "schistosoma_supplement_figure"),
        ("pl", "add_panel_letter"),
        ("pl", "contrast_text_color"),
        ("pl", "draw_banner_axis"),
        ("pl", "filled_label_axis"),
        ("pl", "panel_grid"),
        ("pl", "plot_centroid_heatmap"),
        ("pl", "save_fig"),
        ("pl", "side_banner"),
        ("tl", "KHOP_FEATURE_NAMES"),
        ("tl", "compute_khop_ball_extras"),
        ("tl", "compute_khop_ball_extras_cohort"),
        ("tl", "compute_khop_features"),
        ("tl", "compute_khop_features_cohort"),
        ("tl", "allocate_quotas"),
        ("tl", "build_donor_descriptor"),
        ("tl", "build_node_descriptor"),
        ("tl", "compute_group_centroid_similarity"),
        ("tl", "find_representative_donors"),
        ("tl", "hierarchical_reorder"),
        ("tl", "leiden_motifs"),
        ("tl", "motif_composition_by_group"),
        ("tl", "predict_motifs_cohort"),
        ("tl", "predict_motifs_knn"),
        ("tl", "stratified_subsample"),
        ("tl", "consensus_labels"),
        ("tl", "permutation_test"),
        ("tl", "ecm_niche_k_scan"),
        ("pl", "ecm_niche_k_scan"),
        ("pp", "filter_by_msf"),
        ("palette", "TISSUE"),
        ("palette", "NICHE"),
        ("palette", "TISSUE_CROSS_TISSUE_PUBLISHED"),
        ("palette", "TOL_BRIGHT_12"),
        ("palette", "SEQUENTIAL_IMPORTANCE"),
        ("palette", "CHANNEL_HABP"),
        ("palette", "cmap_intensity"),
        ("palette", "cmap_importance"),
        ("palette", "cmap_diverging"),
    ],
)
def test_obsolete_apis_are_removed(namespace: str, name: str) -> None:
    assert not hasattr(getattr(mt, namespace), name)


@pytest.mark.parametrize(
    "name",
    [
        "Bunch",
        "ColIVIntestineBunch",
        "CELL",
        "CONTROL",
        "DISEASE",
        "DIVERGING",
        "ECM",
        "ECM_CLUSTERS",
        "INTESTINE_LAYERS",
        "KO",
        "SEQUENTIAL_INTENSITY",
        "WT",
    ],
)
def test_namespaced_apis_are_not_reexported_at_package_root(name: str) -> None:
    assert not hasattr(mt, name)


@pytest.mark.parametrize(
    ("namespace", "name"),
    [
        ("pp", "build_ecm_patch_dataframe"),
        ("pp", "compute_features"),
        ("gr", "build_cell_nx_graph"),
        ("gr", "delaunay_edges"),
    ],
)
def test_low_level_helpers_are_not_declared_public(namespace: str, name: str) -> None:
    assert name not in getattr(mt, namespace).__all__
