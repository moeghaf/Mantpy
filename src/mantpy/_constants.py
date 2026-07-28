CELLTYPE_COL = "cell_type"
X_COL = "x"
Y_COL = "y"
SAMPLE_ID_COL = "sample_id"
CONDITION_COL = "condition"

SPATIAL_KEY = "spatial"
RAW_LAYER = "raw"

CELL_GRAPH_KEY = "cell_graph"
ECM_GRAPH_KEY = "ecm_graph"
CELL_ECM_GRAPH_KEY = "cell_ecm_graph"

#: squidpy's canonical connectivity slot (``squidpy._constants._pkg_constants
#: .Key.obsp.spatial_conn``). Every ``sq.gr.*`` function defaults to it, so a
#: graph published here is visible to the whole squidpy toolchain.
SQUIDPY_CONN_KEY = "spatial_connectivities"

ECM_PATCHES_KEY = "ecm_patches"
ECM_CLUSTER_COL = "ecm_cluster"
ECM_IMAGE_KEY = "ecm_image"
IMAGE_CONTAINER_KEY = "image_container"

INTERACTION_TEST_KEY = "interaction_test"
NEIGHBOURHOOD_CLUSTERS_KEY = "neighbourhood_clusters"

MANTPY_UNS_KEY = "mantpy"

NODE_TYPE_CELL = "cell"
NODE_TYPE_ECM = "ecm"

EDGE_TYPE_CC = "cell-cell"
EDGE_TYPE_EE = "ecm-ecm"
EDGE_TYPE_CE = "cell-ecm"
