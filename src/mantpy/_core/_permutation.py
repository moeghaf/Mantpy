"""Permutation test for cell-ECM interaction significance.

Internal module — not part of the public API.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from mantpy._constants import NODE_TYPE_CELL, NODE_TYPE_ECM


def count_interactions(
    G: nx.Graph,
) -> pd.DataFrame:
    """Count observed interactions per (cell_type, ecm_cluster) pair.

    Parameters
    ----------
    G
        Heterogeneous cell-ECM graph.  Nodes must have ``node_type``,
        ``cell_type`` (for cell nodes), and ``ecm_cluster`` (for ECM nodes).

    Returns
    -------
    DataFrame indexed by cell_type, columns are ecm_cluster strings.
    """
    cell_nodes = {n: d for n, d in G.nodes(data=True) if d.get("node_type") == NODE_TYPE_CELL}
    ecm_nodes = {n: d for n, d in G.nodes(data=True) if d.get("node_type") == NODE_TYPE_ECM}

    if not cell_nodes or not ecm_nodes:
        return pd.DataFrame()

    cell_types = sorted({d["cell_type"] for d in cell_nodes.values()})
    ecm_clusters = sorted({str(d["ecm_cluster"]) for d in ecm_nodes.values()})

    counts = pd.DataFrame(0, index=cell_types, columns=ecm_clusters, dtype=int)

    for n1, n2 in G.edges():
        d1 = G.nodes[n1]
        d2 = G.nodes[n2]

        if d1.get("node_type") == NODE_TYPE_CELL and d2.get("node_type") == NODE_TYPE_ECM:
            ct = d1["cell_type"]
            ec = str(d2["ecm_cluster"])
        elif d2.get("node_type") == NODE_TYPE_CELL and d1.get("node_type") == NODE_TYPE_ECM:
            ct = d2["cell_type"]
            ec = str(d1["ecm_cluster"])
        else:
            continue

        if ct in counts.index and ec in counts.columns:
            counts.loc[ct, ec] += 1

    return counts


def permutation_test(
    G: nx.Graph,
    n_iter: int = 1000,
    p: float = 0.05,
    n_jobs: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Test whether cell-ECM interactions are significantly enriched or depleted.

    For each (cell_type, ecm_cluster) pair, builds a null distribution by
    permuting ECM cluster labels ``n_iter`` times, then computes empirical
    p-values.

    Parameters
    ----------
    G
        Heterogeneous cell-ECM graph.
    n_iter
        Number of permutations.
    p
        Significance threshold.
    n_jobs
        Parallel jobs for permutation loop (``-1`` = all cores).
    seed
        Seed for the label permutations. Pass an integer for a deterministic
        null; ``None`` draws fresh entropy from the operating system and is not
        reproducible. Each permutation gets an independent stream derived from
        this seed, so results do not depend on ``n_jobs``.

    Returns
    -------
    DataFrame with same shape as ``count_interactions(G)``.
    Values: +1 (significant enrichment), -1 (significant avoidance), 0 (ns).
    """
    from joblib import Parallel, delayed

    observed = count_interactions(G)
    if observed.empty:
        return observed

    ecm_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == NODE_TYPE_ECM]
    ecm_clusters = np.array([G.nodes[n]["ecm_cluster"] for n in ecm_nodes])

    # Independent, reproducible streams per permutation. Deriving them up front
    # (rather than letting each worker touch global state) is what makes the
    # result invariant to n_jobs and to the scheduling order joblib happens to use.
    streams = np.random.SeedSequence(seed).spawn(n_iter)

    null_counts = Parallel(n_jobs=n_jobs)(
        delayed(_one_permutation)(G, ecm_nodes, ecm_clusters, observed.index, observed.columns, ss)
        for ss in streams
    )

    null_stack = np.stack(null_counts, axis=0)  # (n_iter, n_ct, n_ec)

    obs_arr = observed.values

    p_enriched = (null_stack >= obs_arr[np.newaxis]).mean(axis=0)
    p_depleted = (null_stack <= obs_arr[np.newaxis]).mean(axis=0)

    sigval = pd.DataFrame(0, index=observed.index, columns=observed.columns, dtype=int)
    sigval[p_enriched < p] = 1
    sigval[p_depleted < p] = -1

    return sigval


def _one_permutation(
    G: nx.Graph,
    ecm_nodes: list,
    ecm_clusters: np.ndarray,
    cell_types: pd.Index,
    ecm_cluster_cols: pd.Index,
    seed_seq: np.random.SeedSequence,
) -> np.ndarray:
    perm = ecm_clusters.copy()
    np.random.default_rng(seed_seq).shuffle(perm)

    for node, cluster in zip(ecm_nodes, perm, strict=False):
        G.nodes[node]["_perm_cluster"] = int(cluster)

    counts = np.zeros((len(cell_types), len(ecm_cluster_cols)), dtype=int)
    ct_idx = {ct: i for i, ct in enumerate(cell_types)}
    ec_idx = {ec: i for i, ec in enumerate(ecm_cluster_cols)}

    for n1, n2 in G.edges():
        d1, d2 = G.nodes[n1], G.nodes[n2]
        if d1.get("node_type") == NODE_TYPE_CELL and d2.get("node_type") == NODE_TYPE_ECM:
            ct, ec = d1["cell_type"], str(d2.get("_perm_cluster", d2["ecm_cluster"]))
        elif d2.get("node_type") == NODE_TYPE_CELL and d1.get("node_type") == NODE_TYPE_ECM:
            ct, ec = d2["cell_type"], str(d1.get("_perm_cluster", d1["ecm_cluster"]))
        else:
            continue
        if ct in ct_idx and ec in ec_idx:
            counts[ct_idx[ct], ec_idx[ec]] += 1

    for node in ecm_nodes:
        G.nodes[node].pop("_perm_cluster", None)

    return counts
