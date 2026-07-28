"""Lightweight anomaly scorers used by ECM-cluster denoising workflows.

The public neighbour-composition model and the internal label-frequency
reference share the duck-typed contract consumed by
:func:`mantpy.tl.loo_denoise_evaluation`: ``fit(pristine_pairs, ...)`` and
``score_anomaly(cell_adata, ecm_adata, ...)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from mantpy._constants import (
    CELL_ECM_GRAPH_KEY,
    ECM_PATCHES_KEY,
    EDGE_TYPE_CE,
    EDGE_TYPE_EE,
    NODE_TYPE_CELL,
    NODE_TYPE_ECM,
)

_EPS = 1e-9


def _roc_pr_from_anomaly(
    anomaly: np.ndarray,
    patches: pd.DataFrame,
    *,
    truth_col: str,
    cluster_key: str,
    signal_only: bool,
) -> dict[str, Any]:
    """Build the ROC/PR metrics dict from a per-patch anomaly score.

    Uses shared masking, degenerate-class handling, and return keys so all
    baseline results are computed on identical footing.
    """
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    if truth_col not in patches.columns:
        raise KeyError(
            f"ecm_adata.uns['ecm_patches'] missing required '{truth_col}' column (typically the artefact-overlay flag)."
        )

    anomaly = np.asarray(anomaly, dtype=np.float64)
    y_true_all = patches[truth_col].astype(bool).to_numpy()
    if signal_only:
        mask = patches[cluster_key].astype(int).to_numpy() >= 0
    else:
        mask = np.ones_like(y_true_all, dtype=bool)

    y_true = y_true_all[mask].astype(int)
    y_score = anomaly[mask]
    scores = pd.DataFrame({"anomaly_score": anomaly.astype(np.float32)})

    if y_true.size == 0 or y_true.sum() == 0 or y_true.sum() == y_true.size:
        empty = np.array([], dtype=np.float64)
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "fpr": empty,
            "tpr": empty,
            "thresholds_roc": empty,
            "precision": empty,
            "recall": empty,
            "thresholds_pr": empty,
            "scores": scores,
            "y_true": y_true,
            "y_score": y_score,
        }

    fpr, tpr, thr_roc = roc_curve(y_true, y_score)
    prec, rec, thr_pr = precision_recall_curve(y_true, y_score)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds_roc": thr_roc,
        "precision": prec,
        "recall": rec,
        "thresholds_pr": thr_pr,
        "scores": scores,
        "y_true": y_true,
        "y_score": y_score,
    }


def _patch_index(node: str) -> int:
    """Return the positional index encoded in a graph node such as ``'ecm_7'``."""
    return int(str(node).split("_")[1])


def _row_norm(comp: np.ndarray) -> np.ndarray:
    """L1-normalise each row of a composition matrix (zero rows left at zero)."""
    row = comp.sum(axis=1, keepdims=True)
    return comp / np.where(row > 0, row, 1.0)


class _BaselineScorer:
    """Shared key-handling + ``device`` attribute for the duck-typed contract."""

    def __init__(self) -> None:
        self._cluster_key = "ecm_cluster"
        self._celltype_key = "cell_type"
        self._graph_key = CELL_ECM_GRAPH_KEY
        self.device = "cpu"

    def _store_keys(self, cluster_key: str, celltype_key: str, graph_key: str) -> None:
        self._cluster_key = cluster_key
        self._celltype_key = celltype_key
        self._graph_key = graph_key


class PriorFrequencyBaseline(_BaselineScorer):
    """Score anomaly as ``-log`` global frequency of the observed cluster label.

    This internal reference model knows nothing about the graph, only how rare
    each ECM cluster is across the training cohort.
    """

    def __init__(self, **_: Any) -> None:
        super().__init__()
        self._logp: np.ndarray | None = None

    def fit(
        self,
        pristine_pairs: list[tuple[AnnData, AnnData]],
        *,
        cluster_key: str = "ecm_cluster",
        celltype_key: str = "cell_type",
        graph_key: str = CELL_ECM_GRAPH_KEY,
        **_: Any,
    ) -> PriorFrequencyBaseline:
        self._store_keys(cluster_key, celltype_key, graph_key)
        labels: list[np.ndarray] = []
        for _cell, ecm in pristine_pairs:
            cl = ecm.uns[ECM_PATCHES_KEY][cluster_key].astype(int).to_numpy()
            labels.append(cl[cl >= 0])
        all_cl = np.concatenate(labels) if labels else np.empty(0, dtype=int)
        if all_cl.size == 0:
            raise ValueError("No signal ECM cluster labels in pristine_pairs.")
        K = int(all_cl.max()) + 1
        freq = np.bincount(all_cl, minlength=K).astype(np.float64)
        freq = freq / freq.sum()
        self._logp = -np.log(freq + _EPS)
        return self

    def score_anomaly(
        self,
        cell_adata: AnnData,
        ecm_adata: AnnData,
        *,
        truth_col: str = "is_artifact",
        signal_only: bool = True,
    ) -> dict[str, Any]:
        if self._logp is None:
            raise RuntimeError("PriorFrequencyBaseline is not trained — call .fit().")
        patches = ecm_adata.uns[ECM_PATCHES_KEY]
        cl = patches[self._cluster_key].astype(int).to_numpy()
        anom = np.zeros(cl.size, dtype=np.float64)
        valid = (cl >= 0) & (cl < self._logp.size)
        anom[valid] = self._logp[cl[valid]]
        return _roc_pr_from_anomaly(
            anom,
            patches,
            truth_col=truth_col,
            cluster_key=self._cluster_key,
            signal_only=signal_only,
        )


class NeighbourCompositionBaseline(_BaselineScorer):
    """Logistic regression predicting an ECM patch's cluster from its neighbours' composition.

    ``context`` selects which neighbours it sees.
    This single class is the apples-to-apples ablation that isolates the value
    of cross-compartment (cell<->ECM) information. All variants share the SAME
    model (multinomial logistic regression) and differ ONLY in their features,
    so any gap is attributable to information content, not model capacity:

    - ``context='cell'``  : neighbouring cell-type composition (cell->ECM edges).
    - ``context='ecm'``   : neighbouring ECM-cluster composition (ECM-ECM edges).
    - ``context='joint'`` : both concatenated.

    anomaly = ``-log P(observed label | neighbour composition)``. The
    ``'cell'`` / ``'joint'`` variants add cross-compartment information to the
    ECM-neighbour context.
    """

    def __init__(self, *, context: str = "cell", max_iter: int = 1000, **_: Any) -> None:
        super().__init__()
        if context not in ("cell", "ecm", "joint"):
            raise ValueError(f"context must be 'cell', 'ecm' or 'joint'; got {context!r}.")
        # max_iter: LogisticRegression solver cap, 1000 (above scikit-learn's
        # default 100) so the multinomial fit converges on the dense, often
        # collinear composition features without a convergence warning.
        self.context = context
        self.max_iter = max_iter
        self._clf: Any = None
        self._cell_types: list[str] = []
        self._K: int | None = None

    def _cell_comp(self, G: Any, cell_adata: AnnData, n_ecm: int) -> np.ndarray:
        ct = cell_adata.obs[self._celltype_key].astype(str).to_numpy()
        ct_to_i = {t: i for i, t in enumerate(self._cell_types)}
        comp = np.zeros((n_ecm, len(self._cell_types)), dtype=np.float64)
        for u, v, attrs in G.edges(data=True):
            if attrs.get("edge_type") != EDGE_TYPE_CE:
                continue
            ru, rv = G.nodes[u].get("node_type"), G.nodes[v].get("node_type")
            if ru == NODE_TYPE_CELL and rv == NODE_TYPE_ECM:
                ci, ei = _patch_index(u), _patch_index(v)
            elif rv == NODE_TYPE_CELL and ru == NODE_TYPE_ECM:
                ci, ei = _patch_index(v), _patch_index(u)
            else:
                continue
            if 0 <= ei < n_ecm and 0 <= ci < ct.size and ct[ci] in ct_to_i:
                comp[ei, ct_to_i[ct[ci]]] += 1.0
        return _row_norm(comp)

    def _ecm_comp(self, G: Any, labels: np.ndarray, n_ecm: int) -> np.ndarray:
        K = int(self._K)
        comp = np.zeros((n_ecm, K), dtype=np.float64)
        for u, v, attrs in G.edges(data=True):
            if attrs.get("edge_type") != EDGE_TYPE_EE:
                continue
            ui, vi = _patch_index(u), _patch_index(v)
            if not (0 <= ui < n_ecm and 0 <= vi < n_ecm):
                continue
            # Each endpoint contributes its (possibly corrupted) label to the
            # other's neighbour composition; self is excluded (no self-loops).
            for src, dst in ((ui, vi), (vi, ui)):
                cl = int(labels[dst])
                if 0 <= cl < K:
                    comp[src, cl] += 1.0
        return _row_norm(comp)

    def _features(self, cell_adata: AnnData, ecm_adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
        patches = ecm_adata.uns[ECM_PATCHES_KEY]
        labels = patches[self._cluster_key].astype(int).to_numpy()
        n_ecm = labels.size
        G = cell_adata.uns[self._graph_key]
        cols: list[np.ndarray] = []
        if self.context in ("cell", "joint"):
            cols.append(self._cell_comp(G, cell_adata, n_ecm))
        if self.context in ("ecm", "joint"):
            cols.append(self._ecm_comp(G, labels, n_ecm))
        X = np.hstack(cols) if len(cols) > 1 else cols[0]
        return X, labels

    def fit(
        self,
        pristine_pairs: list[tuple[AnnData, AnnData]],
        *,
        cluster_key: str = "ecm_cluster",
        celltype_key: str = "cell_type",
        graph_key: str = CELL_ECM_GRAPH_KEY,
        **_: Any,
    ) -> NeighbourCompositionBaseline:
        from sklearn.linear_model import LogisticRegression

        self._store_keys(cluster_key, celltype_key, graph_key)
        vocab: set[str] = set()
        max_cl = -1
        for cell, ecm in pristine_pairs:
            vocab.update(cell.obs[celltype_key].astype(str).to_numpy().tolist())
            cl = ecm.uns[ECM_PATCHES_KEY][cluster_key].astype(int).to_numpy()
            if cl.size:
                max_cl = max(max_cl, int(cl.max()))
        self._cell_types = sorted(vocab)
        self._K = max_cl + 1

        feats: list[np.ndarray] = []
        labs: list[np.ndarray] = []
        for cell, ecm in pristine_pairs:
            f, lab = self._features(cell, ecm)
            sig = lab >= 0
            feats.append(f[sig])
            labs.append(lab[sig])
        X = np.vstack(feats)
        y = np.concatenate(labs)
        if np.unique(y).size < 2:
            raise ValueError("NeighbourCompositionBaseline needs >= 2 signal clusters to train.")
        self._clf = LogisticRegression(max_iter=self.max_iter).fit(X, y)
        return self

    def score(self, cell_adata: AnnData, ecm_adata: AnnData) -> pd.DataFrame:
        """Return per-patch outputs for the denoising scorer contract.

        The baseline is a drop-in for :func:`mantpy.tl.denoise_ecm_clusters` and
        related denoising diagnostics.
        Columns: ``anomaly_score`` (0–1, ``-log p_actual`` min-max normalised over
        signal patches; background patches 0), ``denoised_cluster`` (argmax over
        signal clusters; background patches keep their label), ``p_actual``
        (``P(observed label | neighbour composition)``), ``p_pred`` (max class
        probability).
        """
        if self._clf is None:
            raise RuntimeError("NeighbourCompositionBaseline is not trained — call .fit().")
        f, lab = self._features(cell_adata, ecm_adata)
        proba = self._clf.predict_proba(f)
        classes = np.asarray([int(c) for c in self._clf.classes_], dtype=int)
        col_of = {int(c): i for i, c in enumerate(classes)}
        n = int(lab.size)
        sig = lab >= 0
        p_actual = np.full(n, _EPS, dtype=np.float64)
        for j in range(n):
            c = int(lab[j])
            if c in col_of:
                p_actual[j] = proba[j, col_of[c]]
        # Denoised = argmax class for signal patches; background keeps its label.
        denoised = lab.astype(int).copy()
        if sig.any():
            denoised[sig] = classes[proba.argmax(axis=1)[sig]]
        p_pred = proba.max(axis=1)
        # Anomaly = -log p_actual, min-max normalised within ROI over signal
        # patches; background patches stay at 0.
        anomaly = np.zeros(n, dtype=np.float64)
        if sig.any():
            neglog = -np.log(np.maximum(p_actual[sig], _EPS))
            lo, hi = float(neglog.min()), float(neglog.max())
            anomaly[sig] = (neglog - lo) / (hi - lo) if hi > lo else 0.0
        return pd.DataFrame(
            {
                "anomaly_score": anomaly,
                "denoised_cluster": denoised,
                "p_actual": p_actual,
                "p_pred": p_pred,
            }
        )

    def score_anomaly(
        self,
        cell_adata: AnnData,
        ecm_adata: AnnData,
        *,
        truth_col: str = "is_artifact",
        signal_only: bool = True,
    ) -> dict[str, Any]:
        if self._clf is None:
            raise RuntimeError("NeighbourCompositionBaseline is not trained — call .fit().")
        patches = ecm_adata.uns[ECM_PATCHES_KEY]
        f, lab = self._features(cell_adata, ecm_adata)
        proba = self._clf.predict_proba(f)
        col_of = {int(c): i for i, c in enumerate(self._clf.classes_)}
        anom = np.zeros(lab.size, dtype=np.float64)
        for j in range(lab.size):
            c = int(lab[j])
            p = proba[j, col_of[c]] if c in col_of else _EPS
            anom[j] = -np.log(max(p, _EPS))
        return _roc_pr_from_anomaly(
            anom,
            patches,
            truth_col=truth_col,
            cluster_key=self._cluster_key,
            signal_only=signal_only,
        )
