"""GraphSAGE node classifier for ECM-cluster graphs.

Use-case: attribute *which ECM clusters and which spatial patches* separate
experimental groups, when the only node feature is the one-hot ECM-cluster
label.  Each node inherits its ROI's group label; a shallow GraphSAGE predicts
that label per node from its local ECM-cluster neighbourhood.

Why node- rather than graph-level: on a mean-pooled graph classifier every node
contributes ~1/N to the pooled embedding, so per-node integrated-gradients
attributions are diffuse and do not localise.  A node classifier attributes each
node's own logit directly to its own features, so a discriminative rare cluster
(e.g. an infection-induced fibrillar niche) lights up cleanly both in the
per-node class probability and in integrated gradients.

The honest unit of replication is the biological group supplied to ``fit``
(for example, a mouse), or otherwise the ROI--never the node. Evaluation keeps
those units disjoint between folds and reports unit-level performance alongside
the descriptive node-level score.

Requires the ``[gnn]`` extra: ``pip install "mantpy[gnn]"``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import SAGEConv
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError('NodeClassifier requires PyTorch Geometric. Install with: pip install "mantpy[gnn]"') from e

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from mantpy._constants import ECM_CLUSTER_COL, ECM_GRAPH_KEY
from mantpy.ds import Bunch

_log = logging.getLogger(__name__)


def _stratified_group_folds(
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    n_splits: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic class-balanced folds with disjoint groups.

    Groups are sorted within each class and assigned round-robin to test folds.
    Every group must have exactly one label, and every class must contribute at
    least ``n_splits`` groups. This explicit construction is useful for small
    biological cohorts where each animal contributes multiple ROIs.
    """
    labels_array = np.asarray(labels, dtype=int)
    groups_array = np.asarray(groups, dtype=object)
    if labels_array.ndim != 1 or groups_array.ndim != 1:
        raise ValueError("labels and groups must be one-dimensional.")
    if labels_array.size != groups_array.size:
        raise ValueError("labels and groups must have the same length.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    group_labels: dict[str, int] = {}
    for label, group in zip(labels_array, groups_array, strict=True):
        key = str(group)
        previous = group_labels.setdefault(key, int(label))
        if previous != int(label):
            raise ValueError(f"Group {key!r} has more than one class label.")

    test_groups: list[set[str]] = [set() for _ in range(n_splits)]
    for label in sorted(set(group_labels.values())):
        class_groups = sorted(
            group for group, group_label in group_labels.items() if group_label == label
        )
        if len(class_groups) < n_splits:
            raise ValueError(
                f"Class {label} has {len(class_groups)} groups; "
                f"at least {n_splits} are required."
            )
        for index, group in enumerate(class_groups):
            test_groups[index % n_splits].add(group)

    all_indices = np.arange(labels_array.size, dtype=int)
    folds = []
    for fold_groups in test_groups:
        test_mask = np.isin(groups_array.astype(str), sorted(fold_groups))
        test_indices = all_indices[test_mask]
        train_indices = all_indices[~test_mask]
        if set(groups_array[train_indices]) & set(groups_array[test_indices]):
            raise RuntimeError("A group appears in both train and test indices.")
        folds.append((train_indices, test_indices))
    return folds


class ECMClusterGraphBundle(Bunch):
    """PyG graph cohort with a compact, notebook-friendly summary."""

    def __repr__(self) -> str:
        return "\n".join(
            [
                "ECM graph-learning cohort",
                f"  ROIs           {len(self.graphs)}",
                f"  classes        {len(self.class_names)}",
                f"  ECM states     {len(self.cluster_ids)}",
                "  node features  one-hot ECM state",
            ]
        )

    def _repr_html_(self) -> str:
        return f"<pre>{repr(self)}</pre>"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_ohe_cluster_graphs(
    adatas: Mapping[str, object],
    sample_meta: pd.DataFrame,
    *,
    group_col: str = "group",
    group_order: Sequence[str] | None = None,
    graph_key: str = ECM_GRAPH_KEY,
    cluster_attr: str = ECM_CLUSTER_COL,
    pos_attrs: tuple[str, str] = ("x", "y"),
    drop_background: bool = True,
    background: int = -1,
) -> Bunch:
    """Build per-ROI PyG graphs with one-hot ECM-cluster node features.

    For every ROI the NetworkX graph at ``adata.uns[graph_key]`` is converted to
    a :class:`torch_geometric.data.Data` whose node features are the one-hot
    encoding of the integer ``cluster_attr`` node attribute (encoded over the
    sorted set of cluster ids seen across the whole cohort, so every graph shares
    a column layout).  The undirected edges are emitted in both directions.

    Each graph carries:

    - ``x`` — ``(N, n_ohe)`` one-hot cluster features
    - ``edge_index`` — ``(2, 2E)``
    - ``pos`` — ``(N, 2)`` patch coordinates (for plotting; not a feature)
    - ``cluster`` — ``(N,)`` original integer cluster id per node
    - ``y`` — scalar group label (index into the returned ``class_names``)
    - ``roi_id`` — the ROI key

    Parameters
    ----------
    adatas
        Mapping ``{roi_id: AnnData}`` (e.g. ``data.adatas`` from a cohort loader).
    sample_meta
        Per-ROI metadata indexed by ROI id with a ``group_col`` column.
    group_col, group_order
        Column partitioning the cohort and the desired class order.  When
        ``group_order`` is ``None`` the sorted unique groups are used.
    graph_key, cluster_attr, pos_attrs
        Where to read the graph, cluster label, and coordinates.
    drop_background, background
        When ``drop_background`` (default), nodes whose ``cluster_attr`` equals
        ``background`` (default ``-1``) are removed entirely: they are not graph
        nodes, the one-hot is built over the ECM clusters only, and edges are the
        **induced subgraph** among the kept nodes (any edge touching a background
        node is dropped). Set ``drop_background=False`` to keep background as its
        own one-hot column.

    Returns
    -------
    Bunch with fields ``graphs`` (list[Data]), ``labels`` (np.ndarray),
    ``roi_ids`` (list[str]), ``class_names`` (list[str]),
    ``cluster_ids`` (list[int] — column meaning of the one-hot), and
    ``feature_names`` (list[str]).
    """
    classes = list(group_order) if group_order is not None else sorted(sample_meta[group_col].unique())
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    # Cohort-wide cluster vocabulary -> stable one-hot column layout (background excluded).
    cluster_vals: set[int] = set()
    for roi in sample_meta.index:
        G = adatas[roi].uns[graph_key]
        for _, d in G.nodes(data=True):
            c = int(d[cluster_attr])
            if drop_background and c == background:
                continue
            cluster_vals.add(c)
    cluster_ids = sorted(cluster_vals)
    col_of = {c: i for i, c in enumerate(cluster_ids)}
    n_ohe = len(cluster_ids)
    feature_names = [("bg" if c < 0 else f"ECM{c}") for c in cluster_ids]

    graphs, labels, roi_ids = [], [], []
    xk, yk = pos_attrs
    for roi in sample_meta.index:
        G = adatas[roi].uns[graph_key]
        nodes = list(G.nodes(data=True))
        pos_of = {n: i for i, (n, _) in enumerate(nodes)}
        cl_all = np.array([int(d[cluster_attr]) for _, d in nodes], dtype=int)
        keep = (cl_all != background) if drop_background else np.ones(len(nodes), dtype=bool)
        new_idx = -np.ones(len(nodes), dtype=np.int64)
        new_idx[np.where(keep)[0]] = np.arange(int(keep.sum()))

        cl = cl_all[keep]
        xs = np.array([float(d[xk]) for _, d in nodes], dtype=np.float32)[keep]
        ys = np.array([float(d[yk]) for _, d in nodes], dtype=np.float32)[keep]
        ohe = np.zeros((len(cl), n_ohe), dtype=np.float32)
        ohe[np.arange(len(cl)), [col_of[c] for c in cl]] = 1.0

        # induced subgraph: keep only edges with both endpoints kept, then relabel.
        ei = np.array([(pos_of[u], pos_of[v]) for u, v in G.edges()], dtype=np.int64)
        if ei.size:
            em = keep[ei[:, 0]] & keep[ei[:, 1]]
            e = ei[em]
            su, sv = new_idx[e[:, 0]], new_idx[e[:, 1]]
            src = np.concatenate([su, sv])
            dst = np.concatenate([sv, su])
        else:
            src = dst = np.zeros(0, dtype=np.int64)
        g = Data(
            x=torch.from_numpy(ohe),
            edge_index=torch.from_numpy(np.stack([src, dst])).long(),
            pos=torch.from_numpy(np.stack([xs, ys], axis=1)),
        )
        g.cluster = torch.from_numpy(cl).long()
        g.y = torch.tensor(cls_to_idx[sample_meta.loc[roi, group_col]], dtype=torch.long)
        g.roi_id = roi
        graphs.append(g)
        labels.append(int(g.y.item()))
        roi_ids.append(roi)

    return ECMClusterGraphBundle(
        graphs=graphs,
        labels=np.asarray(labels),
        roi_ids=roi_ids,
        class_names=list(classes),
        cluster_ids=cluster_ids,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class _NodeSAGE(nn.Module):
    def __init__(self, in_ch: int, hid: int = 64, n_cls: int = 2, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_ch, hid)
        self.convs = nn.ModuleList([SAGEConv(hid, hid) for _ in range(n_layers)])
        self.dropout = dropout
        self.head = nn.Linear(hid, n_cls)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.proj(x))
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)


@dataclass(frozen=True)
class _HeldOutFit:
    """One seed/fold model and its train-fold-only feature transform."""

    net: _NodeSAGE
    mu: torch.Tensor
    sd: torch.Tensor
    seed: int
    fold: int


class NodeClassifier:
    """Per-node GraphSAGE classifier with unit-disjoint cross-validation.

    Parameters
    ----------
    graphs
        List of PyG ``Data`` with one-hot node features ``x``, ``edge_index``,
        per-graph label ``y``, and (for attribution) per-node ``cluster`` /
        ``pos``. Build them with :func:`build_ohe_cluster_graphs`.
    labels
        Per-graph integer labels.  Defaults to ``[g.y for g in graphs]``.
    n_classes, hidden_dim, n_layers, dropout
        Architecture.  ``in_channels`` is inferred from ``graphs[0].x``.
    device
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto: cuda if available).  Use ``"cpu"``
        for bit-reproducible results — CUDA scatter/aggregation in GraphSAGE is
        non-deterministic even with a fixed seed, which matters on small cohorts.

    Examples
    --------
    >>> bundle = build_ohe_cluster_graphs(data.adatas, data.sample_meta)
    >>> clf = NodeClassifier(bundle.graphs, n_classes=len(bundle.class_names))
    >>> clf.fit(n_splits=4, epochs=120, seed=0)
    >>> clf.cv_metrics()["roi_macro_f1"]  # doctest: +SKIP
    >>> node_imp, feat_attr = clf.integrated_gradients(0, target_class=3)  # doctest: +SKIP
    """

    def __init__(
        self,
        graphs: list[Data],
        *,
        labels: np.ndarray | None = None,
        n_classes: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        device: str | None = None,
    ):
        self._graphs = graphs
        self._labels = np.asarray(labels if labels is not None else [int(g.y.item()) for g in graphs])
        self._in_channels = int(graphs[0].x.size(1))
        self._n_classes = n_classes
        self._arch = {
            "in_ch": self._in_channels,
            "hid": hidden_dim,
            "n_cls": n_classes,
            "n_layers": n_layers,
            "dropout": dropout,
        }
        self._device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._oof: list[np.ndarray] | None = None
        self._oof_by_seed: dict[int, list[np.ndarray]] = {}
        self._held_out_fits: dict[int, list[_HeldOutFit]] = {}
        self._groups: np.ndarray | None = None
        self._fold_ids: np.ndarray | None = None
        self._seeds: tuple[int, ...] = ()
        self._splits: list[tuple[np.ndarray, np.ndarray]] = []
        self._is_fit = False

    # ------------------------------------------------------------------
    def _standardize_stats(
        self,
        train_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        allx = torch.cat([self._graphs[index].x for index in train_indices], 0).numpy()
        mu, sd = allx.mean(0), allx.std(0)
        sd[sd == 0] = 1.0
        return (
            torch.tensor(mu, device=self._device),
            torch.tensor(sd, device=self._device),
        )

    def fit(
        self,
        *,
        n_splits: int = 4,
        epochs: int = 120,
        lr: float = 5e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        seeds: Sequence[int] | None = None,
        groups: Sequence[str] | None = None,
        splits: Sequence[tuple[Sequence[int], Sequence[int]]] | None = None,
        verbose: bool = False,
    ) -> NodeClassifier:
        """Train cross-validated models and store out-of-fold probabilities.

        By default, folds are stratified over graphs. Pass biological-unit IDs in
        ``groups`` for deterministic class-balanced group folds, or pass explicit
        ``splits``. Feature mean and standard deviation are fitted using training
        graphs separately in every fold and retained with that fold's model for
        held-out prediction and integrated gradients. ``seeds`` fits an ensemble
        over the same fixed folds and averages its out-of-fold probabilities.
        """
        if groups is not None and splits is not None:
            raise ValueError("Pass groups or splits, not both.")
        if groups is not None:
            groups_array = np.asarray(groups, dtype=object)
            if groups_array.shape != self._labels.shape:
                raise ValueError("groups must have one value per graph.")
            resolved_splits = _stratified_group_folds(
                self._labels,
                groups_array,
                n_splits=n_splits,
            )
            self._groups = groups_array.astype(str)
        elif splits is not None:
            resolved_splits = [
                (np.asarray(train, dtype=int), np.asarray(test, dtype=int))
                for train, test in splits
            ]
            self._groups = None
        else:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            resolved_splits = list(
                splitter.split(np.zeros(len(self._graphs)), self._labels)
            )
            self._groups = None
        self._validate_splits(resolved_splits)

        resolved_seeds = tuple(int(value) for value in (seeds or (seed,)))
        if not resolved_seeds:
            raise ValueError("At least one model seed is required.")
        if len(set(resolved_seeds)) != len(resolved_seeds):
            raise ValueError("Model seeds must be unique.")

        self._held_out_fits = {index: [] for index in range(len(self._graphs))}
        self._oof_by_seed = {}
        fold_ids = np.full(len(self._graphs), -1, dtype=int)
        for fold, (_, test_indices) in enumerate(resolved_splits, start=1):
            fold_ids[test_indices] = fold

        for model_seed in resolved_seeds:
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)
            seed_oof: list[np.ndarray | None] = [None] * len(self._graphs)
            for fold, (train_indices, test_indices) in enumerate(
                resolved_splits,
                start=1,
            ):
                mu, sd = self._standardize_stats(train_indices)
                net = _NodeSAGE(**self._arch).to(self._device)
                optimizer = torch.optim.Adam(
                    net.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                )
                batch = Batch.from_data_list(
                    [self._graphs[index] for index in train_indices]
                ).to(self._device)
                batch_x = (batch.x - mu) / sd
                batch_y = batch.y[batch.batch]
                counts = np.bincount(
                    self._labels[train_indices],
                    minlength=self._n_classes,
                ).astype(float)
                weights = torch.tensor(
                    counts.sum() / (counts + 1e-6),
                    dtype=torch.float32,
                    device=self._device,
                )
                weights = weights / weights.sum() * self._n_classes
                net.train()
                for _ in range(epochs):
                    optimizer.zero_grad()
                    loss = F.cross_entropy(
                        net(batch_x, batch.edge_index),
                        batch_y,
                        weight=weights,
                    )
                    loss.backward()
                    optimizer.step()
                net.eval()
                held_out_fit = _HeldOutFit(net, mu, sd, model_seed, fold)
                with torch.no_grad():
                    for index in test_indices:
                        graph = self._graphs[index]
                        graph_x = (graph.x.to(self._device) - mu) / sd
                        edge_index = graph.edge_index.to(self._device)
                        seed_oof[index] = (
                            torch.softmax(net(graph_x, edge_index), 1).cpu().numpy()
                        )
                        self._held_out_fits[index].append(held_out_fit)
                if verbose:
                    _log.info(
                        "seed %d fold %d/%d trained (%d test graphs)",
                        model_seed,
                        fold,
                        len(resolved_splits),
                        len(test_indices),
                    )
            self._oof_by_seed[model_seed] = seed_oof  # type: ignore[assignment]

        self._oof = [
            np.mean(
                [self._oof_by_seed[value][index] for value in resolved_seeds],
                axis=0,
            )
            for index in range(len(self._graphs))
        ]
        self._fold_ids = fold_ids
        self._seeds = resolved_seeds
        self._splits = resolved_splits
        self._is_fit = True
        return self

    # ------------------------------------------------------------------
    def oof_proba(self) -> list[np.ndarray]:
        """Out-of-fold per-node class probabilities, aligned to ``graphs``."""
        self._check_fit()
        return self._oof  # type: ignore[return-value]

    def _validate_splits(
        self,
        splits: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        if len(splits) < 2:
            raise ValueError("At least two folds are required.")
        n_graphs = len(self._graphs)
        test_counts = np.zeros(n_graphs, dtype=int)
        expected_classes = set(range(self._n_classes))
        for train_indices, test_indices in splits:
            if not len(train_indices) or not len(test_indices):
                raise ValueError("Every fold needs non-empty train and test sets.")
            if np.any(train_indices < 0) or np.any(train_indices >= n_graphs):
                raise ValueError("A training index is out of range.")
            if np.any(test_indices < 0) or np.any(test_indices >= n_graphs):
                raise ValueError("A test index is out of range.")
            if set(train_indices) & set(test_indices):
                raise ValueError("Train and test indices overlap within a fold.")
            if set(self._labels[train_indices]) != expected_classes:
                raise ValueError("Every training fold must contain every class.")
            if set(self._labels[test_indices]) != expected_classes:
                raise ValueError("Every test fold must contain every class.")
            if self._groups is not None:
                train_groups = set(self._groups[train_indices])
                test_groups = set(self._groups[test_indices])
                if train_groups & test_groups:
                    raise ValueError("A group appears on both sides of a fold.")
            test_counts[test_indices] += 1
        if not np.all(test_counts == 1):
            raise ValueError("Every graph must occur in exactly one test fold.")

    @staticmethod
    def _prediction_level(
        oof: Sequence[np.ndarray],
        labels: np.ndarray,
        groups: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        graph_probability = np.stack([values.mean(axis=0) for values in oof])
        if groups is None:
            ids = np.asarray([str(index) for index in range(len(labels))], dtype=object)
            return ids, labels.copy(), graph_probability.argmax(axis=1), graph_probability

        ordered_groups = list(dict.fromkeys(groups.tolist()))
        true_labels = []
        probabilities = []
        for group in ordered_groups:
            mask = groups == group
            group_labels = np.unique(labels[mask])
            if len(group_labels) != 1:  # pragma: no cover - validated on input
                raise RuntimeError(f"Group {group!r} has multiple labels.")
            true_labels.append(int(group_labels[0]))
            probabilities.append(graph_probability[mask].mean(axis=0))
        probability_array = np.stack(probabilities)
        return (
            np.asarray(ordered_groups, dtype=object),
            np.asarray(true_labels, dtype=int),
            probability_array.argmax(axis=1),
            probability_array,
        )

    def cv_metrics(self) -> dict:
        """Node-, graph-, and optional biological-group-level OOF summaries.

        Graph probabilities average nodes; grouped probabilities then average
        graphs within each group. When ``groups`` was passed to :meth:`fit`, the
        group-level metrics are the primary biological-unit summaries.
        """
        self._check_fit()
        oof = self._oof  # type: ignore[assignment]
        node_true = np.concatenate([np.full(oof[i].shape[0], self._labels[i]) for i in range(len(oof))])
        node_pred = np.concatenate([oof[i].argmax(1) for i in range(len(oof))])
        roi_pred = np.array([oof[i].mean(0).argmax() for i in range(len(oof))])
        roi_confusion = confusion_matrix(
            self._labels,
            roi_pred,
            labels=list(range(self._n_classes)),
        )
        metrics = {
            "node_macro_f1": float(f1_score(node_true, node_pred, average="macro")),
            "roi_macro_f1": float(f1_score(self._labels, roi_pred, average="macro")),
            "roi_accuracy": float((roi_pred == self._labels).mean()),
            "roi_pred": roi_pred,
            "roi_confusion": roi_confusion,
        }
        if self._groups is not None:
            group_ids, group_true, group_pred, group_probability = self._prediction_level(
                oof,
                self._labels,
                self._groups,
            )
            group_confusion = confusion_matrix(
                group_true,
                group_pred,
                labels=list(range(self._n_classes)),
            )
            metrics.update(
                {
                    "group_ids": group_ids,
                    "group_true": group_true,
                    "group_pred": group_pred,
                    "group_probability": group_probability,
                    "group_macro_f1": float(
                        f1_score(group_true, group_pred, average="macro")
                    ),
                    "group_accuracy": float((group_pred == group_true).mean()),
                    "group_confusion": group_confusion,
                }
            )
        return metrics

    def seed_metrics(self) -> pd.DataFrame:
        """Return graph/group performance for every fitted model seed."""
        self._check_fit()
        rows = []
        for seed in self._seeds:
            oof = self._oof_by_seed[seed]
            _, graph_true, graph_pred, _ = self._prediction_level(
                oof,
                self._labels,
                None,
            )
            row = {
                "seed": seed,
                "graph_macro_f1": float(
                    f1_score(graph_true, graph_pred, average="macro")
                ),
                "graph_accuracy": float((graph_true == graph_pred).mean()),
            }
            if self._groups is not None:
                _, group_true, group_pred, _ = self._prediction_level(
                    oof,
                    self._labels,
                    self._groups,
                )
                row.update(
                    {
                        "group_macro_f1": float(
                            f1_score(group_true, group_pred, average="macro")
                        ),
                        "group_accuracy": float((group_true == group_pred).mean()),
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def prediction_table(self, *, level: str = "group") -> pd.DataFrame:
        """Return seed-specific and ensemble predictions for graphs or groups."""
        self._check_fit()
        if level not in {"graph", "group"}:
            raise ValueError("level must be 'graph' or 'group'.")
        if level == "group" and self._groups is None:
            raise ValueError("Group predictions require groups passed to fit().")
        prediction_groups = self._groups if level == "group" else None
        sources = [("ensemble", None, self._oof)] + [
            ("seed", seed, self._oof_by_seed[seed]) for seed in self._seeds
        ]
        rows = []
        for estimate, seed, oof in sources:
            ids, true_labels, predictions, probabilities = self._prediction_level(
                oof,  # type: ignore[arg-type]
                self._labels,
                prediction_groups,
            )
            for index, unit_id in enumerate(ids):
                if level == "group":
                    group_folds = np.unique(self._fold_ids[self._groups == unit_id])
                    if len(group_folds) != 1:  # pragma: no cover - validated splits
                        raise RuntimeError(f"Group {unit_id!r} spans folds.")
                    fold = int(group_folds[0])
                else:
                    fold = int(self._fold_ids[int(unit_id)])
                row = {
                    "estimate": estimate,
                    "seed": seed,
                    "unit_id": str(unit_id),
                    "fold": fold,
                    "true_class_idx": int(true_labels[index]),
                    "predicted_class_idx": int(predictions[index]),
                    "correct": bool(true_labels[index] == predictions[index]),
                }
                row.update(
                    {
                        f"probability_class_{class_index}": float(
                            probabilities[index, class_index]
                        )
                        for class_index in range(self._n_classes)
                    }
                )
                rows.append(row)
        return pd.DataFrame(rows)

    def split_assignments(self) -> pd.DataFrame:
        """Return one row per graph with its fixed held-out fold and group."""
        self._check_fit()
        return pd.DataFrame(
            {
                "graph_index": np.arange(len(self._graphs), dtype=int),
                "class_idx": self._labels,
                "group_id": self._groups if self._groups is not None else None,
                "fold": self._fold_ids,
            }
        )

    def fold_transform_table(self) -> pd.DataFrame:
        """Return the train-fold mean/SD paired with every seed and fold."""
        self._check_fit()
        rows = []
        seen = set()
        for fits in self._held_out_fits.values():
            for fit in fits:
                key = (fit.seed, fit.fold)
                if key in seen:
                    continue
                seen.add(key)
                for feature, (mean, sd) in enumerate(
                    zip(fit.mu.cpu().numpy(), fit.sd.cpu().numpy(), strict=True)
                ):
                    rows.append(
                        {
                            "seed": fit.seed,
                            "fold": fit.fold,
                            "feature": feature,
                            "train_mean": float(mean),
                            "train_sd": float(sd),
                        }
                    )
        return pd.DataFrame(rows).sort_values(["seed", "fold", "feature"])

    def cluster_probability_table(self) -> pd.DataFrame:
        """Return biological-unit OOF probabilities for every ECM cluster.

        Ensemble out-of-fold node probabilities are first averaged within each
        ECM state and graph (ROI). When ``groups`` was passed to :meth:`fit`,
        those ROI means are then averaged equally within each group; otherwise
        each graph is its own unit. Thus neither large ROIs nor groups with more
        ROIs receive extra weight. A ROI without a state is omitted from that
        state's mean, and a group with no ROI containing the state has no row;
        absent states are never represented by an imputed zero.

        Returns
        -------
        pandas.DataFrame
            One row per target class, ECM cluster, and observed biological unit
            with columns ``class_idx``, ``cluster``, ``unit_id``,
            ``mean_probability``, ``n_rois``, and ``n_nodes``. ``n_rois`` and
            ``n_nodes`` count only ROIs/nodes in which the state is present.
            Without groups, ``unit_id`` is the graph's zero-based index as a
            string and ``n_rois`` is one.
        """
        self._check_fit()
        graph_clusters = [graph.cluster.cpu().numpy() for graph in self._graphs]
        clusters = sorted(
            {
                int(cluster)
                for values in graph_clusters
                for cluster in values
            }
        )
        if self._groups is None:
            unit_ids = np.asarray(
                [str(index) for index in range(len(self._graphs))],
                dtype=object,
            )
        else:
            unit_ids = self._groups
        ordered_units = list(dict.fromkeys(unit_ids.tolist()))

        rows: list[dict[str, str | int | float]] = []
        for cluster in clusters:
            present = [
                index
                for index, values in enumerate(graph_clusters)
                if np.any(values == cluster)
            ]
            for class_idx in range(self._n_classes):
                roi_means: dict[int, float] = {}
                roi_nodes: dict[int, int] = {}
                for index in present:
                    mask = graph_clusters[index] == cluster
                    values = self._oof[index][mask, class_idx]  # type: ignore[index]
                    roi_means[index] = float(values.mean())
                    roi_nodes[index] = int(values.size)

                for unit_id in ordered_units:
                    indices = [
                        index
                        for index in present
                        if unit_ids[index] == unit_id
                    ]
                    if not indices:
                        continue
                    rows.append(
                        {
                            "class_idx": class_idx,
                            "cluster": cluster,
                            "unit_id": str(unit_id),
                            "mean_probability": float(
                                np.mean([roi_means[index] for index in indices])
                            ),
                            "n_rois": len(indices),
                            "n_nodes": sum(roi_nodes[index] for index in indices),
                        }
                    )
        return pd.DataFrame(
            rows,
            columns=[
                "class_idx",
                "cluster",
                "unit_id",
                "mean_probability",
                "n_rois",
                "n_nodes",
            ],
        )

    def cluster_probability_summary(self) -> pd.DataFrame:
        """Summarize cluster probabilities with equal biological-unit weight.

        The unit-level values come from :meth:`cluster_probability_table`.
        Their arithmetic mean and sample standard deviation (``ddof=1``) are
        reported for each target class and ECM cluster. A single observed unit
        has ``unit_sd=0``. Counts remain descriptive and do not weight either
        statistic.
        """
        table = self.cluster_probability_table()
        rows: list[dict[str, int | float]] = []
        for (class_idx, cluster), values in table.groupby(
            ["class_idx", "cluster"],
            sort=True,
        ):
            probabilities = values["mean_probability"].to_numpy(dtype=float)
            rows.append(
                {
                    "class_idx": int(class_idx),
                    "cluster": int(cluster),
                    "mean_probability": float(probabilities.mean()),
                    "unit_sd": (
                        float(np.std(probabilities, ddof=1))
                        if probabilities.size > 1
                        else 0.0
                    ),
                    "n_units": int(probabilities.size),
                    "n_rois": int(values["n_rois"].sum()),
                    "n_nodes": int(values["n_nodes"].sum()),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "class_idx",
                "cluster",
                "mean_probability",
                "unit_sd",
                "n_units",
                "n_rois",
                "n_nodes",
            ],
        )

    # ------------------------------------------------------------------
    def integrated_gradients(self, index: int, target_class: int, *, n_steps: int = 50):
        """Per-node integrated gradients toward ``target_class`` for graph ``index``.

        Attributes the mean (over nodes) ``target_class`` logit to the node
        features, using every seed-specific fold network that held ``index``
        out and its train-fold transform. The zero baseline in standardised
        space corresponds to the training-fold mean state-frequency vector.

        Returns
        -------
        node_importance : np.ndarray, shape (N,)
            Per-node summed absolute attribution (a spatial importance map).
        feat_attr : np.ndarray, shape (N, F)
            Signed per-node-per-feature attribution.  With one-hot inputs the
            mass concentrates on each node's own cluster column, so aggregating
            ``feat_attr`` by ``graph.cluster`` gives per-cluster importance.
        """
        self._check_fit()
        from captum.attr import IntegratedGradients

        fits = self._held_out_fits.get(index, [])
        if not fits:
            raise ValueError(f"graph {index} was never held out; call fit() first")
        g = self._graphs[index]
        ei = g.edge_index.to(self._device)
        attrs = []
        for fit in fits:
            fit.net.eval()
            x = ((g.x.to(self._device) - fit.mu) / fit.sd).detach()

            def _fwd(
                xb: torch.Tensor,
                held_out_fit: _HeldOutFit = fit,
            ) -> torch.Tensor:
                outputs = [
                    held_out_fit.net(xb[batch], ei)[:, target_class]
                    .mean()
                    .unsqueeze(0)
                    for batch in range(xb.shape[0])
                ]
                return torch.cat(outputs)

            ig = IntegratedGradients(_fwd)
            attrs.append(
                ig.attribute(
                    x.unsqueeze(0),
                    baselines=torch.zeros_like(x).unsqueeze(0),
                    n_steps=n_steps,
                )
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
        attr = np.mean(attrs, axis=0)
        return np.abs(attr).sum(1), attr

    def cluster_importance(self, *, n_steps: int = 50) -> pd.DataFrame:
        """Per-class × per-cluster integrated-gradients importance over all ROIs.

        For every ROI and class, integrated gradients toward that class are
        attributed to the node features; each node's contribution is read off its
        own one-hot column and grouped by its cluster id.  ``importance_signed``
        (the headline) is the mean signed self-column attribution — positive means
        the cluster *pushes toward* the class — and is class-specific.
        ``importance_abs`` is the magnitude (rare clusters are amplified by
        standardisation, so it is less class-specific).  Both are averaged over
        ROIs that contain the cluster.

        Returns a tidy DataFrame ``[class_idx, cluster, importance_signed,
        importance_abs, n_nodes]`` sorted by class then signed importance.
        """
        self._check_fit()
        n_cls = self._n_classes
        acc: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
        for i in range(len(self._graphs)):
            clu = self._graphs[i].cluster.numpy()
            own_col = self._graphs[i].x.numpy().argmax(1)  # each node's active OHE column
            rows = np.arange(len(clu))
            for c in range(n_cls):
                _, attr = self.integrated_gradients(i, c, n_steps=n_steps)
                self_attr = attr[rows, own_col]  # signed self-column attribution
                abs_node = np.abs(attr).sum(1)  # total magnitude per node
                for k in np.unique(clu):
                    m = clu == k
                    acc.setdefault((c, int(k)), []).append(
                        (float(self_attr[m].mean()), float(abs_node[m].mean()), int(m.sum()))
                    )
        out = []
        for (c, k), vals in acc.items():
            arr = np.array(vals)
            out.append(
                {
                    "class_idx": c,
                    "cluster": k,
                    "importance_signed": float(arr[:, 0].mean()),
                    "importance_abs": float(arr[:, 1].mean()),
                    "n_nodes": int(arr[:, 2].sum()),
                }
            )
        return (
            pd.DataFrame(out)
            .sort_values(["class_idx", "importance_signed"], ascending=[True, False])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def _check_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError("Call .fit() first.")

    @property
    def is_fit(self) -> bool:
        return self._is_fit
