"""Node-level masked-feature graph autoencoder for spatial-domain / niche detection.

``GraphMAE`` learns a per-node embedding by reconstructing masked node features through a
message-passing encoder (GraphMAE-style self-supervision). It keeps every node in the
embedding, which can then be clustered to assign each node (e.g. an image patch or a cell)
to a spatial domain. It trains on **one** graph (which may have several disconnected
components, e.g. several samples embedded jointly) with a lightweight loop, so large graphs
with millions of edges train in seconds per epoch.

References
----------
.. [1] Hou et al. (2022), "GraphMAE: Self-Supervised Masked Graph Autoencoders", KDD.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
from anndata import AnnData
from tqdm.auto import trange

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
except ImportError as e:  # pragma: no cover - exercised only without PyG
    raise ImportError('GraphMAE requires PyTorch Geometric. Install with: pip install "mantpy[gnn]"') from e

from mantpy._core._graph_mae_module import EDGE_ENCODERS, ECMGraphMAEModule
from mantpy.nn._utils import _package_version, resolve_accelerator

logger = logging.getLogger(__name__)


def _zscore(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(0)) / (a.std(0) + 1e-8)


class GraphMAE:
    """Node-level masked-feature graph autoencoder.

    Parameters
    ----------
    hidden_dim
        Width of the encoder layers and of the returned node embedding.
    n_layers
        Number of message-passing layers.
    encoder
        Message-passing convolution: ``"gine"`` (edge-aware GIN; needs edge features),
        ``"gin"``, ``"sage"``, ``"gcn"`` (node-only), or ``"gatv2"`` / ``"transformerconv"``
        (attention, optionally edge-aware).
    decoder
        Feature decoder: ``"mlp2"``, ``"linear"``, or ``"graphmae_remask"`` (re-masks the
        latent before a GNN decode, as in the original GraphMAE formulation).
    norm
        Per-layer normalisation: ``"batchnorm"``, ``"layernorm"``, ``"graphnorm"``, or
        ``"pairnorm"``.
    mask_ratio
        Fraction of nodes whose features are masked and reconstructed each step.
    mitigation
        Over-smoothing mitigation: ``"none"``, ``"dropedge"``, ``"residual"``, or ``"jk"``.

    Examples
    --------
    >>> model = mt.nn.GraphMAE(hidden_dim=64, n_layers=2, encoder="gine")  # doctest: +SKIP
    >>> model.train(joint_graph, max_epochs=150, rng=0)  # doctest: +SKIP
    >>> node_emb = model.get_node_latent()  # (n_nodes, hidden_dim)         # doctest: +SKIP

    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        n_layers: int = 2,
        encoder: Literal["gine", "gin", "sage", "gcn", "gatv2", "transformerconv"] = "gine",
        decoder: Literal["mlp2", "linear", "graphmae_remask"] = "mlp2",
        norm: Literal["batchnorm", "layernorm", "graphnorm", "pairnorm"] = "batchnorm",
        mask_ratio: float = 0.5,
        mitigation: Literal["none", "dropedge", "residual", "jk"] = "none",
        module_factory: Callable | None = None,
        module_kwargs: Mapping[str, Any] | None = None,
    ):
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim={hidden_dim!r}: expected positive int. Use hidden_dim >= 1.")
        if n_layers < 1:
            raise ValueError(f"n_layers={n_layers!r}: expected positive int. Use n_layers >= 1.")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio={mask_ratio!r}: expected a fraction in (0, 1).")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.encoder = encoder
        self.decoder = decoder
        self.norm = norm
        self.mask_ratio = mask_ratio
        self.mitigation = mitigation
        self._module_factory = module_factory
        self._module_kwargs = dict(module_kwargs or {})
        self._module: ECMGraphMAEModule | None = None
        self._device = "cpu"
        self._history: dict[str, list[float]] = {"train_loss": []}
        self._optimizer_updates = 0
        self._feature_transform: dict[str, np.ndarray] = {}
        self._latent_transform: dict[str, np.ndarray] = {}

    def __repr__(self) -> str:
        fitted = self._module is not None and hasattr(self, "_train_graph")
        status = "fitted" if fitted else "unfitted"
        details = f"hidden_dim={self.hidden_dim}, n_layers={self.n_layers}, encoder={self.encoder!r}, status={status!r}"
        if fitted:
            details += f", device={self._device!r}, n_nodes={int(self._train_graph[0].shape[0])}"
        return f"GraphMAE({details})"

    def _tensors(
        self, graph: Data, standardize: bool, *, feature_transform: str = "graph",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = np.asarray(graph.x.cpu().numpy() if torch.is_tensor(graph.x) else graph.x, dtype=np.float32)
        ei = np.asarray(graph.edge_index.cpu().numpy() if torch.is_tensor(graph.edge_index) else graph.edge_index)
        ea = getattr(graph, "edge_attr", None)
        if ea is not None:
            ea = np.asarray(ea.cpu().numpy() if torch.is_tensor(ea) else ea, dtype=np.float32)
        if self.encoder in EDGE_ENCODERS and ea is None:
            raise ValueError(f"encoder={self.encoder!r} needs edge features, but graph.edge_attr is None.")
        if standardize:
            if feature_transform == "graph":
                x = _zscore(x)
                if ea is not None:
                    ea = _zscore(ea)
            elif feature_transform == "fitted":
                if not self._feature_transform:
                    raise RuntimeError("No fitted feature transform; train the model first.")
                if x.shape[1] != len(self._feature_transform["node_mean"]):
                    raise ValueError("Node feature columns differ from the fitted graph.")
                x = (x - self._feature_transform["node_mean"]) / self._feature_transform["node_scale"]
                if ea is not None:
                    if "edge_mean" not in self._feature_transform:
                        raise ValueError("The fitted graph did not have edge features.")
                    if ea.shape[1] != len(self._feature_transform["edge_mean"]):
                        raise ValueError("Edge feature columns differ from the fitted graph.")
                    ea = (ea - self._feature_transform["edge_mean"]) / self._feature_transform["edge_scale"]
            else:
                raise ValueError("feature_transform must be 'graph' or 'fitted'.")
        xt = torch.tensor(x, device=self._device)
        eit = torch.tensor(ei, dtype=torch.long, device=self._device)
        eat = torch.tensor(ea, device=self._device) if ea is not None else None
        return xt, eit, eat

    def train(
        self,
        graph: Data,
        *,
        max_epochs: int = 150,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        recon_loss: Literal["mse", "scaled_cosine"] = "mse",
        standardize: bool = True,
        rng: int | np.random.Generator | None = None,
        accelerator: str = "auto",
        enable_progress_bar: bool = True,
        optimizer_factory: Callable | None = None,
        optimizer_kwargs: Mapping[str, Any] | None = None,
        scheduler: Literal["cosine"] | None = "cosine",
        scheduler_factory: Callable | None = None,
        scheduler_kwargs: Mapping[str, Any] | None = None,
        reconstruction_loss: Callable | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> GraphMAE:
        """Fit the autoencoder on a single graph by masked-feature reconstruction.

        Parameters
        ----------
        graph
            A PyG :class:`~torch_geometric.data.Data` with node features ``x`` of shape
            ``(n_nodes, n_features)``, ``edge_index`` of shape ``(2, n_edges)``, and (for
            edge-aware encoders) ``edge_attr`` of shape ``(n_edges, n_edge_features)``.
            Disconnected components are fine — concatenate several samples into one graph to
            embed them jointly in a shared latent space.
        max_epochs
            Number of full-graph optimisation steps.
        lr, weight_decay
            AdamW settings; a cosine schedule decays ``lr`` to 0.
        recon_loss
            ``"mse"`` (default) or ``"scaled_cosine"`` reconstruction error on masked nodes.
        standardize
            Z-score node and edge features before training (recommended; matches the scale
            the encoder expects).
        rng
            Seed or :class:`numpy.random.Generator` controlling stochastic
            initialisation and masking. The latent is stochastic across seeds;
            exact cross-run GPU reproducibility can depend on the selected
            kernels. Use ``accelerator="cpu"`` for a reproducible CPU run.
        accelerator
            ``"auto"`` (GPU when available), ``"gpu"``/``"cuda"``, or ``"cpu"``.
        enable_progress_bar
            Show epoch and reconstruction-loss progress. Disable for quiet or
            non-interactive runs; this does not affect model fitting.

        Returns
        -------
        GraphMAE
            ``self``, fitted, for chaining.
        """
        seed = rng if isinstance(rng, int | np.integer) else np.random.default_rng(rng).integers(0, 2**31 - 1)
        seed = int(seed)
        if max_epochs < 1 or scheduler not in ("cosine", None):
            raise ValueError("max_epochs must be positive and scheduler must be 'cosine' or None.")
        torch.manual_seed(seed)
        self._device = resolve_accelerator(accelerator)
        self._latent_transform = {}
        raw_x = np.asarray(graph.x.detach().cpu().numpy() if torch.is_tensor(graph.x) else graph.x, dtype=np.float32)
        self._feature_transform = {
            "node_mean": raw_x.mean(axis=0),
            "node_scale": raw_x.std(axis=0) + 1e-8,
        }
        raw_ea = getattr(graph, "edge_attr", None)
        if raw_ea is not None:
            raw_ea = np.asarray(
                raw_ea.detach().cpu().numpy() if torch.is_tensor(raw_ea) else raw_ea, dtype=np.float32
            )
            self._feature_transform.update(edge_mean=raw_ea.mean(axis=0), edge_scale=raw_ea.std(axis=0) + 1e-8)
        if not standardize:
            self._feature_transform = {
                key: np.zeros_like(value) if key.endswith("_mean") else np.ones_like(value)
                for key, value in self._feature_transform.items()
            }
        x, ei, ea = self._tensors(graph, standardize)
        n, din = x.shape
        edim = ea.shape[1] if ea is not None else 3

        factory = self._module_factory or ECMGraphMAEModule
        params = {
            "din": din,
            "hid": self.hidden_dim,
            "layers": self.n_layers,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "edim": edim,
            "norm": self.norm,
            "mitigation": self.mitigation,
        }
        self._module = factory(**{**params, **self._module_kwargs}).to(self._device)
        optimizer_type = optimizer_factory or torch.optim.AdamW
        opt = optimizer_type(
            self._module.parameters(), **{"lr": lr, "weight_decay": weight_decay, **dict(optimizer_kwargs or {})}
        )
        if scheduler_factory is not None:
            sched = scheduler_factory(opt, **dict(scheduler_kwargs or {}))
        elif scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, **{"T_max": max_epochs, **dict(scheduler_kwargs or {})}
            )
        else:
            sched = None
        gen = torch.Generator(device=self._device).manual_seed(seed)
        self._history = {"train_loss": []}
        self._optimizer_updates = 0
        self._module.train()
        epochs = trange(max_epochs, desc="GraphMAE", unit="epoch") if enable_progress_bar else range(max_epochs)
        for epoch in epochs:
            opt.zero_grad()
            mask = torch.rand(n, generator=gen, device=self._device) < self.mask_ratio
            rec = self._module.forward_node(x, ei, ea, mask)
            if reconstruction_loss is not None:
                loss = reconstruction_loss(rec[mask], x[mask])
            elif recon_loss == "scaled_cosine":
                loss = ((1 - F.cosine_similarity(rec[mask], x[mask], dim=1)) ** 2).mean()
            else:
                loss = F.mse_loss(rec[mask], x[mask])
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            value = float(loss.detach().cpu())
            self._history["train_loss"].append(value)
            self._optimizer_updates += 1
            if callback is not None:
                callback({
                    "epoch": epoch + 1, "train_loss": value,
                    "optimizer_updates": self._optimizer_updates, "lr": float(opt.param_groups[0]["lr"]),
                })
            if enable_progress_bar:
                epochs.set_postfix(loss=f"{loss.detach().item():.4f}", refresh=False)
        self._module.eval()
        self._train_graph = (x, ei, ea)
        logger.info(
            "GraphMAE trained: %d nodes, %d edges, encoder=%s, hidden_dim=%d, device=%s",
            n,
            ei.shape[1],
            self.encoder,
            self.hidden_dim,
            self._device,
        )
        return self

    @torch.no_grad()
    def get_node_latent(
        self, graph: Data | None = None, *, standardize: bool = True,
        standardize_output: bool = True, feature_transform: Literal["graph", "fitted"] = "graph",
        output_transform: Literal["graph", "fitted"] = "graph",
    ) -> np.ndarray:
        """Return the per-node embedding (z-scored encoder output).

        Parameters
        ----------
        graph
            Graph to embed. ``None`` (default) re-embeds the training graph.
        standardize
            Z-score node/edge features of ``graph`` before encoding (only used when ``graph``
            is given; ignored for the training graph, which is already prepared).

        Returns
        -------
        numpy.ndarray
            Node embedding of shape ``(n_nodes, hidden_dim)``, z-scored per column.
        """
        if self._module is None:
            raise RuntimeError("GraphMAE is not fitted. Call .train(graph) first.")
        if graph is None:
            x, ei, ea = self._train_graph
        else:
            x, ei, ea = self._tensors(graph, standardize, feature_transform=feature_transform)
        h = self._module.encode(x, ei, ea).cpu().numpy()
        if graph is None:
            self._latent_transform = {"mean": h.mean(axis=0), "scale": h.std(axis=0) + 1e-8}
        if not standardize_output:
            return h.astype(np.float32)
        if output_transform == "graph":
            return _zscore(h).astype(np.float32)
        if output_transform != "fitted":
            raise ValueError("output_transform must be 'graph' or 'fitted'.")
        if not self._latent_transform:
            train_x, train_ei, train_ea = self._train_graph
            train_h = self._module.encode(train_x, train_ei, train_ea).cpu().numpy()
            self._latent_transform = {"mean": train_h.mean(axis=0), "scale": train_h.std(axis=0) + 1e-8}
        return ((h - self._latent_transform["mean"]) / self._latent_transform["scale"]).astype(np.float32)

    @property
    def history(self) -> dict[str, list[float]]:
        """Detached reconstruction losses in optimization-step order."""
        return {key: list(values) for key, values in self._history.items()}

    @property
    def optimizer_updates(self) -> int:
        """Number of completed optimizer steps."""
        return self._optimizer_updates

    @property
    def model(self):
        """Fitted masked graph autoencoder module."""
        if self._module is None:
            raise RuntimeError("GraphMAE is not fitted.")
        return self._module

    @property
    def feature_transform_state(self) -> dict[str, np.ndarray]:
        """Copied training-graph node/edge means and scales."""
        return {key: value.copy() for key, value in self._feature_transform.items()}


def encode_graphmae(
    adata: AnnData,
    *,
    graph_key: str = "patch",
    node_feature_key: str | None = "X",
    key_added: str = "X_graphmae",
    hidden_dim: int = 64,
    n_layers: int = 2,
    encoder: Literal["gine", "gin", "sage", "gcn", "gatv2", "transformerconv"] = "gine",
    decoder: Literal["mlp2", "linear", "graphmae_remask"] = "mlp2",
    norm: Literal["batchnorm", "layernorm", "graphnorm", "pairnorm"] = "batchnorm",
    mask_ratio: float = 0.5,
    mitigation: Literal["none", "dropedge", "residual", "jk"] = "none",
    max_epochs: int = 150,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    recon_loss: Literal["mse", "scaled_cosine"] = "mse",
    standardize: bool = True,
    random_state: int = 0,
    accelerator: str = "auto",
    enable_progress_bar: bool = True,
    overwrite: bool = False,
    include_edge_features: bool = True,
    standardize_output: bool = True,
    module_factory: Callable | None = None,
    module_kwargs: Mapping[str, Any] | None = None,
    optimizer_factory: Callable | None = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    scheduler: Literal["cosine"] | None = "cosine",
    scheduler_factory: Callable | None = None,
    scheduler_kwargs: Mapping[str, Any] | None = None,
    reconstruction_loss: Callable | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> GraphMAE:
    """Learn and attach a GraphMAE node representation in one call.

    Only the selected node feature matrix and graph topology are passed to
    GraphMAE; columns in ``adata.obs`` are not model inputs. The embedding is
    written in place to ``obsm[key_added]`` and its serialisable provenance to
    ``uns[f"{key_added}_params"]``.

    Parameters
    ----------
    adata
        Observation-native AnnData containing a graph and node features.
    graph_key
        Graph stored in ``uns`` or ``obsp``. The default matches
        :func:`mantpy.gr.build_patch_graph`.
    node_feature_key
        ``"X"``/``None`` for ``adata.X`` or a key in ``adata.obsm``.
    key_added
        Destination in ``adata.obsm``.
    overwrite
        Replace an existing embedding/provenance pair when ``True``.
    enable_progress_bar
        Show epoch and reconstruction-loss progress while fitting.

    Returns
    -------
    GraphMAE
        The fitted model, reusable with :meth:`GraphMAE.get_node_latent`.
    """
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData object.")
    if not key_added:
        raise ValueError("key_added must be a non-empty string.")
    params_key = f"{key_added}_params"
    if not overwrite and (key_added in adata.obsm or params_key in adata.uns):
        raise ValueError(f"Output {key_added!r} or its provenance already exists. Pass overwrite=True to replace it.")

    from mantpy.gr import to_pyg

    graph = to_pyg(
        adata, graph_key=graph_key, node_feature_key=node_feature_key, include_edge_features=include_edge_features
    )
    # GraphMAE consumes only x/edge_index/edge_attr. Explicitly discard any
    # optional prediction target that a legacy graph export may carry.
    if "y" in graph:
        del graph["y"]
    model = GraphMAE(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        encoder=encoder,
        decoder=decoder,
        norm=norm,
        mask_ratio=mask_ratio,
        mitigation=mitigation,
        module_factory=module_factory,
        module_kwargs=module_kwargs,
    ).train(
        graph,
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
        recon_loss=recon_loss,
        standardize=standardize,
        rng=random_state,
        accelerator=accelerator,
        enable_progress_bar=enable_progress_bar,
        optimizer_factory=optimizer_factory, optimizer_kwargs=optimizer_kwargs,
        scheduler=scheduler, scheduler_factory=scheduler_factory, scheduler_kwargs=scheduler_kwargs,
        reconstruction_loss=reconstruction_loss, callback=callback,
    )
    embedding = model.get_node_latent(standardize_output=standardize_output)
    if embedding.shape[0] != adata.n_obs:
        raise ValueError(
            f"GraphMAE returned {embedding.shape[0]} node embeddings for an AnnData with {adata.n_obs} observations."
        )

    resolved_feature_key = "X" if node_feature_key in (None, "X") else node_feature_key
    provenance = {
        "method": "GraphMAE",
        "graph_key": graph_key,
        "node_feature_key": resolved_feature_key,
        "key_added": key_added,
        "model": {
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "encoder": encoder,
            "decoder": decoder,
            "norm": norm,
            "mask_ratio": mask_ratio,
            "mitigation": mitigation,
            "module_factory": (
                getattr(module_factory, "__qualname__", type(module_factory).__qualname__)
                if module_factory is not None else "ECMGraphMAEModule"
            ),
            "module_kwargs": dict(module_kwargs or {}),
        },
        "training": {
            "max_epochs": max_epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "recon_loss": recon_loss,
            "standardize": standardize,
            "standardize_output": standardize_output,
            "include_edge_features": include_edge_features,
            "optimizer_updates": model.optimizer_updates,
            "optimizer": (
                getattr(optimizer_factory, "__qualname__", type(optimizer_factory).__qualname__)
                if optimizer_factory is not None else "AdamW"
            ),
            "optimizer_kwargs": dict(optimizer_kwargs or {}),
            "scheduler": (
                getattr(scheduler_factory, "__qualname__", type(scheduler_factory).__qualname__)
                if scheduler_factory is not None else scheduler
            ),
            "scheduler_kwargs": dict(scheduler_kwargs or {}),
        },
        "history": model.history,
        "random_state": int(random_state),
        "device": model._device,
        "n_nodes": int(graph.num_nodes),
        "n_directed_edges": int(graph.edge_index.shape[1]),
        "software": {
            "mantpy": _package_version("mantpy"),
            "torch": str(torch.__version__),
            "torch_geometric": _package_version("torch-geometric"),
        },
    }
    adata.obsm[key_added] = embedding
    adata.uns[params_key] = provenance
    return model


__all__ = ["GraphMAE", "encode_graphmae"]
