"""Shared graph-classifier fitting and attribution primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data


def node_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_index: torch.Tensor,
    graph_targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    reduction: Literal["node", "roi"] = "node",
    label_smoothing: float = 0.0,
    roi_class_weight_mode: Literal["after", "node"] = "after",
) -> torch.Tensor:
    """Pool nodes, or average within each ROI before giving ROIs equal weight.

    For ROI reduction, ``after`` applies class weights after averaging the
    unweighted node losses. ``node`` instead applies weights inside CE; these
    differ when label smoothing is nonzero.
    """
    if reduction == "node":
        return F.cross_entropy(logits, targets, weight=class_weights, label_smoothing=label_smoothing)
    if reduction != "roi":
        raise ValueError("reduction must be 'node' or 'roi'.")
    if roi_class_weight_mode not in {"after", "node"}:
        raise ValueError("roi_class_weight_mode must be 'after' or 'node'.")
    losses = F.cross_entropy(
        logits, targets, reduction="none",
        weight=class_weights if roi_class_weight_mode == "node" else None,
        label_smoothing=label_smoothing,
    )
    totals = torch.zeros(len(graph_targets), dtype=losses.dtype, device=losses.device)
    totals.index_add_(0, batch_index, losses)
    counts = torch.bincount(batch_index, minlength=len(graph_targets)).to(losses.dtype)
    if torch.any(counts == 0):
        raise ValueError("Every training ROI must have at least one node.")
    per_roi = totals / counts
    if class_weights is not None and roi_class_weight_mode == "after":
        per_roi = per_roi * class_weights[graph_targets]
    return per_roi.mean()


@dataclass(frozen=True)
class NodeAttribution:
    """Signed attribution plus its spatial magnitude and completeness diagnostic."""

    signed: np.ndarray
    node_importance: np.ndarray
    convergence_delta: np.ndarray
    target_class: int
    node_ids: tuple[str, ...]
    baseline: str = "zero_standardized"
    per_seed: tuple[np.ndarray, ...] = ()
    seeds: tuple[int, ...] = ()


def node_integrated_gradients(
    model: nn.Module,
    graph: Data,
    *,
    mean: torch.Tensor | np.ndarray,
    scale: torch.Tensor | np.ndarray,
    target_class: int,
    n_steps: int = 50,
    internal_batch_size: int | None = None,
    baseline: torch.Tensor | np.ndarray | None = None,
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray] | NodeAttribution:
    """Attribute the mean target logit to node features in fitted standardized space.

    No evaluation labels enter this operation. The default zero baseline is
    the training-feature mean before standardization. Node order is unchanged.
    """
    from captum.attr import IntegratedGradients

    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    device = next(model.parameters()).device
    model.eval()
    x = graph.x.to(device)
    mu = torch.as_tensor(mean, device=device, dtype=x.dtype)
    sd = torch.as_tensor(scale, device=device, dtype=x.dtype)
    if mu.shape != (x.shape[1],) or sd.shape != mu.shape or torch.any(sd <= 0):
        raise ValueError("mean and scale must align with feature columns, with positive scales.")
    standardized = ((x - mu) / sd).detach()
    edges = graph.edge_index.to(device)
    reference = torch.zeros_like(standardized) if baseline is None else torch.as_tensor(
        baseline, device=device, dtype=x.dtype
    )
    if reference.shape != standardized.shape:
        raise ValueError("baseline must have the same node/feature shape as graph.x.")

    def forward(items: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            model(items[index], edges)[:, target_class].mean().unsqueeze(0)
            for index in range(items.shape[0])
        ])

    output = IntegratedGradients(forward).attribute(
        standardized.unsqueeze(0),
        baselines=reference.unsqueeze(0),
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=return_details,
    )
    attr, delta = output if return_details else (output, None)
    signed = attr.squeeze(0).detach().cpu().numpy()
    importance = np.abs(signed).sum(axis=1)
    if not return_details:
        return importance, signed
    ids = getattr(graph, "node_ids", None)
    node_ids = tuple(str(value) for value in ids) if ids is not None else tuple(str(i) for i in range(len(x)))
    return NodeAttribution(
        signed=signed, node_importance=importance,
        convergence_delta=delta.detach().cpu().numpy(),
        target_class=int(target_class), node_ids=node_ids,
        baseline="zero_standardized" if baseline is None else "supplied_standardized",
    )


@dataclass
class GraphClassifierFit:
    """One fitted model, its training-only transform and detached training record."""

    model: nn.Module
    mean: torch.Tensor
    scale: torch.Tensor
    history: dict[str, list[float]]
    class_weights: torch.Tensor | None
    optimizer_updates: int
    loss_reduction: str
    seed: int | None

    @torch.no_grad()
    def predict_proba(self, graphs: Data | Sequence[Data]) -> np.ndarray | list[np.ndarray]:
        """Return per-node probabilities in each supplied graph's original order."""
        single = isinstance(graphs, Data)
        items = [graphs] if single else list(graphs)
        device = next(self.model.parameters()).device
        self.model.eval()
        result = [
            torch.softmax(
                self.model(
                    (graph.x.to(device) - self.mean) / self.scale,
                    graph.edge_index.to(device),
                ), dim=1,
            ).cpu().numpy()
            for graph in items
        ]
        return result[0] if single else result

    def integrated_gradients(self, graph: Data, *, target_class: int, **kwargs: Any):
        """Attribute a supplied graph with this fit's model and training transform."""
        return node_integrated_gradients(
            self.model, graph, mean=self.mean, scale=self.scale,
            target_class=target_class, **kwargs,
        )


def train_node_classifier(
    graphs: Sequence[Data],
    labels: Sequence[int],
    *,
    model_factory: Callable[..., nn.Module],
    model_kwargs: Mapping[str, Any] | None = None,
    n_classes: int = 2,
    epochs: int = 120,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    seed: int | None = None,
    loss_reduction: Literal["node", "roi"] = "node",
    class_weight: Literal["balanced"] | Sequence[float] | torch.Tensor | None = "balanced",
    roi_class_weight_mode: Literal["after", "node"] = "after",
    standardize: bool = True,
    device: str | torch.device | None = "cpu",
    optimizer_factory: Callable | None = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    scheduler: Literal["cosine"] | None = None,
    scheduler_factory: Callable | None = None,
    scheduler_kwargs: Mapping[str, Any] | None = None,
    input_dropout: float = 0.0,
    label_smoothing: float = 0.0,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> GraphClassifierFit:
    """Fit one node classifier using only the supplied training graphs.

    This primitive does no splitting or checkpoint orchestration. Reset the
    RNG with ``seed`` for each independent fit, or pass ``None`` to preserve
    a caller-managed historical random stream.
    """
    items = list(graphs)
    targets = np.asarray(labels, dtype=int)
    if not items or targets.shape != (len(items),):
        raise ValueError("labels must contain one target per non-empty training graph.")
    if any(graph.x.ndim != 2 or graph.num_nodes == 0 for graph in items):
        raise ValueError("Training graphs require non-empty two-dimensional node features.")
    if np.any(targets < 0) or np.any(targets >= n_classes):
        raise ValueError("labels must lie within [0, n_classes).")
    if epochs < 1 or not 0 <= input_dropout < 1 or not 0 <= label_smoothing <= 1:
        raise ValueError("epochs must be positive and dropout/smoothing must be valid fractions.")
    if scheduler not in (None, "cosine"):
        raise ValueError("scheduler must be 'cosine' or None.")
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    all_x = torch.cat([graph.x.detach().cpu() for graph in items], dim=0).numpy()
    mu, sd = all_x.mean(axis=0), all_x.std(axis=0)
    sd[sd == 0] = 1.0
    if not standardize:
        mu, sd = np.zeros_like(mu), np.ones_like(sd)
    mean = torch.as_tensor(mu, device=resolved_device)
    scale = torch.as_tensor(sd, device=resolved_device)
    model = model_factory(**dict(model_kwargs or {}))
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module.")
    model = model.to(resolved_device)
    factory = optimizer_factory or torch.optim.Adam
    optimizer = factory(
        model.parameters(), **{"lr": lr, "weight_decay": weight_decay, **dict(optimizer_kwargs or {})}
    )
    if scheduler_factory is not None:
        schedule = scheduler_factory(optimizer, **dict(scheduler_kwargs or {}))
    elif scheduler == "cosine":
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **{"T_max": epochs, **dict(scheduler_kwargs or {})}
        )
    else:
        schedule = None
    batch = Batch.from_data_list(items).to(resolved_device)
    features = (batch.x - mean) / scale
    graph_targets = torch.as_tensor(targets, dtype=torch.long, device=resolved_device)
    node_targets = graph_targets[batch.batch]
    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError("class_weight must be 'balanced', a vector, or None.")
        counts = np.bincount(targets, minlength=n_classes).astype(float)
        if np.any(counts == 0):
            raise ValueError("Balanced class weights require every class in the training fold.")
        weights = torch.tensor(counts.sum() / (counts + 1e-6), dtype=torch.float32, device=resolved_device)
        weights = weights / weights.sum() * n_classes
    elif class_weight is None:
        weights = None
    else:
        weights = torch.as_tensor(class_weight, dtype=torch.float32, device=resolved_device)
        if weights.shape != (n_classes,) or not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("class_weight must have one finite, non-negative entry per class.")
    history: dict[str, list[float]] = {"train_loss": []}
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs = features
        if input_dropout:
            keep = (torch.rand(len(features), 1, device=resolved_device) >= input_dropout).to(features.dtype)
            inputs = features * keep / (1 - input_dropout)
        logits = model(inputs, batch.edge_index)
        if logits.shape != (len(node_targets), n_classes):
            raise ValueError("The classifier must return (n_nodes, n_classes) logits.")
        loss = node_classification_loss(
            logits, node_targets, batch_index=batch.batch, graph_targets=graph_targets,
            class_weights=weights, reduction=loss_reduction,
            label_smoothing=label_smoothing, roi_class_weight_mode=roi_class_weight_mode,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("The node-classification loss is not finite.")
        loss.backward()
        optimizer.step()
        if schedule is not None:
            schedule.step()
        history["train_loss"].append(float(loss.detach().cpu()))
        if callback is not None:
            callback({
                "epoch": epoch + 1, "train_loss": history["train_loss"][-1],
                "optimizer_updates": epoch + 1, "lr": float(optimizer.param_groups[0]["lr"]),
            })
    model.eval()
    return GraphClassifierFit(
        model=model, mean=mean, scale=scale, history=history,
        class_weights=weights, optimizer_updates=epochs,
        loss_reduction=loss_reduction, seed=seed,
    )
