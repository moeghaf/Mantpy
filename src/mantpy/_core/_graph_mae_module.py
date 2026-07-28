"""Pure PyTorch modules for the node-level masked-feature graph autoencoder (GraphMAE).

Lowest layer — no AnnData, no Lightning, no training loop. A config-switched encoder
(message-passing conv x norm x over-smoothing mitigation) and decoder, plus the
masked-feature autoencoder that wires them together. The user-facing wrapper with the
training loop lives in :mod:`mantpy.nn._graph_mae`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import (
        GATv2Conv,
        GCNConv,
        GINConv,
        GINEConv,
        GraphNorm,
        PairNorm,
        SAGEConv,
        TransformerConv,
    )
    from torch_geometric.utils import dropout_edge
except ImportError as e:  # pragma: no cover - exercised only without PyG
    raise ImportError(
        'Graph neural network modules require PyTorch Geometric. Install with: pip install "mantpy[gnn]"'
    ) from e

EDGE_ENCODERS = {"gine", "gatv2", "transformerconv"}  # consume edge_attr
ENCODERS = EDGE_ENCODERS | {"gin", "sage", "gcn"}
DECODERS = {"linear", "mlp2", "graphmae_remask"}
NORMS = {"batchnorm", "layernorm", "graphnorm", "pairnorm"}


def _mlp(d: int, h: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h))


def _make_conv(kind: str, d: int, h: int, edim: int | None) -> tuple[nn.Module, bool]:
    """Build one message-passing layer; returns (conv, uses_edges). ``edim=None`` → node-only."""
    if kind == "gine":
        if edim is None:
            raise ValueError("encoder='gine' requires edge features; pass edge_attr or pick another encoder.")
        return GINEConv(_mlp(d, h), edge_dim=edim), True
    if kind == "gin":
        return GINConv(_mlp(d, h)), False
    if kind == "gatv2":
        return GATv2Conv(d, h, heads=4, concat=False, edge_dim=edim), edim is not None
    if kind == "sage":
        return SAGEConv(d, h), False
    if kind == "gcn":
        return GCNConv(d, h), False
    if kind == "transformerconv":
        return TransformerConv(d, h // 4, heads=4, concat=True, edge_dim=edim), edim is not None
    raise ValueError(f"encoder={kind!r}: expected one of {sorted(ENCODERS)}.")


def _make_norm(kind: str, h: int) -> nn.Module:
    if kind == "batchnorm":
        return nn.BatchNorm1d(h)
    if kind == "layernorm":
        return nn.LayerNorm(h)
    if kind == "graphnorm":
        return GraphNorm(h)
    if kind == "pairnorm":
        return PairNorm()
    raise ValueError(f"norm={kind!r}: expected one of {sorted(NORMS)}.")


class Encoder(nn.Module):
    """Stacked message passing with switchable conv / norm / over-smoothing mitigation."""

    def __init__(
        self,
        din: int,
        hid: int,
        layers: int,
        encoder: str = "gine",
        edim: int | None = 3,
        norm: str = "batchnorm",
        mitigation: str = "none",
        dropedge: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.mitigation = mitigation
        self.dropedge_p = dropedge
        self.uses_edges = False
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        d = din
        for _ in range(layers):
            conv, ue = _make_conv(encoder, d, hid, edim)
            self.uses_edges = self.uses_edges or ue
            self.convs.append(conv)
            self.norms.append(_make_norm(norm, hid))
            d = hid
        self.jk = mitigation == "jk"
        self.residual = mitigation == "residual"
        self.jk_lin = nn.Linear(hid * layers, hid) if self.jk else None

    def forward(self, x: torch.Tensor, ei: torch.Tensor, ea: torch.Tensor | None = None) -> torch.Tensor:
        h, prev, hs = x, None, []
        for conv, norm in zip(self.convs, self.norms, strict=False):
            ei_l, ea_l = ei, ea
            if self.training and self.mitigation == "dropedge" and self.dropedge_p > 0:
                ei_l, keep = dropout_edge(ei, p=self.dropedge_p)
                ea_l = ea[keep] if (ea is not None and self.uses_edges) else ea
            m = conv(h, ei_l, ea_l) if self.uses_edges else conv(h, ei_l)
            m = F.gelu(norm(m))
            if self.residual and prev is not None and prev.shape == m.shape:
                m = m + prev
            prev, h = m, m
            hs.append(h)
        return self.jk_lin(torch.cat(hs, dim=1)) if self.jk else h


class Decoder(nn.Module):
    """linear | mlp2 | graphmae_remask (GNN decoder with re-masking)."""

    def __init__(self, kind: str, hid: int, dout: int):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = nn.Linear(hid, dout)
        elif kind == "mlp2":
            self.net = nn.Sequential(nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, dout))
        elif kind == "graphmae_remask":
            self.remask = nn.Parameter(torch.zeros(hid))
            self.conv = GINConv(_mlp(hid, hid))
            self.lin = nn.Linear(hid, dout)
        else:
            raise ValueError(f"decoder={kind!r}: expected one of {sorted(DECODERS)}.")

    def forward(
        self, h: torch.Tensor, ei: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.kind in ("linear", "mlp2"):
            return self.net(h)
        z = h if mask is None else torch.where(mask[:, None], self.remask[None, :], h)
        return self.lin(F.gelu(self.conv(z, ei)))


class ECMGraphMAEModule(nn.Module):
    """Masked-feature graph autoencoder; encoder and decoder fully switchable."""

    def __init__(
        self,
        din: int,
        hid: int = 64,
        layers: int = 2,
        encoder: str = "gine",
        decoder: str = "mlp2",
        use_edges: bool = True,
        edim: int = 3,
        norm: str = "batchnorm",
        mitigation: str = "none",
        dropedge: float = 0.2,
    ):
        super().__init__()
        if encoder == "gine":
            use_edges = True
        edim_eff = edim if (use_edges and encoder in EDGE_ENCODERS) else None
        self.din = din
        self.mask_token = nn.Parameter(torch.zeros(din))
        self.enc = Encoder(din, hid, layers, encoder, edim_eff, norm, mitigation, dropedge)
        self.dec = Decoder(decoder, hid, din)

    def encode(self, x: torch.Tensor, ei: torch.Tensor, ea: torch.Tensor | None = None) -> torch.Tensor:
        return self.enc(x, ei, ea if self.enc.uses_edges else None)

    def forward_node(
        self, x: torch.Tensor, ei: torch.Tensor, ea: torch.Tensor | None, mask: torch.Tensor
    ) -> torch.Tensor:
        """Masked-feature reconstruction: replace masked rows with the [MASK] token, encode, decode back to x."""
        # ``where`` preserves broadcasting under CUDA deterministic mode.
        # Boolean indexed assignment can trigger a PyTorch CUDA shape assert
        # after a Lightning model enables deterministic algorithms globally.
        xin = torch.where(mask[:, None], self.mask_token[None, :], x)
        h = self.encode(xin, ei, ea)
        return self.dec(h, ei=ei, mask=mask)
