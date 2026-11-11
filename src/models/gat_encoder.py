from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, JumpingKnowledge


NormName = Literal["batch", "layer", "none"]
GATName = Literal["gat", "gatv2"]
JKName = Literal["cat", "max", "lstm", "last"]


class GATEncoder(nn.Module):
    """Edge-aware GAT encoder for ogbl-collab link prediction.

    ``hidden_channels`` is interpreted as the total width of each hidden layer,
    not the width per attention head.  For example, ``hidden_channels=256`` and
    ``heads=4`` creates four 64-dimensional heads whose outputs are concatenated
    back to 256 dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int = 4,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        edge_dim: int | None = None,
        gat_type: GATName = "gatv2",
        residual: bool = True,
        norm: NormName = "layer",
        jk: JKName = "cat",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads.")
        if jk not in {"cat", "max", "lstm", "last"}:
            raise ValueError(f"Unsupported Jumping Knowledge mode: {jk}")

        self.dropout = dropout
        self.residual = residual
        self.jk_mode = jk
        self.out_channels = out_channels

        conv_cls = GATv2Conv if gat_type == "gatv2" else GATConv
        head_channels = hidden_channels // heads

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                conv_cls(
                    hidden_channels,
                    head_channels,
                    heads=heads,
                    concat=True,
                    dropout=attention_dropout,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
            )
            self.norms.append(_make_norm(norm, hidden_channels))
            self.skips.append(nn.Identity())

        self.jk = (
            nn.Identity()
            if jk == "last"
            else JumpingKnowledge(jk, channels=hidden_channels, num_layers=num_layers)
        )
        jk_channels = hidden_channels * num_layers if jk == "cat" else hidden_channels
        self.output_proj = nn.Linear(jk_channels, out_channels)

    def reset_parameters(self) -> None:
        self.input_proj.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        if hasattr(self.jk, "reset_parameters"):
            self.jk.reset_parameters()
        self.output_proj.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        layer_outputs = []

        for conv, norm, skip in zip(self.convs, self.norms, self.skips):
            residual = skip(x)
            out = conv(x, edge_index, edge_attr=edge_attr)
            if self.residual:
                out = out + residual
            out = norm(out)
            out = F.elu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            x = out
            layer_outputs.append(x)

        x = layer_outputs[-1] if self.jk_mode == "last" else self.jk(layer_outputs)
        x = self.output_proj(x)
        return x


def _make_norm(name: NormName, channels: int) -> nn.Module:
    if name == "batch":
        return nn.BatchNorm1d(channels)
    if name == "layer":
        return nn.LayerNorm(channels)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {name}")
