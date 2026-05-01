from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkPredictor(nn.Module):
    """Symmetric MLP decoder for undirected collaboration links.

    The forward pass returns logits.  Training uses
    ``binary_cross_entropy_with_logits`` and evaluation ranks the raw logits,
    which is numerically safer than applying a sigmoid before the loss.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        use_batch_norm: bool = True,
        edge_feature_channels: int = 0,
        edge_skip_weights: list[float] | None = None,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.edge_feature_channels = edge_feature_channels
        edge_channels = 4 * in_channels + edge_feature_channels
        channels = [edge_channels]
        channels.extend([hidden_channels] * max(num_layers - 1, 0))
        channels.append(1)

        self.lins = nn.ModuleList(
            nn.Linear(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        )
        self.norms = nn.ModuleList(
            nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity()
            for _ in range(max(num_layers - 1, 0))
        )
        self.dropout = dropout
        self.zero_output = zero_output
        self.edge_skip = (
            nn.Linear(edge_feature_channels, 1, bias=False)
            if edge_feature_channels and edge_skip_weights is not None
            else None
        )
        if self.edge_skip is not None:
            if len(edge_skip_weights) != edge_feature_channels:
                raise ValueError("edge_skip_weights must match edge_feature_channels.")
            initial = torch.tensor(edge_skip_weights, dtype=torch.float).view(1, -1)
            self.register_buffer("edge_skip_initial", initial)
        else:
            self.edge_skip_initial = None

    def reset_parameters(self) -> None:
        for lin in self.lins:
            lin.reset_parameters()
        if self.zero_output:
            nn.init.zeros_(self.lins[-1].weight)
            nn.init.zeros_(self.lins[-1].bias)
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        if self.edge_skip is not None:
            with torch.no_grad():
                self.edge_skip.weight.copy_(self.edge_skip_initial)

    def forward(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pieces = [
            x_i * x_j,
            torch.abs(x_i - x_j),
            x_i + x_j,
            torch.maximum(x_i, x_j),
        ]
        if self.edge_feature_channels:
            if edge_features is None:
                raise ValueError("edge_features are required by this predictor.")
            pieces.append(edge_features)
        x = torch.cat(pieces, dim=-1)

        for lin, norm in zip(self.lins[:-1], self.norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.lins[-1](x).view(-1)
        if self.edge_skip is not None:
            out = out + self.edge_skip(edge_features).view(-1)
        return out

    @torch.no_grad()
    def predict_proba(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_i, x_j, edge_features=edge_features))
