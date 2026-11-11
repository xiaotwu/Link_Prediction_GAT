from __future__ import annotations

import torch
import torch.nn as nn

from .gat_encoder import GATEncoder


class LinkPredictionGAT(nn.Module):
    """Feature adapter plus GAT encoder.

    The model uses the real OGB author features and can concatenate a learned
    transductive node embedding.  Keeping this adapter inside the model ensures
    those parameters are optimized jointly with the GAT rather than being a
    one-off random projection.
    """

    def __init__(
        self,
        num_nodes: int,
        input_channels: int,
        feature_channels: int,
        embedding_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int,
        dropout: float,
        input_dropout: float,
        attention_dropout: float,
        edge_dim: int | None,
        gat_type: str,
        residual: bool,
        norm: str,
        jk: str,
    ) -> None:
        super().__init__()
        self.input_dropout = input_dropout
        self.feature_proj = nn.Linear(input_channels, feature_channels)
        self.node_embedding = (
            nn.Embedding(num_nodes, embedding_channels)
            if embedding_channels > 0
            else None
        )
        encoder_in_channels = feature_channels + max(embedding_channels, 0)
        self.encoder = GATEncoder(
            in_channels=encoder_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            edge_dim=edge_dim,
            gat_type=gat_type,
            residual=residual,
            norm=norm,
            jk=jk,
        )
        self.dropout = nn.Dropout(input_dropout)
        self.out_channels = out_channels

    def reset_parameters(self) -> None:
        self.feature_proj.reset_parameters()
        if self.node_embedding is not None:
            nn.init.xavier_uniform_(self.node_embedding.weight)
        self.encoder.reset_parameters()

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.feature_proj(x)
        if self.node_embedding is not None:
            embedding = self.node_embedding.weight
            x = torch.cat([x, embedding], dim=-1)
        x = self.dropout(x)
        return self.encoder(x, edge_index, edge_attr=edge_attr)
