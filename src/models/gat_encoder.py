import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class GATEncoder(nn.Module):
    """
    Enhanced GAT encoder with:
    - GATv2Conv (dynamic attention) or standard GATConv
    - Residual (skip) connections
    - Layer normalization
    - Jumping Knowledge aggregation (cat, max, last)
    - Learnable node embeddings for featureless graphs
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout=0.3,
        attn_dropout=0.1,
        heads=4,
        use_gatv2=True,
        residual=True,
        layer_norm=True,
        jk_mode="cat",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.use_layer_norm = layer_norm
        self.jk_mode = jk_mode

        ConvLayer = GATv2Conv if use_gatv2 else GATConv

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_lins = nn.ModuleList()

        # First layer
        self.convs.append(
            ConvLayer(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=attn_dropout,
                concat=True,
            )
        )
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        self.skip_lins.append(nn.Linear(in_channels, hidden_channels * heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                ConvLayer(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=attn_dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
            self.skip_lins.append(
                nn.Linear(hidden_channels * heads, hidden_channels * heads)
            )

        # Last layer
        self.convs.append(
            ConvLayer(
                hidden_channels * heads,
                out_channels,
                heads=heads,
                dropout=attn_dropout,
                concat=True,
            )
        )
        self.norms.append(nn.LayerNorm(out_channels * heads))
        self.skip_lins.append(
            nn.Linear(hidden_channels * heads, out_channels * heads)
        )

        # Jumping Knowledge projection
        if jk_mode == "cat":
            self.jk_lin = nn.Linear(
                out_channels * heads * num_layers, out_channels * heads
            )
        elif jk_mode == "max":
            # All intermediate representations must have the same dim
            # We project each layer to the final dim
            self.jk_projs = nn.ModuleList()
            for i in range(num_layers - 1):
                self.jk_projs.append(
                    nn.Linear(hidden_channels * heads, out_channels * heads)
                )
            self.jk_projs.append(nn.Identity())
        # jk_mode == "last" needs no extra parameters

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        for lin in self.skip_lins:
            lin.reset_parameters()
        if self.jk_mode == "cat":
            self.jk_lin.reset_parameters()
        elif self.jk_mode == "max":
            for proj in self.jk_projs:
                if hasattr(proj, "reset_parameters"):
                    proj.reset_parameters()

    def forward(self, x, adj_t):
        layer_outputs = []

        for i in range(self.num_layers):
            x_in = x
            x = self.convs[i](x, adj_t)

            if self.use_layer_norm:
                x = self.norms[i](x)

            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.residual:
                x = x + self.skip_lins[i](x_in)

            layer_outputs.append(x)

        # Jumping Knowledge aggregation
        if self.jk_mode == "cat":
            x = torch.cat(layer_outputs, dim=-1)
            x = self.jk_lin(x)
        elif self.jk_mode == "max":
            projected = [
                self.jk_projs[i](layer_outputs[i])
                for i in range(self.num_layers)
            ]
            x = torch.stack(projected, dim=0).max(dim=0)[0]
        else:  # "last"
            x = layer_outputs[-1]

        return x

    @property
    def out_dim(self):
        return self.convs[-1].out_channels * self.convs[-1].heads
