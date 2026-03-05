import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkPredictor(nn.Module):
    """
    Enhanced link predictor that combines multiple edge features:
    - Hadamard (element-wise) product
    - L1 distance
    - Average of embeddings

    Uses a multi-layer MLP with batch normalization for scoring.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout):
        super().__init__()

        # Input: hadamard + l1_dist + avg = 3 * in_channels
        combined_dim = 3 * in_channels

        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.lins.append(nn.Linear(combined_dim, hidden_channels))
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x_i, x_j):
        hadamard = x_i * x_j
        l1_dist = torch.abs(x_i - x_j)
        avg = (x_i + x_j) / 2.0

        x = torch.cat([hadamard, l1_dist, avg], dim=-1)

        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)
        return torch.sigmoid(x)
