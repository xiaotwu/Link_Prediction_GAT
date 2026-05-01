from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch_geometric.utils import degree


SamplerName = Literal["random", "degree", "mixed"]


@dataclass
class NegativeSampler:
    num_nodes: int
    ratio: float = 1.0
    mode: SamplerName = "mixed"
    degree_fraction: float = 0.5
    degree_prob: torch.Tensor | None = None

    @classmethod
    def from_graph(
        cls,
        edge_index: torch.Tensor,
        num_nodes: int,
        ratio: float,
        mode: SamplerName,
        degree_fraction: float,
    ) -> "NegativeSampler":
        deg = degree(edge_index[0].cpu(), num_nodes=num_nodes, dtype=torch.float)
        deg = deg.clamp_min(1.0)
        degree_prob = deg / deg.sum()
        return cls(
            num_nodes=num_nodes,
            ratio=ratio,
            mode=mode,
            degree_fraction=degree_fraction,
            degree_prob=degree_prob,
        )

    def to(self, device: torch.device) -> "NegativeSampler":
        if self.degree_prob is not None:
            self.degree_prob = self.degree_prob.to(device)
        return self

    def sample(self, num_positive: int, device: torch.device) -> torch.Tensor:
        num_negative = max(1, math.ceil(num_positive * self.ratio))
        if self.mode == "random":
            edge = self._sample_random(num_negative, device)
        elif self.mode == "degree":
            edge = self._sample_degree(num_negative, device)
        elif self.mode == "mixed":
            degree_count = int(num_negative * self.degree_fraction)
            random_count = num_negative - degree_count
            edge = torch.cat(
                [
                    self._sample_random(random_count, device),
                    self._sample_degree(degree_count, device),
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unsupported negative sampler: {self.mode}")

        return _remove_self_samples(edge, self.num_nodes)

    def _sample_random(self, count: int, device: torch.device) -> torch.Tensor:
        if count <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.randint(
            0,
            self.num_nodes,
            (2, count),
            dtype=torch.long,
            device=device,
        )

    def _sample_degree(self, count: int, device: torch.device) -> torch.Tensor:
        if count <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        if self.degree_prob is None:
            return self._sample_random(count, device)
        prob = self.degree_prob.to(device)
        samples = torch.multinomial(prob, 2 * count, replacement=True)
        return samples.view(2, count)


def _remove_self_samples(edge: torch.Tensor, num_nodes: int) -> torch.Tensor:
    same = edge[0] == edge[1]
    if same.any():
        edge[1, same] = (edge[1, same] + 1) % num_nodes
    return edge
