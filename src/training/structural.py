from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy import sparse


DEFAULT_FEATURES = ("cn", "aa", "ra", "jaccard", "pa")


@dataclass
class StructuralFeatureStore:
    """Sparse common-neighbor style features for candidate links."""

    adj: sparse.csr_matrix
    feature_names: tuple[str, ...] = DEFAULT_FEATURES

    def __post_init__(self) -> None:
        degree = np.asarray(self.adj.sum(axis=1)).reshape(-1).astype(np.float64)
        self.degree = degree

        inv_degree = np.zeros_like(degree)
        mask = degree > 0
        inv_degree[mask] = 1.0 / degree[mask]
        self.inv_degree = inv_degree

        inv_log_degree = np.zeros_like(degree)
        mask = degree > 1
        inv_log_degree[mask] = 1.0 / np.log(degree[mask])
        self.inv_log_degree = inv_log_degree

    @classmethod
    def from_edge_index(
        cls,
        edge_index: torch.Tensor,
        num_nodes: int,
        feature_names: Iterable[str] = DEFAULT_FEATURES,
    ) -> "StructuralFeatureStore":
        edge_index = edge_index.detach().cpu()
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        values = np.ones(src.shape[0], dtype=np.float32)
        adj = sparse.csr_matrix((values, (src, dst)), shape=(num_nodes, num_nodes))
        adj.sum_duplicates()
        adj.data[:] = 1.0
        return cls(adj=adj, feature_names=tuple(feature_names))

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def edge_features(
        self,
        edges: torch.Tensor,
        device: torch.device | None = None,
        chunk_size: int = 200_000,
    ) -> torch.Tensor:
        if edges.dim() != 2:
            raise ValueError("edges must be a 2D tensor shaped [num_edges, 2] or [2, num_edges].")
        if edges.size(0) == 2 and edges.size(1) != 2:
            edges = edges.t()
        edges_np = edges.detach().cpu().numpy()

        chunks = []
        for start in range(0, edges_np.shape[0], chunk_size):
            chunk = edges_np[start : start + chunk_size]
            chunks.append(self._edge_features_numpy(chunk[:, 0], chunk[:, 1]))

        out = np.concatenate(chunks, axis=0) if chunks else np.empty((0, self.num_features))
        tensor = torch.from_numpy(out.astype(np.float32, copy=False))
        return tensor if device is None else tensor.to(device)

    def feature_scores(
        self,
        edges: torch.Tensor,
        feature_name: str,
        chunk_size: int = 200_000,
    ) -> torch.Tensor:
        features = self.edge_features(edges, chunk_size=chunk_size)
        try:
            idx = self.feature_names.index(feature_name)
        except ValueError as exc:
            raise ValueError(f"Unknown structural feature: {feature_name}") from exc
        return features[:, idx]

    def _edge_features_numpy(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        intersection = self.adj[src].multiply(self.adj[dst])
        cn = np.asarray(intersection.sum(axis=1)).reshape(-1).astype(np.float64)
        ra = np.asarray(intersection @ self.inv_degree).reshape(-1).astype(np.float64)
        aa = np.asarray(intersection @ self.inv_log_degree).reshape(-1).astype(np.float64)

        deg_src = self.degree[src]
        deg_dst = self.degree[dst]
        union = deg_src + deg_dst - cn
        jaccard = np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)
        pa = deg_src * deg_dst

        values = {
            "cn": np.log1p(cn),
            "aa": np.log1p(aa),
            "ra": np.log1p(ra),
            "jaccard": jaccard,
            "pa": np.log1p(pa),
        }
        return np.stack([values[name] for name in self.feature_names], axis=1)
