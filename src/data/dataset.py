from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, degree


@dataclass(frozen=True)
class EdgeFeatureStats:
    weight_mean: torch.Tensor
    weight_std: torch.Tensor
    year_mean: torch.Tensor
    year_std: torch.Tensor


def load_dataset(cfg: dict[str, Any]) -> tuple[Data, dict[str, dict[str, torch.Tensor]], PygLinkPropPredDataset]:
    """Load ogbl-collab and build train/full message-passing graphs.

    OGB already returns the transductive training graph for ogbl-collab.  This
    function keeps the official split untouched, uses node features supplied by
    OGB, and creates a second graph that optionally includes validation edges
    for test-time message passing, matching the standard OGB baseline protocol.
    """
    dataset_cfg = cfg["dataset"]
    dataset, data, split_edge = _load_ogb_dataset(
        name=dataset_cfg.get("name", "ogbl-collab"),
        root=dataset_cfg.get("root", "dataset"),
    )

    if data.x is None:
        raise ValueError("ogbl-collab is expected to provide node features, but data.x is None.")

    data.x = data.x.to(torch.float)
    if dataset_cfg.get("normalize_node_features", True):
        data.x = _standardize(data.x)

    train_edge_index = data.edge_index.to(torch.long)
    train_edge_attr = None

    if dataset_cfg.get("use_edge_features", True):
        stats = _edge_feature_stats(data)
        train_edge_attr = _make_edge_features(
            weight=getattr(data, "edge_weight", None),
            year=getattr(data, "edge_year", None),
            num_edges=train_edge_index.size(1),
            stats=stats,
        )
        full_edge_attr = train_edge_attr
    else:
        stats = None
        full_edge_attr = None

    full_edge_index = train_edge_index
    if dataset_cfg.get("use_valedges_as_input", True):
        valid_edge_index, valid_edge_attr = _validation_edges_for_input(
            split_edge=split_edge,
            stats=stats,
            use_edge_features=dataset_cfg.get("use_edge_features", True),
            device=train_edge_index.device,
        )
        full_edge_index = torch.cat([train_edge_index, valid_edge_index], dim=1)
        if train_edge_attr is not None and valid_edge_attr is not None:
            full_edge_attr = torch.cat([train_edge_attr, valid_edge_attr], dim=0)

    if dataset_cfg.get("add_degree_features", True):
        data.x = _append_degree_features(data.x, train_edge_index, data.num_nodes)

    if dataset_cfg.get("add_self_loops", True):
        fill_value = float(dataset_cfg.get("self_loop_fill_value", 0.0))
        train_edge_index, train_edge_attr = _add_self_loops(
            train_edge_index,
            train_edge_attr,
            data.num_nodes,
            fill_value,
        )
        full_edge_index, full_edge_attr = _add_self_loops(
            full_edge_index,
            full_edge_attr,
            data.num_nodes,
            fill_value,
        )

    data.train_edge_index = train_edge_index
    data.train_edge_attr = train_edge_attr
    data.full_edge_index = full_edge_index
    data.full_edge_attr = full_edge_attr
    data.edge_attr_dim = 0 if train_edge_attr is None else train_edge_attr.size(-1)

    return data, split_edge, dataset


def _load_ogb_dataset(
    name: str,
    root: str,
) -> tuple[PygLinkPropPredDataset, Data, dict[str, dict[str, torch.Tensor]]]:
    original_load = torch.load

    def compatible_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        try:
            return original_load(*args, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return original_load(*args, **kwargs)

    torch.load = compatible_load
    try:
        dataset = PygLinkPropPredDataset(name=name, root=root)
        data = dataset[0]
        split_edge = dataset.get_edge_split()
    finally:
        torch.load = original_load

    return dataset, data, split_edge


def _standardize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _append_degree_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    row = edge_index[0]
    deg = degree(row, num_nodes=num_nodes, dtype=x.dtype)
    deg = deg.clamp_min(0)
    log_deg = torch.log1p(deg)
    norm_deg = deg / deg.max().clamp_min(1)
    degree_features = torch.stack([log_deg, norm_deg], dim=-1)
    return torch.cat([x, degree_features.to(x.device)], dim=-1)


def _edge_feature_stats(data: Data, eps: float = 1e-12) -> EdgeFeatureStats:
    edge_count = data.edge_index.size(1)
    weight = _coerce_edge_vector(getattr(data, "edge_weight", None), edge_count, default=1.0)
    year = _coerce_edge_vector(getattr(data, "edge_year", None), edge_count, default=0.0)
    log_weight = torch.log1p(weight)
    year = year.to(torch.float)
    return EdgeFeatureStats(
        weight_mean=log_weight.mean(),
        weight_std=log_weight.std().clamp_min(eps),
        year_mean=year.mean(),
        year_std=year.std().clamp_min(eps),
    )


def _make_edge_features(
    weight: torch.Tensor | None,
    year: torch.Tensor | None,
    num_edges: int,
    stats: EdgeFeatureStats,
) -> torch.Tensor:
    weight_vec = _coerce_edge_vector(weight, num_edges, default=1.0)
    year_vec = _coerce_edge_vector(year, num_edges, default=0.0)
    log_weight = (torch.log1p(weight_vec) - stats.weight_mean) / stats.weight_std
    norm_year = (year_vec.to(torch.float) - stats.year_mean) / stats.year_std
    return torch.stack([log_weight, norm_year], dim=-1).to(torch.float)


def _coerce_edge_vector(
    value: torch.Tensor | None,
    num_edges: int,
    default: float,
) -> torch.Tensor:
    if value is None:
        return torch.full((num_edges,), default, dtype=torch.float)
    value = value.view(-1).to(torch.float)
    if value.numel() != num_edges:
        raise ValueError(f"Expected {num_edges} edge values, got {value.numel()}.")
    return value


def _validation_edges_for_input(
    split_edge: dict[str, dict[str, torch.Tensor]],
    stats: EdgeFeatureStats | None,
    use_edge_features: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    valid = split_edge["valid"]
    edge = valid["edge"].to(device).t().contiguous()
    edge = torch.cat([edge, edge.flip(0)], dim=1)

    if not use_edge_features:
        return edge, None

    if stats is None:
        raise ValueError("Edge feature statistics are required when use_edge_features=True.")

    num_valid = valid["edge"].size(0)
    attr = _make_edge_features(
        weight=valid.get("weight"),
        year=valid.get("year"),
        num_edges=num_valid,
        stats=stats,
    )
    attr = torch.cat([attr, attr], dim=0).to(device)
    return edge, attr


def _add_self_loops(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    num_nodes: int,
    fill_value: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return add_remaining_self_loops(
        edge_index=edge_index,
        edge_attr=edge_attr,
        fill_value=fill_value,
        num_nodes=num_nodes,
    )
