from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from src.models import LinkPredictionGAT, LinkPredictor
from src.training.sampling import NegativeSampler
from src.training.structural import StructuralFeatureStore


def train_epoch(
    model: LinkPredictionGAT,
    predictor: LinkPredictor,
    data: Data,
    split_edge: dict[str, dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    negative_sampler: NegativeSampler,
    grad_clip: float | None = 1.0,
    encode_once: bool = False,
    structural_features: StructuralFeatureStore | None = None,
) -> float:
    """Train for one epoch.

    The default path mirrors the official OGB full-batch GNN baseline: each
    edge mini-batch gets a fresh graph encoding and an optimizer step.  The
    optional ``encode_once`` mode computes the graph encoding once and performs
    a single optimizer step per epoch; it is faster but changes the optimizer
    schedule, so it is opt-in.
    """
    model.train()
    predictor.train()

    pos_train_edge = split_edge["train"]["edge"]
    loader = DataLoader(
        range(pos_train_edge.size(0)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    if encode_once:
        return _train_epoch_encode_once(
            model,
            predictor,
            data,
            pos_train_edge,
            optimizer,
            loader,
            negative_sampler,
            grad_clip,
            structural_features,
        )

    total_loss = 0.0
    total_examples = 0
    for perm in loader:
        optimizer.zero_grad(set_to_none=True)
        h = model.encode(data.x, data.train_edge_index, data.train_edge_attr)
        loss, examples = _batch_loss(
            predictor=predictor,
            h=h,
            pos_edge=pos_train_edge[perm].to(h.device).t(),
            negative_sampler=negative_sampler,
            structural_features=structural_features,
        )
        loss.backward()
        _clip_gradients(model, predictor, grad_clip)
        optimizer.step()

        total_loss += loss.item() * examples
        total_examples += examples

    return total_loss / max(total_examples, 1)


def _train_epoch_encode_once(
    model: LinkPredictionGAT,
    predictor: LinkPredictor,
    data: Data,
    pos_train_edge: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loader: Iterable[torch.Tensor],
    negative_sampler: NegativeSampler,
    grad_clip: float | None,
    structural_features: StructuralFeatureStore | None,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    h = model.encode(data.x, data.train_edge_index, data.train_edge_attr)

    batches = list(loader)
    total_loss = 0.0
    total_examples = 0
    total_edges = pos_train_edge.size(0)

    for index, perm in enumerate(batches):
        loss, examples = _batch_loss(
            predictor=predictor,
            h=h,
            pos_edge=pos_train_edge[perm].to(h.device).t(),
            negative_sampler=negative_sampler,
            structural_features=structural_features,
        )
        weighted_loss = loss * (examples / total_edges)
        weighted_loss.backward(retain_graph=index + 1 < len(batches))
        total_loss += loss.item() * examples
        total_examples += examples

    _clip_gradients(model, predictor, grad_clip)
    optimizer.step()
    return total_loss / max(total_examples, 1)


def _batch_loss(
    predictor: LinkPredictor,
    h: torch.Tensor,
    pos_edge: torch.Tensor,
    negative_sampler: NegativeSampler,
    structural_features: StructuralFeatureStore | None = None,
) -> tuple[torch.Tensor, int]:
    neg_edge = negative_sampler.sample(pos_edge.size(1), h.device)
    pos_edge_features = _edge_features(structural_features, pos_edge, h.device)
    neg_edge_features = _edge_features(structural_features, neg_edge, h.device)

    pos_logits = predictor(h[pos_edge[0]], h[pos_edge[1]], pos_edge_features)
    neg_logits = predictor(h[neg_edge[0]], h[neg_edge[1]], neg_edge_features)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits,
        torch.ones_like(pos_logits),
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits,
        torch.zeros_like(neg_logits),
    )
    return pos_loss + neg_loss, pos_logits.numel()


def _edge_features(
    structural_features: StructuralFeatureStore | None,
    edge: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    if structural_features is None:
        return None
    return structural_features.edge_features(edge, device=device)


def _clip_gradients(
    model: LinkPredictionGAT,
    predictor: LinkPredictor,
    grad_clip: float | None,
) -> None:
    if grad_clip is None or grad_clip <= 0:
        return
    parameters = list(model.parameters()) + list(predictor.parameters())
    torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
