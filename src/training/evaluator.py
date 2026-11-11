from __future__ import annotations

import torch
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from src.models import LinkPredictionGAT, LinkPredictor
from src.training.structural import StructuralFeatureStore


@torch.no_grad()
def evaluate(
    model: LinkPredictionGAT,
    predictor: LinkPredictor,
    data: Data,
    split_edge: dict[str, dict[str, torch.Tensor]],
    evaluator: Evaluator,
    batch_size: int,
    metrics: list[int],
    max_train_edges: int | None = None,
    use_full_graph_for_test: bool = True,
    train_structural_features: StructuralFeatureStore | None = None,
    full_structural_features: StructuralFeatureStore | None = None,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, torch.Tensor]]:
    """Evaluate with the official OGB fixed negative edges."""
    model.eval()
    predictor.eval()

    h_train = model.encode(data.x, data.train_edge_index, data.train_edge_attr)

    pos_train_edge = _select_train_edges(
        split_edge["train"]["edge"],
        max_train_edges,
    )
    pos_valid_edge = split_edge["valid"]["edge"]
    neg_valid_edge = split_edge["valid"]["edge_neg"]
    pos_test_edge = split_edge["test"]["edge"]
    neg_test_edge = split_edge["test"]["edge_neg"]

    pos_train_pred = predict_edges(
        predictor, h_train, pos_train_edge, batch_size, train_structural_features
    )
    pos_valid_pred = predict_edges(
        predictor, h_train, pos_valid_edge, batch_size, train_structural_features
    )
    neg_valid_pred = predict_edges(
        predictor, h_train, neg_valid_edge, batch_size, train_structural_features
    )

    if use_full_graph_for_test:
        h_test = model.encode(data.x, data.full_edge_index, data.full_edge_attr)
    else:
        h_test = h_train
    test_structural_features = (
        full_structural_features
        if use_full_graph_for_test and full_structural_features is not None
        else train_structural_features
    )

    pos_test_pred = predict_edges(
        predictor, h_test, pos_test_edge, batch_size, test_structural_features
    )
    neg_test_pred = predict_edges(
        predictor, h_test, neg_test_edge, batch_size, test_structural_features
    )

    results = {}
    for k in metrics:
        evaluator.K = int(k)
        key = f"hits@{int(k)}"
        train_hits = evaluator.eval(
            {"y_pred_pos": pos_train_pred, "y_pred_neg": neg_valid_pred}
        )[key]
        valid_hits = evaluator.eval(
            {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred}
        )[key]
        test_hits = evaluator.eval(
            {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred}
        )[key]
        results[f"Hits@{int(k)}"] = (train_hits, valid_hits, test_hits)

    predictions = {
        "pos_train": pos_train_pred,
        "pos_valid": pos_valid_pred,
        "neg_valid": neg_valid_pred,
        "pos_test": pos_test_pred,
        "neg_test": neg_test_pred,
    }
    return results, predictions


@torch.no_grad()
def predict_edges(
    predictor: LinkPredictor,
    h: torch.Tensor,
    edges: torch.Tensor,
    batch_size: int,
    structural_features: StructuralFeatureStore | None = None,
) -> torch.Tensor:
    preds = []
    loader = DataLoader(range(edges.size(0)), batch_size=batch_size, num_workers=0)
    for perm in loader:
        edge = edges[perm].to(h.device).t()
        edge_features = (
            None
            if structural_features is None
            else structural_features.edge_features(edge, device=h.device)
        )
        preds.append(predictor(h[edge[0]], h[edge[1]], edge_features).detach().cpu())
    return torch.cat(preds, dim=0)


def _select_train_edges(
    edges: torch.Tensor,
    max_train_edges: int | None,
) -> torch.Tensor:
    if max_train_edges is None or max_train_edges <= 0 or edges.size(0) <= max_train_edges:
        return edges
    return edges[:max_train_edges]
