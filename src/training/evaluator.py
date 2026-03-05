import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, predictor, data, split_edge, ogb_evaluator,
             batch_size, metrics, node_emb=None):
    """Evaluate model on train/valid/test splits for multiple Hits@K metrics."""
    model.eval()
    predictor.eval()

    x = data.x
    if node_emb is not None:
        x = x + node_emb.weight

    h = model(x, data.adj_t)

    pos_train_edge = split_edge["train"]["edge"].to(h.device)
    pos_valid_edge = split_edge["valid"]["edge"].to(h.device)
    neg_valid_edge = split_edge["valid"]["edge_neg"].to(h.device)
    pos_test_edge = split_edge["test"]["edge"].to(h.device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(h.device)

    def predict_edges(h, edges, batch_size):
        preds = []
        loader = DataLoader(range(edges.size(0)), batch_size)
        for perm in loader:
            e = edges[perm].t()
            preds.append(predictor(h[e[0]], h[e[1]]).squeeze().cpu())
        return torch.cat(preds, dim=0)

    pos_train_pred = predict_edges(h, pos_train_edge, batch_size)
    pos_valid_pred = predict_edges(h, pos_valid_edge, batch_size)
    neg_valid_pred = predict_edges(h, neg_valid_edge, batch_size)

    # For test: use full_adj_t (includes validation edges)
    h_full = model(x, data.full_adj_t)
    pos_test_pred = predict_edges(h_full, pos_test_edge, batch_size)
    neg_test_pred = predict_edges(h_full, neg_test_edge, batch_size)

    results = {}
    for metric in metrics:
        K = int(metric.split("@")[1])
        ogb_evaluator.K = K
        key = f"hits@{K}"

        train_hits = ogb_evaluator.eval({
            "y_pred_pos": pos_train_pred,
            "y_pred_neg": neg_valid_pred,
        })[key]

        valid_hits = ogb_evaluator.eval({
            "y_pred_pos": pos_valid_pred,
            "y_pred_neg": neg_valid_pred,
        })[key]

        test_hits = ogb_evaluator.eval({
            "y_pred_pos": pos_test_pred,
            "y_pred_neg": neg_test_pred,
        })[key]

        results[f"Hits@{K}"] = (train_hits, valid_hits, test_hits)

    # Also return raw predictions for visualization
    raw_preds = {
        "pos_train_pred": pos_train_pred,
        "pos_valid_pred": pos_valid_pred,
        "neg_valid_pred": neg_valid_pred,
        "pos_test_pred": pos_test_pred,
        "neg_test_pred": neg_test_pred,
    }

    return results, raw_preds
