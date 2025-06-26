import torch
from torch.utils.data import DataLoader

def train(model, predictor, data, split_edge, optimizer, batch_size, epoch):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    loader = DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True)

    for perm in loader:
        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long, device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_pred = torch.cat([
        predictor(h[e[0]], h[e[1]]).squeeze().cpu()
        for e in [pos_train_edge[perm].t() for perm in DataLoader(range(pos_train_edge.size(0)), batch_size)]
    ], dim=0)

    pos_valid_pred = torch.cat([
        predictor(h[e[0]], h[e[1]]).squeeze().cpu()
        for e in [pos_valid_edge[perm].t() for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size)]
    ], dim=0)

    neg_valid_pred = torch.cat([
        predictor(h[e[0]], h[e[1]]).squeeze().cpu()
        for e in [neg_valid_edge[perm].t() for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size)]
    ], dim=0)

    h = model(data.x, data.full_adj_t)

    pos_test_pred = torch.cat([
        predictor(h[e[0]], h[e[1]]).squeeze().cpu()
        for e in [pos_test_edge[perm].t() for perm in DataLoader(range(pos_test_edge.size(0)), batch_size)]
    ], dim=0)

    neg_test_pred = torch.cat([
        predictor(h[e[0]], h[e[1]]).squeeze().cpu()
        for e in [neg_test_edge[perm].t() for perm in DataLoader(range(neg_test_edge.size(0)), batch_size)]
    ], dim=0)

    evaluator.K = 50
    results = {
        'Hits@50': (
            evaluator.eval({'y_pred_pos': pos_train_pred, 'y_pred_neg': neg_valid_pred})['hits@50'],
            evaluator.eval({'y_pred_pos': pos_valid_pred, 'y_pred_neg': neg_valid_pred})['hits@50'],
            evaluator.eval({'y_pred_pos': pos_test_pred,  'y_pred_neg': neg_test_pred})['hits@50']
        )
    }

    return results
