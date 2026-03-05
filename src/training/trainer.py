import torch
from torch.utils.data import DataLoader


def train_epoch(model, predictor, data, split_edge, optimizer,
                batch_size, node_emb=None):
    """Run one training epoch with improved negative sampling."""
    model.train()
    predictor.train()

    pos_train_edge = split_edge["train"]["edge"].to(data.x.device)

    total_loss = total_examples = 0
    loader = DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True)

    for perm in loader:
        optimizer.zero_grad()

        # Build input features
        x = data.x
        if node_emb is not None:
            x = x + node_emb.weight

        h = model(x, data.adj_t)

        # Positive edges
        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Hard negative sampling: mix random negatives with degree-biased negatives
        num_neg = edge.size(1)
        num_random = num_neg // 2
        num_degree = num_neg - num_random

        # Random negatives
        neg_random = torch.randint(
            0, data.num_nodes, (2, num_random),
            dtype=torch.long, device=h.device
        )

        # Degree-biased negatives (higher-degree nodes are harder negatives)
        if num_degree > 0:
            deg = data.adj_t.sum(dim=1).to(torch.float)
            deg_prob = deg / deg.sum()
            deg_idx = torch.multinomial(
                deg_prob, num_degree * 2, replacement=True
            )
            neg_degree = deg_idx.view(2, num_degree)
        else:
            neg_degree = torch.zeros(
                (2, 0), dtype=torch.long, device=h.device
            )

        neg_edge = torch.cat([neg_random, neg_degree], dim=1)
        neg_out = predictor(h[neg_edge[0]], h[neg_edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        if node_emb is not None:
            torch.nn.utils.clip_grad_norm_(node_emb.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples
