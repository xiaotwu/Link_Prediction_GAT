import torch
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset


def load_dataset(cfg):
    """Load and prepare the ogbl-collab dataset."""
    # Patch torch.load for compatibility
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load

    dataset = PygLinkPropPredDataset(name=cfg["dataset"]["name"],
                                     root=cfg["dataset"]["root"])
    data = dataset[0]
    edge_index = data.edge_index
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = T.ToSparseTensor()(data)

    split_edge = dataset.get_edge_split()

    # Optionally use validation edges as input (common trick for ogbl-collab)
    if cfg["dataset"]["use_valedges_as_input"]:
        val_edge_index = split_edge["valid"]["edge"].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t

    return data, split_edge, dataset


def prepare_features(data, cfg, device):
    """Prepare node features: degree-based features + optional learnable embeddings."""
    # Compute degree features from adjacency
    adj_t = data.adj_t
    deg = adj_t.sum(dim=1).to(torch.float)

    # Log-scale degree (more stable for high-degree nodes)
    log_deg = torch.log(deg + 1.0).unsqueeze(-1)

    # Normalized degree
    max_deg = deg.max()
    norm_deg = (deg / max_deg).unsqueeze(-1)

    # Combine into feature matrix
    x = torch.cat([log_deg, norm_deg], dim=-1).to(device)

    data.x = x
    data = data.to(device)

    return data
