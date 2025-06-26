import torch
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import argparse
from tqdm import trange, tqdm

from logger import Logger
from model import GAT, LinkPredictor
from train_eval import train, test

original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GAT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--use_valedges_as_input', action='store_true')
    args = parser.parse_args([])

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset = PygLinkPropPredDataset(name='ogbl-collab')
    data = dataset[0]
    edge_index = data.edge_index
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = T.ToSparseTensor()(data)

    split_edge = dataset.get_edge_split()

    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t

    data = data.to(device)

    if not hasattr(data, 'x') or data.x is None:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float).to(device)

    in_channels = data.x.size(-1)
    out_channels = args.hidden_channels * args.heads

    model = GAT(in_channels, args.hidden_channels,
                args.hidden_channels, args.num_layers,
                args.dropout, heads=args.heads).to(device)

    predictor = LinkPredictor(out_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@50': Logger(args.runs, args),
    }

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in trange(1, 1 + args.epochs, desc=f'Run {run+1}'):
            loss = train(model, predictor, data, split_edge,
                         optimizer, args.batch_size, epoch)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, split_edge, evaluator,
                               args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        tqdm.write(key)
                        tqdm.write(f'Run: {run + 1:02d}, '
                                   f'Epoch: {epoch:03d}, '
                                   f'Loss: {loss:.4f}, '
                                   f'Train: {100 * train_hits:.2f}%, '
                                   f'Valid: {100 * valid_hits:.2f}%, '
                                   f'Test: {100 * test_hits:.2f}%')
                    tqdm.write('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

if __name__ == "__main__":
    main()
