"""
Main training script for Link Prediction with GAT on ogbl-collab.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --epochs 200 --lr 0.0005
"""

import argparse
import os
import sys
import json
import math

import yaml
import torch
from ogb.linkproppred import Evaluator
from tqdm import trange, tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import GATEncoder, LinkPredictor
from src.data import load_dataset, prepare_features
from src.training import train_epoch, evaluate
from src.utils import Logger, set_seed


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs,
                                     min_lr=1e-5):
    """Cosine annealing LR scheduler with linear warmup."""
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / base_lr, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Link Prediction with GAT")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    # Allow CLI overrides for key hyperparameters
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_channels", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--device", type=int, default=None)
    cli_args = parser.parse_args()

    # Load config
    cfg = load_config(cli_args.config)

    # Apply CLI overrides
    if cli_args.epochs is not None:
        cfg["training"]["epochs"] = cli_args.epochs
    if cli_args.lr is not None:
        cfg["training"]["lr"] = cli_args.lr
    if cli_args.hidden_channels is not None:
        cfg["model"]["hidden_channels"] = cli_args.hidden_channels
    if cli_args.heads is not None:
        cfg["model"]["heads"] = cli_args.heads
    if cli_args.runs is not None:
        cfg["experiment"]["runs"] = cli_args.runs
    if cli_args.device is not None:
        cfg["experiment"]["device"] = cli_args.device

    # Setup
    set_seed(cfg["experiment"]["seed"])
    device_id = cfg["experiment"]["device"]
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading dataset...")
    data, split_edge, dataset = load_dataset(cfg)
    data = prepare_features(data, cfg, device)
    print(f"Nodes: {data.num_nodes}, Features: {data.x.size(1)}")

    # Model config
    mcfg = cfg["model"]
    pcfg = cfg["predictor"]
    in_channels = data.x.size(-1)
    emb_dim = cfg["node_embedding"]["embedding_dim"]

    # If using learnable node embeddings, we use a linear projection to
    # map structural features to embedding dim, then add learnable embeddings.
    # The projection is a fixed (non-learned) layer applied once.
    if cfg["node_embedding"]["use_embedding"]:
        feature_proj = torch.nn.Linear(in_channels, emb_dim, bias=False).to(device)
        torch.nn.init.xavier_uniform_(feature_proj.weight)
        with torch.no_grad():
            data.x = feature_proj(data.x)
        gat_in_channels = emb_dim
    else:
        feature_proj = None
        gat_in_channels = in_channels

    out_channels = mcfg["hidden_channels"]
    encoder = GATEncoder(
        in_channels=gat_in_channels,
        hidden_channels=mcfg["hidden_channels"],
        out_channels=out_channels,
        num_layers=mcfg["num_layers"],
        dropout=mcfg["dropout"],
        attn_dropout=mcfg["attn_dropout"],
        heads=mcfg["heads"],
        use_gatv2=mcfg["use_gatv2"],
        residual=mcfg["residual"],
        layer_norm=mcfg["layer_norm"],
        jk_mode=mcfg["jk_mode"],
    ).to(device)

    encoder_out_dim = encoder.out_dim
    predictor = LinkPredictor(
        encoder_out_dim,
        pcfg["hidden_channels"],
        1,
        pcfg["num_layers"],
        pcfg["dropout"],
    ).to(device)

    # Optional learnable node embeddings
    node_emb = None
    if cfg["node_embedding"]["use_embedding"]:
        node_emb = torch.nn.Embedding(data.num_nodes, emb_dim).to(device)
        torch.nn.init.xavier_uniform_(node_emb.weight)

    ogb_evaluator = Evaluator(name=cfg["dataset"]["name"])
    metrics = cfg["evaluation"]["metrics"]

    # Logger
    runs = cfg["experiment"]["runs"]
    logger = Logger(runs, [f"Hits@{m.split('@')[1]}" for m in metrics])

    print(f"\nModel: GATEncoder ({mcfg['num_layers']} layers, "
          f"{mcfg['hidden_channels']}d, {mcfg['heads']} heads, "
          f"{'GATv2' if mcfg['use_gatv2'] else 'GAT'})")
    print(f"JK mode: {mcfg['jk_mode']}, Residual: {mcfg['residual']}, "
          f"LayerNorm: {mcfg['layer_norm']}")
    print(f"Predictor: {pcfg['num_layers']} layers, {pcfg['hidden_channels']}d")
    if node_emb:
        print(f"Node Embeddings: {emb_dim}d")

    total_params = (sum(p.numel() for p in encoder.parameters())
                    + sum(p.numel() for p in predictor.parameters()))
    if node_emb:
        total_params += node_emb.weight.numel()
    print(f"Total parameters: {total_params:,}\n")

    # Training loop
    es_cfg = cfg["early_stopping"]
    tcfg = cfg["training"]

    for run in range(runs):
        set_seed(cfg["experiment"]["seed"] + run)

        encoder.reset_parameters()
        predictor.reset_parameters()
        if node_emb is not None:
            torch.nn.init.xavier_uniform_(node_emb.weight)

        # Build parameter groups
        params = list(encoder.parameters()) + list(predictor.parameters())
        if node_emb is not None:
            params += list(node_emb.parameters())

        optimizer = torch.optim.Adam(params, lr=tcfg["lr"],
                                     weight_decay=tcfg["weight_decay"])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            cfg["scheduler"]["warmup_epochs"],
            tcfg["epochs"],
            cfg["scheduler"]["min_lr"],
        )

        best_valid = 0.0
        patience_counter = 0
        best_state = None

        for epoch in trange(1, 1 + tcfg["epochs"], desc=f"Run {run + 1}"):
            loss = train_epoch(encoder, predictor, data, split_edge,
                               optimizer, tcfg["batch_size"], node_emb)
            logger.add_loss(run, loss)
            scheduler.step()

            if epoch % cfg["evaluation"]["eval_steps"] == 0:
                results, _ = evaluate(
                    encoder, predictor, data, split_edge,
                    ogb_evaluator, tcfg["batch_size"], metrics, node_emb
                )
                logger.add_result(run, results)

                if epoch % cfg["experiment"]["log_steps"] == 0:
                    for key, (train_h, valid_h, test_h) in results.items():
                        tqdm.write(
                            f"Run {run+1:02d} | Epoch {epoch:03d} | "
                            f"Loss {loss:.4f} | {key}: "
                            f"Train {100*train_h:.2f}% "
                            f"Valid {100*valid_h:.2f}% "
                            f"Test {100*test_h:.2f}%"
                        )
                    tqdm.write(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
                    tqdm.write("---")

                # Early stopping on monitored metric
                monitor_key = f"Hits@{es_cfg['monitor'].split('@')[1]}"
                if monitor_key in results:
                    current_valid = results[monitor_key][1]
                    if current_valid > best_valid:
                        best_valid = current_valid
                        patience_counter = 0
                        if cfg["experiment"]["save_checkpoints"]:
                            best_state = {
                                "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                                "predictor": {k: v.cpu().clone() for k, v in predictor.state_dict().items()},
                                "epoch": epoch,
                                "valid_score": best_valid,
                            }
                            if node_emb is not None:
                                best_state["node_emb"] = {k: v.cpu().clone() for k, v in node_emb.state_dict().items()}
                    else:
                        patience_counter += 1

                    if es_cfg["enabled"] and patience_counter >= es_cfg["patience"]:
                        tqdm.write(
                            f"Early stopping at epoch {epoch} "
                            f"(best valid {monitor_key}: {100*best_valid:.2f}%)"
                        )
                        break

        # Save best checkpoint
        if best_state is not None and cfg["experiment"]["save_checkpoints"]:
            ckpt_dir = cfg["experiment"]["checkpoint_dir"]
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"best_run{run+1}.pt")
            torch.save(best_state, ckpt_path)
            tqdm.write(f"Saved checkpoint: {ckpt_path}")

        for metric_key in logger.results:
            logger.print_statistics(run, metric=metric_key)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for metric_key in logger.results:
        logger.print_statistics(metric=metric_key)

    # Export training logs
    log_dir = cfg["experiment"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    logger.export_json(os.path.join(log_dir, "training_log.json"))
    print(f"\nTraining logs saved to {log_dir}/training_log.json")

    # Save config used
    with open(os.path.join(log_dir, "config_used.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


if __name__ == "__main__":
    main()
