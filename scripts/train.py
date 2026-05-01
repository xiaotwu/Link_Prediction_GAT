#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from ogb.linkproppred import Evaluator
from tqdm import trange

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset
from src.models import LinkPredictionGAT, LinkPredictor
from src.training import NegativeSampler, StructuralFeatureStore, evaluate, train_epoch
from src.utils import Logger, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAT link prediction on ogbl-collab")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--no-valedges-as-input", action="store_true")
    parser.add_argument("--encode-once", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    return parser.parse_args()


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if args.device is not None:
        cfg["experiment"]["device"] = args.device
    if args.accelerator is not None:
        cfg["experiment"]["accelerator"] = args.accelerator
    if args.runs is not None:
        cfg["experiment"]["runs"] = args.runs
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.eval_steps is not None:
        cfg["evaluation"]["eval_steps"] = args.eval_steps
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.hidden_channels is not None:
        cfg["model"]["hidden_channels"] = args.hidden_channels
    if args.heads is not None:
        cfg["model"]["heads"] = args.heads
    if args.dropout is not None:
        cfg["model"]["dropout"] = args.dropout
        cfg["predictor"]["dropout"] = args.dropout
    if args.no_valedges_as_input:
        cfg["dataset"]["use_valedges_as_input"] = False
    if args.encode_once:
        cfg["training"]["encode_once_per_epoch"] = True
    if args.no_checkpoints:
        cfg["experiment"]["save_checkpoints"] = False
    return cfg


def build_model(
    cfg: dict[str, Any],
    data,
    edge_feature_channels: int = 0,
) -> tuple[LinkPredictionGAT, LinkPredictor]:
    model_cfg = cfg["model"]
    predictor_cfg = cfg["predictor"]
    edge_dim = int(getattr(data, "edge_attr_dim", 0)) or None

    model = LinkPredictionGAT(
        num_nodes=data.num_nodes,
        input_channels=data.x.size(-1),
        feature_channels=model_cfg["feature_channels"],
        embedding_channels=model_cfg["node_embedding_channels"],
        hidden_channels=model_cfg["hidden_channels"],
        out_channels=model_cfg["out_channels"],
        num_layers=model_cfg["num_layers"],
        heads=model_cfg["heads"],
        dropout=model_cfg["dropout"],
        input_dropout=model_cfg["input_dropout"],
        attention_dropout=model_cfg["attention_dropout"],
        edge_dim=edge_dim,
        gat_type=model_cfg["gat_type"],
        residual=model_cfg["residual"],
        norm=model_cfg["norm"],
        jk=model_cfg["jk"],
    )
    predictor = LinkPredictor(
        in_channels=model_cfg["out_channels"],
        hidden_channels=predictor_cfg["hidden_channels"],
        num_layers=predictor_cfg["num_layers"],
        dropout=predictor_cfg["dropout"],
        use_batch_norm=predictor_cfg["batch_norm"],
        edge_feature_channels=edge_feature_channels,
        edge_skip_weights=cfg.get("structural_features", {}).get("skip_weights"),
        zero_output=predictor_cfg.get("zero_output", False),
    )
    return model, predictor


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR | None:
    scheduler_cfg = cfg["scheduler"]
    if scheduler_cfg.get("type", "cosine") == "none":
        return None

    total_epochs = cfg["training"]["epochs"]
    warmup_epochs = scheduler_cfg.get("warmup_epochs", 0)
    min_lr = scheduler_cfg.get("min_lr", 0.0)
    base_lr = cfg["training"]["lr"]

    def lr_lambda(epoch: int) -> float:
        step = epoch + 1
        if warmup_epochs > 0 and step <= warmup_epochs:
            return step / warmup_epochs
        progress = (step - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def checkpoint_state(
    model: LinkPredictionGAT,
    predictor: LinkPredictor,
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    run: int,
    epoch: int,
    results: dict[str, tuple[float, float, float]],
) -> dict[str, Any]:
    return {
        "run": run,
        "epoch": epoch,
        "config": cfg,
        "model": model.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "results": results,
    }


def select_device(exp_cfg: dict[str, Any]) -> torch.device:
    accelerator = exp_cfg.get("accelerator", "auto")
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(f"cuda:{exp_cfg.get('device', 0)}")
    if accelerator == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if accelerator != "auto":
        raise ValueError(f"Unsupported accelerator: {accelerator}")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{exp_cfg.get('device', 0)}")
    return torch.device("cpu")


def configure_torch_runtime(exp_cfg: dict[str, Any]) -> None:
    precision = exp_cfg.get("matmul_precision")
    if precision:
        torch.set_float32_matmul_precision(precision)
    if torch.cuda.is_available() and exp_cfg.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


def format_score(values: list[float]) -> str:
    if not values:
        return ""
    scores = torch.tensor(values, dtype=torch.float) * 100
    if scores.numel() == 1:
        return f"{scores.item():.4f}"
    return f"{scores.mean().item():.4f} +/- {scores.std(unbiased=False).item():.4f}"


def write_result_sheet(
    cfg: dict[str, Any],
    logger: Logger,
    total_params: int,
    hardware: str,
) -> None:
    sheet_cfg = cfg.get("result_sheet", {})
    if not sheet_cfg:
        return

    path = PROJECT_ROOT / sheet_cfg.get("path", "results/results.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    monitor = "Hits@50" if "Hits@50" in logger.results else logger.monitor
    best = []
    for run in range(logger.runs):
        item = logger.best_for_run(run, monitor)
        if item["epoch_index"] >= 0:
            best.append(item)
    if not best:
        return

    row = {
        "Method": sheet_cfg.get("method", "GATv2"),
        "Ext. data": sheet_cfg.get("external_data", "No"),
        "Test Hits@50": format_score([item["test_at_best_valid"] for item in best]),
        "Validation Hits@50": format_score([item["best_valid"] for item in best]),
        "#Params": str(total_params),
        "Hardware": sheet_cfg.get("hardware") or hardware,
        "Date": datetime.now().strftime("%Y-%m-%d"),
    }

    fieldnames = [
        "Method",
        "Ext. data",
        "Test Hits@50",
        "Validation Hits@50",
        "#Params",
        "Hardware",
        "Date",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Result sheet updated: {path}")


def build_structural_features(
    cfg: dict[str, Any],
    data,
) -> tuple[StructuralFeatureStore | None, StructuralFeatureStore | None, int]:
    structural_cfg = cfg.get("structural_features", {})
    if not structural_cfg.get("enabled", False):
        return None, None, 0

    names = tuple(structural_cfg.get("features", ["cn", "aa", "ra", "jaccard", "pa"]))
    print(f"Building structural link features: {', '.join(names)}")
    train_store = StructuralFeatureStore.from_edge_index(
        data.train_edge_index,
        data.num_nodes,
        feature_names=names,
    )
    full_store = StructuralFeatureStore.from_edge_index(
        data.full_edge_index,
        data.num_nodes,
        feature_names=names,
    )
    return train_store, full_store, train_store.num_features


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    exp_cfg = cfg["experiment"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]

    configure_torch_runtime(exp_cfg)
    set_seed(exp_cfg["seed"], deterministic=exp_cfg.get("deterministic", False))
    device = select_device(exp_cfg)

    print(f"Using device: {device}")
    print("Loading ogbl-collab...")
    data, split_edge, _ = load_dataset(cfg)
    data = data.to(device)
    train_structural_features, full_structural_features, edge_feature_channels = (
        build_structural_features(cfg, data)
    )
    print(
        f"Graph: {data.num_nodes:,} nodes, "
        f"{data.train_edge_index.size(1):,} train message edges, "
        f"{data.x.size(-1)} input features, edge_dim={data.edge_attr_dim}"
    )

    evaluator = Evaluator(name=cfg["dataset"]["name"])
    metrics = [int(k) for k in eval_cfg["metrics"]]
    monitor_k = int(cfg["early_stopping"]["monitor"].split("@")[1])
    logger = Logger(exp_cfg["runs"], metrics, monitor=monitor_k)

    checkpoint_dir = PROJECT_ROOT / exp_cfg["checkpoint_dir"]
    log_dir = PROJECT_ROOT / exp_cfg["log_dir"]
    if exp_cfg.get("save_checkpoints", True):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_params = 0

    for run in range(exp_cfg["runs"]):
        run_seed = exp_cfg["seed"] + run
        set_seed(run_seed, deterministic=exp_cfg.get("deterministic", False))

        model, predictor = build_model(
            cfg,
            data,
            edge_feature_channels=edge_feature_channels,
        )
        model = model.to(device)
        predictor = predictor.to(device)
        model.reset_parameters()
        predictor.reset_parameters()

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(predictor.parameters()),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        scheduler = build_scheduler(optimizer, cfg)
        negative_sampler = NegativeSampler.from_graph(
            edge_index=data.train_edge_index.detach().cpu(),
            num_nodes=data.num_nodes,
            ratio=train_cfg["neg_sampling_ratio"],
            mode=train_cfg["negative_sampler"],
            degree_fraction=train_cfg["degree_negative_fraction"],
        ).to(device)

        best_valid = -1.0
        patience = 0
        best_path = checkpoint_dir / f"best_run{run + 1}.pt"
        total_params = sum(p.numel() for p in model.parameters()) + sum(
            p.numel() for p in predictor.parameters()
        )
        report_params = total_params
        print(f"\nRun {run + 1}/{exp_cfg['runs']} | seed={run_seed} | params={total_params:,}")

        if eval_cfg.get("eval_at_start", False):
            results, _ = evaluate(
                model=model,
                predictor=predictor,
                data=data,
                split_edge=split_edge,
                evaluator=evaluator,
                batch_size=eval_cfg["batch_size"],
                metrics=metrics,
                max_train_edges=eval_cfg["max_train_edges"],
                use_full_graph_for_test=eval_cfg["use_full_graph_for_test"],
                train_structural_features=train_structural_features,
                full_structural_features=full_structural_features,
            )
            lr = optimizer.param_groups[0]["lr"]
            logger.add_result(run, 0, 0.0, lr, results)
            monitor_name = f"Hits@{monitor_k}"
            best_valid = results[monitor_name][1]
            if exp_cfg.get("save_checkpoints", True):
                torch.save(
                    checkpoint_state(model, predictor, optimizer, cfg, run, 0, results),
                    best_path,
                )
            metric_text = " | ".join(
                f"{name}: train {100 * vals[0]:.2f} "
                f"valid {100 * vals[1]:.2f} test {100 * vals[2]:.2f}"
                for name, vals in results.items()
            )
            print(f"Run {run + 1:02d} Epoch 000 lr {lr:.2e} | {metric_text}")

        for epoch in trange(1, train_cfg["epochs"] + 1, desc=f"run {run + 1}"):
            loss = train_epoch(
                model=model,
                predictor=predictor,
                data=data,
                split_edge=split_edge,
                optimizer=optimizer,
                batch_size=train_cfg["batch_size"],
                negative_sampler=negative_sampler,
                grad_clip=train_cfg["grad_clip"],
                encode_once=train_cfg["encode_once_per_epoch"],
                structural_features=train_structural_features,
            )
            logger.add_loss(run, loss)
            if scheduler is not None:
                scheduler.step()

            if epoch % eval_cfg["eval_steps"] != 0:
                continue

            results, _ = evaluate(
                model=model,
                predictor=predictor,
                data=data,
                split_edge=split_edge,
                evaluator=evaluator,
                batch_size=eval_cfg["batch_size"],
                metrics=metrics,
                max_train_edges=eval_cfg["max_train_edges"],
                use_full_graph_for_test=eval_cfg["use_full_graph_for_test"],
                train_structural_features=train_structural_features,
                full_structural_features=full_structural_features,
            )
            lr = optimizer.param_groups[0]["lr"]
            logger.add_result(run, epoch, loss, lr, results)

            monitor_name = f"Hits@{monitor_k}"
            current_valid = results[monitor_name][1]
            if current_valid > best_valid:
                best_valid = current_valid
                patience = 0
                if exp_cfg.get("save_checkpoints", True):
                    torch.save(
                        checkpoint_state(model, predictor, optimizer, cfg, run, epoch, results),
                        best_path,
                    )
            else:
                patience += 1

            if epoch % exp_cfg["log_steps"] == 0:
                metric_text = " | ".join(
                    f"{name}: train {100 * vals[0]:.2f} "
                    f"valid {100 * vals[1]:.2f} test {100 * vals[2]:.2f}"
                    for name, vals in results.items()
                )
                print(
                    f"Run {run + 1:02d} Epoch {epoch:03d} "
                    f"loss {loss:.4f} lr {lr:.2e} | {metric_text}"
                )

            early_cfg = cfg["early_stopping"]
            if early_cfg["enabled"] and patience >= early_cfg["patience"]:
                print(
                    f"Early stopping run {run + 1} at epoch {epoch}; "
                    f"best valid {monitor_name}={100 * best_valid:.2f}%"
                )
                break

        logger.print_run(run, f"Hits@{monitor_k}")

    print("\nFinal summary")
    for k in metrics:
        logger.print_summary(f"Hits@{k}")

    logger.export_json(str(log_dir / "training_log.json"))
    with open(log_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Logs written to {log_dir}")
    write_result_sheet(cfg, logger, report_params, device_name(device))


if __name__ == "__main__":
    main()
