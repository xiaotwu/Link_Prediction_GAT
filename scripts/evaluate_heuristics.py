#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from ogb.linkproppred import Evaluator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset
from src.training import StructuralFeatureStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate structural heuristics on ogbl-collab")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--features", nargs="+", default=["cn", "aa", "ra", "jaccard", "pa"])
    parser.add_argument("--batch-size", type=int, default=200_000)
    return parser.parse_args()


def score(
    store: StructuralFeatureStore,
    edges: torch.Tensor,
    feature_name: str,
    batch_size: int,
) -> torch.Tensor:
    return store.feature_scores(edges, feature_name, chunk_size=batch_size)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["structural_features"] = {"enabled": True, "features": args.features}
    cfg["dataset"]["use_valedges_as_input"] = True

    data, split_edge, _ = load_dataset(cfg)
    train_store = StructuralFeatureStore.from_edge_index(
        data.train_edge_index,
        data.num_nodes,
        feature_names=args.features,
    )
    full_store = StructuralFeatureStore.from_edge_index(
        data.full_edge_index,
        data.num_nodes,
        feature_names=args.features,
    )

    evaluator = Evaluator(name=cfg["dataset"]["name"])
    evaluator.K = 50
    pos_valid = split_edge["valid"]["edge"]
    neg_valid = split_edge["valid"]["edge_neg"]
    pos_test = split_edge["test"]["edge"]
    neg_test = split_edge["test"]["edge_neg"]

    print("Feature,Valid Hits@50,Test Hits@50")
    for feature_name in args.features:
        valid = evaluator.eval(
            {
                "y_pred_pos": score(train_store, pos_valid, feature_name, args.batch_size),
                "y_pred_neg": score(train_store, neg_valid, feature_name, args.batch_size),
            }
        )["hits@50"]
        test = evaluator.eval(
            {
                "y_pred_pos": score(full_store, pos_test, feature_name, args.batch_size),
                "y_pred_neg": score(full_store, neg_test, feature_name, args.batch_size),
            }
        )["hits@50"]
        print(f"{feature_name},{100 * valid:.4f},{100 * test:.4f}")


if __name__ == "__main__":
    main()
