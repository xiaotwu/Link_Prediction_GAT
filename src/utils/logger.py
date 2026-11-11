from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import torch


class Logger:
    """Run logger for OGB Hits@K metrics."""

    def __init__(self, runs: int, metrics: list[int], monitor: int = 50) -> None:
        self.runs = runs
        self.metric_names = [f"Hits@{int(k)}" for k in metrics]
        self.monitor = f"Hits@{int(monitor)}"
        self.results: dict[str, list[list[tuple[float, float, float]]]] = {
            name: [[] for _ in range(runs)] for name in self.metric_names
        }
        self.losses: list[list[float]] = [[] for _ in range(runs)]
        self.records: list[list[dict[str, Any]]] = [[] for _ in range(runs)]

    def add_loss(self, run: int, loss: float) -> None:
        self.losses[run].append(float(loss))

    def add_result(
        self,
        run: int,
        epoch: int,
        loss: float,
        lr: float,
        results: dict[str, tuple[float, float, float]],
    ) -> None:
        record = {"epoch": epoch, "loss": float(loss), "lr": float(lr), "metrics": {}}
        for name, values in results.items():
            values = tuple(float(v) for v in values)
            self.results[name][run].append(values)
            record["metrics"][name] = {
                "train": values[0],
                "valid": values[1],
                "test": values[2],
            }
        self.records[run].append(record)

    def best_for_run(self, run: int, metric: str | None = None) -> dict[str, float]:
        metric = metric or self.monitor
        values = self.results[metric][run]
        if not values:
            return {"best_valid": 0.0, "test_at_best_valid": 0.0, "epoch_index": -1}
        result = torch.tensor(values)
        valid = result[:, 1]
        best_idx = int(valid.argmax().item())
        return {
            "best_valid": float(result[best_idx, 1].item()),
            "test_at_best_valid": float(result[best_idx, 2].item()),
            "epoch_index": best_idx,
        }

    def print_run(self, run: int, metric: str | None = None) -> None:
        metric = metric or self.monitor
        best = self.best_for_run(run, metric)
        print(
            f"Run {run + 1:02d} [{metric}] | "
            f"Best valid: {100 * best['best_valid']:.2f}% | "
            f"Test at best valid: {100 * best['test_at_best_valid']:.2f}%"
        )

    def print_summary(self, metric: str | None = None) -> None:
        metric = metric or self.monitor
        best = [self.best_for_run(run, metric) for run in range(self.runs)]
        best = [item for item in best if item["epoch_index"] >= 0]
        if not best:
            print(f"No evaluation records for {metric}.")
            return
        valid = torch.tensor([item["best_valid"] for item in best]) * 100
        test = torch.tensor([item["test_at_best_valid"] for item in best]) * 100
        valid_std = valid.std(unbiased=False).item() if valid.numel() > 1 else 0.0
        test_std = test.std(unbiased=False).item() if test.numel() > 1 else 0.0
        print(
            f"All runs [{metric}] | "
            f"Valid: {valid.mean():.2f} +/- {valid_std:.2f} | "
            f"Test: {test.mean():.2f} +/- {test_std:.2f}"
        )

    def export_json(self, path: str) -> None:
        export = {
            "losses": self.losses,
            "records": self.records,
            "best": defaultdict(list),
        }
        for metric in self.metric_names:
            export["best"][metric] = [
                self.best_for_run(run, metric) for run in range(self.runs)
            ]
        export["best"] = dict(export["best"])

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2)
