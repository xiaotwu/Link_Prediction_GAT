import json
import os
from collections import defaultdict

import torch


class Logger:
    """Training logger that tracks metrics across runs and supports JSON export."""

    def __init__(self, runs, metrics):
        self.runs = runs
        self.metrics = metrics
        self.results = {m: [[] for _ in range(runs)] for m in metrics}
        self.train_losses = [[] for _ in range(runs)]

    def add_result(self, run, results_dict):
        for metric, values in results_dict.items():
            assert len(values) == 3  # (train, valid, test)
            self.results[metric][run].append(values)

    def add_loss(self, run, loss):
        self.train_losses[run].append(loss)

    def print_statistics(self, run=None, metric="Hits@50"):
        if metric not in self.results:
            return
        if run is not None:
            result = 100 * torch.tensor(self.results[metric][run])
            print(f"Run {run + 1:02d} [{metric}]:")
            print(f"  Best Valid: {result[:, 1].max():.2f}")
            idx = result[:, 1].argmax()
            print(f"  Test @ Best Valid: {result[idx, 2]:.2f}")
        else:
            result = 100 * torch.tensor(self.results[metric])
            best_results = []
            for r in result:
                valid = r[:, 1]
                test = r[:, 2]
                best_val = valid.max().item()
                best_test = test[valid.argmax()].item()
                best_results.append((best_val, best_test))
            best_val = torch.tensor(best_results)[:, 0]
            best_test = torch.tensor(best_results)[:, 1]
            print(f"All Runs [{metric}]:")
            print(f"  Valid: {best_val.mean():.2f} +/- {best_val.std():.2f}")
            print(f"  Test:  {best_test.mean():.2f} +/- {best_test.std():.2f}")

    def get_best_results(self, metric="Hits@50"):
        """Return per-run best valid and corresponding test scores."""
        result = 100 * torch.tensor(self.results[metric])
        bests = []
        for r in result:
            valid = r[:, 1]
            test = r[:, 2]
            idx = valid.argmax()
            bests.append({
                "best_valid": valid[idx].item(),
                "test_at_best_valid": test[idx].item(),
                "best_epoch_idx": idx.item(),
            })
        return bests

    def export_json(self, path):
        """Export all logged data to JSON for visualization."""
        export = {
            "losses": [losses for losses in self.train_losses],
            "metrics": {},
        }
        for metric in self.metrics:
            export["metrics"][metric] = []
            for run_data in self.results[metric]:
                export["metrics"][metric].append(
                    [list(v) for v in run_data]
                )
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(export, f, indent=2)
