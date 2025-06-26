class Logger(object):
    def __init__(self, runs, args):
        self.runs = runs
        self.results = [[] for _ in range(runs)]
        self.args = args

    def add_result(self, run, result):
        assert len(result) == 3
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            print(f'Run {run + 1:02d}:')
            print(f'  Best valid: {result[:, 1].max():.2f}')
            idx = result[:, 1].argmax()
            print(f'  Test at best valid: {result[idx, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                valid = r[:, 1]
                test = r[:, 2]
                best_val = valid.max().item()
                best_test = test[valid.argmax()].item()
                best_results.append((best_val, best_test))
            best_val = torch.tensor(best_results)[:, 0]
            best_test = torch.tensor(best_results)[:, 1]
            print(f'All runs:')
            print(f'  Valid: {best_val.mean():.2f} ± {best_val.std():.2f}')
            print(f'  Test: {best_test.mean():.2f} ± {best_test.std():.2f}')
