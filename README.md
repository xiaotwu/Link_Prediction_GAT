# Link Prediction with GAT on ogbl-collab

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

The default configuration targets an Nvidia RTX 5070 Ti CUDA GPU:

```text
configs/default.yaml
```

## Train

```bash
uv run python scripts/train.py
```

Useful overrides:

```bash
uv run python scripts/train.py --runs 1
uv run python scripts/train.py --epochs 50
uv run python scripts/train.py --batch-size 131072
```

## Validate And Test

Validation and test evaluation run automatically during training with the OGB
fixed negative edges. The monitored metric is `Hits@50`.

Artifacts:

```text
checkpoints/rtx5070ti/best_run*.pt
logs/rtx5070ti/training_log.json
logs/rtx5070ti/config_used.yaml
```

To verify structural heuristic baselines:

```bash
uv run python scripts/evaluate_heuristics.py
```

## Results Sheet

After validation/test verification completes, the best result is appended to:

```text
results/results.csv
```

| Method | Ext. data | Test Hits@50 | Validation Hits@50 | #Params | Hardware | Date |
|---|---|---:|---:|---:|---|---|
|  |  |  |  |  |  |  |
