# Link Prediction with GAT on ogbl-collab

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

The only configuration file is `configs/default.yaml`. It targets CUDA training.

If CUDA memory is tight, first reduce `training.batch_size`, then reduce
`model.hidden_channels`.

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

Validation and test evaluation run automatically during training:

- evaluation uses the official OGB fixed negative edges
- validation monitors `Hits@50`
- test scores are reported at the best validation checkpoint
- epoch `0` is evaluated first so the structural AA/RA prior is checkpointed
  before neural training starts

Artifacts:

```text
checkpoints/default/best_run*.pt
logs/default/training_log.json
logs/default/config_used.yaml
```

To validate the structural heuristic baseline independently:

```bash
uv run python scripts/evaluate_heuristics.py
```

## Results Sheet

After validation/test verification completes, the best result is appended to:

```text
results/results.csv
```

The file is intentionally kept as a separate results sheet:

| Method | Ext. data | Test Hits@50 | Validation Hits@50 | #Params | Hardware | Date |
|---|---|---:|---:|---:|---|---|
|  |  |  |  |  |  |  |
