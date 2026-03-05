# Link Prediction with Enhanced GAT on ogbl-collab

This project implements an **enhanced Graph Attention Network (GAT)** for link prediction on the [`ogbl-collab`](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab) dataset from the Open Graph Benchmark (OGB). The task is to predict future co-authorship links between researchers.

## Key Features

- **GATv2Conv** encoder with dynamic attention (Brody et al., 2021)
- **Residual connections** with learnable skip projections
- **Layer normalization** for stable training
- **Jumping Knowledge** aggregation (cat / max / last)
- **Learnable node embeddings** + degree-based features for featureless graphs
- **Enhanced link predictor** with Hadamard product, L1 distance, and average features
- **Degree-biased negative sampling** for harder training
- **Cosine LR scheduler** with linear warmup
- **Early stopping** with patience-based monitoring
- **Multi-metric evaluation**: Hits@10, Hits@50, Hits@100
- **YAML-based configuration** for reproducible experiments
- **Jupyter notebook** for comprehensive result visualization

## Project Structure

```
Link_Prediction_GAT/
├── configs/
│   └── default.yaml          # Hyperparameter configuration
├── src/
│   ├── models/
│   │   ├── gat_encoder.py     # Enhanced GAT encoder
│   │   └── link_predictor.py  # Multi-feature link predictor
│   ├── data/
│   │   └── dataset.py         # Data loading & feature engineering
│   ├── training/
│   │   ├── trainer.py         # Training loop with improved neg sampling
│   │   └── evaluator.py       # Multi-metric evaluation
│   └── utils/
│       ├── logger.py          # Metric logger with JSON export
│       └── seed.py            # Reproducibility utilities
├── scripts/
│   └── train.py               # Main training entry point
├── notebooks/
│   └── evaluation.ipynb       # Result visualization notebook
├── checkpoints/               # Saved model checkpoints
├── logs/                      # Training logs & plots
├── dataset/                   # Cached OGB dataset
│   └── ogbl_collab/
├── configs/
│   └── default.yaml
├── environment.yml            # Conda environment specification
├── requirements.txt           # Pip dependencies
├── README.md
└── License
```

## Requirements

- Python 3.12
- CUDA 12.8+
- PyTorch 2.7+
- PyTorch Geometric 2.6+

## Setup

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate lp
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```

## Training

```bash
# Run with default config
python scripts/train.py

# Override hyperparameters via CLI
python scripts/train.py --epochs 200 --lr 0.0005 --heads 8

# Use a custom config
python scripts/train.py --config configs/default.yaml --runs 5
```

Training logs are saved to `logs/training_log.json` and best model checkpoints to `checkpoints/`.

## Evaluation & Visualization

Open the Jupyter notebook for comprehensive result analysis:

```bash
jupyter notebook notebooks/evaluation.ipynb
```

The notebook includes:
- Training loss curves (raw + smoothed)
- Hits@K metrics over epochs with best-valid annotations
- Results summary table across all runs
- Prediction score distributions (positive vs. negative)
- ROC and Precision-Recall curves
- Degree vs. prediction score analysis
- Model architecture and parameter breakdown

## Model Architecture

### GAT Encoder
The encoder uses GATv2Conv layers with:
- Multi-head attention (4 heads by default)
- Residual connections via learned linear skip projections
- Layer normalization after each attention layer
- Jumping Knowledge (JK) to aggregate across all layers

### Link Predictor
The predictor combines three edge interaction features:
- **Hadamard product**: element-wise `x_i * x_j`
- **L1 distance**: `|x_i - x_j|`
- **Average**: `(x_i + x_j) / 2`

These are concatenated and fed through a BatchNorm-MLP for scoring.

### Node Features
Since ogbl-collab has no node features, we construct:
- **Log-degree**: `log(degree + 1)` for scale-invariant representation
- **Normalized degree**: `degree / max_degree`
- **Learnable embeddings**: Xavier-initialized, added to structural features

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.num_layers` | 3 | GAT layers |
| `model.hidden_channels` | 256 | Hidden dimension |
| `model.heads` | 4 | Attention heads |
| `model.jk_mode` | cat | JK aggregation |
| `training.epochs` | 400 | Max epochs |
| `training.lr` | 0.001 | Learning rate |
| `scheduler.warmup_epochs` | 10 | LR warmup |
| `early_stopping.patience` | 50 | Early stop patience |

## References

- Hu et al. (2020). *Open Graph Benchmark: Datasets for Machine Learning on Graphs*. [arXiv:2005.00687](https://arxiv.org/abs/2005.00687)
- Velickovic et al. (2018). *Graph Attention Networks*. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
- Brody et al. (2021). *How Attentive are Graph Attention Networks?*. [arXiv:2105.14491](https://arxiv.org/abs/2105.14491)
- Xu et al. (2018). *Representation Learning on Graphs with Jumping Knowledge Networks*. [arXiv:1806.03536](https://arxiv.org/abs/1806.03536)

## License

See [License](License) for details.
