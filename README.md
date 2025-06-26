# Link Prediction with GAT on ogbl-collab

This project implements a link prediction model using **Graph Attention Networks (GAT)** on the [`ogbl-collab`](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab) dataset from the Open Graph Benchmark (OGB).

The goal is to predict potential co-authorship links between authors using a GAT encoder and a link predictor, and evaluate the model using Hits@50.

---

## Requirements
- python 3.12.11
- cuda = 12.9

To install all dependencies: `pip install -r requirements.txt`

## Reference

Hu et al. (2020). *Open Graph Benchmark: Datasets for Machine Learning on Graphs*. [arXiv:2005.00687](https://arxiv.org/abs/2005.00687)
