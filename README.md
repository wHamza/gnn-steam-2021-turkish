# Steam GNN Recommender

A heterogeneous graph neural network for game recommendation using Steam review data. The project represents users and games as graph nodes, enriches interactions with review embeddings and behavioral signals, and learns recommendation scores with PyTorch Geometric.

## Highlights

- Heterogeneous user–game graph construction
- Multilingual review embeddings with Sentence Transformers
- Temporal, engagement, and user-authority features
- Graph Attention Network layers
- Recommendation evaluation with ROC AUC

## Requirements

```bash
pip install torch torch-geometric sentence-transformers scikit-learn pandas numpy
```

## Usage

Place a compatible Steam reviews dataset at `steam_reviews.csv`, then run:

```bash
python recommendsteam.py
```

The dataset and generated `embedding.npy` file are intentionally not included. Review the expected columns in `recommendsteam.py` before using a different dataset.

## Status

Experimental research code intended for learning and iteration rather than production deployment.
