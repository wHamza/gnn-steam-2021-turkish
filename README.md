# Steam GNN Recommender

Personalised game recommendations from **Turkish Steam reviews**, built as a heterogeneous graph neural network. Users and games are graph nodes; recommendation is framed as **link prediction** on user–game review edges.

| | |
|---|---|
| **Model** | Graph Attention Network (2 layers, dropout 0.5) lifted to a heterogeneous graph with `to_hetero` |
| **Task** | Link prediction (will this user review this game?) |
| **Edge features** | Sentence-Transformer review embeddings concatenated with behavioural and temporal features |
| **Split** | `RandomLinkSplit`, 10% validation / 10% test, 1:1 negative sampling |
| **Metric** | ROC AUC |

## Pipeline

```
raw reviews -> filtering and ID mapping
            -> review embeddings (Sentence Transformers, cached as .npy)
            -> HeteroData: user --reviews--> app (+ reverse edges via ToUndirected)
            -> GNNEncoder (GAT x2, to_hetero, aggr="sum")
            -> EdgeDecoder: Linear(2H -> H) -> ReLU -> Linear(H -> 1)
            -> recommendation score
```

## Features

- **Filtering:** keep reviews with `weighted_vote_score > 0.4`, drop rows with missing values.
- **User authority:** `auth_score = log1p(games_owned) * (1 + games_owned / (num_reviews + 1))`.
- **Temporal:** recency weight `exp(-0.005 * days)`, whether the review was edited, weekend flag, days played after the review.
- **Scaling:** `log1p` then MinMax for durations and times, MinMax for counts.

## Expected data columns

`app_id`, `app_name`, `author.steamid`, `review`, `recommended`, `weighted_vote_score`, `author.playtime_forever`, `author.num_games_owned`, `author.num_reviews`, `timestamp_created`, `timestamp_updated`, `author.last_played`

## Run

```bash
pip install torch torch-geometric sentence-transformers scikit-learn pandas numpy
python recommendsteam.py
```

Put a compatible reviews file at `steam_reviews.csv` first. The dataset and the generated `embedding.npy` are not included. Training logs validation AUC every 100 epochs.

## Status and next steps

Experimental research code for learning and iteration, not a production system.

- [ ] Cold-start handling for new users
- [ ] Compare other architectures (GraphSAGE, HGT)
- [ ] Serve recommendations through an API
