# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

df

!pip install torch_geometric

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
import torch_geometric.transforms as T

if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

df = pd.read_csv('#senin indirdigin review dataset')

app_dict = dict(zip(df['app_id'], df['app_name']))

df_new = df.drop(columns = ['app_name', 'language', 'steam_purchase', 'received_for_free', 'written_during_early_access'])

df_new['author.last_played'] = pd.to_datetime(df['author.last_played'], unit='s')
df_new['timestamp_updated'] = pd.to_datetime(df['timestamp_updated'], unit='s')
df_new['timestamp_created'] = pd.to_datetime(df['timestamp_created'], unit='s')

df_new = df_new[df_new['weighted_vote_score']>0.4]

df_new = df_new.dropna()

#authority bununla saglandi

df_new['user_authority'] = df_new['author.num_games_owned'] / (df_new['author.num_reviews']+ 1)

df_new['auth_score'] = np.log1p(df['author.num_games_owned']) * (1 + df_new['user_authority'])

df_new['auth_score'] = (df_new['auth_score'] - df_new['auth_score'].min()) / (df_new['auth_score'].max() - df_new['auth_score'].min())

"""zamansal

"""

dataset_today = df_new['timestamp_created'].max()

df_new['days_relative_to_end'] = (dataset_today - df_new['timestamp_created']).dt.days

df_new['recency_weight'] = np.exp(-0.005 * df_new['days_relative_to_end'])

df_new['is_updated'] = (df_new['timestamp_updated'].values > df_new['timestamp_created'].values).astype(int)

df_new['final_time_score'] = 1.0 * (1 + (df_new['is_updated'] * 0.15))

df_new['days_played_after_review'] = (df_new['author.last_played'] - df_new['timestamp_created']).dt.days.clip(lower=0)

df_new['is_weekend'] = df_new['timestamp_created'].dt.dayofweek.isin([5, 6]).astype(int)

# Sonuçlara bir bakalım
print(df_new[['is_updated', 'final_time_score', 'days_played_after_review', 'is_weekend']].head())

dataset = df_new.drop(columns = ['timestamp_created', 'timestamp_updated' , 'author.last_played'])

dataset.info()

"""# **EMBEDDINGS**"""

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(list(dataset['review']), show_progress_bar=True)

embedding_16 = embeddings.astype(np.float32)
np.save('embedding.npy', embedding_16)

"""preprocessing

"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#review vectorize,


#+Süre ve Zamanlar: author.playtime_forever, author.playtime_last_two_weeks, author.playtime_at_review, days_played_after_review, days_relative_to_end.
#@Sayımlar: comment_count, votes_helpful, votes_funny, author.num_games_owned, author.num_reviews.
#Skorlar:weighted_vote_score, /user_authority, auth_score, recency_weight, is_updated
#final_time_score drop

#log1p uyguluyoruz

dataset['author.playtime_forever'] = np.log1p(dataset['author.playtime_forever'])
dataset['author.playtime_last_two_weeks'] = np.log1p(dataset['author.playtime_last_two_weeks'])
dataset['author.playtime_at_review'] = np.log1p(dataset['author.playtime_at_review'])
dataset['days_played_after_review'] = np.log1p(dataset['days_played_after_review'])
dataset['days_relative_to_end'] = np.log1p(dataset['days_relative_to_end'])

#min max scale

dataset['author.playtime_forever'] = scaler.fit_transform(dataset[['author.playtime_forever']])
dataset['author.playtime_last_two_weeks'] = scaler.fit_transform(dataset[['author.playtime_last_two_weeks']])
dataset['author.playtime_at_review'] = scaler.fit_transform(dataset[['author.playtime_at_review']])
dataset['days_played_after_review'] = scaler.fit_transform(dataset[['days_played_after_review']])
dataset['days_relative_to_end'] = scaler.fit_transform(dataset[['days_relative_to_end']])

#min max scale

dataset['comment_count'] = scaler.fit_transform(dataset[['comment_count']])
dataset['votes_helpful'] = scaler.fit_transform(dataset[['votes_helpful']])
dataset['votes_funny'] = scaler.fit_transform(dataset[['votes_funny']])
dataset['author.num_games_owned'] = scaler.fit_transform(dataset[['author.num_games_owned']])
dataset['author.num_reviews'] = scaler.fit_transform(dataset[['author.num_reviews']])

#id mapping yapacagiz
#review_id, app_id, author.steamid

unique_ids = dataset['author.steamid'].unique()
steamIds = {val: i for i, val in enumerate(unique_ids)}

unique_idsrev = dataset['review_id'].unique()
revIds = {val: i for i, val in enumerate(unique_idsrev)}

unique_idsapp = dataset['app_id'].unique()
appIds = {val: i for i, val in enumerate(unique_idsapp)}

dataset['author.steamid'] = dataset['author.steamid'].map(steamIds)
dataset['review_id'] = dataset['review_id'].map(revIds)
dataset['app_id'] = dataset['app_id'].map(appIds)

len(dataset['author.steamid'].unique()) / len(dataset)

tersSteam[2005]

embedding_32 = np.load('embedding.npy')
embedding_32 = embedding_32.astype(np.float32)

"""# **HETERO DATA**"""

user_cols = ['author.num_games_owned', 'author.num_reviews', 'author.playtime_forever', 'user_authority']
user_df = dataset.groupby('author.steamid')[user_cols].last()
user_df = torch.tensor(user_df.values)

tersSteam = {v: k for k, v in steamIds.items()}

embedding_tensor =  torch.tensor(embedding_32)

edge_index = torch.tensor(dataset[['author.steamid', 'app_id']].values).T.contiguous()
edge_attrs = dataset.drop(columns = ['author.num_games_owned', 'author.num_reviews', 'author.playtime_forever', 'user_authority', 'auth_score', 'author.steamid', 'app_id', 'column00', 'review_id', 'review'])
edge_attrs['recommended'] = edge_attrs['recommended'].astype(int)
edge_attrs_tensor = torch.tensor(edge_attrs.values)
edge_attr = torch.cat([embedding_tensor, edge_attrs_tensor], dim=-1)

edge_attr.shape

xnum_apps = edge_attr.shape[1]
app_features = torch.randn(xnum_apps, 64)

data = HeteroData()

# Düğümleri ekle
data['user'].x = user_df
data['app'].x = app_features
# İlişkileri (Review) ekle
data['user', 'reviews', 'app'].edge_index = edge_index
data['user', 'reviews', 'app'].edge_attr = edge_attr

data = T.ToUndirected()(data)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    add_negative_train_samples=True,
    neg_sampling_ratio=1.0,
    edge_types=[('user', 'reviews', 'app')],
    rev_edge_types=[('app', 'rev_reviews', 'user')],
)

train_data, val_data, test_data = transform(data)

train_data['user'].x = train_data['user'].x.float()

train_data['app'].x = train_data['app'].x.float()

train_data['user', 'reviews', 'app'].edge_attr = train_data['user', 'reviews', 'app'].edge_attr.float()

val_data['user'].x = val_data['user'].x.float()
val_data['app'].x = val_data['app'].x.float()
val_data['user', 'reviews', 'app'].edge_attr = val_data['user', 'reviews', 'app'].edge_attr.float()

datasets = [train_data, val_data, test_data]

for d in datasets:
    for node_type in d.node_types:
        if hasattr(d[node_type], 'x') and d[node_type].x is not None:
            d[node_type].x = d[node_type].x.to(torch.float32)

    for edge_type in d.edge_types:
        if hasattr(d[edge_type], 'edge_attr') and d[edge_type].edge_attr is not None:
            d[edge_type].edge_attr = d[edge_type].edge_attr.to(torch.float32)

        if hasattr(d[edge_type], 'edge_label') and d[edge_type].edge_label is not None:
            d[edge_type].edge_label = d[edge_type].edge_label.to(torch.float32)

"""# **MODEL**"""

from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge, GATConv,SAGEConv
from torch_geometric.nn import global_mean_pool
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.utils import to_networkx
from torch_geometric.explain import CaptumExplainer, Explainer
from torch_geometric.nn import SAGEConv, to_hetero

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_dim):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, edge_dim=edge_dim, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), out_channels, edge_dim=edge_dim, add_self_loops=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index[0], edge_label_index[1]

        z = torch.cat([z_dict['user'][row], z_dict['app'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class SteamGNN(torch.nn.Module):
    def __init__(self, hidden_channels, edge_dim, metadata):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, edge_dim)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')

        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)

        return self.decoder(z_dict, edge_label_index)

HIDDEN_CHANNELS = 64
EDGE_DIM = edge_attr.shape[1]
METADATA = train_data.metadata()

model = SteamGNN(
    hidden_channels=HIDDEN_CHANNELS,
    edge_dim=EDGE_DIM,
    metadata=METADATA
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss()

#to device
model = model.to(device)
model = model.float()
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

def train():
  model.train()

  optimizer.zero_grad()

  pred = model(
        train_data.x_dict,
        train_data.edge_index_dict,
        train_data.edge_attr_dict,
        train_data['user', 'reviews', 'app'].edge_label_index
    )

  target = train_data['user', 'reviews', 'app'].edge_label.float()
  loss = criterion(pred.view(-1), target)
  loss.backward()
  optimizer.step()

  return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    pred = model(
        data.x_dict,
        data.edge_index_dict,
        data.edge_attr_dict,
        data['user', 'reviews', 'app'].edge_label_index
    )
    pred = torch.sigmoid(pred)
    target = data['user', 'reviews', 'app'].edge_label.float()

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())
    except:
        auc = 0.5

    return auc

for epoch in range(1, 1001):
  loss = train()
  val_auc = test(val_data)
  print('sa')

  if epoch % 100 == 0:
    print(f'epoch {epoch}, train loss: {loss:.4f}, val auc: {val_auc:.4f}')

def recommend_apps(model, data, user_id, top_k=10):
    model.eval()

    num_apps = data['app'].x.shape[0]

    edge_index = data['user', 'reviews', 'app'].edge_index
    owned_apps = edge_index[1][edge_index[0] == user_id]  # bu user'ın app'leri
    owned_mask = torch.zeros(num_apps, dtype=torch.bool)
    owned_mask[owned_apps] = True  # sahip olunanları işaretle

    candidate_apps = torch.where(~owned_mask)[0]

    user_ids = torch.full((len(candidate_apps),), user_id, dtype=torch.long)
    edge_label_index = torch.stack([user_ids, candidate_apps])

    with torch.no_grad():
        scores = model(
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict,
            edge_label_index
        )
        probs = torch.sigmoid(scores)

    top_k_local  = probs.topk(top_k)
    top_k_app_ids = candidate_apps[top_k_local.indices]  # global app id'lerine çevir
    top_k_scores  = top_k_local.values

    return top_k_app_ids, top_k_scores

edges = recommend_apps(model, val_data, user_id=6810, top_k=50)
print(f"Önerilen oyunlar: {app_ids.tolist()}")
print(f"Skorlar: {scores.tolist()}")

ters1 = {v: k for k, v in appIds.items()}

def find_games(ids, list1, list2):
  ids = app_ids.tolist()
  games = []
  for i in ids:
    s1 = int(list1[i])
    s2 = list2[s1]
    games.append(s2)
  return games

def get_user(id, df):
  id = tersSteam[id]
  user = df[df['author.steamid'] == id]
  return user

get_user(6810, df)['app_name'].unique()

find_games(app_ids, ters1, app_dict)

en_aktif_yazar = df['author.steamid'].value_counts().idxmax()

en_aktif_yazar

steamIds[76561198083413093]

crazy = data.to_homogeneous()

networkx_graph = to_networkx(crazy)
nx.draw(networkx_graph)

