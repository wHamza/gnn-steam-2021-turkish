# -*- coding: utf-8 -*-
"""
Steam Oyun Öneri Sistemi - Heterogeneous Graph Neural Network
=============================================================
Bu script, Steam kullanıcı yorumlarından oluşan bir veri setiyle
kullanıcı-oyun etkileşimlerini modelleyen bir GNN öneri sistemi kurar.

Gereksinimler:
    pip install torch torch_geometric sentence-transformers scikit-learn pandas numpy
"""

# 1. KÜTÜPHANELER

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.utils import to_networkx
from torch_geometric.explain import CaptumExplainer, Explainer
import torch_geometric.transforms as T
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score


# =============================================================================
# 2. ORTAM AYARLARI
# =============================================================================

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Kullanılan cihaz: {device}")


# 3. VERİ YÜKLEME VE ÖN İŞLEME
# =============================================================================
# --- 3.1 Ham Veriyi Oku ---
df = pd.read_csv("steam_reviews.csv")  # Kendi dosya yolunu buraya yaz

# Oyun ID → isim haritası (orijinal df üzerinden)
app_dict = dict(zip(df["app_id"], df["app_name"]))

# --- 3.2 Gereksiz Sütunları Sil ---
drop_cols = ["app_name", "language", "steam_purchase",
             "received_for_free", "written_during_early_access"]
df_new = df.drop(columns=drop_cols)

# --- 3.3 Zaman Damgalarını Dönüştür ---
for col in ["author.last_played", "timestamp_updated", "timestamp_created"]:
    df_new[col] = pd.to_datetime(df[col], unit="s")

# --- 3.4 Düşük Kaliteli Yorumları Filtrele ---
df_new = df_new[df_new["weighted_vote_score"] > 0.4]
df_new = df_new.dropna()


# 4. ÖZELLİK MÜHENDİSLİĞİ

# --- 4.1 Kullanıcı Otoritesi ---
df_new["user_authority"] = (
    df_new["author.num_games_owned"] / (df_new["author.num_reviews"] + 1)
)
df_new["auth_score"] = (
    np.log1p(df["author.num_games_owned"]) * (1 + df_new["user_authority"])
)
df_new["auth_score"] = (
    (df_new["auth_score"] - df_new["auth_score"].min())
    / (df_new["auth_score"].max() - df_new["auth_score"].min())
)

# --- 4.2 Zamansal Özellikler ---
dataset_today = df_new["timestamp_created"].max()
df_new["days_relative_to_end"] = (
    dataset_today - df_new["timestamp_created"]
).dt.days
df_new["recency_weight"] = np.exp(-0.005 * df_new["days_relative_to_end"])

df_new["is_updated"] = (
    df_new["timestamp_updated"].values > df_new["timestamp_created"].values
).astype(int)
df_new["final_time_score"] = 1.0 * (1 + (df_new["is_updated"] * 0.15))

df_new["days_played_after_review"] = (
    df_new["author.last_played"] - df_new["timestamp_created"]
).dt.days.clip(lower=0)

df_new["is_weekend"] = (
    df_new["timestamp_created"].dt.dayofweek.isin([5, 6]).astype(int)
)

print(df_new[["is_updated", "final_time_score",
              "days_played_after_review", "is_weekend"]].head())

# --- 4.3 Zaman Sütunlarını Bırak ---
dataset = df_new.drop(
    columns=["timestamp_created", "timestamp_updated", "author.last_played"]
)
dataset.info()


# 5. METİN GÖMÜLEMLERİ (EMBEDDINGS)

EMBEDDING_PATH = "embedding.npy"

if os.path.exists(EMBEDDING_PATH):
    # Daha önce kaydedilmişse doğrudan yükle
    print("Kayıtlı embedding yükleniyor...")
    embeddings = np.load(EMBEDDING_PATH).astype(np.float32)
else:
    # Yeni hesapla ve kaydet
    print("Embedding hesaplanıyor...")
    embed_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embeddings = embed_model.encode(
        list(dataset["review"]), show_progress_bar=True
    ).astype(np.float32)
    np.save(EMBEDDING_PATH, embeddings)
    print(f"Embedding kaydedildi: {EMBEDDING_PATH}")


# 6. SAYISAL ÖZELLİK ÖLÇEKLEME

scaler = MinMaxScaler()

# Log1p dönüşümü (çarpık dağılımları düzelt)
log_cols = [
    "author.playtime_forever",
    "author.playtime_last_two_weeks",
    "author.playtime_at_review",
    "days_played_after_review",
    "days_relative_to_end",
]
for col in log_cols:
    dataset[col] = np.log1p(dataset[col])

# Min-Max ölçekleme (süre & zamansal)
for col in log_cols:
    dataset[col] = scaler.fit_transform(dataset[[col]])

# Min-Max ölçekleme (sayımlar)
count_cols = [
    "comment_count",
    "votes_helpful",
    "votes_funny",
    "author.num_games_owned",
    "author.num_reviews",
]
for col in count_cols:
    dataset[col] = scaler.fit_transform(dataset[[col]])


# 7. ID HARITALAMA

def build_id_map(series):
    """Benzersiz değerleri 0-tabanlı indekslere eşler."""
    unique_vals = series.unique()
    return {val: i for i, val in enumerate(unique_vals)}


steamIds = build_id_map(dataset["author.steamid"])
revIds   = build_id_map(dataset["review_id"])
appIds   = build_id_map(dataset["app_id"])

# Ters haritalar (öneride kullanmak için)
tersSteam = {v: k for k, v in steamIds.items()}
tersApp   = {v: k for k, v in appIds.items()}

dataset["author.steamid"] = dataset["author.steamid"].map(steamIds)
dataset["review_id"]      = dataset["review_id"].map(revIds)
dataset["app_id"]         = dataset["app_id"].map(appIds)

print(f"Kullanıcı başına ortalama yorum sayısı: "
      f"{len(dataset) / len(dataset['author.steamid'].unique()):.2f}")


# 8. HETEROJENEİK GRAFIK VERISI (HeteroData)

# --- 8.1 Kullanıcı Düğüm Özellikleri ---
user_cols = [
    "author.num_games_owned",
    "author.num_reviews",
    "author.playtime_forever",
    "user_authority",
]
user_df = dataset.groupby("author.steamid")[user_cols].last()
user_features = torch.tensor(user_df.values, dtype=torch.float32)

# --- 8.2 Kenar Özellikleri ---
embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)

edge_index = torch.tensor(
    dataset[["author.steamid", "app_id"]].values
).T.contiguous()

edge_attrs_df = dataset.drop(
    columns=[
        "author.num_games_owned", "author.num_reviews",
        "author.playtime_forever", "user_authority",
        "auth_score", "author.steamid", "app_id",
        "review_id", "review",
    ],
    errors="ignore",
)
edge_attrs_df["recommended"] = edge_attrs_df["recommended"].astype(int)
edge_attrs_tensor = torch.tensor(edge_attrs_df.values, dtype=torch.float32)

# Embedding + sayısal özellikler birleştir
edge_attr = torch.cat([embedding_tensor, edge_attrs_tensor], dim=-1)
print(f"Kenar özellik boyutu: {edge_attr.shape}")

# --- 8.3 Uygulama Düğüm Özellikleri (rastgele başlangıç) ---
num_apps    = len(appIds)
app_features = torch.randn(num_apps, 64, dtype=torch.float32)

# --- 8.4 HeteroData Nesnesi ---
data = HeteroData()
data["user"].x                             = user_features
data["app"].x                              = app_features
data["user", "reviews", "app"].edge_index  = edge_index
data["user", "reviews", "app"].edge_attr   = edge_attr

# Yönsüz hale getir (rev_reviews kenarlarını otomatik ekler)
data = T.ToUndirected()(data)


# 9. EĞİTİM / DOĞRULAMA / TEST BÖLÜMLEME

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    add_negative_train_samples=True,
    neg_sampling_ratio=1.0,
    edge_types=[("user", "reviews", "app")],
    rev_edge_types=[("app", "rev_reviews", "user")],
)

train_data, val_data, test_data = transform(data)

# Tüm split'lerde float32 garantisi
for split in [train_data, val_data, test_data]:
    for node_type in split.node_types:
        if hasattr(split[node_type], "x") and split[node_type].x is not None:
            split[node_type].x = split[node_type].x.to(torch.float32)
    for edge_type in split.edge_types:
        et = split[edge_type]
        if hasattr(et, "edge_attr") and et.edge_attr is not None:
            et.edge_attr = et.edge_attr.to(torch.float32)
        if hasattr(et, "edge_label") and et.edge_label is not None:
            et.edge_label = et.edge_label.to(torch.float32)


# 10. MODEL TANIMI

class GNNEncoder(torch.nn.Module):
    """İki katmanlı GAT tabanlı mesaj geçişi."""

    def __init__(self, hidden_channels: int, out_channels: int, edge_dim: int):
        super().__init__()
        self.conv1 = GATConv(
            (-1, -1), hidden_channels, edge_dim=edge_dim, add_self_loops=False
        )
        self.conv2 = GATConv(
            (-1, -1), out_channels, edge_dim=edge_dim, add_self_loops=False
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x


class EdgeDecoder(torch.nn.Module):
    """Kullanıcı ve uygulama gömmelerinden bağlantı olasılığı üretir."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index[0], edge_label_index[1]
        z = torch.cat([z_dict["user"][row], z_dict["app"][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class SteamGNN(torch.nn.Module):
    """Heterogeneous GNN: Encoder + Decoder."""

    def __init__(self, hidden_channels: int, edge_dim: int, metadata):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, edge_dim)
        self.encoder = to_hetero(self.encoder, metadata, aggr="sum")
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        return self.decoder(z_dict, edge_label_index)


# 11. EĞİTİM DÖNGÜSÜ

HIDDEN_CHANNELS = 64
EDGE_DIM        = edge_attr.shape[1]
METADATA        = train_data.metadata()
EPOCHS          = 1000
LR              = 1e-4

model = SteamGNN(
    hidden_channels=HIDDEN_CHANNELS,
    edge_dim=EDGE_DIM,
    metadata=METADATA,
).to(device).float()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss()

# Veriyi GPU'ya taşı
train_data = train_data.to(device)
val_data   = val_data.to(device)
test_data  = test_data.to(device)


def train_step() -> float:
    """Tek bir eğitim adımı; kayıp değerini döndürür."""
    model.train()
    optimizer.zero_grad()

    pred = model(
        train_data.x_dict,
        train_data.edge_index_dict,
        train_data.edge_attr_dict,
        train_data["user", "reviews", "app"].edge_label_index,
    )
    target = train_data["user", "reviews", "app"].edge_label.float()
    loss = criterion(pred.view(-1), target)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(split_data) -> float:
    """Verilen split için ROC-AUC skorunu döndürür."""
    model.eval()
    pred = model(
        split_data.x_dict,
        split_data.edge_index_dict,
        split_data.edge_attr_dict,
        split_data["user", "reviews", "app"].edge_label_index,
    )
    pred   = torch.sigmoid(pred).cpu().numpy()
    target = split_data["user", "reviews", "app"].edge_label.float().cpu().numpy()

    try:
        return roc_auc_score(target, pred)
    except ValueError:
        return 0.5


print("Eğitim başlıyor...")
for epoch in range(1, EPOCHS + 1):
    loss    = train_step()
    val_auc = evaluate(val_data)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

# Final test skoru
test_auc = evaluate(test_data)
print(f"\nTest AUC: {test_auc:.4f}")


# 12. ÖNERİ FONKSİYONU

def recommend_apps(
    model: torch.nn.Module,
    split_data,
    user_id: int,
    top_k: int = 10,
):
    """
    Belirtilen kullanıcı için henüz sahip olmadığı oyunları önerir.

    Args:
        model:      Eğitilmiş SteamGNN modeli.
        split_data: Kullanılacak veri split'i (örn. val_data).
        user_id:    İç kullanıcı indeksi (steamIds haritasından).
        top_k:      Kaç oyun önerilecek.

    Returns:
        (app_ids, scores): Global uygulama indeksleri ve olasılık skorları.
    """
    model.eval()
    num_apps = split_data["app"].x.shape[0]

    # Kullanıcının mevcut oyunlarını bul
    ei          = split_data["user", "reviews", "app"].edge_index
    owned_apps  = ei[1][ei[0] == user_id]
    owned_mask  = torch.zeros(num_apps, dtype=torch.bool, device=device)
    owned_mask[owned_apps] = True

    # Aday oyunlar üzerinde tahmin yap
    candidate_apps = torch.where(~owned_mask)[0]
    user_ids       = torch.full((len(candidate_apps),), user_id,
                                dtype=torch.long, device=device)
    edge_label_index = torch.stack([user_ids, candidate_apps])

    with torch.no_grad():
        scores = model(
            split_data.x_dict,
            split_data.edge_index_dict,
            split_data.edge_attr_dict,
            edge_label_index,
        )
        probs = torch.sigmoid(scores)

    top_k_result  = probs.topk(top_k)
    top_app_ids   = candidate_apps[top_k_result.indices]
    top_scores    = top_k_result.values
    return top_app_ids, top_scores


def get_game_names(app_id_tensor, ters_app: dict, app_dict: dict) -> list:
    """İç uygulama indekslerini gerçek oyun isimlerine çevirir."""
    names = []
    for idx in app_id_tensor.tolist():
        real_app_id = ters_app[idx]
        names.append(app_dict.get(real_app_id, f"Bilinmeyen ({real_app_id})"))
    return names


def get_user_history(internal_user_id: int, original_df, ters_steam: dict):
    """Kullanıcının geçmişteki oynadığı oyunları döndürür."""
    real_steam_id = ters_steam[internal_user_id]
    return original_df[original_df["author.steamid"] == real_steam_id]["app_name"].unique()


# 13. ÖRNEK KULLANIM

TARGET_USER = 6810

print(f"\n{'='*50}")
print(f"Kullanıcı {TARGET_USER} için öneri üretiliyor...")
print(f"{'='*50}")

app_ids, scores = recommend_apps(model, val_data, user_id=TARGET_USER, top_k=10)
game_names      = get_game_names(app_ids, tersApp, app_dict)

print("\nÖnerilen Oyunlar:")
for rank, (name, score) in enumerate(zip(game_names, scores.tolist()), start=1):
    print(f"  {rank:2d}. {name:<45} (skor: {score:.4f})")

print("\nKullanıcının Geçmiş Oyunları:")
history = get_user_history(TARGET_USER, df, tersSteam)
for g in history:
    print(f"  - {g}")
