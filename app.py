# =====================================
# IMPORTS
# =====================================
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import networkx as nx

from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv

# =====================================
# TITLE
# =====================================
st.title("🔍 Bitcoin Fraud Detection (GNN + Autoencoder)")

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    features = pd.read_csv("elliptic_txs_features.csv", header=None)
    classes = pd.read_csv("elliptic_txs_classes.csv")
    edges = pd.read_csv("elliptic_txs_edgelist.csv")

    # Rename columns
    features.columns = ["txId"] + [f"f{i}" for i in range(1, 167)]

    # Merge
    df = features.merge(classes, on="txId")

    # Label processing
    df['class'] = df['class'].replace({"unknown": -1, "1": 1, "2": 0})
    df = df[df['class'] != -1]

    # Reduce size (important)
    df = df.sample(2000, random_state=42)

    return df, edges

df, edges = load_data()

# =====================================
# BUILD GRAPH
# =====================================
@st.cache_data
def build_graph(df, edges):
    G = nx.Graph()

    # Add nodes
    for txId in df['txId']:
        G.add_node(int(txId))

    valid_nodes = set(df['txId'])

    filtered_edges = edges[
        edges['txId1'].isin(valid_nodes) & edges['txId2'].isin(valid_nodes)
    ]

    G.add_edges_from(filtered_edges.values)

    feature_cols = [col for col in df.columns if col.startswith("f")]

    # Add features
    for _, row in df.iterrows():
        txId = int(row['txId'])
        G.nodes[txId]['x'] = row[feature_cols].values.tolist()
        G.nodes[txId]['y'] = int(row['class'])

    return G

G = build_graph(df, edges)

# =====================================
# CONVERT TO PYTORCH
# =====================================
data = from_networkx(G)

data.x = torch.tensor([G.nodes[n]['x'] for n in G.nodes()], dtype=torch.float)
data.y = torch.tensor([G.nodes[n]['y'] for n in G.nodes()], dtype=torch.long)

# =====================================
# MODELS
# =====================================
class GNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, 16)
        self.decoder = nn.Linear(16, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded

# =====================================
# TRAIN MODEL (NO CACHE ❗)
# =====================================
def train_model(data):
    model = GNN(data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train GNN
    for _ in range(10):
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index)
        loss = (embeddings ** 2).mean()
        loss.backward()
        optimizer.step()

    # Autoencoder
    ae = Autoencoder(32)
    optimizer_ae = torch.optim.Adam(ae.parameters(), lr=0.01)

    embeddings_detached = embeddings.detach()

    for _ in range(10):
        optimizer_ae.zero_grad()
        recon = ae(embeddings_detached)
        loss = ((embeddings_detached - recon) ** 2).mean()
        loss.backward()
        optimizer_ae.step()

    return embeddings_detached, ae

embeddings_detached, ae = train_model(data)

# =====================================
# FRAUD DETECTION
# =====================================
error = ((embeddings_detached - ae(embeddings_detached)) ** 2).mean(dim=1)

# 🔥 Better threshold
threshold = np.percentile(error.detach().numpy(), 90)

preds = (error > threshold).int()

# =====================================
# RESULTS TABLE
# =====================================
results = pd.DataFrame({
    "txId": list(G.nodes()),
    "error": error.detach().numpy(),
    "prediction": preds.numpy()
})

# =====================================
# UI
# =====================================
tx_id = st.selectbox("Select Transaction ID", results["txId"])

row = results[results["txId"] == tx_id]

score = row["error"].values[0]
prediction = row["prediction"].values[0]

st.subheader("🔎 Result")

st.write(f"Anomaly Score: {score:.4f}")

st.write("Threshold used:", threshold)   # 👈 ADD HERE

if prediction == 1:
    st.error("🚨 Fraudulent Transaction")
else:
    st.success("✅ Normal Transaction")