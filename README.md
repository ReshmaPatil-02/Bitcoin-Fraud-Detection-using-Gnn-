

````markdown
# 🔍 Bitcoin Fraud Detection using Graph Neural Networks (GNN) & Autoencoder

A graph-based anomaly detection system for identifying fraudulent Bitcoin transactions using **Graph Neural Networks (GCN)** and **Autoencoders**. The project leverages transaction relationships to learn node embeddings and detects suspicious transactions based on reconstruction error. An interactive **Streamlit dashboard** enables real-time fraud score prediction.

---

## 📌 Features

- 📊 Graph-based Bitcoin transaction analysis
- 🧠 Graph Convolutional Network (GCN) for node embedding generation
- 🔍 Autoencoder-based anomaly detection
- ⚡ Real-time fraud score prediction
- 🌐 Interactive Streamlit dashboard
- 📈 Fraud classification using reconstruction error threshold

---

## 🏗️ Project Architecture

```
Bitcoin Transaction Data
          │
          ▼
 Data Preprocessing
          │
          ▼
 Transaction Graph (NetworkX)
          │
          ▼
 Graph Neural Network (GCN)
          │
          ▼
 Node Embeddings
          │
          ▼
 Autoencoder
          │
          ▼
 Reconstruction Error
          │
          ▼
 Fraud / Normal Prediction
          │
          ▼
 Streamlit Dashboard
```

---

## 📂 Dataset

This project uses the **Elliptic Bitcoin Transaction Dataset**, which contains:

- Bitcoin transaction nodes
- Transaction graph edges
- Transaction labels (Fraud / Legitimate / Unknown)

Files used:

- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`
- `elliptic_txs_features.csv`

---

## 🛠️ Tech Stack

- Python
- PyTorch
- PyTorch Geometric
- NetworkX
- Pandas
- NumPy
- Streamlit

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/Bitcoin-Fraud-Detection-GNN.git

cd Bitcoin-Fraud-Detection-GNN
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 📈 Working

1. Load Bitcoin transaction dataset.
2. Build a transaction graph using NetworkX.
3. Convert graph into PyTorch Geometric format.
4. Train a Graph Convolutional Network (GCN).
5. Generate node embeddings.
6. Train an Autoencoder on embeddings.
7. Compute reconstruction error.
8. Classify transactions as **Fraudulent** or **Normal** based on anomaly score.
9. Display predictions through the Streamlit interface.

---

## 📊 Model Pipeline

- **Graph Construction**
- **GCN Embedding Learning**
- **Autoencoder Training**
- **Reconstruction Error Calculation**
- **Threshold-based Fraud Detection**

---

## 💻 Dashboard Features

- Select any transaction ID
- View anomaly score
- Display reconstruction threshold
- Predict whether the transaction is:
  - ✅ Normal
  - 🚨 Fraudulent

---

## 📷 Demo

<img src="images/dashboard.png" width="900">

*(Replace with your project screenshot.)*

---

## 📁 Project Structure

```
Bitcoin-Fraud-Detection-GNN/
│
├── app.py
├── elliptic_txs_classes.csv
├── elliptic_txs_edgelist.csv
├── elliptic_txs_features.csv
├── requirements.txt
├── README.md
└── images/
      └── dashboard.png
```

---

## 🔮 Future Improvements

- Graph Attention Networks (GAT)
- GraphSAGE implementation
- Explainable AI (XAI)
- Real-time blockchain transaction monitoring
- Model performance evaluation metrics
- Docker deployment
- Cloud deployment on AWS or Azure

---

## 👩‍💻 Author

**Reshma Patil**

B.Tech Artificial Intelligence & Machine Learning

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
````
