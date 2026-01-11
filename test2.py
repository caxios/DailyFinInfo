"""
Stock Clustering Comparison: RNN-Correlation vs Plain Correlation

This script compares two correlation-based clustering approaches:
1. RNN-Correlation: Correlation computed on RNN encoder embeddings over time
2. Plain Correlation: Correlation computed on raw daily returns

The key insight:
- Plain Correlation: Do stocks move up/down together on the same days?
- RNN-Correlation: Do stocks exhibit similar learned patterns over time?
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. RNN Encoder Model
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=n_layers, batch_first=True)

    def forward(self, x):
        _, h_next = self.rnn(x)
        return h_next

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_prev):
        out, h_next = self.rnn(x, h_prev)
        return self.fc(out), h_next

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_len = target_len

    def forward(self, x):
        batch_size = x.size(0)
        output_size = self.decoder.fc.out_features
        context_vector = self.encoder(x)
        h_prev = context_vector
        decoder_input = torch.zeros(batch_size, 1, output_size).to(x.device)
        outputs = []
        for t in range(self.target_len):
            pred, h_next = self.decoder(decoder_input, h_prev)
            outputs.append(pred)
            decoder_input = pred
            h_prev = h_next
        return torch.cat(outputs, dim=1)

    def encode_only(self, x):
        """Get flattened encoder embeddings."""
        context = self.encoder(x)
        batch_size = x.size(0)
        return context.transpose(0, 1).reshape(batch_size, -1)

# =============================================================================
# 2. Data Preparation
# =============================================================================
tickers = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'TSLA',
    # Finance
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV',
    # Consumer
    'WMT', 'AMZN', 'HD', 'NKE', 'MCD', 'SBUX',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB',
]

period = '1y'
sequence_length = 30
target_len = 5

print(f"Fetching data for {len(tickers)} stocks...")

stock_data = {}
all_sequences = []
all_targets = []

for ticker in tickers:
    try:
        data = yf.Ticker(ticker).history(period=period)
        if len(data) < sequence_length + target_len:
            print(f"  Skipping {ticker}: Not enough data")
            continue
        
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        X_list = []
        y_list = []
        for i in range(len(scaled_data) - sequence_length - target_len + 1):
            X_list.append(scaled_data[i:i + sequence_length])
            y_list.append(scaled_data[i + sequence_length:i + sequence_length + target_len])
        
        if len(X_list) > 0:
            stock_data[ticker] = {
                'scaler': scaler,
                'X': np.array(X_list),
                'y': np.array(y_list),
                'raw_prices': close_prices,
                'dates': data.index
            }
            all_sequences.extend(X_list)
            all_targets.extend(y_list)
            print(f"  {ticker}: {len(X_list)} sequences")
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")

X = torch.tensor(np.array(all_sequences), dtype=torch.float32)
y = torch.tensor(np.array(all_targets), dtype=torch.float32)

print(f"\nTotal sequences: {X.shape[0]}")

# =============================================================================
# 3. Train RNN Encoder
# =============================================================================
print("\n" + "="*70)
print("TRAINING RNN ENCODER")
print("="*70)

input_size = 1
hidden_size = 64
n_layers = 2
output_size = 1

enc = Encoder(input_size, hidden_size, n_layers)
dec = Decoder(output_size, hidden_size, output_size, n_layers)
model = Seq2Seq(enc, dec, target_len)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
print(f"Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f'  Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print("Training complete.")

# =============================================================================
# 4. Extract Embedding Time Series for Each Stock
# =============================================================================
print("\n" + "="*70)
print("RNN-CORRELATION CLUSTERING")
print("="*70)

model.eval()

# For RNN-Correlation, we need the sequence of embeddings over time (not just mean)
# Each stock has multiple sliding window embeddings - this forms a "latent time series"
stock_embedding_series = {}

with torch.no_grad():
    for ticker, data in stock_data.items():
        X_stock = torch.tensor(data['X'], dtype=torch.float32)
        # Shape: (num_windows, embedding_dim) where embedding_dim = n_layers * hidden_size
        embeddings = model.encode_only(X_stock).numpy()
        stock_embedding_series[ticker] = embeddings
        print(f"  {ticker}: embedding series shape = {embeddings.shape}")

ticker_list = list(stock_embedding_series.keys())
n_stocks = len(ticker_list)

# =============================================================================
# 5. Compute RNN-Correlation Matrix
# =============================================================================
print("\nComputing RNN-based correlation matrix...")

# We need all stocks to have the same number of time points for correlation
# Find minimum length and truncate
min_len = min(emb.shape[0] for emb in stock_embedding_series.values())
print(f"  Using {min_len} time points per stock")

# For each embedding dimension, compute correlation across time
# Then aggregate (mean correlation across all embedding dimensions)
embedding_dim = hidden_size * n_layers

rnn_corr_matrix = np.zeros((n_stocks, n_stocks))

for i, ticker_i in enumerate(ticker_list):
    for j, ticker_j in enumerate(ticker_list):
        if i == j:
            rnn_corr_matrix[i, j] = 1.0
        elif i < j:
            emb_i = stock_embedding_series[ticker_i][:min_len]  # (time, dim)
            emb_j = stock_embedding_series[ticker_j][:min_len]  # (time, dim)
            
            # Compute correlation for each embedding dimension, then average
            correlations = []
            for d in range(embedding_dim):
                corr = np.corrcoef(emb_i[:, d], emb_j[:, d])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            avg_corr = np.mean(correlations) if correlations else 0
            rnn_corr_matrix[i, j] = avg_corr
            rnn_corr_matrix[j, i] = avg_corr

print("  RNN correlation matrix computed.")

# =============================================================================
# 6. Plain Correlation Matrix (Daily Returns)
# =============================================================================
print("\n" + "="*70)
print("PLAIN CORRELATION CLUSTERING")
print("="*70)

# Compute daily returns for each stock
returns_dict = {}
for ticker in ticker_list:
    prices = stock_data[ticker]['raw_prices'].flatten()
    returns = np.diff(prices) / (prices[:-1] + 1e-8)
    returns_dict[ticker] = returns

# Align all returns to same length
min_return_len = min(len(r) for r in returns_dict.values())
returns_matrix = np.array([returns_dict[t][:min_return_len] for t in ticker_list])

# Compute correlation matrix
plain_corr_matrix = np.corrcoef(returns_matrix)
print(f"  Plain correlation matrix shape: {plain_corr_matrix.shape}")

# =============================================================================
# 7. Hierarchical Clustering on Both Correlation Matrices
# =============================================================================
n_clusters = 4

def cluster_from_corr_matrix(corr_matrix, n_clusters, method='ward'):
    """Perform hierarchical clustering on a correlation matrix."""
    # Convert correlation to distance: d = 1 - corr
    distance_matrix = 1 - corr_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Handle any remaining numerical issues
    distance_matrix = np.clip(distance_matrix, 0, 2)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Convert to condensed form for linkage
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    Z = linkage(condensed_dist, method=method)
    labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-indexed
    
    return labels, Z

# Cluster using RNN-Correlation
print("\nClustering with RNN-Correlation...")
labels_rnn_corr, linkage_rnn = cluster_from_corr_matrix(rnn_corr_matrix, n_clusters)
clusters_rnn_corr = {i: [] for i in range(n_clusters)}
for ticker, label in zip(ticker_list, labels_rnn_corr):
    clusters_rnn_corr[label].append(ticker)

print(f"\nRNN-Correlation Clusters:")
for cluster_id, stocks in clusters_rnn_corr.items():
    print(f"  Cluster {cluster_id}: {stocks}")

# Cluster using Plain Correlation
print("\nClustering with Plain Correlation...")
labels_plain_corr, linkage_plain = cluster_from_corr_matrix(plain_corr_matrix, n_clusters)
clusters_plain_corr = {i: [] for i in range(n_clusters)}
for ticker, label in zip(ticker_list, labels_plain_corr):
    clusters_plain_corr[label].append(ticker)

print(f"\nPlain Correlation Clusters:")
for cluster_id, stocks in clusters_plain_corr.items():
    print(f"  Cluster {cluster_id}: {stocks}")

# =============================================================================
# 8. Comparison
# =============================================================================
print("\n" + "="*70)
print("COMPARISON: RNN-Correlation vs Plain Correlation")
print("="*70)

print("\n{:<8} | {:<18} | {:<18}".format("Ticker", "RNN-Correlation", "Plain Correlation"))
print("-" * 50)
for i, ticker in enumerate(ticker_list):
    rnn_cl = labels_rnn_corr[i]
    plain_cl = labels_plain_corr[i]
    match = "✓" if rnn_cl == plain_cl else ""
    print(f"{ticker:<8} | Cluster {rnn_cl:<10} | Cluster {plain_cl:<10} {match}")

ari_score = adjusted_rand_score(labels_rnn_corr, labels_plain_corr)
print(f"\nAdjusted Rand Index: {ari_score:.4f}")
print("(1.0 = identical, 0.0 = random)")

# =============================================================================
# 9. Visualization: Correlation Matrices Heatmaps
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# RNN Correlation Matrix
im1 = axes[0].imshow(rnn_corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
axes[0].set_xticks(range(n_stocks))
axes[0].set_yticks(range(n_stocks))
axes[0].set_xticklabels(ticker_list, rotation=45, ha='right', fontsize=8)
axes[0].set_yticklabels(ticker_list, fontsize=8)
axes[0].set_title('RNN-Correlation Matrix\n(Correlation of Encoder Embeddings Over Time)')
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Plain Correlation Matrix
im2 = axes[1].imshow(plain_corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
axes[1].set_xticks(range(n_stocks))
axes[1].set_yticks(range(n_stocks))
axes[1].set_xticklabels(ticker_list, rotation=45, ha='right', fontsize=8)
axes[1].set_yticklabels(ticker_list, fontsize=8)
axes[1].set_title('Plain Correlation Matrix\n(Correlation of Daily Returns)')
plt.colorbar(im2, ax=axes[1], shrink=0.8)

plt.tight_layout()
plt.savefig('correlation_matrices_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nCorrelation matrices saved to 'correlation_matrices_comparison.png'")

# =============================================================================
# 10. Visualization: Dendrograms
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# RNN-Correlation Dendrogram
axes[0].set_title('RNN-Correlation Dendrogram')
dendrogram(linkage_rnn, labels=ticker_list, ax=axes[0], leaf_rotation=45)
axes[0].set_ylabel('Distance (1 - correlation)')

# Plain Correlation Dendrogram
axes[1].set_title('Plain Correlation Dendrogram')
dendrogram(linkage_plain, labels=ticker_list, ax=axes[1], leaf_rotation=45)
axes[1].set_ylabel('Distance (1 - correlation)')

plt.tight_layout()
plt.savefig('dendrograms_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Dendrograms saved to 'dendrograms_comparison.png'")

# =============================================================================
# 11. Visualization: Cluster Price Plots (Side by Side)
# =============================================================================
fig, axes = plt.subplots(n_clusters, 2, figsize=(18, 4 * n_clusters))

for cluster_id in range(n_clusters):
    # RNN-Correlation clusters (left)
    ax_rnn = axes[cluster_id, 0] if n_clusters > 1 else axes[0]
    for ticker in clusters_rnn_corr[cluster_id]:
        prices = stock_data[ticker]['raw_prices'].flatten()
        normalized = (prices - prices.min()) / (prices.max() - prices.min())
        ax_rnn.plot(normalized, label=ticker, alpha=0.7)
    ax_rnn.set_title(f'RNN-Corr Cluster {cluster_id}: {clusters_rnn_corr[cluster_id]}')
    ax_rnn.set_xlabel('Trading Days')
    ax_rnn.set_ylabel('Normalized Price')
    ax_rnn.legend(loc='upper left', fontsize=7)
    ax_rnn.grid(True, alpha=0.3)
    
    # Plain Correlation clusters (right)
    ax_plain = axes[cluster_id, 1] if n_clusters > 1 else axes[1]
    for ticker in clusters_plain_corr[cluster_id]:
        prices = stock_data[ticker]['raw_prices'].flatten()
        normalized = (prices - prices.min()) / (prices.max() - prices.min())
        ax_plain.plot(normalized, label=ticker, alpha=0.7)
    ax_plain.set_title(f'Plain Corr Cluster {cluster_id}: {clusters_plain_corr[cluster_id]}')
    ax_plain.set_xlabel('Trading Days')
    ax_plain.set_ylabel('Normalized Price')
    ax_plain.legend(loc='upper left', fontsize=7)
    ax_plain.grid(True, alpha=0.3)

plt.suptitle('RNN-Correlation (Left) vs Plain Correlation (Right)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('correlation_clusters_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nCluster visualization saved to 'correlation_clusters_comparison.png'")

# =============================================================================
# 12. Summary Statistics
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
Method Comparison:
-----------------
• RNN-Correlation:
  - Uses encoder embeddings (learned representations)
  - Captures correlation of "latent pattern dynamics" over time
  - May find deeper structural similarities
  
• Plain Correlation:
  - Uses raw daily returns
  - Captures stocks that move up/down together on same days
  - Traditional approach, highly interpretable

Adjusted Rand Index: {ari_score:.4f}
{"→ Clusters are very similar!" if ari_score > 0.7 else "→ Methods produce different groupings."}
""")
