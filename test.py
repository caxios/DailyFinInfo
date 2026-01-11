import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =============================================================================
# 1. Encoder - Extracts features from price sequences
# =============================================================================
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=n_layers, batch_first=True)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Size)
        # h_n: (Num_Layers, Batch, Hidden_Size)
        _, h_next = self.rnn(x)
        return h_next  # Context Vector (hidden state)

# =============================================================================
# 2. Decoder - Predicts future prices
# =============================================================================
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_prev):
        out, h_next = self.rnn(x, h_prev)
        pred = self.fc(out)
        return pred, h_next

# =============================================================================
# 3. Seq2Seq Model with encode_only for clustering
# =============================================================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, target_len):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_len = target_len

    def forward(self, x, target=None):
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
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def encode_only(self, x):
        """
        Extract encoder's context vector for clustering.
        Returns: (Batch, n_layers * hidden_size) flattened vector
        """
        context_vector = self.encoder(x)  # (n_layers, batch, hidden_size)
        batch_size = x.size(0)
        return context_vector.transpose(0, 1).reshape(batch_size, -1)

# =============================================================================
# 4. Data Preparation - Multiple Stocks
# =============================================================================
# List of stock tickers to cluster
tickers = [
    # Tech stocks
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'TSLA',
    # Financial stocks
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C',
    # Healthcare stocks
    'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV',
    # Consumer stocks
    'WMT', 'AMZN', 'HD', 'NKE', 'MCD', 'SBUX',
    # Energy stocks
    'XOM', 'CVX', 'COP', 'SLB',
]

period = '1y'
sequence_length = 30
target_len = 5

print(f"Fetching data for {len(tickers)} stocks...")

# Store processed data for each stock
stock_data = {}
all_sequences = []
all_targets = []
stock_indices = []  # Track which sequences belong to which stock

for ticker in tickers:
    try:
        data = yf.Ticker(ticker).history(period=period)
        if len(data) < sequence_length + target_len:
            print(f"  Skipping {ticker}: Not enough data")
            continue
        
        close_prices = data['Close'].values.reshape(-1, 1)
        
        # Normalize each stock's prices individually
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        # Create sequences for this stock
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
                'raw_prices': close_prices
            }
            all_sequences.extend(X_list)
            all_targets.extend(y_list)
            stock_indices.extend([ticker] * len(X_list))
            print(f"  {ticker}: {len(X_list)} sequences")
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")

# Convert to tensors
X = torch.tensor(np.array(all_sequences), dtype=torch.float32)
y = torch.tensor(np.array(all_targets), dtype=torch.float32)

print(f"\nTotal sequences: {X.shape[0]}")
print(f"Sequence shape: {X.shape}")
print(f"Target shape: {y.shape}")

# =============================================================================
# 5. Model Setup
# =============================================================================
input_size = 1  # Price is 1-dimensional
hidden_size = 64
n_layers = 2
output_size = 1

enc = Encoder(input_size, hidden_size, n_layers)
dec = Decoder(output_size, hidden_size, output_size, n_layers)
model = Seq2Seq(enc, dec, target_len)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =============================================================================
# 6. Training
# =============================================================================
num_epochs = 100
print(f"\nTraining for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print("Training complete.\n")

# =============================================================================
# 7. Extract Encoder Embeddings for Each Stock
# =============================================================================
print("--- Extracting Stock Embeddings ---")

model.eval()
stock_embeddings = {}

with torch.no_grad():
    for ticker, data in stock_data.items():
        X_stock = torch.tensor(data['X'], dtype=torch.float32)
        
        # Get encoder embeddings for all sequences of this stock
        embeddings = model.encode_only(X_stock).numpy()
        
        # Aggregate: Use mean embedding to represent the stock's overall pattern
        # You could also use: max, last, or weighted mean
        stock_embedding = embeddings.mean(axis=0)
        stock_embeddings[ticker] = stock_embedding
        
        print(f"  {ticker}: embedding shape = {stock_embedding.shape}")

# Convert to matrix for clustering
embedding_matrix = np.array(list(stock_embeddings.values()))
ticker_list = list(stock_embeddings.keys())

print(f"\nEmbedding matrix shape: {embedding_matrix.shape}")

# =============================================================================
# 8. K-Means Clustering of Stocks
# =============================================================================
print("\n--- K-Means Clustering Results ---")

n_clusters = 4  # Adjust based on how many groups you want
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embedding_matrix)

# Group stocks by cluster
clusters = {i: [] for i in range(n_clusters)}
for ticker, label in zip(ticker_list, labels):
    clusters[label].append(ticker)

# Print clustering results
print(f"\nStocks grouped into {n_clusters} clusters:\n")
for cluster_id, stocks in clusters.items():
    print(f"Cluster {cluster_id}: {stocks}")

# =============================================================================
# 9. Visualization - Plot stocks by cluster
# =============================================================================
fig, axes = plt.subplots(n_clusters, 1, figsize=(14, 4 * n_clusters))

for cluster_id in range(n_clusters):
    ax = axes[cluster_id] if n_clusters > 1 else axes
    cluster_stocks = clusters[cluster_id]
    
    for ticker in cluster_stocks:
        # Normalize prices to 0-1 range for comparison
        prices = stock_data[ticker]['raw_prices'].flatten()
        normalized = (prices - prices.min()) / (prices.max() - prices.min())
        ax.plot(normalized, label=ticker, alpha=0.7)
    
    ax.set_title(f'Cluster {cluster_id}: {cluster_stocks}')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Normalized Price')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stock_clusters_rnn_kmeans.png', dpi=150)
plt.show()

print("\nRNN-KMeans visualization saved to 'stock_clusters_rnn_kmeans.png'")

# =============================================================================
# 10. PLAIN K-MEANS (Without RNN) - For Comparison
# =============================================================================
print("\n" + "="*70)
print("PLAIN K-MEANS CLUSTERING (Without RNN)")
print("="*70)

def extract_simple_features(prices):
    """
    Extract simple statistical features from price series.
    These features capture basic characteristics of price movement.
    """
    prices = prices.flatten()
    
    # Normalize prices to 0-1 range
    normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
    
    # Daily returns
    returns = np.diff(prices) / (prices[:-1] + 1e-8)
    
    features = [
        # Basic statistics
        np.mean(returns),           # Average daily return
        np.std(returns),            # Volatility
        np.min(returns),            # Worst day
        np.max(returns),            # Best day
        
        # Trend indicators
        (prices[-1] - prices[0]) / (prices[0] + 1e-8),  # Total return
        np.mean(normalized[:len(normalized)//2]),  # First half avg
        np.mean(normalized[len(normalized)//2:]),  # Second half avg
        
        # Momentum
        np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns),  # Recent momentum
        np.mean(returns[:20]) if len(returns) >= 20 else np.mean(returns),   # Early momentum
        
        # Shape of the curve (sample points from normalized curve)
        normalized[0],                          # Start
        normalized[len(normalized)//4],         # Q1
        normalized[len(normalized)//2],         # Mid
        normalized[3*len(normalized)//4],       # Q3
        normalized[-1],                         # End
    ]
    
    return np.array(features)

# Extract simple features for each stock
print("\nExtracting simple statistical features...")
simple_features = {}
for ticker in stock_data.keys():
    prices = stock_data[ticker]['raw_prices']
    features = extract_simple_features(prices)
    simple_features[ticker] = features
    print(f"  {ticker}: {len(features)} features")

# Convert to matrix
simple_feature_matrix = np.array(list(simple_features.values()))
simple_ticker_list = list(simple_features.keys())

# Normalize features for better clustering
from sklearn.preprocessing import StandardScaler
scaler_features = StandardScaler()
simple_feature_matrix_scaled = scaler_features.fit_transform(simple_feature_matrix)

print(f"\nSimple feature matrix shape: {simple_feature_matrix_scaled.shape}")

# Plain K-Means clustering
kmeans_plain = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_plain = kmeans_plain.fit_predict(simple_feature_matrix_scaled)

# Group stocks by cluster (Plain K-Means)
clusters_plain = {i: [] for i in range(n_clusters)}
for ticker, label in zip(simple_ticker_list, labels_plain):
    clusters_plain[label].append(ticker)

print(f"\nPlain K-Means Clustering Results ({n_clusters} clusters):\n")
for cluster_id, stocks in clusters_plain.items():
    print(f"Cluster {cluster_id}: {stocks}")

# =============================================================================
# 11. COMPARISON: RNN-KMeans vs Plain K-Means
# =============================================================================
print("\n" + "="*70)
print("COMPARISON: RNN-KMeans vs Plain K-Means")
print("="*70)

# Create comparison dataframe
print("\n{:<8} | {:<15} | {:<15}".format("Ticker", "RNN-KMeans", "Plain K-Means"))
print("-" * 45)
for ticker in ticker_list:
    rnn_cluster = labels[ticker_list.index(ticker)]
    plain_cluster = labels_plain[simple_ticker_list.index(ticker)]
    match = "✓" if rnn_cluster == plain_cluster else ""
    print(f"{ticker:<8} | Cluster {rnn_cluster:<7} | Cluster {plain_cluster:<7} {match}")

# Calculate agreement (note: cluster IDs may differ, so we use adjusted rand score)
from sklearn.metrics import adjusted_rand_score
ari_score = adjusted_rand_score(labels, labels_plain)
print(f"\nAdjusted Rand Index (clustering similarity): {ari_score:.4f}")
print("(1.0 = identical clustering, 0.0 = random, negative = worse than random)")

# =============================================================================
# 12. Side-by-Side Visualization
# =============================================================================
fig, axes = plt.subplots(n_clusters, 2, figsize=(18, 4 * n_clusters))

for cluster_id in range(n_clusters):
    # RNN-KMeans (left column)
    ax_rnn = axes[cluster_id, 0] if n_clusters > 1 else axes[0]
    cluster_stocks_rnn = clusters[cluster_id]
    
    for ticker in cluster_stocks_rnn:
        prices = stock_data[ticker]['raw_prices'].flatten()
        normalized = (prices - prices.min()) / (prices.max() - prices.min())
        ax_rnn.plot(normalized, label=ticker, alpha=0.7)
    
    ax_rnn.set_title(f'RNN-KMeans Cluster {cluster_id}: {cluster_stocks_rnn}')
    ax_rnn.set_xlabel('Trading Days')
    ax_rnn.set_ylabel('Normalized Price')
    ax_rnn.legend(loc='upper left', fontsize=7)
    ax_rnn.grid(True, alpha=0.3)
    
    # Plain K-Means (right column)
    ax_plain = axes[cluster_id, 1] if n_clusters > 1 else axes[1]
    cluster_stocks_plain = clusters_plain[cluster_id]
    
    for ticker in cluster_stocks_plain:
        prices = stock_data[ticker]['raw_prices'].flatten()
        normalized = (prices - prices.min()) / (prices.max() - prices.min())
        ax_plain.plot(normalized, label=ticker, alpha=0.7)
    
    ax_plain.set_title(f'Plain K-Means Cluster {cluster_id}: {cluster_stocks_plain}')
    ax_plain.set_xlabel('Trading Days')
    ax_plain.set_ylabel('Normalized Price')
    ax_plain.legend(loc='upper left', fontsize=7)
    ax_plain.grid(True, alpha=0.3)

plt.suptitle('Comparison: RNN-KMeans (Left) vs Plain K-Means (Right)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('stock_clusters_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nComparison visualization saved to 'stock_clusters_comparison.png'")
