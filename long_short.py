"""
Long-Short Strategy Comparison Using Clustering

This script compares 4 clustering methods for a long-short strategy:
1. RNN-KMeans: Clusters based on RNN encoder embeddings
2. Plain K-Means: Clusters based on statistical features
3. RNN-Correlation: Clusters based on RNN embedding correlations
4. Plain Correlation: Clusters based on raw return correlations

Strategy:
- LONG the cluster with highest momentum/expected returns
- SHORT the cluster with lowest momentum/expected returns
- The portfolio return = Long returns - Short returns

Compare: Returns, Volatility, Sharpe Ratio, Max Drawdown, Win Rate
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Seq2Seq Model (Same as other files)
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
        context = self.encoder(x)
        batch_size = x.size(0)
        return context.transpose(0, 1).reshape(batch_size, -1)

# =============================================================================
# 2. Data Preparation
# =============================================================================
print("="*70)
print("LONG-SHORT STRATEGY COMPARISON")
print("="*70)

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
n_clusters = 4

print(f"\nFetching data for {len(tickers)} stocks...")

stock_data = {}
all_sequences = []
all_targets = []
returns_dict = {}

for ticker in tickers:
    try:
        data = yf.Ticker(ticker).history(period=period)
        if len(data) < sequence_length + target_len:
            print(f"  Skipping {ticker}: Not enough data")
            continue
        
        close_prices = data['Close'].values.reshape(-1, 1)
        returns = np.diff(close_prices.flatten()) / (close_prices.flatten()[:-1] + 1e-8)
        returns_dict[ticker] = returns
        
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
                'returns': returns,
                'dates': data.index
            }
            all_sequences.extend(X_list)
            all_targets.extend(y_list)
            print(f"  {ticker}: loaded")
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")

X = torch.tensor(np.array(all_sequences), dtype=torch.float32)
y = torch.tensor(np.array(all_targets), dtype=torch.float32)

ticker_list = list(stock_data.keys())
n_stocks = len(ticker_list)
print(f"\nLoaded {n_stocks} stocks, {X.shape[0]} total sequences")

# =============================================================================
# 3. Train RNN Model
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
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 25 == 0:
        print(f'  Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print("Training complete.")
model.eval()

# =============================================================================
# 4. Extract Features for All Clustering Methods
# =============================================================================
print("\n" + "="*70)
print("EXTRACTING FEATURES FOR CLUSTERING")
print("="*70)

# 4.1 RNN Embeddings
rnn_embeddings = {}
rnn_embedding_series = {}

with torch.no_grad():
    for ticker in ticker_list:
        X_stock = torch.tensor(stock_data[ticker]['X'], dtype=torch.float32)
        embeddings = model.encode_only(X_stock).numpy()
        rnn_embedding_series[ticker] = embeddings
        rnn_embeddings[ticker] = embeddings.mean(axis=0)

# 4.2 Simple Statistical Features
def extract_simple_features(prices):
    prices = prices.flatten()
    normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
    returns = np.diff(prices) / (prices[:-1] + 1e-8)
    
    features = [
        np.mean(returns), np.std(returns), np.min(returns), np.max(returns),
        (prices[-1] - prices[0]) / (prices[0] + 1e-8),
        np.mean(normalized[:len(normalized)//2]),
        np.mean(normalized[len(normalized)//2:]),
        np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns),
        np.mean(returns[:20]) if len(returns) >= 20 else np.mean(returns),
        normalized[0], normalized[len(normalized)//4], normalized[len(normalized)//2],
        normalized[3*len(normalized)//4], normalized[-1],
    ]
    return np.array(features)

simple_features = {t: extract_simple_features(stock_data[t]['raw_prices']) for t in ticker_list}

# 4.3 Correlation Matrices
min_len = min(emb.shape[0] for emb in rnn_embedding_series.values())
embedding_dim = hidden_size * n_layers
rnn_corr_matrix = np.zeros((n_stocks, n_stocks))

for i, ti in enumerate(ticker_list):
    for j, tj in enumerate(ticker_list):
        if i == j:
            rnn_corr_matrix[i, j] = 1.0
        elif i < j:
            emb_i = rnn_embedding_series[ti][:min_len]
            emb_j = rnn_embedding_series[tj][:min_len]
            correlations = []
            for d in range(embedding_dim):
                corr = np.corrcoef(emb_i[:, d], emb_j[:, d])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            avg_corr = np.mean(correlations) if correlations else 0
            rnn_corr_matrix[i, j] = avg_corr
            rnn_corr_matrix[j, i] = avg_corr

min_return_len = min(len(returns_dict[t]) for t in ticker_list)
returns_matrix_all = np.array([returns_dict[t][:min_return_len] for t in ticker_list])
plain_corr_matrix = np.corrcoef(returns_matrix_all)

print("Feature extraction complete.")

# =============================================================================
# 5. Perform All 4 Clustering Methods
# =============================================================================
print("\n" + "="*70)
print("PERFORMING 4 CLUSTERING METHODS")
print("="*70)

clustering_results = {}

# 5.1 RNN-KMeans
rnn_embedding_matrix = np.array([rnn_embeddings[t] for t in ticker_list])
kmeans_rnn = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_rnn_kmeans = kmeans_rnn.fit_predict(rnn_embedding_matrix)
clustering_results['RNN-KMeans'] = {t: labels_rnn_kmeans[i] for i, t in enumerate(ticker_list)}
print("  ✓ RNN-KMeans complete")

# 5.2 Plain K-Means
simple_feature_matrix = np.array([simple_features[t] for t in ticker_list])
scaler_features = StandardScaler()
simple_feature_matrix_scaled = scaler_features.fit_transform(simple_feature_matrix)
kmeans_plain = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_plain_kmeans = kmeans_plain.fit_predict(simple_feature_matrix_scaled)
clustering_results['Plain-KMeans'] = {t: labels_plain_kmeans[i] for i, t in enumerate(ticker_list)}
print("  ✓ Plain K-Means complete")

# 5.3 RNN-Correlation
def cluster_from_corr_matrix(corr_matrix, n_clusters):
    distance_matrix = 1 - corr_matrix
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.clip(distance_matrix, 0, 2)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
    return labels

labels_rnn_corr = cluster_from_corr_matrix(rnn_corr_matrix, n_clusters)
clustering_results['RNN-Correlation'] = {t: labels_rnn_corr[i] for i, t in enumerate(ticker_list)}
print("  ✓ RNN-Correlation complete")

# 5.4 Plain Correlation
labels_plain_corr = cluster_from_corr_matrix(plain_corr_matrix, n_clusters)
clustering_results['Plain-Correlation'] = {t: labels_plain_corr[i] for i, t in enumerate(ticker_list)}
print("  ✓ Plain-Correlation complete")

# =============================================================================
# 6. Long-Short Strategy Construction
# =============================================================================
print("\n" + "="*70)
print("LONG-SHORT STRATEGY CONSTRUCTION")
print("="*70)
print("\nStrategy:")
print("  • LONG: Cluster with highest recent momentum (past 20-day returns)")
print("  • SHORT: Cluster with lowest recent momentum")
print("  • Portfolio Return = Long Returns - Short Returns")

def calculate_cluster_momentum(cluster_stocks, returns_dict, lookback=20):
    """Calculate average momentum for stocks in a cluster."""
    momentums = []
    for ticker in cluster_stocks:
        returns = returns_dict[ticker]
        if len(returns) >= lookback:
            momentum = np.mean(returns[-lookback:])
        else:
            momentum = np.mean(returns)
        momentums.append(momentum)
    return np.mean(momentums) if momentums else 0

def calculate_cluster_returns(cluster_stocks, returns_dict):
    """Calculate equal-weighted returns for a cluster."""
    min_len = min(len(returns_dict[t]) for t in cluster_stocks)
    returns_matrix = np.array([returns_dict[t][:min_len] for t in cluster_stocks])
    return returns_matrix.mean(axis=0)

# For each clustering method, identify long/short clusters
long_short_portfolios = {}

for method, cluster_dict in clustering_results.items():
    # Group stocks by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for ticker, label in cluster_dict.items():
        clusters[label].append(ticker)
    
    # Calculate momentum for each cluster
    cluster_momentums = {}
    for cluster_id, stocks in clusters.items():
        if stocks:
            cluster_momentums[cluster_id] = calculate_cluster_momentum(stocks, returns_dict)
    
    # Sort clusters by momentum
    sorted_clusters = sorted(cluster_momentums.keys(), key=lambda x: cluster_momentums[x], reverse=True)
    
    # Long highest momentum cluster, Short lowest momentum cluster
    long_cluster = sorted_clusters[0]
    short_cluster = sorted_clusters[-1]
    
    long_short_portfolios[method] = {
        'long_cluster': long_cluster,
        'short_cluster': short_cluster,
        'long_stocks': clusters[long_cluster],
        'short_stocks': clusters[short_cluster],
        'long_momentum': cluster_momentums[long_cluster],
        'short_momentum': cluster_momentums[short_cluster],
    }
    
    print(f"\n{method}:")
    print(f"  LONG Cluster {long_cluster}: {clusters[long_cluster]} (momentum: {cluster_momentums[long_cluster]:.4f})")
    print(f"  SHORT Cluster {short_cluster}: {clusters[short_cluster]} (momentum: {cluster_momentums[short_cluster]:.4f})")

# =============================================================================
# 7. Calculate Long-Short Portfolio Returns
# =============================================================================
print("\n" + "="*70)
print("LONG-SHORT PORTFOLIO PERFORMANCE")
print("="*70)

def calculate_long_short_metrics(long_stocks, short_stocks, returns_dict):
    """Calculate long-short portfolio metrics."""
    # Get aligned returns
    all_stocks = long_stocks + short_stocks
    min_len = min(len(returns_dict[t]) for t in all_stocks)
    
    # Long returns (equal-weighted)
    long_returns = np.array([returns_dict[t][:min_len] for t in long_stocks]).mean(axis=0)
    
    # Short returns (equal-weighted)
    short_returns = np.array([returns_dict[t][:min_len] for t in short_stocks]).mean(axis=0)
    
    # Long-Short portfolio: Long returns - Short returns
    # (Shorting means we profit when short portfolio goes down)
    ls_returns = long_returns - short_returns
    
    # Metrics
    total_return = (1 + ls_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(ls_returns)) - 1
    volatility = ls_returns.std() * np.sqrt(252)
    
    risk_free_rate = 0.02
    excess_returns = ls_returns - risk_free_rate / 252
    sharpe = np.mean(excess_returns) / np.std(ls_returns) * np.sqrt(252) if np.std(ls_returns) > 0 else 0
    
    # Max Drawdown
    cumulative = (1 + ls_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win Rate (days with positive return)
    win_rate = (ls_returns > 0).sum() / len(ls_returns)
    
    # Information Ratio (vs long-only benchmark)
    benchmark_returns = long_returns  # Long-only as benchmark
    tracking_error = (ls_returns - benchmark_returns).std() * np.sqrt(252)
    active_return = annualized_return - ((1 + benchmark_returns).prod() - 1)
    info_ratio = active_return / tracking_error if tracking_error > 0 else 0
    
    # Long vs Short performance
    long_cumulative = (1 + long_returns).cumprod()
    short_cumulative = (1 + short_returns).cumprod()
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Information Ratio': info_ratio,
        'Num Long': len(long_stocks),
        'Num Short': len(short_stocks),
        'Long-Short Returns': ls_returns,
        'Long Returns': long_returns,
        'Short Returns': short_returns,
        'LS Cumulative': cumulative,
        'Long Cumulative': long_cumulative,
        'Short Cumulative': short_cumulative,
    }

# Calculate metrics for all methods
results = {}
for method, portfolio_info in long_short_portfolios.items():
    results[method] = calculate_long_short_metrics(
        portfolio_info['long_stocks'],
        portfolio_info['short_stocks'],
        returns_dict
    )

# Also create baselines
# Baseline 1: Long-only all stocks
all_returns_matrix = np.array([returns_dict[t][:min_return_len] for t in ticker_list])
long_only_returns = all_returns_matrix.mean(axis=0)
long_only_cumulative = (1 + long_only_returns).cumprod()
results['Long-Only (Baseline)'] = {
    'Total Return': (1 + long_only_returns).prod() - 1,
    'Annualized Return': ((1 + long_only_returns).prod()) ** (252 / len(long_only_returns)) - 1,
    'Volatility': long_only_returns.std() * np.sqrt(252),
    'Sharpe Ratio': (np.mean(long_only_returns) - 0.02/252) / long_only_returns.std() * np.sqrt(252),
    'Max Drawdown': ((long_only_cumulative - np.maximum.accumulate(long_only_cumulative)) / 
                     np.maximum.accumulate(long_only_cumulative)).min(),
    'Win Rate': (long_only_returns > 0).sum() / len(long_only_returns),
    'Information Ratio': 0,
    'Num Long': len(ticker_list),
    'Num Short': 0,
    'LS Cumulative': long_only_cumulative,
}

# =============================================================================
# 8. Results Comparison Table
# =============================================================================
print("\n" + "-"*100)
print(f"{'Method':<20} | {'Return':<10} | {'Vol':<10} | {'Sharpe':<10} | {'MaxDD':<10} | {'WinRate':<10} | {'L/S':<10}")
print("-"*100)

for method, metrics in results.items():
    ls_info = f"{metrics.get('Num Long', '-')}/{metrics.get('Num Short', '-')}"
    print(f"{method:<20} | {metrics['Annualized Return']*100:>8.2f}% | "
          f"{metrics['Volatility']*100:>8.2f}% | {metrics['Sharpe Ratio']:>8.3f} | "
          f"{metrics['Max Drawdown']*100:>8.2f}% | {metrics['Win Rate']*100:>8.1f}% | "
          f"{ls_info:<10}")
print("-"*100)

# =============================================================================
# 9. Rankings
# =============================================================================
print("\n" + "="*70)
print("RANKINGS BY METRIC")
print("="*70)

method_names = [m for m in results.keys() if 'Baseline' not in m]

rankings = {
    'Sharpe Ratio (↑)': sorted(method_names, key=lambda m: results[m]['Sharpe Ratio'], reverse=True),
    'Annualized Return (↑)': sorted(method_names, key=lambda m: results[m]['Annualized Return'], reverse=True),
    'Volatility (↓)': sorted(method_names, key=lambda m: results[m]['Volatility']),
    'Max Drawdown (↓)': sorted(method_names, key=lambda m: results[m]['Max Drawdown'], reverse=True),
    'Win Rate (↑)': sorted(method_names, key=lambda m: results[m]['Win Rate'], reverse=True),
}

for metric, ranking in rankings.items():
    print(f"\n{metric}:")
    for i, method in enumerate(ranking, 1):
        print(f"  {i}. {method}")

# =============================================================================
# 10. Visualization: Cumulative Returns
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors = {'RNN-KMeans': '#FF6B6B', 'Plain-KMeans': '#4ECDC4', 
          'RNN-Correlation': '#45B7D1', 'Plain-Correlation': '#96CEB4'}

for idx, (method, metrics) in enumerate([(m, results[m]) for m in method_names]):
    ax = axes[idx // 2, idx % 2]
    
    # Plot Long, Short, and Long-Short cumulative returns
    ax.plot(metrics['Long Cumulative'], label='Long Portfolio', color='green', alpha=0.7)
    ax.plot(metrics['Short Cumulative'], label='Short Portfolio', color='red', alpha=0.7)
    ax.plot(metrics['LS Cumulative'], label='Long-Short', color='blue', linewidth=2)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title(f'{method}\nSharpe: {metrics["Sharpe Ratio"]:.2f}, Return: {metrics["Annualized Return"]*100:.1f}%')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('long_short_cumulative.png', dpi=150)
plt.show()
print("\nCumulative returns saved to 'long_short_cumulative.png'")

# =============================================================================
# 11. Visualization: All Long-Short Strategies Comparison
# =============================================================================
plt.figure(figsize=(14, 8))

for method in method_names:
    metrics = results[method]
    label = f"{method} (Sharpe: {metrics['Sharpe Ratio']:.2f})"
    plt.plot(metrics['LS Cumulative'], label=label, linewidth=2)

# Add baseline
plt.plot(results['Long-Only (Baseline)']['LS Cumulative'], 
         label='Long-Only Baseline', linewidth=1, linestyle='--', color='gray', alpha=0.7)
plt.axhline(y=1, color='black', linestyle=':', alpha=0.3)

plt.title('Long-Short Strategy Comparison: All Methods', fontsize=14)
plt.xlabel('Trading Days')
plt.ylabel('Cumulative Return')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('long_short_comparison.png', dpi=150)
plt.show()
print("Strategy comparison saved to 'long_short_comparison.png'")

# =============================================================================
# 12. Visualization: Metrics Bar Charts
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

metrics_to_plot = [
    ('Annualized Return', 'Annualized Return (%)', lambda x: x * 100),
    ('Volatility', 'Volatility (%)', lambda x: x * 100),
    ('Sharpe Ratio', 'Sharpe Ratio', lambda x: x),
    ('Max Drawdown', 'Max Drawdown (%)', lambda x: x * 100),
    ('Win Rate', 'Win Rate (%)', lambda x: x * 100),
    ('Information Ratio', 'Information Ratio', lambda x: x),
]

method_labels = method_names
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for idx, (metric_key, title, transform) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    values = [transform(results[m][metric_key]) for m in method_labels]
    bars = ax.bar(range(len(method_labels)), values, color=colors_bar)
    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels([m.replace('-', '\n') for m in method_labels], fontsize=8, rotation=45, ha='right')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('long_short_metrics.png', dpi=150)
plt.show()
print("Metrics comparison saved to 'long_short_metrics.png'")

# =============================================================================
# 13. Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

best_sharpe_method = max(method_names, key=lambda m: results[m]['Sharpe Ratio'])
best_return_method = max(method_names, key=lambda m: results[m]['Annualized Return'])
lowest_vol_method = min(method_names, key=lambda m: results[m]['Volatility'])
best_winrate_method = max(method_names, key=lambda m: results[m]['Win Rate'])

print(f"""
Long-Short Strategy Results:
----------------------------
• Best Sharpe Ratio: {best_sharpe_method}
  → Sharpe: {results[best_sharpe_method]['Sharpe Ratio']:.3f}

• Best Annualized Return: {best_return_method}
  → Return: {results[best_return_method]['Annualized Return']*100:.2f}%

• Lowest Volatility: {lowest_vol_method}
  → Vol: {results[lowest_vol_method]['Volatility']*100:.2f}%

• Best Win Rate: {best_winrate_method}
  → Win Rate: {results[best_winrate_method]['Win Rate']*100:.1f}%

Strategy Logic:
---------------
• Long the cluster with highest momentum → Bet on winners continuing
• Short the cluster with lowest momentum → Bet on losers continuing to lose
• Different clustering methods identify different "winner" and "loser" groups
• RNN-based methods may capture more complex patterns than plain methods

Comparison vs Long-Only Baseline:
""")

baseline_sharpe = results['Long-Only (Baseline)']['Sharpe Ratio']
baseline_return = results['Long-Only (Baseline)']['Annualized Return']

for method in method_names:
    m = results[method]
    sharpe_diff = m['Sharpe Ratio'] - baseline_sharpe
    return_diff = m['Annualized Return'] - baseline_return
    print(f"  {method}:")
    print(f"    Sharpe vs Baseline: {'+' if sharpe_diff >= 0 else ''}{sharpe_diff:.3f}")
    print(f"    Return vs Baseline: {'+' if return_diff >= 0 else ''}{return_diff*100:.2f}%")

print("\n" + "="*70)
print("END OF ANALYSIS")
print("="*70)
