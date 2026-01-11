"""
Portfolio Diversification Strategy Comparison

This script compares 4 clustering methods for portfolio diversification:
1. RNN-KMeans: Clusters based on RNN encoder embeddings
2. Plain K-Means: Clusters based on statistical features
3. RNN-Correlation: Clusters based on RNN embedding correlations
4. Plain Correlation: Clusters based on raw return correlations

Strategy: Select one stock from each cluster to create a diversified portfolio.
Compare: Returns, Volatility, Sharpe Ratio, Max Drawdown, Diversification Ratio
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Seq2Seq Model (Same as test.py and test2.py)
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
print("PORTFOLIO DIVERSIFICATION STRATEGY COMPARISON")
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
        
        # Calculate daily returns
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

# 4.1 RNN Embeddings (for RNN-KMeans and RNN-Correlation)
rnn_embeddings = {}
rnn_embedding_series = {}

with torch.no_grad():
    for ticker in ticker_list:
        X_stock = torch.tensor(stock_data[ticker]['X'], dtype=torch.float32)
        embeddings = model.encode_only(X_stock).numpy()
        rnn_embedding_series[ticker] = embeddings
        rnn_embeddings[ticker] = embeddings.mean(axis=0)  # Mean for K-Means

# 4.2 Simple Statistical Features (for Plain K-Means)
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
# RNN-Correlation
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

# Plain Correlation
min_return_len = min(len(returns_dict[t]) for t in ticker_list)
returns_matrix = np.array([returns_dict[t][:min_return_len] for t in ticker_list])
plain_corr_matrix = np.corrcoef(returns_matrix)

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
print("  ✓ RNN-KMeans clustering complete")

# 5.2 Plain K-Means
simple_feature_matrix = np.array([simple_features[t] for t in ticker_list])
scaler_features = StandardScaler()
simple_feature_matrix_scaled = scaler_features.fit_transform(simple_feature_matrix)
kmeans_plain = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_plain_kmeans = kmeans_plain.fit_predict(simple_feature_matrix_scaled)
clustering_results['Plain-KMeans'] = {t: labels_plain_kmeans[i] for i, t in enumerate(ticker_list)}
print("  ✓ Plain K-Means clustering complete")

# 5.3 RNN-Correlation (Hierarchical)
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
print("  ✓ RNN-Correlation clustering complete")

# 5.4 Plain Correlation (Hierarchical)
labels_plain_corr = cluster_from_corr_matrix(plain_corr_matrix, n_clusters)
clustering_results['Plain-Correlation'] = {t: labels_plain_corr[i] for i, t in enumerate(ticker_list)}
print("  ✓ Plain-Correlation clustering complete")

# Print clusters for each method
print("\nClustering Results:")
for method, cluster_dict in clustering_results.items():
    clusters = {i: [] for i in range(n_clusters)}
    for ticker, label in cluster_dict.items():
        clusters[label].append(ticker)
    print(f"\n{method}:")
    for cluster_id, stocks in clusters.items():
        print(f"  Cluster {cluster_id}: {stocks}")

# =============================================================================
# 6. Portfolio Construction Strategy
# =============================================================================
print("\n" + "="*70)
print("PORTFOLIO CONSTRUCTION STRATEGY")
print("="*70)
print("\nStrategy: Select ONE stock from EACH cluster for diversification")
print("Selection criterion: Stock with highest Sharpe ratio in each cluster")

def calculate_sharpe(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    if np.std(returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)

# For each clustering method, select best stock from each cluster
portfolios = {}

for method, cluster_dict in clustering_results.items():
    # Group stocks by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for ticker, label in cluster_dict.items():
        clusters[label].append(ticker)
    
    # Select best stock (highest Sharpe) from each cluster
    selected_stocks = []
    for cluster_id, stocks in clusters.items():
        if stocks:
            best_stock = max(stocks, key=lambda t: calculate_sharpe(stock_data[t]['returns']))
            selected_stocks.append(best_stock)
    
    portfolios[method] = selected_stocks
    print(f"\n{method} Portfolio: {selected_stocks}")

# Also create a baseline: Equal-weight all stocks
portfolios['All-Stocks (Baseline)'] = ticker_list

# =============================================================================
# 7. Portfolio Performance Metrics
# =============================================================================
print("\n" + "="*70)
print("PORTFOLIO PERFORMANCE ANALYSIS")
print("="*70)

def calculate_portfolio_metrics(selected_tickers, all_returns_dict, all_dates=None):
    """Calculate portfolio performance metrics."""
    # Align returns to same length
    min_len = min(len(all_returns_dict[t]) for t in selected_tickers)
    returns_matrix = np.array([all_returns_dict[t][:min_len] for t in selected_tickers])
    
    # Equal-weighted portfolio returns
    portfolio_returns = returns_matrix.mean(axis=0)
    
    # Metrics
    total_return = (1 + portfolio_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe = calculate_sharpe(portfolio_returns)
    
    # Max Drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Portfolio Correlation (average pairwise correlation)
    if len(selected_tickers) > 1:
        corr_matrix = np.corrcoef(returns_matrix)
        # Get upper triangle (excluding diagonal)
        upper_triangle = corr_matrix[np.triu_indices(len(selected_tickers), k=1)]
        avg_correlation = upper_triangle.mean()
    else:
        avg_correlation = 1.0
    
    # Diversification Ratio = weighted avg volatility / portfolio volatility
    individual_vols = [all_returns_dict[t][:min_len].std() * np.sqrt(252) for t in selected_tickers]
    avg_individual_vol = np.mean(individual_vols)
    diversification_ratio = avg_individual_vol / volatility if volatility > 0 else 1
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_drawdown,
        'Avg Correlation': avg_correlation,
        'Diversification Ratio': diversification_ratio,
        'Num Stocks': len(selected_tickers),
        'Portfolio Returns': portfolio_returns,
        'Cumulative Returns': cumulative
    }

# Calculate metrics for all portfolios
results = {}
for method, selected_stocks in portfolios.items():
    results[method] = calculate_portfolio_metrics(selected_stocks, returns_dict)

# =============================================================================
# 8. Results Comparison Table
# =============================================================================
print("\n" + "-"*90)
print(f"{'Method':<20} | {'Return':<10} | {'Vol':<10} | {'Sharpe':<10} | {'MaxDD':<10} | {'AvgCorr':<10} | {'DivRatio':<10}")
print("-"*90)

for method, metrics in results.items():
    print(f"{method:<20} | {metrics['Annualized Return']*100:>8.2f}% | "
          f"{metrics['Volatility']*100:>8.2f}% | {metrics['Sharpe Ratio']:>8.3f} | "
          f"{metrics['Max Drawdown']*100:>8.2f}% | {metrics['Avg Correlation']:>8.3f} | "
          f"{metrics['Diversification Ratio']:>8.3f}")
print("-"*90)

# =============================================================================
# 9. Rankings
# =============================================================================
print("\n" + "="*70)
print("RANKINGS BY METRIC")
print("="*70)

# Exclude baseline for fair comparison
method_names = [m for m in results.keys() if m != 'All-Stocks (Baseline)']

rankings = {
    'Sharpe Ratio (↑)': sorted(method_names, key=lambda m: results[m]['Sharpe Ratio'], reverse=True),
    'Volatility (↓)': sorted(method_names, key=lambda m: results[m]['Volatility']),
    'Max Drawdown (↓)': sorted(method_names, key=lambda m: results[m]['Max Drawdown'], reverse=True),
    'Avg Correlation (↓)': sorted(method_names, key=lambda m: results[m]['Avg Correlation']),
    'Diversification Ratio (↑)': sorted(method_names, key=lambda m: results[m]['Diversification Ratio'], reverse=True),
}

for metric, ranking in rankings.items():
    print(f"\n{metric}:")
    for i, method in enumerate(ranking, 1):
        print(f"  {i}. {method}")

# =============================================================================
# 10. Visualization: Cumulative Returns
# =============================================================================
plt.figure(figsize=(14, 8))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
for i, (method, metrics) in enumerate(results.items()):
    cumulative = metrics['Cumulative Returns']
    label = f"{method} (Sharpe: {metrics['Sharpe Ratio']:.2f})"
    plt.plot(cumulative, label=label, linewidth=2 if 'Baseline' not in method else 1,
             alpha=0.5 if 'Baseline' in method else 1, color=colors[i % len(colors)])

plt.title('Portfolio Cumulative Returns Comparison', fontsize=14)
plt.xlabel('Trading Days')
plt.ylabel('Cumulative Return')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('portfolio_cumulative_returns.png', dpi=150)
plt.show()
print("\nCumulative returns chart saved to 'portfolio_cumulative_returns.png'")

# =============================================================================
# 11. Visualization: Metrics Comparison Bar Chart
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

metrics_to_plot = [
    ('Annualized Return', 'Annualized Return (%)', lambda x: x * 100),
    ('Volatility', 'Volatility (%)', lambda x: x * 100),
    ('Sharpe Ratio', 'Sharpe Ratio', lambda x: x),
    ('Max Drawdown', 'Max Drawdown (%)', lambda x: x * 100),
    ('Avg Correlation', 'Avg Pairwise Correlation', lambda x: x),
    ('Diversification Ratio', 'Diversification Ratio', lambda x: x),
]

method_labels = list(results.keys())
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

for idx, (metric_key, title, transform) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    values = [transform(results[m][metric_key]) for m in method_labels]
    bars = ax.bar(range(len(method_labels)), values, color=colors_bar)
    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels([m.replace('-', '\n') for m in method_labels], fontsize=8, rotation=45, ha='right')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('portfolio_metrics_comparison.png', dpi=150)
plt.show()
print("Metrics comparison chart saved to 'portfolio_metrics_comparison.png'")

# =============================================================================
# 12. Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

# Find overall best method (by Sharpe)
best_sharpe_method = max(method_names, key=lambda m: results[m]['Sharpe Ratio'])
best_diversification_method = max(method_names, key=lambda m: results[m]['Diversification Ratio'])
lowest_correlation_method = min(method_names, key=lambda m: results[m]['Avg Correlation'])

print(f"""
Portfolio Diversification Strategy Results:
-------------------------------------------
• Best Sharpe Ratio: {best_sharpe_method}
  → Sharpe: {results[best_sharpe_method]['Sharpe Ratio']:.3f}

• Best Diversification Ratio: {best_diversification_method}
  → Div Ratio: {results[best_diversification_method]['Diversification Ratio']:.3f}

• Lowest Avg Correlation: {lowest_correlation_method}
  → Avg Corr: {results[lowest_correlation_method]['Avg Correlation']:.3f}

Key Insights:
-------------
• Higher Diversification Ratio = more effective diversification
• Lower Avg Correlation = less overlap between stock movements
• Higher Sharpe Ratio = better risk-adjusted returns

Comparison vs Baseline (All-Stocks):
• The clustering-based portfolios use only {n_clusters} stocks vs {len(ticker_list)} stocks
• Compare if diversification is as effective with fewer, strategically selected stocks
""")

# Final comparison table
print("\nFinal Verdict:")
print("-"*50)
baseline = results['All-Stocks (Baseline)']
for method in method_names:
    m = results[method]
    sharpe_diff = m['Sharpe Ratio'] - baseline['Sharpe Ratio']
    div_diff = m['Diversification Ratio'] - baseline['Diversification Ratio']
    print(f"{method}:")
    print(f"  Sharpe vs Baseline: {'+' if sharpe_diff >= 0 else ''}{sharpe_diff:.3f}")
    print(f"  Div Ratio vs Baseline: {'+' if div_diff >= 0 else ''}{div_diff:.3f}")
