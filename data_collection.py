import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# List of 30 cryptocurrency tickers
tickers = [
    "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD",
    "DOGE-USD", "DOT-USD", "LTC-USD", "BCH-USD", "SOL-USD",
    "AVAX-USD", "LINK-USD", "UNI-USD", "XLM-USD", "TRX-USD",
    "ETC-USD", "XMR-USD", "ALGO-USD", "VET-USD", "FIL-USD",
    "MATIC-USD", "EOS-USD", "AAVE-USD", "SUSHI-USD", "MKR-USD",
    "ZEC-USD", "DASH-USD", "THETA-USD", "ICP-USD", "KSM-USD"
]

# Download 5-year historical data (daily frequency) for OHLCV.
print("Downloading data for 5 years...")
# Use group_by="ticker" to get a separate DataFrame per coin.
data = yf.download(tickers, period="5y", group_by="ticker")

# Organize data per ticker and show initial preprocessing information.
all_ticker_data = {}

for ticker in tickers:
    # Extract OHLCV columns (we exclude 'Adj Close' as it is redundant with Close)
    df = data[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
    print(f"{ticker}: initial data shape = {df.shape}")
    before_drop = df.shape[0]
    df = df.dropna()
    after_drop = df.shape[0]
    dropped = before_drop - after_drop
    if dropped > 0:
        print(f"{ticker}: Dropped {dropped} rows due to missing values.")
    all_ticker_data[ticker] = df

# Ensure a consistent date range across all tickers by intersecting the dates.
common_dates = set.intersection(*(set(df.index) for df in all_ticker_data.values()))
print("Common dates across all tickers:", len(common_dates))
for ticker in tickers:
    all_ticker_data[ticker] = all_ticker_data[ticker].loc[sorted(common_dates)]

# Feature extraction.
# For each ticker, compute daily percent changes for Open, High, Low, Close, and Volume,
# then flatten the resulting DataFrame so each ticker is represented as a single feature vector.
features = {}
for ticker in tickers:
    df = all_ticker_data[ticker]
    # Compute daily percent change for each metric.
    df_returns = df.pct_change().dropna()
    # Replace infinite values with NaN and then fill NaN with 0.
    df_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_returns.fillna(0, inplace=True)
    # Flatten row-wise into one long feature vector.
    features[ticker] = df_returns.values.flatten()
    print(f"{ticker}: Feature vector length = {features[ticker].shape[0]}")

# Build a DataFrame where each row is a cryptocurrency.
features_df = pd.DataFrame.from_dict(features, orient='index')
print("Shape of feature matrix (cryptos x features):", features_df.shape)

# Preprocessing - Standardize features.
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)
print("Features have been standardized.")

# Dimensionality Reduction via PCA.
# First, run PCA with all components to compute the cumulative explained variance.
pca_full = PCA().fit(features_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
# Choose the minimum number of components needed to explain at least 90% of the variance.
optimal_n_components = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Optimal number of PCA components to retain >=90% variance: {optimal_n_components}")
print("Cumulative explained variance for selected components:", cumulative_variance[:optimal_n_components])

# For visualization, compute PCA with 2 components.
pca_2 = PCA(n_components=2)
pca_result_2 = pca_2.fit_transform(features_scaled)

# For clustering, use the optimal PCA features.
pca_opt = PCA(n_components=optimal_n_components)
features_pca = pca_opt.fit_transform(features_scaled)

# Clustering - Compare KMeans and Agglomerative Clustering.
num_clusters = 4

# KMeans clustering.
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(features_pca)
silhouette_kmeans = silhouette_score(features_pca, kmeans_labels)
print(f"KMeans silhouette score with {num_clusters} clusters: {silhouette_kmeans:.4f}")

# Agglomerative Clustering.
agg_clust = AgglomerativeClustering(n_clusters=num_clusters)
agg_labels = agg_clust.fit_predict(features_pca)
silhouette_agg = silhouette_score(features_pca, agg_labels)
print(f"Agglomerative Clustering silhouette score with {num_clusters} clusters: {silhouette_agg:.4f}")

# Choose the algorithm with the higher silhouette score.
if silhouette_kmeans >= silhouette_agg:
    best_labels = kmeans_labels
    clustering_algo = "KMeans"
    print("KMeans clustering selected based on silhouette score.")
else:
    best_labels = agg_labels
    clustering_algo = "AgglomerativeClustering"
    print("Agglomerative Clustering selected based on silhouette score.")

# Map tickers to clusters.
cluster_df = pd.DataFrame({"Ticker": tickers, "Cluster": best_labels})
print("\nCluster assignments:")
print(cluster_df.sort_values("Cluster"))

# Group tickers by cluster.
groups = cluster_df.groupby("Cluster")["Ticker"].apply(list)
print("\nCryptocurrency groups by cluster:")
for cluster, coins in groups.items():
    print(f"Cluster {cluster}: {coins}")

# Select one representative cryptocurrency per cluster.
# For KMeans, we choose the coin closest to the cluster centroid.
selected = []
if clustering_algo == "KMeans":
    for cluster in groups.index:
        cluster_indices = [i for i, label in enumerate(best_labels) if label == cluster]
        cluster_features = features_pca[cluster_indices]
        centroid = np.mean(cluster_features, axis=0)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        best_index = cluster_indices[np.argmin(distances)]
        selected.append(tickers[best_index])
else:
    # For Agglomerative, simply choose the first coin in each cluster.
    selected = groups.apply(lambda x: x[0]).tolist()
print("\nSelected cryptocurrencies for further analysis:")
print(selected)

# Correlation Analysis (using Close price returns).
# For consistency, we compute percent changes for Close prices from the common dates.
close_returns = {}
for ticker in tickers:
    df = all_ticker_data[ticker]
    close_returns[ticker] = df['Close'].pct_change().dropna()
close_returns_df = pd.DataFrame(close_returns)
# Use only the selected tickers.
selected_close_returns = close_returns_df[selected]

corr_matrix = selected_close_returns.corr()
print("\nCorrelation matrix for selected cryptocurrencies (using Close price returns):")
print(corr_matrix)

# Extract and display the top 4 positively and negatively correlated pairs.
corr_matrix_triu = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
corr_pairs = corr_matrix_triu.unstack().dropna()
top_positive = corr_pairs.sort_values(ascending=False).head(4)
print("\nTop 4 positively correlated pairs:")
print(top_positive)
top_negative = corr_pairs.sort_values(ascending=True).head(4)
print("\nTop 4 negatively correlated pairs:")
print(top_negative)

# Visualization.
# (a) PCA scatter plot (using 2 components) colored by cluster.
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_result_2[:, 0], pca_result_2[:, 1], c=best_labels, cmap="viridis", s=100)
for i, ticker in enumerate(tickers):
    plt.annotate(ticker, (pca_result_2[i, 0], pca_result_2[i, 1]), fontsize=8, alpha=0.75)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(f"PCA of 5-Year Daily OHLCV Returns for 30 Cryptocurrencies\nClustering: {clustering_algo}")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("pca_plot.png")
plt.show()

# (b) Correlation heatmap for the selected cryptocurrencies.
plt.figure(figsize=(6, 5))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation Coefficient")
plt.xticks(range(len(selected)), selected, rotation=45)
plt.yticks(range(len(selected)), selected)
plt.title("Correlation Heatmap for Selected Cryptocurrencies")
plt.tight_layout()
plt.savefig("correlation_plot.png")
plt.show()

print("\nAnalysis complete.")
