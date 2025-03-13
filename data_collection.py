import os
import time
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

data_file = 'crypto_data.csv'

if os.path.exists(data_file):
    print("Loading cached data from file...")
    combined_data = pd.read_csv(data_file, header=[0, 1], index_col=0, parse_dates=True)
    # Convert all data columns to numeric types
    combined_data = combined_data.apply(pd.to_numeric, errors='coerce')
else:
    print("Cached data not found. Downloading data for 5 years...")
    all_data = {}
    failed_tickers = []
    for ticker in tickers:
        try:
            print(f"Downloading data for {ticker}...")
            df = yf.download(ticker, period="5y")
            if df.empty:
                print(f"No data downloaded for {ticker}.")
                failed_tickers.append(ticker)
                continue
            all_data[ticker] = df
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            failed_tickers.append(ticker)
        time.sleep(2)
    if failed_tickers:
        print("Failed tickers:", failed_tickers)
    if not all_data:
        raise ValueError("No data downloaded. Exiting.")
    combined_data = pd.concat(all_data, axis=1)
    combined_data.to_csv(data_file)
    print(f"Data downloaded and saved to {data_file}.")


# Proceed with your script using combined_data
all_ticker_data = {}

for ticker in tickers:
    if ticker in combined_data.columns.get_level_values(0):
        df = combined_data[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
        print(f"{ticker}: initial data shape = {df.shape}")
        before_drop = df.shape[0]
        df.dropna(inplace=True)
        after_drop = df.shape[0]
        dropped = before_drop - after_drop
        if dropped > 0:
            print(f"{ticker}: Dropped {dropped} rows due to missing values.")
        all_ticker_data[ticker] = df
    else:
        print(f"{ticker}: Data missing from combined dataset, skipped.")

# Ensure a consistent date range across all tickers by intersecting the dates.
if len(all_ticker_data) == 0:
    raise ValueError("No valid data for any ticker after processing.")

common_dates = set.intersection(*(set(df.index) for df in all_ticker_data.values()))
print("Common dates across all tickers:", len(common_dates))
for ticker in all_ticker_data:
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
print(f"Components to explain >=90% variance: {optimal_n_components}")
print("Cumulative explained variance for selected components:", cumulative_variance[:optimal_n_components])

# --- Grid Search for Best Number of PCA Components for Clustering ---
num_clusters = 4
silhouette_scores = {}
max_components_to_try = optimal_n_components  # You can also try a wider range if needed

# Here, we use Agglomerative Clustering since it previously yielded a higher silhouette score
for n in range(2, max_components_to_try + 1):
    pca_temp = PCA(n_components=n)
    features_pca_temp = pca_temp.fit_transform(features_scaled)
    agg_temp = AgglomerativeClustering(n_clusters=num_clusters)
    labels_temp = agg_temp.fit_predict(features_pca_temp)
    score_temp = silhouette_score(features_pca_temp, labels_temp)
    silhouette_scores[n] = score_temp
    print(f"Silhouette score for {n} PCA components: {score_temp:.4f}")

best_n = max(silhouette_scores, key=silhouette_scores.get)
print(f"Best number of PCA components based on silhouette score: {best_n}, score: {silhouette_scores[best_n]:.4f}")

# Use the best number of components for clustering
pca_opt = PCA(n_components=best_n)
features_pca = pca_opt.fit_transform(features_scaled)

# Clustering - Compare KMeans and Agglomerative Clustering.

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
# For consistency, compute percent changes for Close prices from the common dates.
close_returns = {}
for ticker in all_ticker_data:  # iterate over successfully processed tickers only
    df = all_ticker_data[ticker]
    close_returns[ticker] = df['Close'].pct_change().dropna()

close_returns_df = pd.DataFrame(close_returns)
# Use only the selected tickers for correlation analysis.
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

# -----------------------------
# Visualization: PCA Scatter Plot and Correlation Heatmap
# -----------------------------

# Use the best_n PCA transformation for clustering visualization
pca_opt = PCA(n_components=best_n)
features_pca = pca_opt.fit_transform(features_scaled)

# Visualization using best_n PCA components
if best_n == 2:
    # Simple 2D scatter plot
    plt.figure(figsize=(16, 12))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=best_labels, cmap="viridis", s=200)
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (features_pca[i, 0], features_pca[i, 1]), fontsize=12, alpha=0.85)
    plt.xlabel("PCA Component 1", fontsize=14)
    plt.ylabel("PCA Component 2", fontsize=14)
    plt.title(f"PCA (2D) of 5-Year OHLCV Returns\nClustering: {clustering_algo}", fontsize=16)
    plt.colorbar(scatter, label="Cluster", ticks=range(num_clusters))
    plt.tight_layout()
    plt.savefig("pca_plot_bestn.png", dpi=300)
    plt.show()
elif best_n == 3:
    # 3D scatter plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2],
                         c=best_labels, cmap="viridis", s=200)
    for i, ticker in enumerate(tickers):
        ax.text(features_pca[i, 0], features_pca[i, 1], features_pca[i, 2],
                ticker, fontsize=10, alpha=0.85)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    plt.title(f"PCA (3D) of 5-Year OHLCV Returns\nClustering: {clustering_algo}", fontsize=16)
    plt.colorbar(scatter, label="Cluster", ticks=range(num_clusters))
    plt.tight_layout()
    plt.savefig("pca_plot_bestn_3d.png", dpi=300)
    plt.show()
else:
    # For best_n > 3, create a scatter matrix (pair plot)
    # Create a DataFrame from the PCA features
    pca_df = pd.DataFrame(features_pca, index=tickers, 
                          columns=[f"PC{i+1}" for i in range(best_n)])
    pd.plotting.scatter_matrix(pca_df, figsize=(20, 20), diagonal='kde', alpha=0.8)
    plt.suptitle(f"Scatter Matrix for Best {best_n} PCA Components", fontsize=20)
    plt.savefig("pca_scatter_matrix.png", dpi=300)
    plt.show()

# Correlation heatmap for selected cryptocurrencies with higher resolution
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation Coefficient")
plt.xticks(range(len(selected)), selected, rotation=45, fontsize=12)
plt.yticks(range(len(selected)), selected, fontsize=12)
plt.title("Correlation Heatmap for Selected Cryptocurrencies", fontsize=16)
plt.tight_layout()
plt.savefig("correlation_plot.png", dpi=300)
plt.show()

# -----------------------------
# EDA: Exploratory Data Analysis
# -----------------------------
print("\nPerforming EDA for selected cryptocurrencies...")

for ticker in selected:
    print(f"\nEDA for {ticker}:")
    df = all_ticker_data[ticker].copy()
    # Ensure the index is in datetime format
    df.index = pd.to_datetime(df.index)
    
    # 1. Time Series Plot of Close Prices with 50-Day Moving Average
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df['Close'], label=f"{ticker} Close Price", color='tab:blue')
    # Calculate a 50-day moving average and overlay
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    plt.plot(df.index, df['50_MA'], label="50-Day MA", color='tab:orange', linewidth=2)
    plt.title(f"Time Series of {ticker} Close Prices with 50-Day MA")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ticker}_timeseries_ma.png")
    plt.close()
    
    # 2. Distribution of Daily Returns (Histogram + KDE)
    returns = df['Close'].pct_change().dropna()
    plt.figure(figsize=(10, 4))
    plt.hist(returns, bins=50, density=True, alpha=0.6, color='tab:green', label='Histogram')
    returns.plot(kind='kde', color='tab:orange', label='KDE')
    plt.title(f"Distribution of Daily Returns for {ticker}")
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ticker}_returns_distribution.png")
    plt.close()
    
    # 3. Box Plot of Daily Returns Grouped by Year
    df_returns = df['Close'].pct_change().dropna().to_frame(name='Return')
    df_returns['Year'] = pd.to_datetime(df_returns.index).year
    plt.figure(figsize=(10, 6))
    df_returns.boxplot(column='Return', by='Year', grid=False)
    plt.title(f"Yearly Boxplot of Daily Returns for {ticker}")
    plt.suptitle("")
    plt.xlabel("Year")
    plt.ylabel("Daily Return")
    plt.tight_layout()
    plt.savefig(f"{ticker}_yearly_boxplot.png")
    plt.close()
    
    # 4. Rolling Volatility Plot: 30-day Rolling Std Dev of Daily Returns
    rolling_vol = returns.rolling(window=30).std()
    plt.figure(figsize=(10, 4))
    plt.plot(rolling_vol.index, rolling_vol, color='tab:red', label="30-Day Rolling Volatility")
    plt.title(f"30-Day Rolling Volatility of Daily Returns for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Rolling Std Dev")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ticker}_rolling_volatility.png")
    plt.close()
    
    # 5. Autocorrelation Plot for Daily Returns
    plt.figure(figsize=(8, 4))
    pd.plotting.autocorrelation_plot(returns)
    plt.title(f"Autocorrelation of Daily Returns for {ticker}")
    plt.tight_layout()
    plt.savefig(f"{ticker}_autocorrelation.png")
    plt.close()
    
    # 6. Print Descriptive Statistics for Daily Returns
    print(returns.describe())


print("\nAnalysis complete.")
