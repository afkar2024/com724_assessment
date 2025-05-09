# data_collection.py
import os
import time
import logging
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional callback for real-time notifications (e.g., WebSocket emit)
# Should accept (event_name: str, data: dict)

# -----------------------------
# Load configuration
# -----------------------------
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

TICKERS = config['tickers']
PERIOD = config.get('period', '5y')
DATA_FILE = config.get('data_file', 'crypto_data.csv')
MAX_WORKERS = config.get('max_workers', 5)


# -----------------------------
# Data acquisition functions
# -----------------------------
def download_ticker_data(ticker):
    """Download single ticker data and return DataFrame or None."""
    try:
        df = yf.download(ticker, period=PERIOD)
        if df.empty:
            return ticker, None
        # Fill missing values by interpolation + ffill/bfill
        df.interpolate(method='time', inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return ticker, df
    except Exception:
        return ticker, None


def fetch_all_data():
    """Use ThreadPoolExecutor to download all tickers in parallel."""
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_ticker_data, tk): tk for tk in TICKERS}
        for fut in as_completed(futures):
            tk = futures[fut]
            ticker, df = fut.result()
            if df is not None:
                results[ticker] = df
    # Save to CSV for caching
    if results:
        combined = pd.concat(results, axis=1)
        combined.to_csv(DATA_FILE)
    return results


def load_cached_data():
    """Load cached CSV if exists, else fetch new."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, header=[0,1], index_col=0, parse_dates=True)
        return {tk: df[tk] for tk in df.columns.get_level_values(0).unique()}
    return fetch_all_data()

def load_data(tickers, period="5y", cache_file="crypto_data.csv", notify_fn=None):
    """
    Load OHLCV data for given tickers; use cache if available, else download.
    Returns dict[ticker] = DataFrame.
    """
    def notify(event, data):
        if notify_fn:
            notify_fn(event, data)

    if os.path.exists(cache_file):
        notify('data.load', {'status': 'cache_hit'})
        df = pd.read_csv(cache_file, header=[0,1], index_col=0, parse_dates=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        data = {tk: df[tk] for tk in df.columns.get_level_values(0).unique()}
        notify('data.loaded', {'tickers': list(data.keys())})
        return data

    notify('data.load', {'status': 'cache_miss'})
    data = {}
    for ticker in tickers:
        notify('data.download.start', {'ticker': ticker})
        try:
            df = yf.download(ticker, period=period)
            if df.empty:
                notify('data.download.fail', {'ticker': ticker, 'reason': 'empty'})
                continue
            df.dropna(inplace=True)
            data[ticker] = df
            notify('data.download.success', {'ticker': ticker})
        except Exception as e:
            notify('data.download.fail', {'ticker': ticker, 'reason': str(e)})
        time.sleep(1)
    if data:
        combined = pd.concat(data, axis=1)
        combined.to_csv(cache_file)
        notify('data.cached', {'cache_file': cache_file})
    notify('data.loaded', {'tickers': list(data.keys())})
    return data


# -----------------------------
# Technical Indicators
# -----------------------------
def compute_technical_indicators(data_dict):
    """Compute RSI, MACD, Bollinger Bands, and rolling volatility for each ticker."""
    ind_data = {}
    for tk, df in data_dict.items():
        df2 = df.copy()
        # RSI (14)
        delta = df2['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.ewm(com=13, adjust=False).mean()
        roll_down = down.ewm(com=13, adjust=False).mean()
        rs = roll_up / roll_down
        df2['RSI'] = 100 - (100 / (1 + rs))
        # MACD
        ema12 = df2['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df2['Close'].ewm(span=26, adjust=False).mean()
        df2['MACD'] = ema12 - ema26
        df2['MACD_signal'] = df2['MACD'].ewm(span=9, adjust=False).mean()
        # Bollinger Bands (20)
        ma20 = df2['Close'].rolling(window=20).mean()
        std20 = df2['Close'].rolling(window=20).std()
        df2['BB_upper'] = ma20 + 2 * std20
        df2['BB_lower'] = ma20 - 2 * std20
        # Rolling volatility (30-day)
        df2['Volatility'] = df2['Close'].pct_change().rolling(window=30).std()
        # Fill any NAs after indicators
        df2.interpolate(method='time', inplace=True)
        df2.ffill(inplace=True)
        df2.bfill(inplace=True)
        ind_data[tk] = df2
    return ind_data


def align_data(data_dict, metrics=("Open","High","Low","Close","Volume")):
    """
    Align multiple time-series DataFrames on common dates.
    Returns filtered dict.
    """
    # find intersection of indices
    common = set.intersection(*(set(df.index) for df in data_dict.values()))
    aligned = {}
    for tk, df in data_dict.items():
        aligned[tk] = df.loc[sorted(common), metrics].copy()
    return aligned


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(aligned_data):
    """Compute percent-change features and include technical indicators."""
    features = {}
    for tk, df in aligned_data.items():
        # Percent-change of OHLCV
        df_returns = df.pct_change().dropna()
        # Retrieve computed indicators aligned by index
        # Ensure indicators exist in df (after compute_technical_indicators)
        ind_cols = ['RSI','MACD','MACD_signal','BB_upper','BB_lower','Volatility']
        df_ind = df_returns.index.to_series().apply(lambda d: None)
        # Build indicator matrix
        ind_matrix = np.vstack([aligned_data[tk].loc[df_returns.index, col].values for col in ind_cols]).T
        # Flatten OHLCV + indicators
        vec = np.concatenate([df_returns.values.flatten(), ind_matrix.flatten()])
        features[tk] = vec
    features_df = pd.DataFrame.from_dict(features, orient='index')
    return features_df


def standardize_features(features_df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_df)
    return scaled, scaler


def compute_pca(scaled_features, variance_threshold=0.90):
    """
    Compute PCA, return PCA-transformed data and PCA model.
    Also determine n_components for threshold.
    """
    pca_full = PCA().fit(scaled_features)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = np.argmax(cumsum >= variance_threshold) + 1
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(scaled_features)
    return transformed, pca, n_comp, cumsum


def find_best_pca_components(scaled_features, max_comp, n_clusters=4):
    """
    Grid search PCA n_components by silhouette score for Agglomerative.
    Returns best_n and score dict.
    """
    scores = {}
    for n in range(2, max_comp+1):
        tmp = PCA(n_components=n).fit_transform(scaled_features)
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(tmp)
        scores[n] = silhouette_score(tmp, labels)
    best = max(scores, key=scores.get)
    return best, scores


# -----------------------------
# Clustering & grouping
# -----------------------------
def find_best_k_elbow(scaled_features, k_range=range(2,11)):
    """Return list of (k, inertia) for elbow plot."""
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42).fit(scaled_features)
        inertias[k] = km.inertia_
    return inertias

def cluster_data(pca_features, n_clusters=4):
    """
    Runs KMeans and Agglomerative clustering, picks best by silhouette.
    Returns labels, algorithm name.
    """
    kl = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(pca_features)
    al = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(pca_features)
    sk = silhouette_score(pca_features, kl)
    sa = silhouette_score(pca_features, al)
    if sk >= sa:
        return kl, 'KMeans', sk
    else:
        return al, 'Agglomerative', sa


def select_representatives(pca_features, labels, tickers):
    """
    Pick one ticker per cluster: closest to centroid for KMeans-like.
    """
    reps = []
    for c in np.unique(labels):
        idx = [i for i, l in enumerate(labels) if l == c]
        subset = pca_features[idx]
        centroid = subset.mean(axis=0)
        dist = np.linalg.norm(subset - centroid, axis=1)
        reps.append(tickers[idx[np.argmin(dist)]])
    return reps


def correlation_analysis(aligned_data, selected):
    """
    Compute correlation matrix and top positive/negative pairs for selected tickers.
    """
    ret = {tk: df['Close'].pct_change().dropna() for tk, df in aligned_data.items()}
    df_ret = pd.concat(ret, axis=1)[selected]
    corr = df_ret.corr()
    tri = corr.where(np.triu(np.ones(corr.shape),1).astype(bool))
    pairs = tri.unstack().dropna()
    top_pos = pairs.sort_values(ascending=False).head(4)
    top_neg = pairs.sort_values().head(4)
    return corr, top_pos, top_neg


def perform_eda(aligned_data, selected, output_dir="plots"):
    """
    Generate EDA plots for each selected ticker, save to output_dir.
    Returns dict[ticker] = list of filepaths.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for tk in selected:
        df = aligned_data[tk].copy()
        df.index = pd.to_datetime(df.index)
        files = []
        # Time series + 50MA
        plt.figure(); plt.plot(df['Close']); plt.plot(df['Close'].rolling(50).mean());
        f1 = os.path.join(output_dir, f"{tk}_ts_ma.png"); plt.savefig(f1); plt.close()
        files.append(f1)
        # Distribution
        rets = df['Close'].pct_change().dropna()
        plt.figure(); plt.hist(rets, bins=50, density=True); rets.plot(kind='kde');
        f2 = os.path.join(output_dir, f"{tk}_dist.png"); plt.savefig(f2); plt.close()
        files.append(f2)
        # Boxplot by year
        by = rets.to_frame('r'); by['year']=df.index.year
        plt.figure(); by.boxplot(column='r', by='year');
        f3 = os.path.join(output_dir, f"{tk}_box.png"); plt.savefig(f3); plt.close()
        files.append(f3)
        # Rolling vol
        vol = rets.rolling(30).std()
        plt.figure(); plt.plot(vol);
        f4 = os.path.join(output_dir, f"{tk}_vol.png"); plt.savefig(f4); plt.close()
        files.append(f4)
        results[tk] = files
    return results


# -----------------------------
# Pipeline Orchestration
# -----------------------------
def run_full_pipeline(tickers, notify_fn=None):
    data = load_cached_data()
    # Notify load
    if notify_fn: notify_fn('pipeline.stage', {'stage': 'data_loaded'})
    aligned = align_data(data)
    # Compute indicators
    enriched = compute_technical_indicators(aligned)
    if notify_fn: notify_fn('pipeline.stage', {'stage': 'indicators_computed'})
    features_df = extract_features(enriched)
    if notify_fn: notify_fn('pipeline.stage', {'stage': 'features_extracted'})
    scaled, scaler = standardize_features(features_df)
    pca_feats, pca_model, n_comp, var = compute_pca(scaled)
    if notify_fn: notify_fn('pipeline.stage', {'stage': 'pca_completed', 'components': n_comp})
    # Optionally determine best k with elbow
    # inertias = find_best_k_elbow(pca_feats)
    labels, algo, score = cluster_data(pca_feats)
    reps = select_representatives(pca_feats, labels, list(enriched.keys()))
    corr, pos, neg = correlation_analysis(enriched, reps)
    eda_files = perform_eda(enriched, reps)
    return {
        'representatives': reps,
        'clusters': labels.tolist(),
        'clustering_algo': algo,
        'correlation_matrix': corr,
        'top_positive': pos,
        'top_negative': neg,
        'eda_plots': eda_files
    }
