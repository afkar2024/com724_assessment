import os
import yaml
import time
import pandas as pd
import numpy as np
from flask import Flask, jsonify, send_file
from flask_socketio import SocketIO, emit
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

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
# Flask & SocketIO setup
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret!')
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
# Data acquisition functions
# -----------------------------
def download_ticker_data(ticker):
    """Download single ticker data and return DataFrame or None."""
    try:
        df = yf.download(ticker, period=PERIOD)
        if df.empty:
            return ticker, None
        df.dropna(inplace=True)
        return ticker, df
    except Exception as e:
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

# -----------------------------
# REST API endpoints
# -----------------------------
@app.route('/api/status')
def status():
    return jsonify({'status': 'ok'})

@app.route('/api/tickers')
def get_tickers():
    return jsonify({'tickers': TICKERS})

@app.route('/api/data/cached')
def data_cached():
    """Return a list of tickers available in cache."""
    data = load_cached_data()
    return jsonify({'available': list(data.keys())})

@app.route('/api/data/refresh')
def data_refresh():
    """Trigger a fresh download of all data."""
    socketio.start_background_task(fetch_and_notify)
    return jsonify({'task': 'refresh started'})

def fetch_and_notify():
    data = fetch_all_data()
    socketio.emit('data_refreshed', {'tickers': list(data.keys())})

@app.route('/api/data/<string:ticker>')
def get_ticker_data(ticker):
    """Return OHLCV JSON for a specific ticker from cache."""
    data = load_cached_data()
    if ticker not in data:
        return jsonify({'error': 'ticker not found'}), 404
    df = data[ticker]
    # Simplify with date and Close price
    payload = df['Close'].reset_index().to_dict(orient='records')
    return jsonify({ticker: payload})

# -----------------------------
# WebSocket events
# -----------------------------
@socketio.on('connect')
def on_connect():
    emit('message', {'data': 'Connected to Crypto API'})

@socketio.on('refresh_data')
def on_refresh(json):
    socketio.start_background_task(fetch_and_notify)

# -----------------------------
# Main entry
# -----------------------------
if __name__ == '__main__':
    # For development: use eventlet or gevent for async support
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)