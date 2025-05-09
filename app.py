import os
import yaml
from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
from data_collection import (
    load_data, run_full_pipeline
)

# -----------------------------
# Load configuration
# -----------------------------
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

TICKERS = config['tickers']
PERIOD = config.get('period', '5y')
CACHE_FILE = config.get('data_file', 'crypto_data.csv')

# -----------------------------
# Flask & SocketIO setup
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret!')
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory cache
_cached_data = None

# Helper: emit via SocketIO
notify = lambda evt, data: socketio.emit(evt, data)

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
    global _cached_data
    if _cached_data is None:
        _cached_data = load_data(
            TICKERS, period=PERIOD, cache_file=CACHE_FILE, notify_fn=None
        )
    return jsonify({'available': list(_cached_data.keys())})

@app.route('/api/data/refresh')
def data_refresh():
    """Trigger async data reload"""
    socketio.start_background_task(_refresh_data_task)
    return jsonify({'task': 'refresh_started'})


def _refresh_data_task():
    global _cached_data
    _cached_data = load_data(
        TICKERS, period=PERIOD, cache_file=CACHE_FILE, notify_fn=notify
    )

@app.route('/api/data/<string:ticker>')
def get_ticker_data(ticker):
    global _cached_data
    if _cached_data is None:
        _cached_data = load_data(TICKERS, period=PERIOD, cache_file=CACHE_FILE)
    if ticker not in _cached_data:
        return jsonify({'error': 'ticker not found'}), 404
    df = _cached_data[ticker]
    payload = df.reset_index()[['Date', 'Close']].rename(
        columns={'Date': 'date', 'Close': 'close'}
    ).to_dict(orient='records')
    return jsonify({ticker: payload})

@app.route('/api/pipeline')
def pipeline_api():
    """Run full data pipeline in background and emit results"""
    socketio.start_background_task(_pipeline_task)
    return jsonify({'task': 'pipeline_started'})


def _pipeline_task():
    results = run_full_pipeline(
        TICKERS, period=PERIOD, cache_file=CACHE_FILE, notify_fn=notify
    )
    # Prepare payload
    payload = {
        'representatives': results['representatives'],
        'clusters': results['clusters'],
        'clustering_algo': results['clustering_algo'],
        'correlation_matrix': results['correlation_matrix'].to_dict(),
        'top_positive': results['top_positive'].to_dict(),
        'top_negative': results['top_negative'].to_dict(),
        'eda_plots': results['eda_plots']
    }
    notify('pipeline_complete', payload)

# -----------------------------
# WebSocket events
# -----------------------------
@socketio.on('connect')
def on_connect():
    emit('message', {'data': 'Connected to Crypto API'})

@socketio.on('run_pipeline')
def on_run_pipeline(_json):
    socketio.start_background_task(_pipeline_task)

# -----------------------------
# Main entry
# -----------------------------
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
